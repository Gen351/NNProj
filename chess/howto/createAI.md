# createAI.md -- How your neural network becomes the chess engine's brain

## The big picture (read this first)

The engine is a **search** wrapped around **your network**:

```
            ChessGame (rules: legal moves, FEN, state encoding)
                 |
            Searcher  <-- "if I play e2e4, then he plays g8f6, ..."
                 |          tries millions of move sequences (alpha-beta)
                 v
        Evaluator  <-- scores ONE position with a single number
                  |          (this is where YOUR net lives)
        material+pst  OR   NetEvaluator(your .backprop_net)
```

A common misconception to clear up first: **the neural network does NOT pick
moves.** It never sees "list of moves" and never outputs a move.

- Your net answers exactly one question: *"given this board, how good is it
  for White, in centipawns?"* (+100 ~ white is up a pawn).
- The search asks that question about the positions after each candidate line,
  and picks the move that leads to the best worst-case score (minimax).

So your job as the AI creator is only this: **train a function
`781 floats -> 1 float`** that maps a position to its value. The engine does
everything else. This split is exactly how real engines (Stockfish + NNUE)
work.

## The input contract: 781 floats

`ChessGame::getState()` / `getState(Position&, std::vector<float>&)` produces
the vector your net receives, always in this exact layout:

| indices       | meaning                                                        |
|---------------|----------------------------------------------------------------|
| `0 .. 767`    | 12 one-hot piece planes. Index = `plane*64 + square`.           |
|               | planes 0..5  = white P, N, B, R, Q, K                          |
|               | planes 6..11 = black P, N, B, R, Q, K                          |
|               | squares are `a1=0 .. h8=63` (`file = sq & 7`, `rank = sq >> 3`)|
|               | value 1.0 where that piece stands, 0 elsewhere                  |
| `768`         | side to move: +1 if White, -1 if Black                          |
| `769..772`    | castling rights, one-hot: [WK] [WQ] [BK] [BQ], 1.0 if right exists |
| `773..780`    | en-passant FILE one-hot: index `773+file` (a=0..h=7), all zeros if no ep square |

Total: `12*64 + 1 + 4 + 8 = 781`. The engine **rejects any net whose first
Dense layer doesn't take exactly 781 inputs** at load time, so a size bug is
caught immediately rather than silently corrupting play.

## The output contract: one number, White's point of view

Your net must end with a Dense layer producing exactly **1 output**, and that
number is interpreted as centipawns from White's perspective (positive =
White better; it means the same thing regardless of who is to move).

Recommended recipe if you train with squashing activations: make the last
layer a `Dense(1)` **without activation** acting as an affine scaler on top of
a TANH stack:

```
... -> TANH(hidden) -> Dense(1, no activation)      weight ~= 8000, bias ~= 0
```

That maps the tanh range [-1, 1] linearly onto [-8000, +8000] cp. Because the
final layer has no nonlinearity, gradient descent can also learn to adjust the
scale itself during training -- the fixed 8000 is just a sane starting point.

Notes:
- Mate scores in the search are +-30000; your eval gets clamped to
  +-29000, so you never need to output anything larger than that.
- NaN/inf outputs are treated as 0 by a safety guard.
- The engine calls this function *millions of times per second*, so smaller /
  cheaper nets search deeper. A tiny `781 -> H -> TANH -> 1` often beats a big
  one purely through extra search depth.

## Plugging your net in

Three ways, all equivalent:

```
./chess_engine --net path/to/net.backprop_net          # command line
setoption name EvalFile value path/to/net.backprop_net # inside UCI session
Engine::setEvaluator(loadNetEvaluator(path))         # from C++ code
```

On success you'll see: `info string loaded eval: nnue-acc:path/...` (the load
compiles your net into the NNUE-style sparse-accumulator evaluator; option
`EvalQuant 1` uses the quantized int16/int32 accumulator, and compiling with
`-mavx2` runs that path on AVX2 SIMD).
On failure the engine keeps running with the material+pst fallback and prints
an error explaining exactly what was wrong, e.g.:

- `input size mismatch: expected 781 ... got N` -- fix your first layer.
- `must end with a Dense layer producing exactly 1 output` -- add/fix the head.
- File unreadable / bad format -- BackpropNetReader threw; check the path
  (`BackpropNetReader::load` appends `.backprop_net` itself when used
  internally; here you give the full file name).

## What to train on

You do NOT need self-play games to start. Good escalating options:

1. **Distill the built-in evaluator.** Generate random-ish positions, label
   each with the material+pst score, train your net to match. Cheap,
   supervised, and guarantees your net is at least PST-strength.
   **The bundled trainer does this for you** -- see the next section.
2. **Human/game database labels.** Lichess publishes its analysis database
   (Stockfish centipawn evaluations for millions of positions). Regress your
   net onto those evals -- this is classic supervised evaluation training and
   much stronger than PST. The trainer reads these dumps directly:

   ```
   ./chess_trainer DATASET=dataset/200k_blitz_rapid_classical_bullet.csv \
                   CONTINUE=trained_networks/run_1/net.backprop_net \
                   SAMPLES=1000000 EPOCHS=8 LR=0.0005
   ```

   Each row of the CSV is one game: SAN moves + White-POV evals per ply. The
   CSV stores evals in **PAWNS** (e.g. `0.25` = +25cp; `"#N"` = mate scores,
   dropped by default via SKIP_MATES=1); the trainer multiplies by 100 to get
   centipawns at load time. It replays every game with its own move generator
   (see dataset_tools.h), samples state/label pairs, mirrors each position for
   free symmetry data, and randomly selects games across the WHOLE file so a
   SAMPLES cap stays unbiased. Expect >99% of games to replay cleanly; anything
   that fails to parse is dropped whole and counted in the load stats.
3. **Self-play results.** Play your current net against the previous one with
   the engine's own search (short time controls), evolve weights the same way
   the snakes trainer does. Slow but closes the loop on "eval good == wins
   games".

Whatever the source, remember the engine/evaluator contract is scores in
centipawns, White POV. See "Label normalization" below -- the trainer does NOT
regress raw centipawns anymore.

## The bundled trainer (chess/train.cpp)

`chess_trainer` distills the material+pst evaluator into a `BackpropNet::Net`
end-to-end: it generates capture-biased random playouts, labels every position
with the PST evaluator (plus a color-mirrored copy of each position), and
trains with `BackpropNet::Net::train_v2` (Adam, minibatch SGD, per-epoch
validation checkpoints, LR decay on plateau).

```
-- compile (from chess/):
g++ -std=c++17 -DNDEBUG -O3 -pthread train.cpp -o chess_trainer

-- train fresh (writes trained_networks/run_N/net.backprop_net):
./chess_trainer SAMPLES=50000 EPOCHS=40 BATCH=32

-- resume / finetune later (e.g. onto Lichess evals):
./chess_trainer CONTINUE=trained_networks/run_1/net.backprop_net EPOCHS=10 LR=0.0003

-- sanity probes (startpos ~ 0, queen-up positive, mirror flips sign):
./chess_trainer --probe NET=trained_networks/run_1/net.backprop_net
```

Tunables (`KEY=VALUE`, see `./chess_trainer --help`): `SAMPLES EPOCHS BATCH
LR PATIENCE CLAMP PST_SCALE SEED MIRROR CAPTURE_BIAS MAX_PLIES OUT`.

## Label normalization (win-probability targets)

The trainer no longer regresses raw centipawns. Instead targets go through
`tanh(cp / S)` with `S = 400` (see `cpWinProb()`/`winProbToCp()` in
`evaluator.h`):

- Labels are clamped to `+-CLAMP` (default 2500) centipawns, then squashed to
  `(-1, 1)` with tanh. Near 0 the map is ~1:1, so opening positions keep their
  ordering; the noisy, heavy tails of real datasets (Stockfish evals run 2-4x
  hotter than material/PST) saturate instead of owning the MSE gradient.
- **The net's architecture ends with a TANH activation** (the output head is
  `Dense -> TANH`), so its output natively matches the `(-1,1)` target scale
  and the MSE gradient stays well-scaled for every sample. Without this, raw-MSE
  training regresses the whole net toward ~0 (predicting "draw" everywhere),
  which shows up as a stalled plateau and timid shuffle-draw play.
- At eval time `NetEvaluator` auto-detects the trailing TANH and inverts the
  bounded output back to centipawns via `S * atanh(y)`, so the engine/search
  still sees a normal centipawn score. Legacy linear-head nets are still
  detected and evaluated as raw centipawns (no retraining on them needed).
- Because targets are bounded, `Val MSE` is reported in win-prob units
  (expect ~0.01-0.05 after a good fit -- NOT comparable to the old raw-cp MSE).
- `CLAMP` bounds inputs; `PST_SCALE` (PST mode only) multiplies the teacher's
  centipawns by 1.5-2.0 before the tanh transform to approximate the hotter
  Stockfish magnitudes on the eval CSV -- the magnitude gap between material/PST
  and a good engine is nonlinear (largest near 0).

`create_architecture()` seeds the final head weights small (`+-0.5`) so the
pre-activation starts near 0 and the final tanh begins in its sensitive region;
keep that property if you edit the architecture or the head starts saturated
and training stalls at epoch 0.

## Testing ladder (do these in order)

1. `./chess_engine --selftest` -- rules/zobrist/perft sanity (no net needed).
2. `./chess_test` -- dataset-tools unit tests (CSV parsing, SAN replay,
   eval alignment, mirror augmentation). Required before DATASET= training.
2. Play vs the PST fallback: load your net, run
   `go movetime 200` twice from `position startpos` swapping sides, or just
   eyeball that it prefers reasonable developing/capturing moves at low depth.
3. Sanity probes: `./chess_trainer --probe NET=path/to/net` (or a small C++
   test harness using `getState()` + `net.predict()` directly):
   - startpos should evaluate near 0;
   - white up a queen should be strongly positive;
   - the mirror of a position should flip the sign (your net must understand
     both colors -- the side-to-move plane matters!).
4. Strength match: net-vs-PST at equal fixed depth (e.g. `go depth 6`), ~20
   games alternating colors. If your net can't beat PST, keep training --
   don't tune the search.
5. When it's strong: plug into a real GUI (any UCI app) via
   `chess_engine --net your.backprop_net`, or an online bot bridge such as
   lichess-bot, which talks UCI to this exact executable.

## Quick reference for C++ side integration

```cpp
#include "../nn_engine/backprop_net/backprop_net.h"
#include "../nn_engine/backprop_net/backprop_net_reader.h"
#include "chess_environment.h"
#include "evaluator.h"

ChessGame game;                                  // rules + state encoding
game.setFen(chess::kStartFen);

auto eval = chess::loadNetEvaluator("best.backprop_net");  // throws w/ message
std::vector<float> state;
chess::getState(game.pos, state);                // the 781 floats
// ... train / predict against `state`, target = centipawns (White POV)
```
