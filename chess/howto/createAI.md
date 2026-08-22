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
        material+pst  OR   NetEvaluator(your .simple_net)
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
./chess_engine --net path/to/net.simple_net          # command line
setoption name EvalFile value path/to/net.simple_net # inside UCI session
Engine::setEvaluator(loadNetEvaluator(path))         # from C++ code
```

On success you'll see: `info string loaded eval: net:path/...`.
On failure the engine keeps running with the material+pst fallback and prints
an error explaining exactly what was wrong, e.g.:

- `input size mismatch: expected 781 ... got N` -- fix your first layer.
- `must end with a Dense layer producing exactly 1 output` -- add/fix the head.
- File unreadable / bad format -- SimpleNetReader threw; check the path
  (`SimpleNetReader::load` appends `.simple_net` itself when used internally;
  here you give the full file name).

## What to train on

You do NOT need self-play games to start. Good escalating options:

1. **Distill the built-in evaluator.** Generate random-ish positions, label
   each with the material+pst score, train your net to match. Cheap,
   supervised, and guarantees your net is at least PST-strength.
2. **Human/game database labels.** Lichess publishes its analysis database
   (Stockfish centipawn evaluations for millions of positions). Regress your
   net onto those evals -- this is classic supervised evaluation training and
   much stronger than PST.
3. **Self-play results.** Play your current net against the previous one with
   the engine's own search (short time controls), evolve weights the same way
   the snakes trainer does. Slow but closes the loop on "eval good == wins
   games".

Whatever the source, remember the target units are centipawns, White POV.

## Testing ladder (do these in order)

1. `./chess_engine --selftest` -- rules/zobrist/perft sanity (no net needed).
2. Play vs the PST fallback: load your net, run
   `go movetime 200` twice from `position startpos` swapping sides, or just
   eyeball that it prefers reasonable developing/capturing moves at low depth.
3. Sanity probes: feed positions via a small C++ test harness using
   `getState()` + `net->predict()` directly:
   - startpos should evaluate near 0;
   - white up a queen should be strongly positive;
   - the mirror of a position should flip the sign (your net must understand
     both colors -- the side-to-move plane matters!).
4. Strength match: net-vs-PST at equal fixed depth (e.g. `go depth 6`), ~20
   games alternating colors. If your net can't beat PST, keep training --
   don't tune the search.
5. When it's strong: plug into a real GUI (any UCI app) via
   `chess_engine --net your.net`, or an online bot bridge such as lichess-bot,
   which talks UCI to this exact executable.

## Quick reference for C++ side integration

```cpp
#include "../nn_engine/simple_net/simple_net.h"
#include "chess_environment.h"
#include "evaluator.h"

ChessGame game;                                  // rules + state encoding
game.setFen(chess::kStartFen);

auto eval = chess::loadNetEvaluator("best.simple_net");  // throws w/ message
std::vector<float> state;
chess::getState(game.pos, state);                // the 781 floats
// ... train / predict against `state`, target = centipawns (White POV)
```
