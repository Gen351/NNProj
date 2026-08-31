# UCI Protocol & NNProj Engine CLI Cheat Sheet

A quick reference guide for interacting with the `NNProj` C++ Chess Engine via standard input (`stdin`) and command-line arguments.

---

## 🚀 1. Command-Line Execution Modes

Run the engine from your terminal using various flags and parameters.

| Launch Command | Description |
| :--- | :--- |
| `./chess_engine` | Starts the engine in default interactive UCI mode. |
| `./chess_engine --net model.backprop_net` | Loads your trained neural network binary at startup. |
| `./chess_engine --perft 5` | Runs perft (move generation count verification) up to depth 5 from the standard start position. |
| `./chess_engine --perft 4 "FEN_STRING"` | Runs perft verification on a custom FEN position up to depth 4. |
| `./chess_engine --selftest` | Executes internal integrity tests (Zobrist hash consistency, perft test suite, sample self-game). |
| `./chess_engine --help` | Displays the help menu and available CLI flags. |

---

## ⚙️ 2. Core UCI Initialization Commands

Commands used by chess GUIs (e.g., Lichess-Bot, Arena, Cutechess) to handshake and configure options.

| UCI Command | Description | Example Engine Response |
| :--- | :--- | :--- |
| `uci` | Requests engine identification and available custom parameters. | `id name NNProj Chess 0.1` ... `uciok` |
| `isready` | Synchronizes engine readiness (waits for background threads to finish loading). | `readyok` |
| `setoption name EvalFile value <path>` | Dynamically loads a new neural network model path during run-time. | `info string loaded eval: nnue-acc:models/v1.backprop_net (win-prob)` |
| `setoption name EvalQuant value <0\|1>` | Sets whether the next EvalFile load quantizes the accumulator's int16 first layer (AVX2-ready NNUE path). Default 0. | `info string EvalQuant set to 1 (applies on next EvalFile load)` |
| `setoption name Threads value <N>` | Sets the number of search threads (Lazy-SMP concurrency, all sharing one transposition table). | `info string Threads set to 8` |
| `setoption name Hash value <MB>` | Stops any running search, then resizes the shared transposition table (default 64 MB, range 4-4096). Clears its contents. | `info string Hash set to 128 MB` |
| `ucinewgame` | Clears the transposition table and resets internal state for a new game session. | *(No output expected)* |
| `quit` | Stops all active background thread pools and exits the program cleanly. | *(Exits program)* |

---

## ♟️ 3. Setting Up Positions (`position`)

Sets the internal game state board before invoking search routines.

```uci
# Set to default starting position
position startpos

# Set to default starting position and apply a list of moves (UCI algebraic format)
position startpos moves e2e4 e7e5 g1f3 b8c6

# Set position from a FEN string
position fen r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3

# Set position from a FEN string AND apply subsequent moves
position fen r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3 moves f8c5 c2c3
```

> **Warm-up:** as soon as a `position` is set, the engine silently starts a
> background search (single-threaded, capped at depth 9) that fills the shared
> transposition table — no `info`/`bestmove` output is emitted. A later `go`
> on the same position reuses that work and searches much faster. `stop`,
> `position`, `go`, `ucinewgame`, `setoption`, and `quit` all abort it cleanly.

---

## 🎯 4. Executing Search & Moving (`go`)

Triggers the searcher thread to evaluate legal moves and return the optimal action (`bestmove`).

### Standard Search Constraints
```uci
# Search up to a fixed depth limit
go depth 6

# Search for a fixed duration in milliseconds
go movetime 2000

# Search until maximum node count target is hit
go nodes 1000000

# Search indefinitely until a 'stop' command is explicitly issued
go infinite
```

### Clock & Time Control Parameters
```uci
# Standard clock controls (time and increment given in milliseconds)
go wtime 300000 btime 300000 winc 2000 binc 2000

# Search under time controls with a specific number of moves remaining until time control
go wtime 120000 btime 120000 movestogo 10
```

### Pondering (Opponent's Turn Calculation)
```uci
# Start searching speculatively on opponent's expected turn
go ponder wtime 300000 btime 300000

# Opponent made the expected move -> convert ponder into normal active search
ponderhit
```

---

## 🛑 5. Search Control & Debugging

Commands to inspect or interrupt internal engine routines.

| Command | Category | Description |
| :--- | :--- | :--- |
| `stop` | **Control** | Immediately interrupts the ongoing `go` search loop and forces the engine to output `bestmove`. Also silently aborts a background warm-up search (no `bestmove`). |
| `d` or `debug` | **Debug** | Prints an ASCII representation of the current board, fullmove count, halfmove clock, game status, and total legal move count. |
| `perft <depth>` | **Debug** | Runs interactive move-generation verification count up to `<depth>` on the current board state. |

---

## 🔄 6. Complete Session Example

Below is a trace of a complete interaction sequence for playing a move:

```text
>>> uci
<<< id name NNProj Chess 0.1
<<< id author NNProj
<<< option name EvalFile type string default
<<< option name Hash type spin default 64 min 4 max 4096
<<< option name Threads type spin default 6 min 1 max 64
<<< option name EvalQuant type spin default 0 min 0 max 1
<<< uciok

>>> setoption name EvalFile value models/v1.backprop_net
<<< info string loaded eval: nnue-acc:models/v1.backprop_net (win-prob)

>>> setoption name Hash value 128
<<< info string Hash set to 128 MB

>>> isready
<<< readyok

>>> position startpos moves e2e4 e7e5
    (engine silently warms the TT on this position -- no output)
>>> go depth 6
<<< info depth 1 score cp 25 nodes 32 time 1 pv g1f3
<<< info depth 2 score cp 30 nodes 128 time 3 pv g1f3 b8c6
...
<<< info depth 6 score cp 35 nodes 45210 time 120 pv g1f3 b8c6 f1b5
<<< bestmove g1f3

>>> quit
```
