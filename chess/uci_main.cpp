// NNProj chess engine entry point.
//
// Modes:
//   chess_engine                        -- UCI protocol on stdin/stdout (default)
//   chess_engine --net model.simple_net -- UCI with your trained net loaded at startup
//   chess_engine --perft 5 ["FEN"]      -- move-generation correctness counts
//   chess_engine --selftest             -- internal consistency checks + quick game
//
// UCI compatibility: works with lichess-bot, Arena, Banksia, cutechess, ...

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "./chess_environment.h"
#include "./evaluator.h"
#include "./search.h"

namespace {

constexpr const char* kEngineName = "NNProj Chess";
constexpr const char* kEngineVersion = "0.1";
constexpr const char* kStartFen =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// ---------------------------------------------------------------------------
// Perft (node counting -- the standard proof that move generation is exact)
// ---------------------------------------------------------------------------
long long perft(const chess::Position& p, const int depth) {
    if (depth <= 0) return 1;
    const std::vector<chess::Move> moves = chess::genLegalMoves(p);
    if (depth == 1) return static_cast<long long>(moves.size());
    long long n = 0;
    for (const chess::Move& m : moves) {
        chess::Position child = p;
        chess::applyMove(child, m);
        n += perft(child, depth - 1);
    }
    return n;
}

// ---------------------------------------------------------------------------
// Selftest
// ---------------------------------------------------------------------------
int g_failures = 0;

void check(const bool ok, const std::string& what) {
    if (!ok) {
        ++g_failures;
        std::cerr << "  FAIL: " << what << "\n";
    }
}

void testZobristConsistency(std::mt19937& rng) {
    std::cerr << "[1] Zobrist incremental-hash consistency (random playouts)\n";
    constexpr int GAMES = 60, PLIES = 150;
    for (int g = 0; g < GAMES; ++g) {
        chess::ChessGame game;
        for (int ply = 0; ply < PLIES; ++ply) {
            check(game.pos.hash == chess::computeHash(game.pos),
                  "hash drift at game " + std::to_string(g) + " ply " + std::to_string(ply));
            const std::vector<chess::Move> moves = game.legalMoves();
            if (moves.empty()) break;
            std::uniform_int_distribution<size_t> pick(0, moves.size() - 1);
            game.make(moves[pick(rng)]);
        }
    }
    std::cerr << "    done\n";
}

void testPerftSuite() {
    struct Case { const char* name; const char* fen; int depth; long long expected; };
    static const Case CASES[] = {
        {"startpos  ", kStartFen, 1, 20},
        {"startpos  ", kStartFen, 2, 400},
        {"startpos  ", kStartFen, 3, 8902},
        {"startpos  ", kStartFen, 4, 197281},
        {"kiwipete  ", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1, 48},
        {"kiwipete  ", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2, 2039},
        {"kiwipete  ", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862},
        {"kiwipete  ", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603},
        {"pos3      ", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 1, 14},
        {"pos3      ", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 2, 191},
        {"pos3      ", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3, 2812},
        {"pos3      ", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4, 43238},
        {"pos4      ", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 1, 6},
        {"pos4      ", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 2, 264},
        {"pos4      ", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3, 9467},
        {"pos4      ", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333},
        {"pos5      ", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 1, 44},
        {"pos5      ", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 2, 1486},
        {"pos5      ", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379},
        {"pos6      ", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 1, 46},
        {"pos6      ", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 2, 2079},
        {"pos6      ", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3, 89890},
    };
    std::cerr << "[2] Perft suite (" << std::size(CASES) << " cases)\n";
    for (const Case& c : CASES) {
        const chess::ChessGame game(c.fen);
        const auto t0 = std::chrono::steady_clock::now();
        const long long got = perft(game.pos, c.depth);
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0).count();
        std::ostringstream label;
        label << c.name << " d" << c.depth
              << ": got " << got << " expected " << c.expected
              << " (" << ms << " ms)";
        check(got == c.expected, label.str());
        std::cerr << (got == c.expected ? "    PASS " : "    ") << label.str() << "\n";
    }
}

void testEngineGame() {
    std::cerr << "[3] Quick material+PST self-game (depth 3)\n";
    chess::MaterialPSTEvaluator eval;
    chess::Searcher searcher(eval);

    chess::ChessGame game;
    chess::SearchLimits lim;
    lim.depthLimit = 3;

    std::vector<std::string> uciMoves;
    for (int ply = 0; ply < 140; ++ply) {
        const chess::GameStatus st = game.status();
        if (st.result != chess::ONGOING) {
            std::cerr << "    game ended: " << st.reason << "\n";
            break;
        }
        const chess::SearchResult r = searcher.think(game, lim);
        if (!r.hasMove) break;

        // Every engine move must be one of the generated legal moves.
        bool legal = false;
        for (const chess::Move& m : game.legalMoves())
            if (m == r.best) { legal = true; break; }
        check(legal, "search returned an illegal move");

        uciMoves.push_back(chess::moveToUci(r.best));
        game.make(r.best);
    }

    std::cerr << "    moves:";
    for (size_t i = 0; i < uciMoves.size(); ++i)
        std::cerr << ((i % 2 == 0) ? " " + std::to_string(i / 2 + 1) + "." : "") << uciMoves[i];
    std::cerr << "\n    final fen: " << game.fen() << "\n";
    check(uciMoves.size() >= 10, "engine should produce a reasonable opening");
}

int runSelftest() {
    std::mt19937 rng(42);
    testZobristConsistency(rng);
    testPerftSuite();
    testEngineGame();

    std::cerr << (g_failures == 0 ? "SELFTEST PASSED\n" : "SELFTEST FAILED\n");
    return g_failures == 0 ? 0 : 1;
}

void runPerftCli(const std::string& fen, const int maxDepth) {
    const chess::ChessGame game(fen);
    for (int d = 1; d <= maxDepth; ++d) {
        const auto t0 = std::chrono::steady_clock::now();
        const long long n = perft(game.pos, d);
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0).count();
        std::cout << "perft " << d << " = " << n << " (" << ms << " ms)\n";
    }
}

// ---------------------------------------------------------------------------
// UCI engine
// ---------------------------------------------------------------------------
struct Engine {
    chess::ChessGame game;
    std::shared_ptr<chess::Evaluator> evaluator = chess::makePstEvaluator();
    std::unique_ptr<chess::Searcher> searcher;
    std::thread worker;
    std::atomic<bool> searching{false};

    chess::SearchLimits ponderRealLimits;   // clocks saved from "go ... ponder"
    bool pondering = false;

    void joinSearch() {
        pondering = false;
        if (worker.joinable()) {
            if (searcher) searcher->stop();
            worker.join();
        }
        searching.store(false);
    }

    void launch(chess::SearchLimits lim) {
        joinSearch();
        searcher = std::make_unique<chess::Searcher>(*evaluator);
        searcher->onInfo = [](const chess::SearchInfo& i) {
            std::ostringstream out;
            out << "info depth " << i.depth << " score ";
            if (i.mate) out << "mate " << i.mateIn;
            else        out << "cp " << static_cast<long long>(i.scoreCp);
            out << " nodes " << i.nodes << " time " << i.timeMs << " pv";
            for (const chess::Move& m : i.pv) out << ' ' << chess::moveToUci(m);
            std::cout << out.str() << std::endl;
        };
        searching.store(true);
        worker = std::thread([this, lim] {
            const chess::SearchResult r = searcher->think(game, lim);
            std::cout << (r.hasMove ? "bestmove " + chess::moveToUci(r.best)
                                    : "bestmove (none)") << std::endl;
            searching.store(false);
        });
    }
};

bool tryLoadNet(Engine& eng, const std::string& path) {
    try {
        auto ev = chess::loadNetEvaluator(path);
        eng.joinSearch();
        eng.evaluator = std::move(ev);
        std::cout << "info string loaded eval: " << eng.evaluator->name() << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cout << "info string ERROR loading '" << path << "': " << e.what()
                  << " -- keeping previous evaluator" << std::endl;
        return false;
    }
}

chess::SearchLimits parseGo(std::istringstream& in, bool& ponder, chess::SearchLimits& saved) {
    using namespace chess;
    SearchLimits lim;
    ponder = false;
    std::string tok;
    while (in >> tok) {
        if (tok == "wtime")       in >> lim.wtime;
        else if (tok == "btime")  in >> lim.btime;
        else if (tok == "winc")   in >> lim.winc;
        else if (tok == "binc")   in >> lim.binc;
        else if (tok == "movestogo") in >> lim.movestogo;
        else if (tok == "movetime")  in >> lim.movetime;
        else if (tok == "depth")     in >> lim.depthLimit;
        else if (tok == "nodes")     in >> lim.nodesLimit;
        else if (tok == "infinite")  lim.infinite = true;
        else if (tok == "ponder")    ponder = true;
        // "ponderhit"/others silently ignored here
    }
    saved = lim;   // raw clocks kept for ponderhit
    if (ponder) lim.infinite = true;   // ponder: think until stop/ponderhit
    return lim;
}

int uciLoop(const std::string& startupNetPath) {
    Engine eng;
    if (!startupNetPath.empty()) tryLoadNet(eng, startupNetPath);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream in(line);
        std::string cmd;
        if (!(in >> cmd)) continue;

        if (cmd == "uci") {
            std::cout << "id name " << kEngineName << ' ' << kEngineVersion << "\n"
                      << "id author NNProj\n"
                      << "option name EvalFile type string default\n"
                      << "uciok" << std::endl;
        } else if (cmd == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (cmd == "setoption") {
            std::string tok, name, value;
            in >> tok;                       // "name"
            in >> name;
            in >> tok;                       // "value" (may be absent)
            std::getline(in, value);
            if (!value.empty() && value.front() == ' ') value.erase(0, 1);
            if (name == "EvalFile" && !value.empty()) tryLoadNet(eng, value);
        } else if (cmd == "ucinewgame") {
            eng.joinSearch();
            eng.game.reset();
        } else if (cmd == "position") {
            eng.joinSearch();
            std::string tok;
            in >> tok;
            if (tok == "startpos") {
                eng.game.reset();
                in >> tok;                   // optional "moves"
            } else if (tok == "fen") {
                std::string fenPart, fen;
                while (in >> fenPart && fenPart != "moves") fen += fenPart + ' ';
                if (!fen.empty()) fen.pop_back();
                try {
                    eng.game.setFen(fen.empty() ? kStartFen : fen);
                } catch (const std::exception& e) {
                    std::cout << "info string bad fen: " << e.what() << std::endl;
                    eng.game.reset();
                }
                tok = fenPart;               // "moves" or consumed
            }
            if (tok == "moves") {
                while (in >> tok) {
                    chess::Move m;
                    if (chess::parseUciMove(tok, eng.game.pos, m)) eng.game.make(m);
                    else std::cout << "info string ignoring illegal/unknown move: "
                                   << tok << std::endl;
                }
            }
        } else if (cmd == "go") {
            bool ponder = false;
            chess::SearchLimits saved;
            const chess::SearchLimits lim = parseGo(in, ponder, saved);
            if (ponder) {
                eng.ponderRealLimits = saved;
                eng.pondering = true;
            }
            eng.launch(lim);
        } else if (cmd == "ponderhit") {
            if (eng.searching.load() && eng.pondering) {
                eng.pondering = false;
                const chess::SearchLimits real = eng.ponderRealLimits;
                eng.launch(real);   // re-launch with the real clock budget
            }
        } else if (cmd == "stop") {
            if (eng.searching.load() && eng.searcher) eng.searcher->stop();
        } else if (cmd == "quit") {
            eng.joinSearch();
            return 0;
        } else if (cmd == "d" || cmd == "debug") {
            const chess::GameStatus st = eng.game.status();
            std::cerr << chess::boardString(eng.game.pos)
                      << "fen: " << eng.game.fen()
                      << "\nhalfmove: " << eng.game.pos.halfmove
                      << "  fullmove: " << eng.game.fullmove
                      << "  status: " << st.reason << (st.reason.empty() ? "ongoing" : "")
                      << "\nlegal moves: " << eng.game.legalMoves().size() << "\n";
        } else if (cmd == "perft") {
            int depth = 1;
            in >> depth;
            const auto t0 = std::chrono::steady_clock::now();
            const long long n = ::perft(eng.game.pos, depth);
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - t0).count();
            std::cout << "info string perft(" << depth << ") = " << n
                      << " (" << ms << " ms)" << std::endl;
        }
        // Unknown commands are ignored per the UCI spec.
    }
    eng.joinSearch();
    return 0;
}

void usage() {
    std::cout
        << kEngineName << ' ' << kEngineVersion << "\n\n"
        << "Usage:\n"
        << "  chess_engine                          UCI mode (default; pipe commands via stdin)\n"
        << "  chess_engine --net FILE.simple_net    UCI mode with your trained net preloaded\n"
        << "  chess_engine --perft DEPTH [\"FEN\"]    perft node counts (default: startpos)\n"
        << "  chess_engine --selftest               rules/search consistency checks\n";
}

}   // namespace

int main(int argc, char* argv[]) {
    std::ios::sync_with_stdio(false);

    std::string netPath, fen = kStartFen;
    int perftDepth = -1;
    bool selftest = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            return (i + 1 < argc) ? argv[++i] : "";
        };
        if (arg == "--net")          netPath = value();
        else if (arg == "--perft")   perftDepth = std::atoi(value().c_str());
        else if (arg == "--fen")     fen = value();
        else if (arg == "--selftest") selftest = true;
        else if (arg == "--help" || arg == "-h") { usage(); return 0; }
        else                          fen = arg;   // bare arg = FEN (quoted)
    }

    if (selftest)              return runSelftest();
    if (perftDepth > 0)        { runPerftCli(fen, perftDepth); return 0; }
    return uciLoop(netPath);
}
