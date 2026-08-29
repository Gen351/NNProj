// self_play.cpp -- pitch two chess evaluators against each other autonomously.
//
// Usage (from chess/):
//   ./self_play NET1=path NET2=path [DEPTH=N | MOVETIME=ms] [MAX_PLIES=N]
//
// NET1 plays White, NET2 plays Black. Either may be the literal value PST to
// use the built-in material+pst evaluator; otherwise it must be a
// .backprop_net file (see howto/createAI.md). If both DEPTH and MOVETIME are
// given, the one written LATER on the command line wins.
//
// Each turn the players alternate: the side to move searches with the Searcher
// under the chosen limit, plays its best move, and the loop repeats until a
// game-ending status (or MAX_PLIES safety cap). Output mimics the engine's
// UCI trace: board ("d"), the chosen move recorded as "position moves ...",
// then the winner and the full move history.

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "./chess_environment.h"
#include "./evaluator.h"
#include "./search.h"
#include "./transposition_table.h"

namespace {

struct Config {
    std::string net1;        // White
    std::string net2;        // Black
    bool use_time = true;    // false => fixed depth
    long long movetime = 1000;
    int depth = -1;
    float contempt = 50.0f;  // draw contempt in cp (see Searcher::setDrawContempt)
    size_t max_plies = 400;
    int threads = 1;         // Lazy-SMP searchers per move (1 = single-threaded)
    size_t hash_mb = 64;     // shared transposition-table size
};

struct Player {
    std::string label;
    std::shared_ptr<chess::Evaluator> eval;      // primary (White/Black) evaluator
    // Builds a fresh, self-contained evaluator for each helper thread.
    // Needed because NetEvaluator keeps a per-instance state buffer and is
    // only safe when one thread owns each instance (targets the same shared net).
    std::function<std::shared_ptr<chess::Evaluator>()> factory;
};

void print_usage() {
    std::cout
        << "NNProj Chess Self-Play\n\n"
        << "Pitches two evaluators against each other. NET1 plays White, NET2 Black.\n"
        << "Either may be 'PST' for the built-in material+pst evaluator, or a path to\n"
        << "a .backprop_net file. If both DEPTH and MOVETIME are passed, the one\n"
        << "written later on the command line wins.\n\n"
        << "Compile (from chess/):\n"
        << "  g++ -std=c++17 -DNDEBUG -O3 -pthread self_play.cpp -o self_play\n\n"
        << "Usage:\n"
        << "  ./self_play NET1=path/to/net.backprop_net NET2=PST DEPTH=6\n"
        << "  ./self_play NET1=run_1/net.backprop_net NET2=run_2/net.backprop_net MOVETIME=3000\n\n"
        << "Options (KEY=VALUE):\n"
        << "  NET1=path        White player (file or PST)\n"
        << "  NET2=path        Black player (file or PST)\n"
        << "  DEPTH=N          search to a fixed ply depth\n"
        << "  MOVETIME=ms      search for a fixed number of milliseconds\n"
        << "                   (default when neither is given: MOVETIME=1000)\n"
        << "  CONTEMPT=cp      draw-avoidance penalty (default 50): a root move that\n"
        << "                   would immediately draw (3rd repetition, fifty-move,\n"
        << "                   insufficient material, stalemate) is compared as if it\n"
        << "                   scored 2*cp worse, so repeats into draws are avoided\n"
        << "  THREADS=N        Lazy-SMP searchers per move around a shared TT\n"
        << "                   (default 1 = single-threaded; each helper gets its own\n"
        << "                   NetEvaluator around the same shared net)\n"
        << "  HASH=MB          shared transposition-table size (default 64)\n"
        << "  MAX_PLIES=N      safety cap (default 400) to avoid endless draws\n";
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { print_usage(); std::exit(0); }

        const size_t eq = arg.find('=');
        if (eq == std::string::npos)
            throw std::runtime_error("parse_args: bad argument '" + arg
                                     + "' (expected KEY=VALUE)");
        const std::string key = arg.substr(0, eq);
        std::string val = arg.substr(eq + 1);
        if (val.size() >= 2 && val.front() == '"' && val.back() == '"')
            val = val.substr(1, val.size() - 2);

        if      (key == "NET1")    cfg.net1 = val;
        else if (key == "NET2")    cfg.net2 = val;
        else if (key == "DEPTH")   { cfg.depth = std::stoi(val); cfg.use_time = false; }
        else if (key == "MOVETIME") { cfg.movetime = std::stoll(val); cfg.use_time = true; }
        else if (key == "CONTEMPT") cfg.contempt = std::stof(val);
        else if (key == "THREADS")  cfg.threads = std::stoi(val);
        else if (key == "HASH")     cfg.hash_mb = std::stoull(val);
        else if (key == "MAX_PLIES") cfg.max_plies = std::stoull(val);
        else throw std::runtime_error("parse_args: unknown option '" + key + "'");
    }
    if (cfg.net1.empty() || cfg.net2.empty())
        throw std::runtime_error("parse_args: both NET1= and NET2= are required");
    if (!cfg.use_time && cfg.depth < 1)
        throw std::runtime_error("parse_args: DEPTH must be >= 1");
    if (cfg.use_time && cfg.movetime < 1)
        throw std::runtime_error("parse_args: MOVETIME must be >= 1");
    if (cfg.threads < 1)
        throw std::runtime_error("parse_args: THREADS must be >= 1");
    return cfg;
}

std::string base_name(const std::string& path) {
    return std::filesystem::path(path).filename().string();
}

Player make_player(const std::string& spec) {
    Player p;
    std::string lower = spec;
    for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (lower == "pst") {
        p.label = "material+pst";
        p.eval = chess::makePstEvaluator();
        // PST is stateless, so every helper can evaluator independently too.
        p.factory = [] { return chess::makePstEvaluator(); };
    } else {
        // Load the net ONCE; helpers wrap the same read-only net in their own
        // NetEvaluator (per-thread state buffers, safe during search).
        auto net = std::make_shared<BackpropNet::Net>(BackpropNetReader::load(spec));
        p.eval = std::make_shared<chess::NetEvaluator>(net, spec);
        p.label = base_name(spec);
        p.factory = [net, spec] {
            return std::shared_ptr<chess::Evaluator>(
                std::make_shared<chess::NetEvaluator>(net, spec));
        };
    }
    return p;
}

chess::SearchLimits make_limits(const Config& cfg) {
    chess::SearchLimits lim;
    if (!cfg.use_time) lim.depthLimit = cfg.depth;
    else lim.movetime = cfg.movetime;
    return lim;
}

}   // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg = parse_args(argc, argv);

        Player white = make_player(cfg.net1);
        Player black = make_player(cfg.net2);
        std::cout << "White: " << white.label << "\n"
                  << "Black: " << black.label << "\n"
                  << "Limit : " << (cfg.use_time
                                        ? "movetime " + std::to_string(cfg.movetime) + " ms"
                                        : "depth " + std::to_string(cfg.depth))
                  << "\nContempt: " << cfg.contempt << " cp\n\n";

        chess::ChessGame game;
        std::vector<std::string> history;
        int lastFrom = -1, lastTo = -1;
        std::string game_over_reason;

        // One table for the WHOLE match: positions transposed earlier in the
        // game (and their subtrees) are reused by every later move.
        chess::TranspositionTable table;
        table.setSize(cfg.hash_mb);

        for (size_t ply = 0;; ++ply) {
            const chess::GameStatus st = game.status();
            if (st.result != chess::ONGOING) { game_over_reason = st.reason; break; }
            if (ply >= cfg.max_plies) { game_over_reason = "max plies reached"; break; }

            const bool whiteMove = game.pos.side > 0;
            Player& mover = whiteMove ? white : black;

            std::cout << "--- " << (whiteMove ? "White" : "Black")
                      << " to move ---\n";
            std::cout << chess::boardString(game.pos, &lastFrom, &lastTo);

            const float eW = white.eval->evaluate(game.pos);
            const float eB = black.eval->evaluate(game.pos);
            std::cout << std::fixed << std::setprecision(1)
                      << "White net " << white.label << ": " << eW << " cp\n"
                      << "Black net " << black.label << ": " << eB << " cp\n";

            // Lazy-SMP: the primary searcher plus N-1 helpers all scan the
            // same tree from the root, every thread writing into the one
            // shared table. The primary returns the first completed iteration;
            // helpers are told to stop and then joined.
            std::vector<std::shared_ptr<chess::Evaluator>> helperEvals;
            std::vector<std::unique_ptr<chess::Searcher>> helpers;
            std::vector<std::unique_ptr<std::thread>> helperThreads;
            helperEvals.reserve(static_cast<size_t>(cfg.threads - 1));
            helpers.reserve(static_cast<size_t>(cfg.threads - 1));
            helperThreads.reserve(static_cast<size_t>(cfg.threads - 1));

            chess::SearchResult res;
            {
                chess::Searcher primary(*mover.eval, &table);
                primary.setDrawContempt(cfg.contempt);
                res = primary.think(game, make_limits(cfg));

                for (int t = 1; t < cfg.threads; ++t) {
                    // The factory's shared_ptr must be retained for the whole
                    // search: Searcher holds an Evaluator&, and NetEvaluator's
                    // per-thread state buffer would be freed with it.
                    auto ev = mover.factory();
                    auto helper = std::make_unique<chess::Searcher>(*ev, &table);
                    helper->setDrawContempt(cfg.contempt);
                    chess::Searcher* raw = helper.get();
                    helperEvals.push_back(std::move(ev));
                    helpers.push_back(std::move(helper));
                    helperThreads.push_back(std::make_unique<std::thread>([&, raw] {
                        (void)raw->think(game, make_limits(cfg));
                    }));
                }
                for (auto& h : helpers) h->stop();
                for (auto& th : helperThreads) if (th) th->join();
            }
            if (!res.hasMove) { game_over_reason = st.reason; break; }

            history.push_back(chess::moveToUci(res.best));
            lastFrom = res.best.from;
            lastTo = res.best.to;
            game.make(res.best);

            std::cout << "position startpos";
            for (const std::string& m : history) std::cout << ' ' << m;
            std::cout << "\n\n";
        }

        const chess::GameStatus st = game.status();
        std::cout << "--- Game over" << (game_over_reason.empty() ? "" : " (" + game_over_reason + ")")
                  << " ---\n";
        std::cout << chess::boardString(game.pos, &lastFrom, &lastTo);

        const float eW = white.eval->evaluate(game.pos);
        const float eB = black.eval->evaluate(game.pos);
        std::cout << std::fixed << std::setprecision(1)
                  << "White net " << white.label << ": " << eW << " cp\n"
                  << "Black net " << black.label << ": " << eB << " cp\n";

        std::cout << "position startpos";
        for (const std::string& m : history) std::cout << ' ' << m;
        std::cout << "\n";

        if (st.result == chess::WHITE_WIN)      std::cout << "Net " << white.label << " Won\n";
        else if (st.result == chess::BLACK_WIN) std::cout << "Net " << black.label << " Won\n";
        else {
            const std::string reason = st.reason.empty() ? game_over_reason : st.reason;
            std::cout << "Draw (" << reason << ")\n";
        }

        std::cout << "History (" << history.size() << " plies):\n";
        for (size_t i = 0; i < history.size(); ++i) {
            if (i % 2 == 0) std::cout << (i / 2 + 1) << ". " << history[i];
            else            std::cout << ' ' << history[i] << '\n';
        }
        if (history.size() % 2 == 1) std::cout << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}