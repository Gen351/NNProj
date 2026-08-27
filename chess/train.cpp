#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../nn_engine/backprop_net/backprop_net.h"
#include "../nn_engine/backprop_net/backprop_net_reader.h"

#include "./chess_environment.h"
#include "./evaluator.h"
#include "./dataset_tools.h"

namespace {

struct Config {
    std::vector<size_t> hidden = {512, 64};
    size_t samples = 50000;
    size_t epochs = 40;
    size_t batch = 32;
    float lr = 0.001f;
    float weight_decay = 0.001f;
    size_t patience = 5;
    float clamp_cp = 2500.0f;
    unsigned seed = 42;
    int mirror = 1;
    float capture_bias = 0.35f;
    int max_plies = 120;
    unsigned threads = 0;
    std::string dataset_path;
    int skip_mates = 1;
    std::string out_dir;
    std::string continue_path;
    std::string net_path;
    bool probe = false;
};

struct Dataset {
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> targets;
};

void print_usage() {
    std::cout
        << "NNProj Chess Trainer\n\n"
        << "Distills the built-in material+PST evaluator into a BackpropNet the\n"
        << "engine can load as its evaluator (see howto/createAI.md).\n\n"
        << "Compile (from chess/):\n"
        << "  g++ -std=c++17 -DNDEBUG -O3 -pthread train.cpp -o chess_trainer\n\n"
        << "Usage:\n"
        << "  chess_trainer                        train a fresh net (PST distillation)\n"
        << "  chess_trainer DATASET=file.csv       finetune on a Lichess-eval CSV dump\n"
        << "  chess_trainer CONTINUE=path/net      resume/finetune an existing net\n"
        << "  chess_trainer --probe NET=path/net   sanity probes on a trained net\n\n"
        << "Options (KEY=VALUE):\n"
        << "  SAMPLES=50000      positions generated (mirror pairs count twice)\n"
        << "  HIDDEN=512,64      hidden layer widths: Dense(781->h1)->TANH->...->Dense(->1)\n"
        << "  THREADS=auto       worker threads for data loading + validation\n"
        << "                     (training itself stays sequential SGD; 1 = old behavior)\n"
        << "  EPOCHS=40          training epochs\n"
        << "  BATCH=32           minibatch size\n"
        << "  LR=0.001           learning rate (halved on plateau)\n"
        << "  WEIGHT_DECAY=0.001 L2 weight decay applied after each epoch\n"
        << "  PATIENCE=5         epochs without val improvement before LR decay\n"
        << "  CLAMP=2500         labels clamped to +/-CLAMP centipawns\n"
        << "  SEED=42            rng seed for generation and shuffling\n"
        << "  MIRROR=1           add a color-flipped mirror of every position\n"
        << "  CAPTURE_BIAS=0.35  chance a playout picks a capture/promotion\n"
        << "  MAX_PLIES=120      playout length cap (PST mode only)\n"
        << "  DATASET=path       Lichess-eval CSV: replay games, label with Stockfish evals\n"
        << "                     instead of PST generation (pairs well with CONTINUE=)\n"
        << "  SKIP_MATES=1       drop '#N' mate-scored plies from DATASET samples\n"
        << "  OUT=path           output dir (default: trained_networks/run_N)\n"
        << "\nOutput: OUT/net.backprop_net, OUT/performance.txt\n"
        << "Play:   ./chess_engine --net trained_networks/run_N/net.backprop_net\n";
}

std::vector<size_t> parse_hidden(const std::string& val) {
    std::vector<size_t> sizes;
    std::string cur;
    auto flush = [&]() {
        if (!cur.empty()) sizes.push_back(static_cast<size_t>(std::stoull(cur)));
        cur.clear();
    };
    for (const char c : val) {
        if (c == ',' || c == ' ') flush();
        else cur += c;
    }
    flush();
    if (sizes.empty() || *std::min_element(sizes.begin(), sizes.end()) < 1)
        throw std::runtime_error("parse_hidden: HIDDEN must list positive widths, e.g. HIDDEN=256,64");
    return sizes;
}

unsigned resolve_threads(const Config& cfg) {
    if (cfg.threads > 0) return cfg.threads;
    const unsigned hw = std::thread::hardware_concurrency();
    return hw > 1 ? hw - 1 : 1;
}

std::string describe_architecture(const std::vector<size_t>& hidden) {
    std::string s = "781";
    for (const size_t h : hidden) s += " -> " + std::to_string(h) + " TANH";
    s += " -> 1 (linear head)";
    return s;
}

BackpropNet::Net create_architecture(const std::vector<size_t>& hidden, const float clamp_cp) {
    using BackpropNet::DenseLayer;
    using BackpropNet::ActivationLayer;

    BackpropNet::Net net;
    size_t prev = chess::STATE_SIZE;
    for (const size_t h : hidden) {
        net.add_layer(std::make_unique<DenseLayer>(prev, h, DenseLayer::InitType::XAVIER));
        net.add_layer(std::make_unique<ActivationLayer>(ActivationLayer::TANH));
        prev = h;
    }

    // The linear output head is initialized at centipawn scale (TANH features
    // live in [-1,1]); without this the first epochs are spent merely growing
    // the head's magnitude instead of learning evaluation.
    // clamp_cp/32 keeps initial outputs moderate (~+-78 cp) so Adam can
    // find its footing before the head grows.
    auto head = std::make_unique<DenseLayer>(prev, 1, DenseLayer::InitType::XAVIER);

    
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-clamp_cp / 32.0f, clamp_cp / 32.0f);
    for (float& w : head->get_weights().data) w = dist(rng);
    head->get_biases()[0] = 0.0f;
    net.add_layer(std::move(head));
    return net;
}

std::string describe_dims(const std::vector<std::pair<size_t, size_t>>& dims) {
    std::string s;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i) s += " -> ";
        s += std::to_string(dims[i].first) + "->" + std::to_string(dims[i].second);
    }
    return s;
}

BackpropNet::Net load_net(const std::string& path, const std::vector<size_t>& hidden) {
    BackpropNet::Net net = BackpropNetReader::load(path);

    // Full shape check against the architecture we intend to train: every
    // DEN layer's dims must match [781, h0..hn, 1] in order. This catches
    // resuming a checkpoint from a DIFFERENT architecture early -- otherwise
    // the mismatch only explodes mid-forward.
    std::vector<std::pair<size_t, size_t>> dims;
    for (const auto& l : net.layers) {
        if (l->get_type() != "DEN") continue;
        const auto* d = static_cast<const BackpropNet::DenseLayer*>(l.get());
        dims.emplace_back(d->get_weights().cols(), d->get_weights().rows());
    }
    std::vector<std::pair<size_t, size_t>> want;
    size_t prev = chess::STATE_SIZE;
    for (const size_t h : hidden) {
        want.emplace_back(prev, h);
        prev = h;
    }
    want.emplace_back(prev, 1);

    if (dims != want) {
        throw std::runtime_error(
            "train: '" + path + "' has a different architecture than HIDDEN implies "
            "(loaded " + describe_dims(dims) + ", wanted "
            + describe_architecture(hidden) + ")");
    }
    return net;
}

// mirror_position lives in dataset_tools.h (shared with test.cpp).
using dataset::mirror_position;

chess::Move pick_move(const chess::Position& pos,
                      const std::vector<chess::Move>& moves,
                      const float capture_bias,
                      std::mt19937& rng) {
    if (capture_bias > 0.0f) {
        std::uniform_real_distribution<float> uni(0.0f, 1.0f);
        if (uni(rng) < capture_bias) {
            std::vector<size_t> tactical;
            for (size_t i = 0; i < moves.size(); ++i) {
                const chess::Move& m = moves[i];
                if (m.promo || (m.flags & chess::F_EP) || pos.board[m.to] != chess::EMPTY)
                    tactical.push_back(i);
            }
            if (!tactical.empty()) {
                std::uniform_int_distribution<size_t> tpick(0, tactical.size() - 1);
                return moves[tpick(rng)];
            }
        }
    }
    std::uniform_int_distribution<size_t> pick(0, moves.size() - 1);
    return moves[pick(rng)];
}

// Workers overshoot the SAMPLES budget by at most one game each; shave the
// excess off the train tail so SAMPLES stays meaningful (val is never cut).
void trim_to_budget(Dataset& train, const Dataset& val, const size_t samples) {
    const size_t total = train.inputs.size() + val.inputs.size();
    if (total <= samples) return;
    const size_t excess = total - samples;
    if (excess > train.inputs.size()) return;
    train.inputs.resize(train.inputs.size() - excess);
    train.targets.resize(train.targets.size() - excess);
}

// ---------------------------------------------------------------------------
// PST-distillation generation, threaded.
//
// Games are completely independent, so each worker runs its own seeded RNG
// and teacher over its own ChessGame and buffers samples locally; finished
// games are spliced into the shared datasets under a mutex. The global
// budget is checked at game granularity -- a thread can overshoot by at most
// one game (~2*MAX_PLIES samples), which trim_to_budget() removes afterwards.
// ---------------------------------------------------------------------------
void generate_dataset(Dataset& train, Dataset& val, const Config& config, const unsigned nthreads) {
    std::atomic<size_t> made{0};
    std::atomic<bool> val_assigned{false};
    std::mutex merge_mutex;
    std::atomic<size_t> games{0};

    const auto worker = [&](const unsigned tid) {
        chess::MaterialPSTEvaluator teacher;
        std::mt19937 rng(config.seed + tid * 0x9E3779B9u);
        std::uniform_real_distribution<float> uni(0.0f, 1.0f);
        Dataset local;
        std::vector<float> state;

        while (made.load(std::memory_order_relaxed) < config.samples) {
            chess::ChessGame game;
            local.inputs.clear();
            local.targets.clear();

            for (int ply = 0; ply < config.max_plies; ++ply) {
                const std::vector<chess::Move> moves = game.legalMoves();
                if (moves.empty()) break;

                if (game.status().result != chess::ONGOING) break;

                const float cp = teacher.evaluate(game.pos);

                state.clear();
                chess::getState(game.pos, state);
                local.inputs.push_back(state);
                local.targets.push_back({std::clamp(cp, -config.clamp_cp, config.clamp_cp)});

                if (config.mirror) {
                    const chess::Position mirrored = mirror_position(game.pos);
                    state.clear();
                    chess::getState(mirrored, state);
                    local.inputs.push_back(state);
                    local.targets.push_back({std::clamp(-cp, -config.clamp_cp, config.clamp_cp)});
                }

                game.make(pick_move(game.pos, moves, config.capture_bias, rng));
            }

            const size_t added = local.inputs.size();
            const bool to_val = !val_assigned.exchange(true) || uni(rng) < 0.05f;
            {
                std::lock_guard<std::mutex> lock(merge_mutex);
                Dataset& dst = to_val ? val : train;
                dst.inputs.insert(dst.inputs.end(),
                                  std::make_move_iterator(local.inputs.begin()),
                                  std::make_move_iterator(local.inputs.end()));
                dst.targets.insert(dst.targets.end(),
                                   std::make_move_iterator(local.targets.begin()),
                                   std::make_move_iterator(local.targets.end()));
            }
            made.fetch_add(added, std::memory_order_relaxed);
            games.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(nthreads);
    for (unsigned t = 0; t < nthreads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();

    trim_to_budget(train, val, config.samples);

    if (train.inputs.empty())
        throw std::runtime_error("generate_dataset: training set came out empty (raise SAMPLES)");
}

// ---------------------------------------------------------------------------
// Lichess-eval CSV ingestion (DATASET=<file.csv>)
//
// One CSV row = one game (see dataset_tools.h for the exact format). Each
// game is replayed from the starting position; for every ply with a usable
// evaluation we emit one sample: state = getState(position), target =
// clamped centipawns from Eval_ply_N. Alignment note: Eval_ply_N labels the
// position BEFORE move N, so we sample first and apply the move second --
// test.cpp pins this contract down on a real game.
//
// Selection under the RAM cap is unbiased: pass 1 records the byte offset of
// every data line (~1.6MB for 200k games), the offsets are shuffled, then
// pass 2 seeks directly to randomly chosen games until SAMPLES is reached.
// This stays correct even if the file groups games by category/time control.
//
// A game whose SAN fails to resolve is dropped WHOLE (never partially) --
// samples are buffered per game and only spliced into the real dataset once
// the replay reaches the end cleanly. Otherwise one bad token mid-game would
// silently mislabel everything recorded before it.
// ---------------------------------------------------------------------------
struct CsvLoadStats {
    size_t games_available = 0;
    size_t games_used = 0;
    size_t bad_san = 0;
    size_t malformed_rows = 0;
    size_t mate_plies_skipped = 0;
    size_t empty_eval_plies = 0;
    size_t samples = 0;
};

// ---------------------------------------------------------------------------
// Threaded pass 2. Games are independent, so each worker gets its OWN
// ifstream (seeking a shared handle would race), its own replay state, and
// its own stats; finished games are spliced into the shared datasets under a
// mutex. Workers pull the next shuffled offset from an atomic cursor and stop
// once the global SAMPLES budget is reached -- overshoot is bounded by one
// game per thread and trimmed afterwards. Unbiasedness is preserved because
// the cursor walks the fully shuffled order regardless of thread count.
// ---------------------------------------------------------------------------
void generate_dataset_csv(Dataset& train,
                          Dataset& val,
                          const Config& cfg,
                          CsvLoadStats& stats,
                          const unsigned nthreads) {
    std::ifstream file(cfg.dataset_path);
    if (!file.is_open())
        throw std::runtime_error("generate_dataset_csv: cannot open '" + cfg.dataset_path + "'");

    // Pass 1 -- byte offset of every non-empty data line (header skipped).
    std::vector<unsigned long long> offsets;
    offsets.reserve(220000);
    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        offsets.push_back(static_cast<unsigned long long>(file.tellg()));
        ++stats.games_available;
    }
    if (offsets.empty())
        throw std::runtime_error("generate_dataset_csv: no data rows in '" + cfg.dataset_path + "'");
    file.close();

    std::mt19937 rng(cfg.seed);
    std::shuffle(offsets.begin(), offsets.end(), rng);

    std::atomic<size_t> next_idx{0};
    std::atomic<size_t> made{0};
    std::atomic<bool> val_assigned{false};
    std::mutex merge_mutex;

    const auto worker = [&](const unsigned tid) {
        std::ifstream in(cfg.dataset_path);
        std::string line;
        chess::ChessGame game;
        std::vector<float> state;
        Dataset local;
        CsvLoadStats lst{};
        std::mt19937 uni_rng(cfg.seed + tid * 0x9E3779B9u);
        std::uniform_real_distribution<float> uni(0.0f, 1.0f);

        auto record_sample = [&](const chess::Position& p, float cp) {
            state.clear();
            chess::getState(p, state);
            local.inputs.push_back(std::move(state));
            local.targets.push_back({std::clamp(cp, -cfg.clamp_cp, cfg.clamp_cp)});
            ++lst.samples;

            if (cfg.mirror) {
                const chess::Position mirrored = dataset::mirror_position(p);
                state.clear();
                chess::getState(mirrored, state);
                local.inputs.push_back(std::move(state));
                local.targets.push_back({std::clamp(-cp, -cfg.clamp_cp, cfg.clamp_cp)});
                ++lst.samples;
            }
        };

        for (;;) {
            if (made.load(std::memory_order_relaxed) >= cfg.samples) break;
            const size_t i = next_idx.fetch_add(1, std::memory_order_relaxed);
            if (i >= offsets.size()) break;

            in.clear();
            in.seekg(static_cast<std::streamoff>(offsets[i]));
            if (!std::getline(in, line)) continue;
            if (!line.empty() && line.back() == '\r') line.pop_back();

            const std::vector<std::string> fields = dataset::parse_csv_line(line);
            if (fields.size() < dataset::MIN_COLS || fields[dataset::MOVE_COL].empty()) {
                ++lst.malformed_rows;
                continue;
            }

            local.inputs.clear();
            local.targets.clear();
            game.reset();

            bool replay_ok = true;
            for (size_t ply = 0; ply < 200; ++ply) {
                const std::string& san = fields[dataset::MOVE_COL + ply];
                const std::string& ev = fields[dataset::EVAL_COL + ply];
                if (san.empty()) break;

                if (!ev.empty() && ev[0] == '#') {
                    if (!cfg.skip_mates) {
                        // "#-N" = Black mates in N -> dead-lost for White.
                        const bool white_mates = !(ev.size() > 1 && ev[1] == '-');
                        record_sample(game.pos, white_mates ? cfg.clamp_cp : -cfg.clamp_cp);
                    } else {
                        ++lst.mate_plies_skipped;
                    }
                } else if (!ev.empty()) {
                    // Dataset convention: evals are in PAWNS ("0.25", "-10.81",
                    // "-70.52"), the engine contract wants CENTIPAWNS -- hence x100.
                    // After conversion CLAMP=2500 finally bounds the extreme tails.
                    try {
                        record_sample(game.pos, std::stof(ev) * 100.0f);
                    } catch (...) {
                        ++lst.empty_eval_plies;
                    }
                } else {
                    ++lst.empty_eval_plies;
                }

                chess::Move m;
                if (!dataset::san_to_move(game.pos, san, m)) {
                    replay_ok = false;
                    ++lst.bad_san;
                    break;
                }
                game.make(m);
            }

            if (!replay_ok || local.inputs.empty()) continue;

            const bool to_val = !val_assigned.exchange(true) || uni(uni_rng) < 0.05f;
            const size_t added = local.inputs.size();
            {
                std::lock_guard<std::mutex> lock(merge_mutex);
                Dataset& dst = to_val ? val : train;
                dst.inputs.insert(dst.inputs.end(),
                                  std::make_move_iterator(local.inputs.begin()),
                                  std::make_move_iterator(local.inputs.end()));
                dst.targets.insert(dst.targets.end(),
                                   std::make_move_iterator(local.targets.begin()),
                                   std::make_move_iterator(local.targets.end()));
                ++stats.games_used;
            }
            made.fetch_add(added, std::memory_order_relaxed);
        }

        std::lock_guard<std::mutex> lock(merge_mutex);
        stats.samples += lst.samples;
        stats.bad_san += lst.bad_san;
        stats.malformed_rows += lst.malformed_rows;
        stats.mate_plies_skipped += lst.mate_plies_skipped;
        stats.empty_eval_plies += lst.empty_eval_plies;
    };

    std::vector<std::thread> pool;
    pool.reserve(nthreads);
    for (unsigned t = 0; t < nthreads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();

    trim_to_budget(train, val, cfg.samples);

    std::cout << "CSV load: " << stats.games_used << '/' << stats.games_available
              << " games used | samples=" << stats.samples
              << " | bad_san_games=" << stats.bad_san
              << " | malformed_rows=" << stats.malformed_rows
              << " | mate_plies_skipped=" << stats.mate_plies_skipped
              << " | unlabeled_plies=" << stats.empty_eval_plies << "\n";

    if (train.inputs.empty())
        throw std::runtime_error("generate_dataset_csv: training set came out empty");
}

void shuffle_together(Dataset& data, std::mt19937& rng) {
    for (size_t i = data.inputs.size(); i > 1; --i) {
        const size_t j = std::uniform_int_distribution<size_t>(0, i - 1)(rng);
        std::swap(data.inputs[i - 1], data.inputs[j]);
        std::swap(data.targets[i - 1], data.targets[j]);
    }
}

// Validation is embarrassingly parallel (read-only forwards), so it is
// chunked across workers too. Layers are switched to predict mode ONCE
// before spawning so threads never touch the mode flags concurrently.
float mse(BackpropNet::Net& net, const Dataset& data, const unsigned nthreads) {
    if (data.inputs.empty()) return 0.0f;

    for (auto& layer : net.layers) layer->predict();

    std::vector<double> partial(nthreads, 0.0);
    const size_t n = data.inputs.size();

    const auto worker = [&](const unsigned t) {
        double sum = 0.0;
        const size_t lo = n * t / nthreads;
        const size_t hi = n * (t + 1) / nthreads;
        for (size_t i = lo; i < hi; ++i) {
            std::vector<float> out = data.inputs[i];
            for (const auto& layer : net.layers) out = layer->forward(out);
            const double d = static_cast<double>(out[0]) - data.targets[i][0];
            sum += d * d;
        }
        partial[t] = sum;
    };

    std::vector<std::thread> pool;
    pool.reserve(nthreads);
    for (unsigned t = 0; t < nthreads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();

    double total = 0.0;
    for (const double p : partial) total += p;
    return static_cast<float>(total / static_cast<double>(n));
}

std::string next_run_dir() {
    int max_ver = 0;
    if (!std::filesystem::exists("trained_networks"))
        return "trained_networks/run_1";
    for (const std::filesystem::directory_entry& entry
            : std::filesystem::directory_iterator("trained_networks")) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("run_", 0) != 0) continue;
        try {
            const int ver = std::stoi(name.substr(4));
            if (ver > max_ver) max_ver = ver;
        } catch (...) {
        }
    }
    return "trained_networks/run_" + std::to_string(max_ver + 1);
}

Config parse_args(int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--probe") { config.probe = true; continue; }
        if (arg == "--help" || arg == "-h") { print_usage(); std::exit(0); }

        const size_t eq = arg.find('=');
        if (eq == std::string::npos)
            throw std::runtime_error("parse_args: bad argument '" + arg
                                     + "' (expected KEY=VALUE or --probe)");
        const std::string key = arg.substr(0, eq);
        std::string val = arg.substr(eq + 1);
        if (val.size() >= 2 && val.front() == '"' && val.back() == '"')
            val = val.substr(1, val.size() - 2);

        if      (key == "CONTINUE")     config.continue_path = val;
        else if (key == "NET")          config.net_path = val;
        else if (key == "OUT")          config.out_dir = val;
        else if (key == "SAMPLES")      config.samples = std::stoull(val);
        else if (key == "EPOCHS")       config.epochs = std::stoull(val);
        else if (key == "BATCH")        config.batch = std::stoull(val);
        else if (key == "LR")           config.lr = std::stof(val);
        else if (key == "WEIGHT_DECAY") config.weight_decay = std::stof(val);
        else if (key == "PATIENCE")     config.patience = std::stoull(val);
        else if (key == "CLAMP")        config.clamp_cp = std::stof(val);
        else if (key == "SEED")         config.seed = static_cast<unsigned>(std::stoul(val));
        else if (key == "MIRROR")       config.mirror = std::stoi(val);
        else if (key == "CAPTURE_BIAS") config.capture_bias = std::stof(val);
        else if (key == "MAX_PLIES")    config.max_plies = std::stoi(val);
        else if (key == "HIDDEN")       config.hidden = parse_hidden(val);
        else if (key == "THREADS")      config.threads = static_cast<unsigned>(std::stoul(val));
        else if (key == "DATASET")      config.dataset_path = val;
        else if (key == "SKIP_MATES")   config.skip_mates = std::stoi(val);
        else throw std::runtime_error("parse_args: unknown option '" + key + "'");
    }
    return config;
}

int run_probe(const Config& config) {
    if (config.net_path.empty())
        throw std::runtime_error("run_probe: pass the net via NET=path/to/net.backprop_net");

    const auto evaluator = chess::loadNetEvaluator(config.net_path);

    chess::ChessGame start;
    const float e_start = evaluator->evaluate(start.pos);

    chess::ChessGame queen_up("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    const float e_queen = evaluator->evaluate(queen_up.pos);

    const chess::Position mirrored = mirror_position(queen_up.pos);
    const float e_mirror = evaluator->evaluate(mirrored);

    std::cout << "startpos eval    : " << e_start << " cp (want ~0)\n"
              << "white up a queen : " << e_queen << " cp (want strongly positive)\n"
              << "mirror of it     : " << e_mirror << " cp (want ~negative of the above)\n";

    int failures = 0;
    const auto check = [&](const bool ok, const char* what) {
        std::cout << (ok ? "PASS " : "FAIL ") << what << "\n";
        if (!ok) ++failures;
    };
    check(std::fabs(e_start) < 150.0f, "startpos near zero");
    check(e_queen > 400.0f, "queen advantage positive");
    check(std::fabs(e_mirror + e_queen) < 600.0f, "mirror flips sign");

    std::cout << (failures == 0 ? "PROBE PASSED\n" : "PROBE FAILED\n");
    return failures == 0 ? 0 : 1;
}

}   // namespace

int main(int argc, char* argv[]) {
    try {
        const Config config = parse_args(argc, argv);
        if (config.probe) return run_probe(config);

        const unsigned nthreads = resolve_threads(config);
        std::cout << (config.continue_path.empty()
                          ? "Architecture: " + describe_architecture(config.hidden)
                          : "Continuing from: " + config.continue_path)
                  << "\nThreads: " << nthreads << " (data loading + validation)\n";

        Dataset train, val;
        if (!config.dataset_path.empty()) {
            std::cout << "Loading Lichess-eval dataset: " << config.dataset_path << "\n";
            CsvLoadStats stats;
            generate_dataset_csv(train, val, config, stats, nthreads);
        } else {
            std::cout << "Generating PST-distillation dataset...\n";
            generate_dataset(train, val, config, nthreads);
        }

        std::cout << "Train: " << train.inputs.size() << " | Val: " << val.inputs.size() << "\n";


        const std::string run_dir = config.out_dir.empty() ? next_run_dir() : config.out_dir;
        std::filesystem::create_directories(run_dir);


        BackpropNet::Net net = config.continue_path.empty()
            ? create_architecture(config.hidden, config.clamp_cp)
            : load_net(config.continue_path, config.hidden);
        BackpropNetReader::save(run_dir + "/net", net);


        float best_val = mse(net, val, nthreads);
        size_t bad_epochs = 0;
        float lr = config.lr;
        std::mt19937 rng(config.seed ^ 0x9E3779B9u);

        std::ofstream perf(run_dir + "/performance.txt");
        if (!perf.is_open())
            throw std::runtime_error("main: failed to open " + run_dir + "/performance.txt");

            perf << "CONTINUE: " << (config.continue_path.empty() ? "(fresh)" : config.continue_path) << "\n"
             << "DATASET: " << (config.dataset_path.empty() ? "(PST distillation)" : config.dataset_path) << "\n"
             << "HIDDEN: " << describe_architecture(config.hidden) << "\n"
             << "THREADS: " << nthreads << "\n"
             << "SAMPLES: " << config.samples << " EPOCHS: " << config.epochs
             << " BATCH: " << config.batch << " LR: " << config.lr
             << " WEIGHT_DECAY: " << config.weight_decay
             << " PATIENCE: " << config.patience << " CLAMP: " << config.clamp_cp
             << " MIRROR: " << config.mirror << " CAPTURE_BIAS: " << config.capture_bias
             << " SKIP_MATES: " << config.skip_mates
             << " SEED: " << config.seed << "\n"
             << "epoch val_mse lr\n";

        std::cout << "Initial val MSE: " << best_val << "\n";

        for (size_t epoch = 1; epoch <= config.epochs; ++epoch) {
            shuffle_together(train, rng);
            net.train_v2(1, config.batch, config.patience, lr,
                         train.inputs, train.targets, nullptr, nullptr);

            // L2 weight decay: manually decay DenseLayer weights after each
            // epoch (backprop_net.h has no weight decay).  Biases are left
            // alone -- regularising them rarely helps.
            if (config.weight_decay > 0.0f) {
                const float scale = 1.0f - lr * config.weight_decay;
                for (auto& layer : net.layers) {
                    if (layer->get_type() != "DEN") continue;
                    auto& w = static_cast<BackpropNet::DenseLayer*>(layer.get())->get_weights();
                    for (float& v : w.data) v *= scale;
                }
            }

            const float val_loss = mse(net, val, nthreads);
            perf << epoch << ' ' << val_loss << ' ' << lr << '\n';
            perf.flush();

            const bool improved = val_loss < best_val;
            if (improved) {
                best_val = val_loss;
                bad_epochs = 0;
                BackpropNetReader::save(run_dir + "/net", net);
            } else {
                ++bad_epochs;
                if (bad_epochs >= config.patience && lr > 1e-6f) {
                    lr *= 0.5f;
                    bad_epochs = 0;
                    std::cout << "--- no improvement, LR decayed to " << lr << " ---\n";
                }
            }

            std::cout << "Epoch " << epoch << '/' << config.epochs
                      << " | Val MSE=" << val_loss
                      << " | Best=" << best_val
                      << " | LR=" << lr
                      << (improved ? " [best saved]" : "")
                      << "\n";
        }

        std::cout << "\nDone. Best val MSE: " << best_val << "\n"
                  << "Load into engine:\n"
                  << "  ./chess_engine --net " << run_dir << "/net.backprop_net\n"
                  << "Sanity probes:\n"
                  << "  ./chess_trainer --probe NET=" << run_dir << "/net\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
