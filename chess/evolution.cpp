#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../nn_engine/backprop_net/backprop_net.h"
#include "../nn_engine/backprop_net/backprop_net_reader.h"

#include "./chess_environment.h"
#include "./evaluator.h"
#include "./dataset_tools.h"
#include "./search.h"
#include "./transposition_table.h"
#include "./nnue_evaluator.h"

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
    int fen_csv = 0;
    int skip_mates = 1;
    float pst_scale = 1.0f;
    std::string out_dir;
    std::string continue_path;
    std::string net_path;
    int win_prob = -1;       // -1 = auto (fresh -> linear, CONTINUE -> checkpoint)
    bool probe = false;

    // Evolution mode (EVO=1).
    int evo = 0;
    std::string evo_opponent = "pst";
    size_t pop = 16;
    size_t generations = 50;
    size_t gate_games = 8;
    size_t h2h_games = 4;
    long long movetime = 1000;
    float sigma = 0.1f;
    float sigma_abs = 0.0f;
    float margin = 0.5f;
    float hmargin = 0.0f;
    float sigma_decay = 0.95f;
    std::string openings_file;
    size_t hash_mb = 16;
    int quant = 0;
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
        << "  PST_SCALE=1.0      multiply PST-teacher labels by this before the win-prob\n"
        << "                     transform (PST mode only; 1.5-2.0 approximates the\n"
        << "                     stockfish-hot magnitudes on the eval CSV, default 1.0)\n"
        << "  SEED=42            rng seed for generation and shuffling\n"
        << "  MIRROR=1           add a color-flipped mirror of every position\n"
        << "  CAPTURE_BIAS=0.35  chance a playout picks a capture/promotion\n"
        << "  MAX_PLIES=120      playout length cap (PST mode only)\n"
        << "  DATASET=path       Lichess-eval CSV: replay games, label with Stockfish evals\n"
        << "                     instead of PST generation (pairs well with CONTINUE=)\n"
        << "  FENCSV=1           with DATASET=, read a FEN,eval CSV (one position + integer\n"
        << "                     centipawn score per row, header row skipped) instead of the\n"
        << "                     Lichess game dump. Rows are uniformly sampled (reservoir)\n"
        << "                     from the WHOLE file, not just the first SAMPLES lines.\n"
        << "  SKIP_MATES=1       drop '#N' mate-scored plies from DATASET samples\n"
        << "  OUT=path           output dir (default: trained_networks/run_N)\n"
        << "  WINPROB=0|1        force a linear (0) or win-prob (1) head for a fresh\n"
        << "                     run. Default: fresh -> linear head, CONTINUE -> the\n"
        << "                     checkpoint's own head. (linear: raw-centipawn output,\n"
        << "                     win-prob: final TANH over tanh(cp/400))\n"
        << "\nEvolution mode (EVO=1): (1+lambda)-ES over a champion net. Each\n"
        << "generation POP mutated children play a fixed suite of GAMES gate games\n"
        << "against a fixed opponent (material+PST by default) -- every opening once\n"
        << "as the candidate and once mirrored as Black, all games run in parallel\n"
        << "across THREADS. The best child is promoted only if it beats the gate by\n"
        << "MARGIN points over the champion AND wins/ties the H2H veto against the\n"
        << "champion (no regression to the incumbent). Requires NET=champion.\n"
        << "  EVO=1              enable evolution mode\n"
        << "  POP=16             children per generation\n"
        << "  GENERATIONS=50     generations to run\n"
        << "  GAMES=8            gate-suite size per net (openings x both colors)\n"
        << "  H2H=4              head-to-head veto games vs the champion\n"
        << "  MOVETIME=1000      per-move search budget (ms)\n"
        << "  OPPONENT=pst       gate opponent: pst, or a .backprop_net path\n"
        << "  SIGMA=0.1          relative mutation std (fraction of each layer's std)\n"
        << "  SIGMA_ABS=0        absolute noise added on top of the relative SIGMA\n"
        << "  MARGIN=0.5         child must outscore the champion's gate score by this\n"
        << "  HMARGIN=0          veto margin: child needs H2H/2 + HMARGIN points\n"
        << "  SIGMA_DECAY=0.95   shrink SIGMA each generation nothing promotes\n"
        << "  OPENINGS=file      one opening-position FEN per line (default: 4 built-ins)\n"
        << "  HASH=16            per-worker transposition-table size (MB)\n"
        << "  QUANT=0            1 = quantized int16 evaluator for the matches\n"
        << "  OUT=path           output dir (default: evolved_networks/run_N)\n"
        << "\nEvolution output: OUT/champ.net.backprop_net (current champion),\n"
        << "OUT/gen_NNN/net.backprop_net (best child of each generation, promoted or\n"
        << "not), and OUT/evolution.txt (columns: gen champ_gate best_child promoted\n"
        << "sigma).\n"
        << "\nLabel convention and loss units: with a win-prob head (WINPROB=1) the\n"
        << "targets are bounded tanh(cp/S), S=400, in (-1,1); NetEvaluator inverts them\n"
        << "back to centipawns automatically, and Val MSE is in win-prob units\n"
        << "(~0.01-0.05 after a good fit). With a linear head (WINPROB=0, the fresh\n"
        << "default) targets are RAW centipawns, so the reported MSE is in cp^2 -- take\n"
        << "sqrt(MSE) to get the RMS error in centipawns (e.g. 40000 -> ~200 cp RMS).\n"
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

std::string describe_architecture(const std::vector<size_t>& hidden, const bool win_prob) {
    std::string s = "781";
    for (const size_t h : hidden) s += " -> " + std::to_string(h) + " TANH";
    s += " -> 1";
    s += win_prob ? " -> TANH (win-prob head)" : " (linear head)";
    return s;
}

BackpropNet::Net create_architecture(const std::vector<size_t>& hidden, const float clamp_cp, bool win_prob=true) {
    using BackpropNet::DenseLayer;
    using BackpropNet::ActivationLayer;

    BackpropNet::Net net;
    size_t prev = chess::STATE_SIZE;
    for (const size_t h : hidden) {
        net.add_layer(std::make_unique<DenseLayer>(prev, h, DenseLayer::InitType::XAVIER));
        net.add_layer(std::make_unique<ActivationLayer>(ActivationLayer::TANH));
        prev = h;
    }

    // Win-probability head: Dense -> TANH. The Dense maps the hidden TANH
    // features to a single pre-activation; the final TANH bounds it to
    // (-1,1), which NetEvaluator scales by kWinScaleCP and inverts back to
    // centipawns. Crucially, the final TANH's 1-cosh^-2 "squashing" kills the
    // gradient for positions where the head is already confident -- exactly
    // the noisy tail that made raw-MSE training regress everything to ~0.
    //
    // Seed the head weights small so the pre-activation starts near 0 and
    // tanh maps it into the sensitive (roughly linear) part of its curve;
    // initializing it to centipawn-scale magnitudes would put the final tanh
    // deep in its saturated flat region and stall learning at epoch 0.
    auto head = std::make_unique<DenseLayer>(prev, 1, DenseLayer::InitType::XAVIER);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (float& w : head->get_weights().data) w = dist(rng);
    head->get_biases()[0] = 0.0f;
    net.add_layer(std::move(head));
    
    // Add TANH for win prob (-1, 1), -1 for black, 1 for white
    if(win_prob) {
        net.add_layer(std::make_unique<ActivationLayer>(ActivationLayer::TANH));
    }
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

BackpropNet::Net load_net(const std::string& path, const std::vector<size_t>& hidden,
                          bool& out_winprob) {
    BackpropNet::Net net = BackpropNetReader::load(path);

    // Win-prob convention: the net was trained against bounded S*tanh(cp/S)
    // targets iff its final layer is a TANH activation. This drives how the
    // training targets must be encoded below (fresh checkerboards always use
    // win-prob, but an old linear-head checkpoint must keep raw-cp targets).
    out_winprob = false;
    for (auto it = net.layers.rbegin(); it != net.layers.rend(); ++it) {
        if ((*it)->get_type() == "DEN") break;
        if ((*it)->get_type() == "ACT") {
            const auto* a = static_cast<const BackpropNet::ActivationLayer*>(it->get());
            out_winprob = (a->get_activation() == BackpropNet::ActivationLayer::TANH);
            break;
        }
    }

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
            + describe_architecture(hidden, out_winprob) + ")");
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
void generate_dataset(Dataset& train, Dataset& val, const Config& config,
                      const bool winprob, const unsigned nthreads) {
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
                const float c = std::clamp(cp * config.pst_scale, -config.clamp_cp, config.clamp_cp);

                state.clear();
                chess::getState(game.pos, state);
                local.inputs.push_back(state);
                local.targets.push_back({winprob ? chess::cpWinProb(c) : c});

                if (config.mirror) {
                    const chess::Position mirrored = mirror_position(game.pos);
                    state.clear();
                    chess::getState(mirrored, state);
                    const float mc = std::clamp(-cp * config.pst_scale, -config.clamp_cp, config.clamp_cp);
                    local.inputs.push_back(state);
                    local.targets.push_back({winprob ? chess::cpWinProb(mc) : mc});
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
                          bool winprob,
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
            const float c = std::clamp(cp, -cfg.clamp_cp, cfg.clamp_cp);
            local.targets.push_back({winprob ? chess::cpWinProb(c) : c});
            ++lst.samples;

            if (cfg.mirror) {
                const chess::Position mirrored = dataset::mirror_position(p);
                state.clear();
                chess::getState(mirrored, state);
                local.inputs.push_back(std::move(state));
                const float mc = std::clamp(-cp, -cfg.clamp_cp, cfg.clamp_cp);
                local.targets.push_back({winprob ? chess::cpWinProb(mc) : mc});
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




struct FenDatasetStats {
    std::string fen;
    int eval;
};

// Load a "FEN",eval CSV -- one position and its integer centipawn score per
// row (the header row "FEN,eval" is skipped). Each row becomes ONE sample:
// the FEN is converted to the 781-float state vector via getState(), and the
// eval is the target (raw centipawns, no x100 -- unlike the Lichess game dump
// whose evals are pawns). `cfg.samples` rows are picked UNIFORMLY from the
// whole file via single-pass reservoir sampling (unbiased regardless of file
// order or length), NOT just the first `cfg.samples` lines. When `winprob` is
// set, targets are mapped through cpWinProb() to match a win-probability
// head. A seeded random split marks ~5% as validation.
void generate_dataset_from_fen_csv(Dataset& train,
                                    Dataset& val,
                                    const Config& cfg,
                                    const bool winprob,
                                    const unsigned nthreads) {
    (void)nthreads;
    std::ifstream file(cfg.dataset_path);
    if (!file.is_open())
        throw std::runtime_error("generate_dataset_from_fen_csv: cannot open '"
                                 + cfg.dataset_path + "'");

    std::vector<FenDatasetStats> raw;
    raw.reserve(cfg.samples);

    std::string temp;
    bool first = true;   // skip the "FEN,eval" header row
    size_t malformed = 0;
    size_t seen = 0;     // valid rows scanned across the WHOLE file

    // Reservoir sampling: keep a uniform cfg.samples-subset of `seen` rows in
    // one pass. The shared split RNG below is seeded from cfg.seed on its own,
    // so a separate RNG here keeps the 95/5 draw stable across file sizes.
    std::mt19937 rs_rng(cfg.seed ^ 0x51ED270bu);

    while (std::getline(file, temp)) {
        if (temp.empty()) continue;
        if (temp.back() == '\r') temp.pop_back();
        if (first) {
            first = false;   // header row
            continue;
        }

        // Optional UTF-8 BOM at the very start of the file.
        if (!temp.empty() && static_cast<unsigned char>(temp[0]) == 0xEF)
            temp.erase(0, 1);

        std::stringstream ss(temp);
        std::string f, e;
        std::getline(ss, f, ',');
        std::getline(ss, e, ',');

        if (f.empty() || e.empty()) { ++malformed; continue; }
        int eval = 0;
        try {
            eval = std::stoi(e);
        } catch (...) {
            ++malformed;
            continue;
        }

        ++seen;   // row's 1-based index among valid rows
        FenDatasetStats rec{ f, eval };
        if (seen <= cfg.samples) {
            raw.push_back(std::move(rec));
        } else {
            std::uniform_int_distribution<size_t> uni(0, seen - 1);
            const size_t j = uni(rs_rng);
            if (j < cfg.samples) raw[j] = std::move(rec);
        }
    }

    if (raw.empty())
        throw std::runtime_error("generate_dataset_from_fen_csv: no data rows in '"
                                 + cfg.dataset_path + "'");

    // Seeded random split: 95% train, 5% validation.
    std::mt19937 rng(cfg.seed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    chess::ChessGame game;
    std::vector<float> state;
    size_t skipped = 0;
    for (const FenDatasetStats& sample : raw) {
        try {
            game.setFen(sample.fen);
        } catch (...) {
            ++skipped;
            continue;
        }

        state.clear();
        chess::getState(game.pos, state);

        const float c = std::clamp(static_cast<float>(sample.eval),
                                   -cfg.clamp_cp, cfg.clamp_cp);
        const float t = winprob ? chess::cpWinProb(c) : c;
        if (uni(rng) < 0.05f) {
            val.inputs.push_back(state);
            val.targets.push_back({ t });
        } else {
            train.inputs.push_back(state);
            train.targets.push_back({ t });
        }
    }

    if (malformed) std::cout << "  FENCSV malformed rows skipped: " << malformed << "\n";
    if (skipped)   std::cout << "  FENCSV unparseable FENs skipped: " << skipped << "\n";
    std::cout << "  FENCSV rows scanned: " << seen << ", sampled: " << raw.size() << "\n";

    if (train.inputs.empty())
        throw std::runtime_error("generate_dataset_from_fen_csv: training set came out empty");
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
        else if (key == "PST_SCALE")    config.pst_scale = std::stof(val);
        else if (key == "SEED")         config.seed = static_cast<unsigned>(std::stoul(val));
        else if (key == "MIRROR")       config.mirror = std::stoi(val);
        else if (key == "CAPTURE_BIAS") config.capture_bias = std::stof(val);
        else if (key == "MAX_PLIES")    config.max_plies = std::stoi(val);
        else if (key == "HIDDEN")       config.hidden = parse_hidden(val);
        else if (key == "THREADS")      config.threads = static_cast<unsigned>(std::stoul(val));
        else if (key == "DATASET")      config.dataset_path = val;
        else if (key == "FENCSV")       config.fen_csv = std::stoi(val);
        else if (key == "WINPROB")      config.win_prob = std::clamp(std::stoi(val), 0, 1);
        else if (key == "SKIP_MATES")   config.skip_mates = std::stoi(val);
        else if (key == "EVO")          config.evo = std::stoi(val);
        else if (key == "OPPONENT")     config.evo_opponent = val;
        else if (key == "POP")          config.pop = std::stoull(val);
        else if (key == "GENERATIONS")  config.generations = std::stoull(val);
        else if (key == "GAMES")        config.gate_games = std::stoull(val);
        else if (key == "H2H")          config.h2h_games = std::stoull(val);
        else if (key == "MOVETIME")     config.movetime = std::stoll(val);
        else if (key == "SIGMA")        config.sigma = std::stof(val);
        else if (key == "SIGMA_ABS")    config.sigma_abs = std::stof(val);
        else if (key == "MARGIN")       config.margin = std::stof(val);
        else if (key == "HMARGIN")      config.hmargin = std::stof(val);
        else if (key == "SIGMA_DECAY")  config.sigma_decay = std::stof(val);
        else if (key == "OPENINGS")     config.openings_file = val;
        else if (key == "HASH")         config.hash_mb = std::stoull(val);
        else if (key == "QUANT")        config.quant = std::stoi(val);
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

// ---------------------------------------------------------------------------
// Evolution mode (EVO=1): (1+lambda)-ES over the champion.
//
// Each generation POP children are created by adding Gaussian noise to the
// champion (SIGMA scaled by each layer's own weight std, so one knob spans
// every layer width). The champion AND all children then play the same fixed
// suite of GAMES gate games against a fixed opponent (material+PST by default)
// on a shared set of openings, each opening once as the candidate and once
// mirrored so the candidate plays Black. The suite is embarrassingly parallel:
// one worker thread per game, each with its own Searcher + transposition table
// (the search itself stays single-threaded to avoid 11x11 oversubscription).
//
// The best-scoring child is promoted only if it clears BOTH bars:
//   [1] PST gate:  best_child_gate >= champion_gate + MARGIN  -- still improving
//       against the "true" teacher;
//   [2] veto:      head-to-head vs the incumbent over H2H games, child must win
//       at least H2H/2 + HMARGIN points (tie goes to the child, boxing out churn).
// Generations with no promotion anneal SIGMA via SIGMA_DECAY.
// ---------------------------------------------------------------------------

std::vector<std::string> load_openings(const std::string& path) {
    static const std::vector<std::string> kDefaultOpenings = {
        // Four balanced mid-opening book positions, White to move and NOT in
        // check (verified via ChessGame::status()==ONGOING). Each is a standard
        // single-move-deviation book line.
        // Ruy Lopez (closed), ~ply 9: 1.d4 d5 2.c3 e5 3.g1 f3
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        // Italian (Giuoco Piano), ~ply 9: 1.d4 d5 2.e6 e6 3.g1 f3 4.e2 e4
        "r1bqkbnr/ppp1pppp/1n6/3Pp3/2P1P3/3P4/PP2NPPP/RNBQKB1R w KQkq - 4 5",
        // Queen's Gambit Declined, ~ply 7: 1.d4 d5 2.c4 c5 3.d3 d4
        "rnbqkbnr/ppp1pppp/8/2pp4/2PPP3/8/PPPN1PPP/RNB1KBNR w KQkq - 2 4",
        // Sicilian Defense, ~ply 9: 1.d4 d5 2.c4 c5 3.g1 f3 4.p2 p3
        "rnbqkbnr/ppp1pp1p/6p1/3p4/3P2P1/5N2/PPPPP1PP/RNBQKB1R w KQkq - 3 4",
    };
    if (path.empty()) return kDefaultOpenings;

    std::ifstream in(path);
    if (!in.is_open())
        throw std::runtime_error("load_openings: cannot open OPENINGS='" + path + "'");
    std::vector<std::string> out;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        out.push_back(line);
    }
    if (out.empty())
        throw std::runtime_error("load_openings: '" + path + "' contains no FEN lines");
    return out;
}

// Net holds vector<unique_ptr<AbstractLayer>> (no copy ctor), so clone a fresh
// Net and copy each layer's trained weights/biases/activation explicitly.
BackpropNet::Net clone_net(const BackpropNet::Net& src) {
    using BackpropNet::DenseLayer;
    using BackpropNet::ActivationLayer;
    BackpropNet::Net out;
    for (const auto& layer : src.layers) {
        if (layer->get_type() == "DEN") {
            const auto* d = static_cast<const DenseLayer*>(layer.get());
            auto copy = std::make_unique<DenseLayer>(d->get_weights().cols(),
                                                     d->get_weights().rows(),
                                                     DenseLayer::InitType::XAVIER);
            copy->get_weights().data = d->get_weights().data;
            copy->get_biases() = d->get_biases();
            out.add_layer(std::move(copy));
        } else if (layer->get_type() == "ACT") {
            const auto* a = static_cast<const ActivationLayer*>(layer.get());
            out.add_layer(std::make_unique<ActivationLayer>(a->get_activation()));
        }
    }
    return out;
}

// Relative perturbation: noise std = SIGMA * (that set's own std) + SIGMA_ABS,
// applied independently to every DenseLayer's weights and biases.
void mutate_net(BackpropNet::Net& net, std::mt19937& rng, const float sigma,
                const float sigma_abs) {
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    for (auto& layer : net.layers) {
        if (layer->get_type() != "DEN") continue;
        auto* d = static_cast<BackpropNet::DenseLayer*>(layer.get());
        auto& w = d->get_weights();
        auto& b = d->get_biases();

        float wstd = 0.0f;
        for (const float v : w.data) wstd += v * v;
        wstd = std::sqrt(wstd / std::max<size_t>(1, w.data.size()));
        const float amp_w = sigma * wstd + sigma_abs;
        for (float& v : w.data) v += gauss(rng) * amp_w;

        float bstd = 0.0f;
        for (const float v : b) bstd += v * v;
        bstd = std::sqrt(bstd / std::max<size_t>(1, b.size()));
        const float amp_b = sigma * bstd + sigma_abs;
        for (float& v : b) v += gauss(rng) * amp_b;
    }
}

struct GameOutcome {
    float score = 0.5f;   // candidate's POV: 1 win / 0.5 draw / 0 loss
    std::string reason;   // checkmate / stalemate / fifty-move / threefold /
                          // insufficient material / max plies
    size_t plies = 0;
};

// One match from position `game`; candidate = net_eval, opponent = opp_eval.
GameOutcome play_game(const std::shared_ptr<chess::Evaluator>& net_eval,
                      const std::shared_ptr<chess::Evaluator>& opp_eval,
                      chess::ChessGame& game, const bool net_is_white,
                      const long long movetime, chess::TranspositionTable& table) {
    chess::Searcher net_search(*net_eval, &table);
    chess::Searcher opp_search(*opp_eval, &table);
    chess::SearchLimits lim;
    lim.movetime = movetime;

    for (size_t ply = 0; ply < 400; ++ply) {
        const chess::GameStatus st = game.status();
        if (st.result != chess::ONGOING) {
            GameOutcome out;
            out.plies = ply;
            out.reason = st.reason.empty() ? "game end" : st.reason;
            if (st.result == chess::WHITE_WIN) out.score = net_is_white ? 1.0f : 0.0f;
            else if (st.result == chess::BLACK_WIN) out.score = net_is_white ? 0.0f : 1.0f;
            else out.score = 0.5f;
            return out;
        }
        const bool white_move = game.pos.side > 0;
        chess::Searcher& mover = (white_move == net_is_white) ? net_search : opp_search;
        const chess::SearchResult res = mover.think(game, lim);
        if (!res.hasMove) { GameOutcome out; out.plies = ply; out.reason = "no legal move"; return out; }
        game.make(res.best);
    }
    GameOutcome out;
    out.plies = 400;
    out.reason = "max plies";
    return out;   // max plies -> draw
}

struct GameTask {
    size_t slot = 0;            // candidate index into slots (0 = champion)
    bool vs_champion = false;   // opponent = slots[0]; else the fixed opponent
    size_t opening = 0;
    bool net_is_white = true;
};

struct SuiteResult {
    std::vector<float> scores;                 // candidate's POV per task (index-aligned)
    size_t wins = 0, draws = 0, losses = 0;
    std::map<std::string, size_t> reasons;
    size_t total_plies = 0;
};

// Run every game in `tasks` across `nthreads` worker threads. Each worker owns
// a private transposition table and a private (stateless) PST teacher; net
// evaluators are built per task so each thread gets its own accumulator state.
SuiteResult run_suite(const std::string& label,
                      const std::vector<GameTask>& tasks,
                      const std::vector<std::shared_ptr<BackpropNet::Net>>& slots,
                      const std::shared_ptr<BackpropNet::Net>& opp_net,
                      const std::vector<std::string>& openings,
                      const Config& config, const unsigned nthreads) {
    SuiteResult out;
    out.scores.assign(tasks.size(), 0.5f);
    if (tasks.empty()) return out;

    const bool quant = config.quant != 0;
    std::atomic<size_t> next{0}, done{0};
    std::mutex cout_mutex;   // guards W/D/L counters and the shared progress line
    const unsigned nw = std::min(nthreads, static_cast<unsigned>(tasks.size()));

    const auto tally = [&](const GameOutcome& o) {
        const size_t i = done.fetch_add(1) + 1;   // games completed so far
        {
            std::lock_guard<std::mutex> lk(cout_mutex);
            if (o.score > 0.7f) ++out.wins;
            else if (o.score < 0.3f) ++out.losses;
            else ++out.draws;
            if (o.plies < 400) ++out.reasons[o.reason];
            out.total_plies += o.plies;
            if (i % 10 == 0 || i == tasks.size()) {
                std::cout << '\r' << label << ": game " << i << '/'
                          << tasks.size() << " (W " << out.wins << " D " << out.draws
                          << " L " << out.losses << ")     ";
                std::cout.flush();
            }
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(nw);
    for (unsigned t = 0; t < nw; ++t) {
        pool.emplace_back([&] {
            chess::TranspositionTable table;
            table.setSize(config.hash_mb);
            std::shared_ptr<chess::Evaluator> pst_teacher;   // lazy, stateless, per worker
            for (;;) {
                const size_t i = next.fetch_add(1);
                if (i >= tasks.size()) break;
                const GameTask& tk = tasks[i];
                const auto net_eval =
                    chess::makeNnueEvaluator(slots[tk.slot], "evolution", quant);
                std::shared_ptr<chess::Evaluator> opp_eval;
                if (tk.vs_champion) {
                    opp_eval = chess::makeNnueEvaluator(slots[0], "evolution", quant);
                } else if (opp_net) {
                    opp_eval = chess::makeNnueEvaluator(opp_net, "opponent", quant);
                } else {
                    if (!pst_teacher) pst_teacher = chess::makePstEvaluator();
                    opp_eval = pst_teacher;
                }
                chess::ChessGame game(openings[tk.opening]);
                if (!tk.net_is_white) game.pos = mirror_position(game.pos);
                const GameOutcome o = play_game(net_eval, opp_eval, game, tk.net_is_white,
                                                config.movetime, table);
                out.scores[i] = o.score;
                tally(o);
            }
        });
    }
    for (auto& th : pool) th.join();

    std::cout << '\r' << label << " done: " << tasks.size() << " games (W " << out.wins
              << " D " << out.draws << " L " << out.losses << "), reasons:";
    for (const auto& [reason, n] : out.reasons) std::cout << ' ' << reason << '=' << n;
    std::cout << ", avg " << (tasks.empty() ? 0.0 : static_cast<double>(out.total_plies) / tasks.size())
              << " plies/game\n";
    return out;
}

std::string zpad3(const size_t n) {
    std::ostringstream o;
    o << (n < 10 ? "00" : n < 100 ? "0" : "") << n;
    return o.str();
}

std::string next_evolved_run_dir() {
    int best = 0;
    if (std::filesystem::exists("evolved_networks")) {
        for (const auto& entry : std::filesystem::directory_iterator("evolved_networks")) {
            const std::string name = entry.path().filename().string();
            if (name.rfind("run_", 0) != 0) continue;
            try {
                best = std::max(best, std::stoi(name.substr(4)));
            } catch (...) { /* not a run_N dir, skip */ }
        }
    }
    return "evolved_networks/run_" + std::to_string(best + 1);
}

int run_evolution(const Config& config) {
    using BackpropNet::Net;

    if (config.net_path.empty())
        throw std::runtime_error("EVO=1 requires NET=path/to/champion/net.backprop_net");
    if (config.pop == 0)
        throw std::runtime_error("EVO: POP must be >= 1");
    if (config.gate_games == 0)
        throw std::runtime_error("EVO: GAMES must be >= 1");
    if (config.h2h_games == 0)
        throw std::runtime_error("EVO: H2H must be >= 1");
    if (config.movetime < 1)
        throw std::runtime_error("EVO: MOVETIME must be >= 1");

    std::string opp_lower = config.evo_opponent;
    for (char& c : opp_lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const bool opp_is_pst = (opp_lower == "pst" || opp_lower.empty());

    const std::vector<std::string> openings = load_openings(config.openings_file);
    if (openings.empty()) throw std::runtime_error("EVO: no openings available");
    for (const std::string& fen : openings) {   // validate early, fail fast
        chess::ChessGame g(fen);
        if (g.status().result != chess::ONGOING)
            throw std::runtime_error("EVO: opening FEN is not an ongoing game: " + fen);
    }

    const std::string run_dir = config.out_dir.empty() ? next_evolved_run_dir() : config.out_dir;
    std::filesystem::create_directories(run_dir);

    auto champion = std::make_shared<Net>(BackpropNetReader::load(config.net_path));
    BackpropNetReader::save(run_dir + "/champ.net", *champion);   // seed the run dir
    const std::shared_ptr<Net> opp_net =
        opp_is_pst ? nullptr
                   : std::make_shared<Net>(BackpropNetReader::load(config.evo_opponent));

    const unsigned nthreads = resolve_threads(config);
    const float veto_need = static_cast<float>(config.h2h_games) / 2.0f + config.hmargin;

    std::ofstream log(run_dir + "/evolution.txt");
    if (!log.is_open())
        throw std::runtime_error("EVO: cannot write " + run_dir + "/evolution.txt");
    log << "gen champ_gate best_child promoted sigma\n";

    float sigma = config.sigma;
    std::mt19937 rng(config.seed ^ 0x0BADF00Du);
    std::vector<std::shared_ptr<Net>> slots;
    size_t games_played = 0, games_total = config.generations * (config.pop + 1) * config.gate_games
                                          + config.generations * config.h2h_games;
    auto t_start = std::chrono::steady_clock::now();

    std::cout << "Champion  : " << config.net_path << "\n"
              << "Opponent  : " << (opp_is_pst ? "material+pst" : config.evo_opponent) << "\n"
              << "Openings  : " << openings.size() << " (each played both colors)\n"
              << "Output    : " << run_dir << "\n"
              << "Threads   : " << nthreads << " (games in parallel, single search each)\n"
              << "Gate      : " << config.gate_games << " games/net/gen, MARGIN=" << config.margin << "\n"
              << "Veto      : " << config.h2h_games << " h2h games, needs " << veto_need << " pts\n"
              << "MOVETIME  : " << config.movetime << " ms\n\n";

    for (size_t gen = 1; gen <= config.generations; ++gen) {
        // slots[0] = champion, slots[1..POP] = this generation's children.
        slots.clear();
        slots.push_back(champion);
        slots.reserve(config.pop + 1);
        for (size_t i = 0; i < config.pop; ++i) {
            auto child = std::make_shared<Net>(clone_net(*champion));
            mutate_net(*child, rng, sigma, config.sigma_abs);
            slots.push_back(std::move(child));
        }

        // Gate suite: champion + every child against the fixed opponent, on the
        // same deterministic openings/colors so scores are comparable.
        std::vector<GameTask> gate_tasks;
        gate_tasks.reserve((config.pop + 1) * config.gate_games);
        for (size_t slot = 0; slot <= config.pop; ++slot) {
            for (size_t g = 0; g < config.gate_games; ++g) {
                GameTask tk;
                tk.slot = slot;
                tk.opening = (g / 2) % openings.size();
                tk.net_is_white = ((g % 2) == 0);
                gate_tasks.push_back(tk);
            }
        }
        const SuiteResult gate =
            run_suite("gate", gate_tasks, slots, opp_net, openings, config, nthreads);
        games_played += gate.scores.size();

        const auto sum_games = [&](const size_t slot) {
            float s = 0.0f;
            for (size_t g = 0; g < config.gate_games; ++g)
                s += gate.scores[slot * config.gate_games + g];
            return s;
        };
        const float champ_gate = sum_games(0);

        size_t best_child = 1;
        float best_score = -1.0f;
        for (size_t c = 1; c <= config.pop; ++c) {
            const float s = sum_games(c);
            if (s > best_score) { best_score = s; best_child = c; }
        }

        // Promotion: PST gate + champion veto (head-to-head vs the incumbent).
        bool promoted = false;
        float child_veto = 0.0f;
        if (best_score >= champ_gate + config.margin) {
            std::vector<GameTask> veto_tasks;
            veto_tasks.reserve(config.h2h_games);
            for (size_t h = 0; h < config.h2h_games; ++h) {
                GameTask tk;
                tk.slot = best_child;
                tk.vs_champion = true;
                tk.opening = (h / 2) % openings.size();
                tk.net_is_white = ((h % 2) == 0);
                veto_tasks.push_back(tk);
            }
            const SuiteResult veto =
                run_suite("veto", veto_tasks, slots, opp_net, openings, config, nthreads);
            games_played += veto.scores.size();
            for (const float s : veto.scores) child_veto += s;
            if (child_veto >= veto_need) promoted = true;
        }

        const auto t_gen = std::chrono::steady_clock::now();
        const double gen_s = std::chrono::duration<double>(t_gen - t_start).count();
        const std::string elapsed = [&] {
            std::ostringstream o;
            o << std::fixed << std::setprecision(1) << (gen_s / 60.0) << "m";
            return o.str();
        }();
        const std::string eta = [&] {
            if (gen < 2 || t_start == t_gen) return std::string("-");
            const double per_gen = gen_s / static_cast<double>(gen);
            double rem = per_gen * static_cast<double>(config.generations - gen);
            std::ostringstream o;
            o << std::fixed << std::setprecision(1) << (rem / 60.0) << "m";
            return o.str();
        }();

        const std::shared_ptr<Net> best_of_gen = slots[best_child];
        if (promoted) {
            std::swap(slots[0], slots[best_child]);
            champion = slots[0];
            BackpropNetReader::save(run_dir + "/champ.net", *champion);
        } else {
            sigma *= config.sigma_decay;
        }

        const std::string gen_dir = run_dir + "/gen_" + zpad3(gen);
        std::filesystem::create_directories(gen_dir);
        BackpropNetReader::save(gen_dir + "/net", *best_of_gen);

        log << gen << ' ' << champ_gate << ' ' << best_score << ' '
            << (promoted ? 1 : 0) << ' ' << sigma << '\n';
        log.flush();

        std::cout << "Gen " << gen << '/' << config.generations
                  << " | champ_gate=" << champ_gate
                  << " best_child=" << best_score
                  << "/" << config.gate_games
                  << " promoted=" << (promoted ? "yes" : "no")
                  << " sigma=" << sigma
                  << " | games " << games_played << '/' << games_total
                  << " elapsed " << elapsed << " | ETA " << eta << (eta == "-" ? "\n" : "\n\n");
    }

    std::cout << "\nDone. Champion saved: " << run_dir << "/champ.net.backprop_net\n"
              << "Best child per gen: " << run_dir << "/gen_NNN/net.backprop_net\n";
    return 0;
}

}   // namespace

int main(int argc, char* argv[]) {
    try {
        const Config config = parse_args(argc, argv);
        if (config.probe) return run_probe(config);
        if (config.evo) return run_evolution(config);

        const unsigned nthreads = resolve_threads(config);

        // Resolve the starting net and its label convention FIRST, because the
        // dataset targets must be encoded to match: a fresh net defaults to a
        // linear head (raw centipawn targets; WINPROB=1 opts into win-prob),
        // while continuing from a checkpoint re-derives the convention from the
        // checkpoint's final activation (NetEvaluator auto-detects it at eval).
        const bool fresh = config.continue_path.empty();
        bool winprob = false;
        if (config.win_prob >= 0) {
            winprob = (config.win_prob != 0);
        } else if (fresh) {
            winprob = false;   // fresh default: linear head
        }
        // else (not fresh, no flag): winprob is re-derived from the checkpoint in load_net().
        if (fresh) {
            std::cout << "Architecture: " << describe_architecture(config.hidden, winprob) << "\n";
        } else {
            std::cout << "Continuing from: " << config.continue_path << "\n";
        }
        std::cout << "Threads: " << nthreads << " (data loading + validation)\n";
        std::cout << (winprob ? "LABELS: win-prob (targets tanh(cp/400); reported MSE is in win-prob units, ~0.01-0.05 good fit)"
                              : "LABELS: linear-cp (raw-centipawn targets; reported MSE is in cp^2, so sqrt(MSE) = RMS centipawns)")
                  << "\n";

        Dataset train, val;
        if (!config.dataset_path.empty() && config.fen_csv) {
            std::cout << "Loading FEN-eval dataset: " << config.dataset_path << "\n";
            generate_dataset_from_fen_csv(train, val, config, winprob, nthreads);
        } else if (!config.dataset_path.empty()) {
            std::cout << "Loading Lichess-eval dataset: " << config.dataset_path << "\n";
            CsvLoadStats stats;
            generate_dataset_csv(train, val, config, winprob, stats, nthreads);
        } else {
            std::cout << "Generating PST-distillation dataset...\n";
            generate_dataset(train, val, config, winprob, nthreads);
        }

        std::cout << "Train: " << train.inputs.size() << " | Val: " << val.inputs.size() << "\n";


        const std::string run_dir = config.out_dir.empty() ? next_run_dir() : config.out_dir;
        std::filesystem::create_directories(run_dir);


        BackpropNet::Net net;
        if (fresh) {
            net = create_architecture(config.hidden, config.clamp_cp, winprob);
        } else {
            net = load_net(config.continue_path, config.hidden, winprob);
            std::cout << (winprob ? "Checkpoint label convention: win-prob (S*tanh(cp/S))"
                                  : "Checkpoint label convention: legacy linear cp")
                      << "\n";
        }
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
             << "HIDDEN: " << describe_architecture(config.hidden, winprob) << "\n"
             << "THREADS: " << nthreads << "\n"
             << "SAMPLES: " << config.samples << " EPOCHS: " << config.epochs
             << " BATCH: " << config.batch << " LR: " << config.lr
             << " WEIGHT_DECAY: " << config.weight_decay
             << " PATIENCE: " << config.patience << " CLAMP: " << config.clamp_cp
             << " PST_SCALE: " << config.pst_scale
             << " MIRROR: " << config.mirror << " CAPTURE_BIAS: " << config.capture_bias
             << " SKIP_MATES: " << config.skip_mates
             << " LABELS: " << (winprob ? "win-prob" : "linear-cp")
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
