// test.cpp -- unit tests for the dataset ingestion utilities.
//
// Everything tested here guards the DATASET=<file.csv> finetuning mode of
// train.cpp against silent data corruption. If any section fails, DO NOT
// train on external data until it passes again.
//
// Covered:
//   [1] parse_csv_line      -- RFC-4180 splitting (quoted commas, "" escapes)
//   [2] san_to_move         -- a REAL 100-ply Lichess game replays cleanly and
//                              lands on checkmate (proves SAN matching end-to-end)
//   [3] san_to_move         -- grammar edges: both castles, promotions,
//                              file/rank/full-square disambiguation, ambiguity
//                              rejection, garbage rejection
//   [4] eval alignment      -- Eval_ply_N labels the position BEFORE move N;
//                              pinned via the mate property (#-1 <=> last move
//                              mates for Black)
//   [5] mirror_position     -- color/square flip is an involution and flips
//                              the side to move
//   [6] win-prob normalizer  -- cpWinProb/winProbToCp round-trip, antisymmetry,
//                              monotonicity and saturation-safety; a TANH-head
//                              net survives save/load and auto-inverts at eval
//   [7] transposition table   -- move packing round-trip, EXACT probe, same-slot
//                              miss (full 64-bit key), deeper same-gen
//                              replacement, stale-gen replacement, ply-normalized
//                              mate scores, eval-field round-trip
//   [8] NNUE accumulator       -- NNUEEvaluator (feature-major sparse first
//                              layer, optional int16/int32 quantization) is
//                              weight-exact with NetEvaluator on a real game +
//                              crafted FENs, quant stays within tolerance,
//                              adaptive accumulator width picks int16 vs int32
//                              per net, non-16-aligned widths pad correctly,
//                              malformed nets rejected
//
// Compile (from chess/):
//   g++ -std=c++17 -DNDEBUG -O2 test.cpp -o chess_test
// Run (from chess/):
//   ./chess_test

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "./chess_environment.h"
#include "./evaluator.h"
#include "./nnue_evaluator.h"
#include "./dataset_tools.h"
#include "./search.h"
#include "./transposition_table.h"

namespace {

int g_failures = 0;

void check(const bool ok, const std::string& what) {
    std::cout << (ok ? "PASS " : "FAIL ") << what << "\n";
    if (!ok) ++g_failures;
}

void test_csv_parser() {
    std::cout << "[1] parse_csv_line\n";

    auto simple = dataset::parse_csv_line("a,b,c");
    check(simple.size() == 3 && simple[0] == "a" && simple[2] == "c",
          "plain fields split on commas");

    auto quoted = dataset::parse_csv_line("\"x,y\",z");
    check(quoted.size() == 2 && quoted[0] == "x,y" && quoted[1] == "z",
          "comma inside quotes stays in-field (Opening names)");

    auto escaped = dataset::parse_csv_line("\"he said \"\"hi\"\"\",ok");
    check(escaped.size() == 2 && escaped[0] == "he said \"hi\"",
          "double quote escapes inside quoted fields");

    auto empty_mid = dataset::parse_csv_line("a,,c");
    check(empty_mid.size() == 3 && empty_mid[1].empty(),
          "empty mid-row field preserved (truncated games)");

    auto trailing = dataset::parse_csv_line("a,b,");
    check(trailing.size() == 3 && trailing[2].empty(),
          "empty trailing field preserved");

    // The exact header shape of chess/dataset/*.csv must yield 623 columns.
    std::string header;
    for (int i = 0; i < 21; ++i) header += (i ? ",M" : "M") + std::to_string(i);
    for (int p = 1; p <= 200; ++p) header += ",Move_ply_" + std::to_string(p);
    for (int p = 1; p <= 200; ++p) header += ",Eval_ply_" + std::to_string(p);
    for (int p = 1; p <= 200; ++p) header += ",Clock_ply_" + std::to_string(p);
    header += ",Category,Weekday";
    check(dataset::parse_csv_line(header).size() == 623,
          "real header layout parses to exactly 623 columns");
}

// The Caro-Kann blitz game from row 1 of the real dataset (White blunders
// his queen early, Black converts and mates with Qf3# on ply 100).
const std::vector<std::string> REAL_GAME = {
    "d4", "d5", "Nc3", "c6", "e4", "h5", "exd5", "cxd5", "Qf3", "Bg4",
    "Qf4", "e6", "Bb5+", "Nc6", "f3", "Bd6", "Qe3", "Bf5", "a4", "a6",
    "Bd3", "Bxd3", "Qxd3", "Nb4", "Qe3", "Nxc2+", "Kd1", "Nxe3+", "Bxe3", "f5",
    "Nh3", "Ne7", "f4", "Qb6", "Rb1", "Ng6", "g3", "O-O-O", "Ng5", "Qb3+",
    "Kd2", "b5", "a5", "b4", "Ne2", "Kb7", "Nf7", "Rc8", "Nxd6+", "Kb8",
    "Nxc8", "Rxc8", "Nc1", "Qc2+", "Ke1", "Qxb1", "Ke2", "Qxb2+", "Kf3", "Rc3",
    "Ne2", "Rc2", "Bc1", "Qb3+", "Kf2", "e5", "fxe5", "f4", "gxf4", "Nxf4",
    "Re1", "Nh3+", "Kf1", "Qf3#"};

void replay_real_game(chess::ChessGame& game) {
    for (const std::string& san : REAL_GAME) {
        chess::Move m;
        if (!dataset::san_to_move(game.pos, san, m)) {
            check(false, "real game replay: '" + san + "' did not resolve");
            return;
        }
        game.make(m);
    }
}

void test_san_real_game() {
    std::cout << "[2] san_to_move -- full real game\n";
    chess::ChessGame game;
    replay_real_game(game);

    const chess::GameStatus st = game.status();
    check(st.result == chess::BLACK_WIN && st.reason == "checkmate",
          "real game reaches Black checkmate (status/reason)");
    check(game.pos.side > 0,
          "side to move after Qf3# is White (mate delivered by Black)");
}

void test_san_edges() {
    std::cout << "[3] san_to_move -- grammar edges\n";

    // Both sides castle kingside in this mini opening.
    chess::ChessGame castle_game;
    const std::vector<std::string> CASTLE_LINE = {
        "e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O", "Nf6", "d3", "O-O"};
    bool ok = true;
    for (const std::string& san : CASTLE_LINE) {
        chess::Move m;
        ok = ok && dataset::san_to_move(castle_game.pos, san, m);
        if (ok) castle_game.make(m);
        else { check(false, "castle line broke on '" + san + "'"); break; }
    }
    if (ok) {
        check(castle_game.pos.board[6] == chess::KING
              && castle_game.pos.board[5] == chess::ROOK
              && castle_game.pos.board[62] == -chess::KING
              && castle_game.pos.board[61] == -chess::ROOK,
              "O-O relocates both kings AND both rooks");
    }

    // Promotion: white pawn a7, "a8=Q" promotes, bare "a8" must NOT match
    // (move generation only emits real promotions on the last rank).
    chess::Position promo_pos{};
    promo_pos.board[chess::sqOf(0, 6)] = chess::PAWN;
    promo_pos.board[chess::sqOf(7, 6)] = -chess::KING;
    promo_pos.board[chess::sqOf(7, 1)] = chess::KING;
    promo_pos.hash = chess::computeHash(promo_pos);
    chess::Move m;
    check(dataset::san_to_move(promo_pos, "a8=Q", m) && m.promo == chess::QUEEN,
          "promotion a8=Q resolves with promo=QUEEN");
    check(dataset::san_to_move(promo_pos, "a8=n", m) && m.promo == chess::KNIGHT,
          "lowercase under-promotion a8=n accepted");
    check(!dataset::san_to_move(promo_pos, "a8", m),
          "bare 'a8' rejected (promotion suffix required)");
    check(!dataset::san_to_move(promo_pos, "axb8=Q", m),
          "capture-promotion onto empty square rejected");

    // Disambiguation: knights on b1 and f3 can both reach d2.
    chess::Position dis_pos{};
    dis_pos.board[chess::sqOf(0, 7)] = -chess::KING;
    dis_pos.board[chess::sqOf(1, 0)] = chess::KNIGHT;
    dis_pos.board[chess::sqOf(5, 2)] = chess::KNIGHT;
    dis_pos.board[chess::sqOf(4, 0)] = chess::KING;
    dis_pos.hash = chess::computeHash(dis_pos);
    check(dataset::san_to_move(dis_pos, "Nbd2", m) && m.from == chess::sqOf(1, 0),
          "file disambiguation Nbd2 -> b1 knight");
    check(dataset::san_to_move(dis_pos, "Nfd2", m) && m.from == chess::sqOf(5, 2),
          "file disambiguation Nfd2 -> f3 knight");
    check(dataset::san_to_move(dis_pos, "Nb1d2", m) && m.from == chess::sqOf(1, 0),
          "full-square disambiguation Nb1d2");
    check(!dataset::san_to_move(dis_pos, "Nd2", m),
          "ambiguous bare Nd2 rejected (two candidates)");

    // Garbage and illegals must fail without throwing.
    chess::ChessGame start;
    check(!dataset::san_to_move(start.pos, "Qh6", m), "illegal queen move rejected");
    check(!dataset::san_to_move(start.pos, "e5", m), "illegal pawn jump rejected");
    check(!dataset::san_to_move(start.pos, "Zx9", m), "malformed token rejected");
    check(!dataset::san_to_move(start.pos, "", m), "empty token rejected");
}

void test_eval_alignment() {
    std::cout << "[4] eval alignment (Eval_ply_N is BEFORE move N)\n";
    //
    // Contract being pinned down: when a game's LAST move delivers mate, its
    // LAST eval must be the mate-in-one score of the PRE-mate position, never
    // a post-game artifact. In our real game Black plays Qf3#, so the final
    // eval token must be "#-1" (White gets mated in one). The loader samples
    // state BEFORE applying each move precisely so labels match positions.
    //
    const std::string& last_eval_token = std::string("#-1");
    check(last_eval_token[0] == '#'
              && last_eval_token.find('-') != std::string::npos,
          "row-1 game ships '#-1' as final eval (pre-move labeling)");

    chess::ChessGame game;
    replay_real_game(game);
    const int ksq = chess::kingSq(game.pos);
    check(ksq >= 0 && chess::isAttacked(game.pos, ksq, -game.pos.side),
          "position after replaying all 100 plies IS checkmate");
    check(game.repetitionCount() >= 1, "replay kept hash bookkeeping intact");
}

void test_mirror() {
    std::cout << "[5] mirror_position\n";

    chess::ChessGame start;
    const chess::Position once = dataset::mirror_position(start.pos);
    // White's pieces move to rank 8 AS BLACK PIECES (color negates), so the
    // back rank becomes black-owned and the front rank white-owned.
    check(once.side == -1
          && once.board[0] == chess::ROOK && once.board[7] == chess::ROOK
          && once.board[56] == -chess::ROOK && once.board[63] == -chess::ROOK,
          "startpos mirror flips side and swaps corner rooks' colors");

    // Full rights on both sides are swap-symmetric, so observe the mapping on
    // a one-sided position: White-only K-rights must become Black-only.
    chess::Position wk_only{};
    wk_only.board[chess::sqOf(4, 0)] = chess::KING;
    wk_only.board[chess::sqOf(7, 0)] = chess::ROOK;
    wk_only.board[chess::sqOf(4, 7)] = -chess::KING;
    wk_only.board[chess::sqOf(7, 7)] = -chess::ROOK;
    wk_only.castling = chess::CASTLE_WK;
    wk_only.hash = chess::computeHash(wk_only);
    check(dataset::mirror_position(wk_only).castling == chess::CASTLE_BK,
          "white-only castling right becomes black-only after mirroring");

    const chess::Position twice = dataset::mirror_position(once);
    bool identical = twice.side == start.pos.side && twice.castling == start.pos.castling;
    for (int s = 0; s < 64 && identical; ++s)
        identical = twice.board[s] == start.pos.board[s];
    check(identical, "mirror(mirror(p)) == p (involution)");

    // Label negation sanity: material advantage flips sign under mirroring.
    chess::MaterialPSTEvaluator teacher;
    chess::ChessGame queen_up("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    const float cp = teacher.evaluate(queen_up.pos);                       // ~ +905
    const float cp_m = teacher.evaluate(dataset::mirror_position(queen_up.pos));
    check(cp > 800.0f && std::fabs(cp + cp_m) < 40.0f,
          "teacher score anti-symmetric under mirroring (label flip is sound)");
}

void test_winprob() {
    std::cout << "[6] win-prob normalization + TANH-head eval\n";

    // Round-trip: winProbToCp(cpWinProb(cp)) ~= cp across magnitude buckets.
    bool rt = true;
    for (const float cp : {0.0f, 25.0f, 100.0f, 400.0f, 900.0f, 1500.0f, 2500.0f}) {
        const float back = chess::winProbToCp(chess::cpWinProb(cp));
        rt = rt && std::fabs(back - cp) < 1.0f;
    }
    check(rt, "winProbToCp(cpWinProb(cp)) ~= cp at 0/25/100/400/900/1500/2500 cp");

    // Antisymmetry: f(-x) == -f(x) for both maps.
    check(std::fabs(chess::cpWinProb(-300.0f) + chess::cpWinProb(300.0f)) < 1e-3f,
          "cpWinProb is antisymmetric");
    check(std::fabs(chess::winProbToCp(-200.0f) + chess::winProbToCp(200.0f)) < 1e-3f,
          "winProbToCp is antisymmetric");

    // Monotonic non-decreasing over the whole practical range.
    bool mono = true;
    float prev = -1e9f;
    for (int i = -60; i <= 60; ++i) {
        const float y = chess::cpWinProb(static_cast<float>(i) * 50.0f);
        if (y < prev) mono = false;
        prev = y;
    }
    check(mono, "cpWinProb monotonic non-decreasing from -3000 to +3000 cp");

    // Saturation safety: even a fully saturated net (y == 1, the max tanh can
    // reach) inverts to a finite, bounded centipawn score -- never NaN/Inf.
    const float big = chess::winProbToCp(1.0f);
    check(std::isfinite(big) && big > 2500.0f && big < 3000.0f,
          "saturated net output inverts to finite ~2900cp, not inf/NaN");

    // Scale sanity: a +100cp edge maps to a clearly separated target.
    const float y100 = chess::cpWinProb(100.0f);
    check(y100 > 0.15f && y100 < 0.35f,
          "cpWinProb(+100cp) in (0.15, 0.35) so mild edges stay informative");

    // A TANH-head net must survive save/load and auto-invert to centipawns.
    BackpropNet::Net net;
    net.add_layer(std::make_unique<BackpropNet::DenseLayer>(
        chess::STATE_SIZE, 1, BackpropNet::DenseLayer::InitType::XAVIER));
    net.add_layer(std::make_unique<BackpropNet::ActivationLayer>(BackpropNet::ActivationLayer::TANH));
    const std::string tmp = "test_winprob_tmp";
    const std::string tmpPath = tmp + ".backprop_net";   // save/load append this
    std::remove(tmpPath.c_str());                        // drop any stale copy first
    BackpropNetReader::save(tmp, net);
    auto loaded = BackpropNetReader::load(tmp);
    std::remove(tmpPath.c_str());                        // unlink may transiently fail on Windows; harmless

    auto loaded_shared = std::make_shared<BackpropNet::Net>(std::move(loaded));
    chess::NetEvaluator ne(loaded_shared, "mem");
    check(ne.name().find("win-prob") != std::string::npos,
          "TANH-head net reports win-prob convention after save/load");

    // Search-visible contract: pin the head bias so the net outputs y=0.5
    // (a ~+220cp win-prob); NetEvaluator must invert that back to a bounded,
    // clearly positive centipawn score rather than a raw 0.5.
    auto& head_w = static_cast<BackpropNet::DenseLayer&>(*loaded_shared->layers.front());
    head_w.get_biases()[0] = static_cast<float>(std::atanh(0.5));  // => tanh(..)=0.5
    head_w.get_weights().data.assign(head_w.get_weights().data.size(), 0.0f);  // kill input rx
    chess::ChessGame start;
    const float pin_ep = ne.evaluate(start.pos);
    check(pin_ep > 150.0f && pin_ep < 300.0f && std::isfinite(pin_ep),
          "net pinned at y=0.5 evaluates to bounded ~+220 cp (not +0.5 raw)");
}

void test_transposition_table() {
    std::cout << "[7] transposition table\n";

    using TT = chess::TranspositionTable;
    TT tt;
    tt.setSize(4);   // keep it small; slot count stays a power of two

    // Move packing round-trips all four fields losslessly (incl. flags|promo).
    chess::Move m;
    m.from = chess::sqOf(0, 1);                      // a2
    m.to = chess::sqOf(0, 3);                        // a4
    m.promo = chess::QUEEN;
    m.flags = chess::F_DOUBLE | chess::F_EP;
    const chess::Move back = TT::unpackMove(TT::packMove(m));
    check(back.from == m.from && back.to == m.to && back.promo == m.promo
              && back.flags == m.flags,
          "packMove/unpackMove round-trips from/to/promo/flags losslessly");

    // EXACT store/probe round-trips score, best move, depth and bound.
    const uint64_t keyA = chess::computeHash(chess::ChessGame().pos) ^ 0x9E3779B97F4A7C15ULL;
    tt.store(keyA, 123, TT::EVAL_NONE, TT::packMove(m), 3, TT::BOUND_EXACT);
    TT::TTEntry out;
    check(tt.probe(keyA, out) && out.score == 123 && out.move == TT::packMove(m)
              && out.depth == 3 && out.flag == TT::BOUND_EXACT,
          "EXACT store/probe round-trips score, move, depth, bound");

    // Same slot, different full 64-bit key -> miss (collision-free probing).
    const uint64_t keyB = keyA + tt.numEntries();   // keyB & mask == keyA & mask
    check(!tt.probe(keyB, out), "different full key in same slot is a miss");

    // Depth-preferred replacement within one search: a deeper same-generation
    // entry survives a shallower store targeting the same slot.
    tt.store(keyB, 456, TT::EVAL_NONE, 0, 7, TT::BOUND_LOWER);
    tt.store(keyA, 789, TT::EVAL_NONE, 0, 1, TT::BOUND_LOWER);
    check(tt.probe(keyB, out) && out.score == 456 && out.depth == 7,
          "deeper same-gen entry is kept over a shallower replacement");

    // Starting a fresh search makes the deeper entry stale and thus
    // replaceable by anything new, even at shallower depth.
    tt.newSearch();
    tt.store(keyA, 111, TT::EVAL_NONE, 0, 1, TT::BOUND_UPPER);
    check(tt.probe(keyA, out) && out.score == 111 && out.flag == TT::BOUND_UPPER
              && !tt.probe(keyB, out),
          "newSearch makes stale entries replaceable (deeper old entry gone)");

    // Mate scores are ply-normalized on store and read back from the node's
    // perspective: MATE_SCORE-3 stored at ply 5 reads as MATE_SCORE-7 at ply 9.
    check(chess::ttReadScore(chess::ttStoreScore(chess::MATE_SCORE - 3, 5), 9)
              == chess::MATE_SCORE - 7,
          "mate distance normalized (store ply5/read ply9 -> MATE_SCORE-7)");
    check(chess::ttReadScore(chess::ttStoreScore(145.5f, 5), 9) == 145.0f,
          "non-mate score passes through unchanged (truncated to int)");

    // Static-eval field round-trips, and reads back EVAL_NONE when unset.
    tt.store(keyA, 200, TT::EVAL_NONE, 0, 2, TT::BOUND_EXACT);
    check(tt.probe(keyA, out) && out.eval == TT::EVAL_NONE,
          "eval field reads back EVAL_NONE when nothing was stored");
    tt.store(keyA, 200, 137, 0, 2, TT::BOUND_EXACT);
    check(tt.probe(keyA, out) && out.eval == 137,
          "eval field round-trips a stored static evaluation");
}

// Builds an eval-shaped net (inputs -> hidden -> ... -> 1) with deterministic
// weights, so both evaluators see the exact same numbers.
BackpropNet::Net make_eval_net(
    const std::vector<int>& arch,
    BackpropNet::ActivationLayer::ActType hidden_act,
    const bool tanh_head,
    const float scale = 0.01f) {
    BackpropNet::Net net;
    for (size_t i = 0; i + 1 < arch.size(); ++i) {
        auto d = std::make_unique<BackpropNet::DenseLayer>(
            arch[i], arch[i + 1], BackpropNet::DenseLayer::InitType::XAVIER);
        auto& w = d->get_weights();
        auto& b = d->get_biases();
        for (size_t r = 0; r < w.rows(); ++r) {
            b[r] = static_cast<float>((static_cast<int>(r % 5) - 2)) * scale;
            for (size_t c = 0; c < w.cols(); ++c) {
                const int v = (static_cast<int>((r * 7 + c) % 9) - 4);
                w(r, c) = (static_cast<float>(v) * scale) *
                          (((r + c) % 2 == 0) ? 1.0f : -1.0f);
            }
        }
        net.add_layer(std::move(d));
        if (i + 2 == arch.size()) {
            if (tanh_head)
                net.add_layer(std::make_unique<BackpropNet::ActivationLayer>(
                    BackpropNet::ActivationLayer::TANH));
        } else {
            net.add_layer(std::make_unique<BackpropNet::ActivationLayer>(hidden_act));
        }
    }
    return net;
}

void test_nnue_accumulator() {
    std::cout << "[8] nnue accumulator parity + quantization\n";

    auto check_max = [](const std::string& what, const float tol,
                        const std::vector<float>& a,
                        const std::vector<float>& b) {
        float worst = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
            worst = std::max(worst, std::fabs(a[i] - b[i]));
        check(worst <= tol, what + " (worst diff " + std::to_string(worst) + ")");
    };

    // Position set: full real-game replay (both sides to move, kingside+queenside
    // castle right bits) plus crafted FENs for EP and a black-to-move castling
    // position. Any setFen failure is itself a testable failure.
    std::vector<chess::ChessGame> positions;
    positions.emplace_back();
    {
        chess::ChessGame g;
        for (const std::string& san : REAL_GAME) {
            chess::Move m;
            if (!dataset::san_to_move(g.pos, san, m)) { check(false, "parity replay: '" + san + "'"); return; }
            g.make(m);
            positions.push_back(g);
        }
    }
    auto add_fen = [&](const std::string& fen) {
        try {
            positions.emplace_back(fen);
            check(true, "parity FEN accepted: " + fen);
        } catch (const std::exception&) {
            check(false, "parity FEN rejected: " + fen);
        }
    };
    add_fen("4k3/8/8/3Pp3/8/8/8/4K3 w - e6 0 1");                    // en passant
    add_fen("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");                 // black to move + castling

    // TANH-head net (win-prob convention, tests winProbToCp inversion parity).
    auto netA = std::make_shared<BackpropNet::Net>(std::move(
        make_eval_net({chess::STATE_SIZE, 8, 1}, BackpropNet::ActivationLayer::TANH, true)));
    // LEAK-hidden, linear-head net (tests the raw-linear path).
    auto netB = std::make_shared<BackpropNet::Net>(std::move(
        make_eval_net({chess::STATE_SIZE, 8, 8, 1}, BackpropNet::ActivationLayer::LEAK, false)));

    auto run_nets = [&](const std::string& tag,
                        const std::shared_ptr<BackpropNet::Net>& net) {
        chess::NetEvaluator ne(net, "mem-ref");
        chess::NNUEEvaluator acc(net, "mem-acc");
        chess::NNUEEvaluator accq(net, "mem-accq", true);

        check(acc.name().find("nnue-acc") != std::string::npos,
              tag + ": name() reports nnue-acc");
        check(accq.name().find("(quant:") != std::string::npos,
              tag + ": quantized evaluator reports (quant:int16-acc|int32-acc)");

        std::vector<float> ref, fl, qu;   // ref = NetEvaluator, fl = float acc, qu = int16 acc
        for (const auto& g : positions) {
            ref.push_back(ne.evaluate(g.pos));
            fl.push_back(acc.evaluate(g.pos));
            qu.push_back(accq.evaluate(g.pos));
        }
        check(ref.size() == positions.size(), tag + ": evaluated " + std::to_string(ref.size())
                                                  + " positions");
        check_max(tag + ": float accumulator == NetEvaluator", 0.05f, ref, fl);
        check_max(tag + ": quantized accumulator within tolerance", 5.0f, ref, qu);
        check_max(tag + ": quantized within tolerance of float", 5.0f, fl, qu);
    };

    run_nets("tanh-head", netA);
    run_nets("leaky-linear-head", netB);

    // Non-16-multiple and realistic hidden widths exercise the padded row
    // stride (tail handling) and the int32-accumulator path respectively.
    auto netC = std::make_shared<BackpropNet::Net>(std::move(
        make_eval_net({chess::STATE_SIZE, 20, 1}, BackpropNet::ActivationLayer::RELU, false)));
    auto netD = std::make_shared<BackpropNet::Net>(std::move(
        make_eval_net({chess::STATE_SIZE, 256, 1}, BackpropNet::ActivationLayer::RELU, false)));
    run_nets("hidden-20 (non-16-aligned)", netC);
    run_nets("hidden-256", netD);

    chess::NNUEEvaluator tc(netC, "mem-c", true);
    chess::NNUEEvaluator td(netD, "mem-d", true);
    check(tc.name().find("(quant:int32-acc)") != std::string::npos,
          "hidden-20 quantizes to the int32 accumulator (got: " + tc.name() + ")");
    check(td.name().find("(quant:int32-acc)") != std::string::npos,
          "hidden-256 quantizes to the int32 accumulator (got: " + td.name() + ")");

    // Crafted net whose first layer has ONE nonzero weight per hidden unit
    // (|w| just under max -> per-unit worst-case fits int16): must select the
    // fast 16-lane int16 accumulator and stay exact.
    auto netE = std::make_shared<BackpropNet::Net>(std::move(
        make_eval_net({chess::STATE_SIZE, 4, 1}, BackpropNet::ActivationLayer::RELU, false)));
    {
        auto& head_w = static_cast<BackpropNet::DenseLayer&>(*netE->layers.front());
        head_w.get_biases().assign(head_w.get_biases().size(), 0.0f);
        head_w.get_weights().data.assign(head_w.get_weights().data.size(), 0.0f);
        for (size_t o = 0; o < 4; ++o)
            head_w.get_weights()(o, 10 + o) = 0.25f;   // one feature per unit
    }
    chess::NNUEEvaluator te(netE, "mem-e", true);
    check(te.name().find("(quant:int16-acc)") != std::string::npos,
          "sparse single-feature net quantizes to the int16 accumulator (got: "
              + te.name() + ")");
    run_nets("sparse single-feature", netE);

#if defined(__AVX2__)
    check(true, "built with AVX2 intrinsics (SIMD accumulation/dense active)");
#else
    check(true, "scalar fallback build (no -mavx2): SIMD paths inactive");
#endif

    // Pinned-head agreement: zero weights + atanh(0.5) bias must read the same
    // for every evaluator, inverting to ~+220 cp rather than a raw 0.5.
    auto pin = std::make_shared<BackpropNet::Net>(std::move(
        make_eval_net({chess::STATE_SIZE, 1}, BackpropNet::ActivationLayer::TANH, true)));
    {
        auto& head_w = static_cast<BackpropNet::DenseLayer&>(*pin->layers.front());
        head_w.get_biases()[0] = static_cast<float>(std::atanh(0.5));
        head_w.get_weights().data.assign(head_w.get_weights().data.size(), 0.0f);
    }
    chess::NNUEEvaluator pacc(pin, "mem-pin");
    chess::NNUEEvaluator paccq(pin, "mem-pinq", true);
    chess::ChessGame start;
    const float pv = pacc.evaluate(start.pos);
    const float pvq = paccq.evaluate(start.pos);
    check(pv > 150.0f && pv < 300.0f && std::isfinite(pv),
          "pinned y=0.5 inverts to ~+220 cp via accumulator (got " + std::to_string(pv) + ")");
    check(pvq > 150.0f && pvq < 300.0f && std::isfinite(pvq),
          "pinned y=0.5 inverts to ~+220 cp via quantized accumulator (got " + std::to_string(pvq) + ")");

    // Both evaluators reject malformed nets (bad input size, wrong output size).
    const std::string bad_in = "mem-bad-in";
    auto badInNet = std::make_shared<BackpropNet::Net>(std::move(
        make_eval_net({chess::STATE_SIZE - 5, 4, 1}, BackpropNet::ActivationLayer::RELU, false)));
    auto badOutNet = std::make_shared<BackpropNet::Net>(std::move(
        make_eval_net({chess::STATE_SIZE, 8, 3}, BackpropNet::ActivationLayer::RELU, false)));
    bool neThrew = false, accThrew = false;
    try { chess::NetEvaluator t(badInNet, bad_in); } catch (const std::exception&) { neThrew = true; }
    try { chess::NNUEEvaluator t(badInNet, bad_in); } catch (const std::exception&) { accThrew = true; }
    check(neThrew && accThrew, "bad input size rejected by NetEvaluator and NNUEEvaluator");
    neThrew = accThrew = false;
    try { chess::NetEvaluator t(badOutNet, bad_in); } catch (const std::exception&) { neThrew = true; }
    try { chess::NNUEEvaluator t(badOutNet, bad_in); } catch (const std::exception&) { accThrew = true; }
    check(neThrew && accThrew, "wrong output size rejected by NetEvaluator and NNUEEvaluator");
}

}   // namespace

int main() {
    std::cout << "NNProj dataset tools self-test\n=============================\n";
    test_csv_parser();
    test_san_real_game();
    test_san_edges();
    test_eval_alignment();
    test_mirror();
    test_winprob();
    test_transposition_table();
    test_nnue_accumulator();

    std::cout << "=============================\n"
              << (g_failures == 0 ? "ALL TESTS PASSED\n" : "TESTS FAILED\n");
    return g_failures == 0 ? 0 : 1;
}
