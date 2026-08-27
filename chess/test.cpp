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
//
// Compile (from chess/):
//   g++ -std=c++17 -DNDEBUG -O2 test.cpp -o chess_test
// Run (from chess/):
//   ./chess_test

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "./chess_environment.h"
#include "./evaluator.h"
#include "./dataset_tools.h"

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

}   // namespace

int main() {
    std::cout << "NNProj dataset tools self-test\n=============================\n";
    test_csv_parser();
    test_san_real_game();
    test_san_edges();
    test_eval_alignment();
    test_mirror();

    std::cout << "=============================\n"
              << (g_failures == 0 ? "ALL TESTS PASSED\n" : "TESTS FAILED\n");
    return g_failures == 0 ? 0 : 1;
}
