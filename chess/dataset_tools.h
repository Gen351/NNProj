#pragma once

// dataset_tools.h -- shared utilities for turning external chess data
// (Lichess eval dumps) into engine-native samples.
//
// Used by:
//   chess/train.cpp   DATASET=<file.csv> finetuning mode
//   chess/test.cpp    unit tests for everything below
//
// ---------------------------------------------------------------------------
// FILE FORMAT (see chess/dataset/*.csv)
// ---------------------------------------------------------------------------
// Standard CSV (RFC-4180): comma-separated, one GAME per line, header row
// first. Fields MAY be double-quoted; inside quotes a comma is literal and
// "" is an escaped quote. Example quoted field:
//     "Caro-Kann Defense: Panov Attack, Modern Defense, Mieses Line"
//
// Column layout is positional (623 columns):
//   [0   ..20 ]  game metadata (ignored)
//   [21  ..220]  Move_ply_1 .. Move_ply_200   -- SAN, e.g. d4 exd5 Nbd7 O-O
//   [221 ..420]  Eval_ply_1 .. Eval_ply_200   -- White-POV evals in PAWNS
//                                                 ("0.25" = +25cp); "#N"/"#-N"
//                                                 = mate in |N| moves. The
//                                                 trainer converts x100 to
//                                                 centipawns at load time.
//   [421 ..620]  Clock_ply_* (ignored)
//   [621..622 ]  Category, Weekday (ignored)
//
// EVAL ALIGNMENT (verified empirically): Eval_ply_N belongs to the position
// BEFORE move N is played (i.e. after N-1 plies), from White's point of
// view. A game whose last move is checkmate therefore ends with an eval of
// "#-1" or "#1", never "#0". test.cpp asserts this property on a real game.
//
// SAN dialect supported by san_to_move():
//   - optional '+' / '#' / '!' / '?' suffixes (stripped)
//   - castling as O-O / O-O-O (also 0-0 / 0-0-0)
//   - promotions as e8=Q (also lowercase q, missing '=' tolerated? NO --
//     the '=' is required because a bare "e8" would otherwise be ambiguous
//     with a normal move; lichess SAN always includes it)
//   - disambiguation by file (Nbd7), rank (R1e2) or both (Qh4e1)
//   - captures with x, including en passant (exd6, no special marker)

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "./chess_environment.h"

namespace dataset {

// Number of the first column holding Move_ply_1 (metadata occupies 0..20).
constexpr size_t MOVE_COL = 21;
// Number of the first column holding Eval_ply_1 (right after Move_ply_200).
constexpr size_t EVAL_COL = MOVE_COL + 200;
// A usable row needs at least one move column and one eval column.
constexpr size_t MIN_COLS = EVAL_COL + 1;

// ---------------------------------------------------------------------------
// parse_csv_line
// Splits one CSV line into fields following RFC-4180:
//   - ',' separates fields (outside quotes)
//   - '"...' opens a quoted field; inside it ',' is literal and '""' is '"'
// A trailing '\r' should already be removed by the caller (getline keeps it
// on Windows line endings).
// ---------------------------------------------------------------------------
inline std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string cur;
    bool in_quotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    cur += '"';
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                cur += c;
            }
        } else {
            if (c == '"') {
                in_quotes = true;
            } else if (c == ',') {
                fields.push_back(std::move(cur));
                cur.clear();
            } else {
                cur += c;
            }
        }
    }
    fields.push_back(std::move(cur));
    return fields;
}

// ---------------------------------------------------------------------------
// san_to_move
// Resolves ONE Standard Algebraic Notation token against the LEGAL moves of
// `pos`. Returns true and fills `out` iff exactly one legal move matches;
// ambiguity (missing disambiguation) and nonsense both return false so the
// caller can drop the whole game rather than desync the replay.
// ---------------------------------------------------------------------------
inline bool san_to_move(const chess::Position& pos, std::string san, chess::Move& out) {
    while (!san.empty()) {
        const char c = san.back();
        if (c == '+' || c == '#' || c == '!' || c == '?' || c == ' ' || c == '\r')
            san.pop_back();
        else
            break;
    }
    for (char& c : san)
        if (c == '0') c = 'O';
    if (san.empty()) return false;

    const std::vector<chess::Move> moves = chess::genLegalMoves(pos);

    // Castling: unique legal move landing on the fixed king destination.
    if (san == "O-O" || san == "O-O-O") {
        const int target = (san == "O-O") ? (pos.side > 0 ? 6 : 62)
                                          : (pos.side > 0 ? 2 : 58);
        for (const chess::Move& m : moves) {
            if ((m.flags & chess::F_CASTLE) && m.to == target) {
                out = m;
                return true;
            }
        }
        return false;
    }

    // Strip promotion suffix "=Q" (case-insensitive) from the back.
    int promo = 0;
    if (!san.empty()) {
        const char c = static_cast<char>(san.back() >= 'a' && san.back() <= 'z'
                                             ? san.back() - 'a' + 'A'
                                             : san.back());
        if (c == 'Q' || c == 'R' || c == 'B' || c == 'N') {
            switch (c) {
                case 'Q': promo = chess::QUEEN; break;
                case 'R': promo = chess::ROOK; break;
                case 'B': promo = chess::BISHOP; break;
                default:  promo = chess::KNIGHT; break;
            }
            san.pop_back();
            if (!san.empty() && san.back() == '=') san.pop_back();
        }
    }

    if (san.size() < 2) return false;

    // Target square = last two characters.
    const char tf = san[san.size() - 2];
    const char tr = san[san.size() - 1];
    if (tf < 'a' || tf > 'h' || tr < '1' || tr > '8') return false;
    const int to_sq = chess::sqOf(tf - 'a', tr - '1');

    // Head = optional piece letter + optional disambiguation + optional 'x'.
    std::string head = san.substr(0, san.size() - 2);
    if (!head.empty() && head.back() == 'x') head.pop_back();

    int piece_type = chess::PAWN;
    if (!head.empty()) {
        switch (head[0]) {
            case 'K': piece_type = chess::KING; break;
            case 'Q': piece_type = chess::QUEEN; break;
            case 'R': piece_type = chess::ROOK; break;
            case 'B': piece_type = chess::BISHOP; break;
            case 'N': piece_type = chess::KNIGHT; break;
            default:  piece_type = chess::PAWN; break;
        }
        if (piece_type != chess::PAWN) head.erase(0, 1);
    }

    int from_file = -1, from_rank = -1;
    if (head.size() >= 1) {
        if (head[0] >= 'a' && head[0] <= 'h') from_file = head[0] - 'a';
        else if (head[0] >= '1' && head[0] <= '8') from_rank = head[0] - '1';
        else return false;
    }
    if (head.size() >= 2) {
        if (head[1] >= '1' && head[1] <= '8') from_rank = head[1] - '1';
        else return false;
    }
    if (head.size() >= 3) return false;

    // Match against legal moves; require EXACTLY one survivor.
    int matches = 0;
    for (const chess::Move& m : moves) {
        const int pc = pos.board[m.from];
        const int t = pc > 0 ? pc : -pc;
        if (t != piece_type) continue;
        if (m.promo != promo) continue;
        if (static_cast<int>(m.to) != to_sq) continue;
        if (from_file >= 0 && chess::fileOf(m.from) != from_file) continue;
        if (from_rank >= 0 && chess::rankOf(m.from) != from_rank) continue;
        ++matches;
        out = m;
    }
    return matches == 1;
}

// ---------------------------------------------------------------------------
// mirror_position
// Vertical flip + color swap: the position that White would face if the
// game continued with colors reversed. Labels negate under this transform,
// which is what makes MIRROR augmentation sound (and what --probe checks).
// ---------------------------------------------------------------------------
inline chess::Position mirror_position(const chess::Position& p) {
    chess::Position m{};
    for (int s = 0; s < 64; ++s) {
        const int pc = p.board[s];
        m.board[chess::mirrorSq(s)] =
            static_cast<int8_t>(pc != chess::EMPTY ? -pc : chess::EMPTY);
    }
    m.side = static_cast<int8_t>(-p.side);
    int c = 0;
    if (p.castling & chess::CASTLE_WK) c |= chess::CASTLE_BK;
    if (p.castling & chess::CASTLE_WQ) c |= chess::CASTLE_BQ;
    if (p.castling & chess::CASTLE_BK) c |= chess::CASTLE_WK;
    if (p.castling & chess::CASTLE_BQ) c |= chess::CASTLE_WQ;
    m.castling = static_cast<int8_t>(c);
    m.ep = (p.ep >= 0) ? static_cast<int8_t>(chess::mirrorSq(p.ep)) : -1;
    m.halfmove = p.halfmove;
    m.hash = chess::computeHash(m);
    return m;
}

}   // namespace dataset
