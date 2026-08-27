#pragma once

// Pure chess rules environment.
// It knows nothing about neural networks, search, or UCI -- callers apply moves,
// generate legal moves, and read state themselves (mirrors snake_environment.h).
//
// Square layout: a1 = 0, b1 = 1, ..., h1 = 7, a2 = 8, ..., h8 = 63.
//   file = sq & 7   (0 = 'a' .. 7 = 'h')
//   rank = sq >> 3  (0 = rank '1' .. 7 = rank '8')
//
// Piece codes: 0 = empty. Positive = white, negative = black.
//   1 = pawn, 2 = knight, 3 = bishop, 4 = rook, 5 = queen, 6 = king.

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

namespace chess {

constexpr int EMPTY = 0;
constexpr int PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6;

inline constexpr int sqOf(const int file, const int rank) { return rank * 8 + file; }
inline constexpr int fileOf(const int s) { return s & 7; }
inline constexpr int rankOf(const int s) { return s >> 3; }
inline constexpr bool onBoard(const int f, const int r) { return f >= 0 && f < 8 && r >= 0 && r < 8; }
inline constexpr int mirrorSq(const int s) { return s ^ 56; }   // vertical flip (a1 <-> a8)

// Castling right bits (stored in Position::castling).
constexpr int CASTLE_WK = 1;   // white may castle king-side
constexpr int CASTLE_WQ = 2;   // white may castle queen-side
constexpr int CASTLE_BK = 4;
constexpr int CASTLE_BQ = 8;

// Move flag bits.
constexpr uint8_t F_NONE = 0;
constexpr uint8_t F_EP = 1;       // en passant capture
constexpr uint8_t F_CASTLE = 2;   // king moves two squares (rook moved by applyMove)
constexpr uint8_t F_DOUBLE = 4;   // pawn double push (sets the ep square)

struct Move {
    uint8_t from = 0;
    uint8_t to = 0;
    int8_t promo = 0;     // 0 or piece TYPE (KNIGHT..QUEEN) without color
    uint8_t flags = F_NONE;

    bool operator==(const Move& o) const {
        return from == o.from && to == o.to && promo == o.promo && flags == o.flags;
    }
    bool operator!=(const Move& o) const { return !(*this == o); }
};

inline constexpr Move NO_MOVE{255, 255, 0, F_NONE};

// Everything the search needs is a cheap POD (~80 bytes): search does copy-make.
struct Position {
    int8_t board[64]{};
    int8_t side = 1;        // 1 = white to move, -1 = black
    int8_t castling = 0;    // bitmask of CASTLE_*
    int8_t ep = -1;         // en passant TARGET square, or -1
    int16_t halfmove = 0;   // plies since last pawn move / capture (fifty-move rule)
    uint64_t hash = 0;      // zobrist hash, maintained incrementally by applyMove
};

// ---------------------------------------------------------------------------
// Zobrist hashing (deterministic: fixed seed, same keys every run)
// ---------------------------------------------------------------------------
namespace zobrist {

    inline uint64_t splitmix64(uint64_t& x) {
        x += 0x9E3779B97F4A7C15ull;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    struct Keys {
        uint64_t psq[64][12];
        uint64_t sideKey;
        uint64_t castle[16];
        uint64_t epFile[8];

        Keys() {
            uint64_t s = 0x1234567890ABCDEFull;
            for (int i = 0; i < 64; ++i)
                for (int j = 0; j < 12; ++j)
                    psq[i][j] = splitmix64(s);
            sideKey = splitmix64(s);
            for (int i = 0; i < 16; ++i) castle[i] = splitmix64(s);
            for (int i = 0; i < 8; ++i)  epFile[i] = splitmix64(s);
        }
    };

    // Zobrist piece index: white type t -> t-1 (0..5); black type t -> t+5 (6..11).
    inline constexpr int pieceIndex(const int p) { return p > 0 ? p - 1 : -p + 5; }

    inline const Keys& keys() {
        static const Keys k;
        return k;
    }
}

// Recompute the hash from scratch (used on setup and by selftests that verify
// the incremental updates in applyMove).
inline uint64_t computeHash(const Position& p) {
    const auto& K = zobrist::keys();
    uint64_t h = 0;
    for (int s = 0; s < 64; ++s)
        if (p.board[s] != EMPTY) h ^= K.psq[s][zobrist::pieceIndex(p.board[s])];
    if (p.side < 0) h ^= K.sideKey;
    h ^= K.castle[p.castling & 15];
    if (p.ep >= 0) h ^= K.epFile[p.ep & 7];
    return h;
}

// ---------------------------------------------------------------------------
// Attack detection
// ---------------------------------------------------------------------------

// Is square `s` attacked by any piece of color `by` (+1 white, -1 black)?
inline bool isAttacked(const Position& p, const int s, const int by) {
    const int f = fileOf(s), r = rankOf(s);

    // Pawn: a white pawn pushes +8 so an attacking white pawn sits one rank below.
    const int pr = r - by;
    if (pr >= 0 && pr < 8) {
        if (f - 1 >= 0 && p.board[sqOf(f - 1, pr)] == by * PAWN) return true;
        if (f + 1 < 8  && p.board[sqOf(f + 1, pr)] == by * PAWN) return true;
    }

    static constexpr int KN[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
    for (const auto& d : KN) {
        const int nf = f + d[0], nr = r + d[1];
        if (onBoard(nf, nr) && p.board[sqOf(nf, nr)] == by * KNIGHT) return true;
    }

    static constexpr int KG[8][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
    for (const auto& d : KG) {
        const int nf = f + d[0], nr = r + d[1];
        if (onBoard(nf, nr) && p.board[sqOf(nf, nr)] == by * KING) return true;
    }

    static constexpr int DIAG[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
    for (const auto& d : DIAG) {
        int nf = f + d[0], nr = r + d[1];
        while (onBoard(nf, nr)) {
            const int pc = p.board[sqOf(nf, nr)];
            if (pc != EMPTY) {
                if (pc == by * BISHOP || pc == by * QUEEN) return true;
                break;
            }
            nf += d[0]; nr += d[1];
        }
    }

    static constexpr int ORTHO[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    for (const auto& d : ORTHO) {
        int nf = f + d[0], nr = r + d[1];
        while (onBoard(nf, nr)) {
            const int pc = p.board[sqOf(nf, nr)];
            if (pc != EMPTY) {
                if (pc == by * ROOK || pc == by * QUEEN) return true;
                break;
            }
            nf += d[0]; nr += d[1];
        }
    }
    return false;
}

inline int kingSq(const Position& p) {
    const int k = p.side * KING;
    for (int s = 0; s < 64; ++s)
        if (p.board[s] == k) return s;
    return -1;   // broken position (should never happen via setFen/reset/make)
}

inline bool inCheck(const Position& p) {
    const int ks = kingSq(p);
    return ks >= 0 && isAttacked(p, ks, -p.side);
}

// ---------------------------------------------------------------------------
// The single core mutator. Applies `m` to `p` in place, keeping the zobrist
// hash incremental. Assumes `m` came from move generation for this position.
// ---------------------------------------------------------------------------
inline void applyMove(Position& p, const Move& m) {
    const auto& K = zobrist::keys();
    const int us = p.side;

    if (p.ep >= 0) p.hash ^= K.epFile[p.ep & 7];
    p.hash ^= K.castle[p.castling & 15];

    bool capture = false;
    const int piece = p.board[m.from];

    if (m.flags & F_EP) {
        const int capSq = m.to - us * 8;   // captured pawn sits behind the target
        capture = true;
        p.hash ^= K.psq[capSq][zobrist::pieceIndex(-us * PAWN)];
        p.board[capSq] = EMPTY;
    } else if (p.board[m.to] != EMPTY) {
        capture = true;
        p.hash ^= K.psq[m.to][zobrist::pieceIndex(p.board[m.to])];
    }

    p.board[m.from] = EMPTY;
    const int placed = m.promo ? us * m.promo : piece;
    p.board[m.to] = static_cast<int8_t>(placed);
    p.hash ^= K.psq[m.from][zobrist::pieceIndex(piece)]
            ^ K.psq[m.to][zobrist::pieceIndex(placed)];

    if (m.flags & F_CASTLE) {
        int rf, rt;
        switch (m.to) {
            case 6:  rf = 7;  rt = 5;  break;   // e1g1 -> rook h1->f1
            case 2:  rf = 0;  rt = 3;  break;   // e1c1 -> rook a1->d1
            case 62: rf = 63; rt = 61; break;   // e8g8 -> rook h8->f8
            default: rf = 56; rt = 59; break;   // e8c8 -> rook a8->d8
        }
        const int rook = p.board[rf];
        p.board[rf] = EMPTY;
        p.board[rt] = static_cast<int8_t>(rook);
        p.hash ^= K.psq[rf][zobrist::pieceIndex(rook)]
                ^ K.psq[rt][zobrist::pieceIndex(rook)];
    }

    // Castling rights: cleared when the king moves off its square or when a
    // rook leaves / is captured on its home square (`from` OR `to` covers both).
    auto clearFor = [](const int s) -> int {
        switch (s) {
            case 4:  return CASTLE_WK | CASTLE_WQ;   // e1
            case 7:  return CASTLE_WK;               // h1
            case 0:  return CASTLE_WQ;               // a1
            case 60: return CASTLE_BK | CASTLE_BQ;   // e8
            case 63: return CASTLE_BK;               // h8
            case 56: return CASTLE_BQ;               // a8
            default: return 0;
        }
    };
    p.castling &= ~(clearFor(m.from) | clearFor(m.to));
    p.hash ^= K.castle[p.castling & 15];

    p.ep = (m.flags & F_DOUBLE) ? static_cast<int8_t>((m.from + m.to) / 2) : -1;
    if (p.ep >= 0) p.hash ^= K.epFile[p.ep & 7];

    const bool pawnMove = (piece == PAWN || piece == -PAWN || m.promo != 0);
    p.halfmove = (pawnMove || capture) ? 0 : static_cast<int16_t>(p.halfmove + 1);

    p.side = static_cast<int8_t>(-us);
    p.hash ^= K.sideKey;
}

// ---------------------------------------------------------------------------
// Move generation (pseudo-legal, then legality filter via copy-make)
// ---------------------------------------------------------------------------

inline void addPawnMoves(std::vector<Move>& out, const Position& p, const int from, const int to, const uint8_t flags) {
    const int promoRank = p.side > 0 ? 7 : 0;
    if (rankOf(to) == promoRank) {
        for (const int t : {QUEEN, ROOK, BISHOP, KNIGHT})
            out.push_back({static_cast<uint8_t>(from), static_cast<uint8_t>(to),
                           static_cast<int8_t>(t), flags});
    } else {
        out.push_back({static_cast<uint8_t>(from), static_cast<uint8_t>(to), 0, flags});
    }
}

inline std::vector<Move> genPseudoMoves(const Position& p) {
    std::vector<Move> out;
    out.reserve(48);

    const int us = p.side, them = -us;
    const int fwd = us * 8;
    const int startRank = us > 0 ? 1 : 6;

    static constexpr int KN[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
    static constexpr int KG[8][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
    static constexpr int DIAG[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
    static constexpr int ORTHO[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};

    for (int s = 0; s < 64; ++s) {
        const int pc = p.board[s];
        if (pc == EMPTY || pc * us < 0) continue;   // not ours
        const int type = pc > 0 ? pc : -pc;
        const int f = fileOf(s), r = rankOf(s);

        switch (type) {
            case PAWN: {
                const int r1 = r + us;
                if (r1 >= 0 && r1 < 8 && p.board[sqOf(f, r1)] == EMPTY) {
                    addPawnMoves(out, p, s, sqOf(f, r1), F_NONE);
                    if (r == startRank && p.board[sqOf(f, r + 2 * us)] == EMPTY)
                        out.push_back({static_cast<uint8_t>(s), static_cast<uint8_t>(sqOf(f, r + 2 * us)), 0, F_DOUBLE});
                }
                for (int df = -1; df <= 1; df += 2) {
                    const int nf = f + df;
                    if (!onBoard(nf, r1)) continue;
                    const int t = sqOf(nf, r1);
                    const int tp = p.board[t];
                    if (tp != EMPTY && tp * us < 0) {
                        addPawnMoves(out, p, s, t, F_NONE);
                    } else if (p.ep >= 0 && t == p.ep
                               && p.board[t - fwd] == them * PAWN) {
                        out.push_back({static_cast<uint8_t>(s), static_cast<uint8_t>(t), 0, F_EP});
                    }
                }
                break;
            }
            case KNIGHT:
            case KING: {
                const auto& table = (type == KNIGHT) ? KN : KG;
                const int n = 8;
                for (int i = 0; i < n; ++i) {
                    const int nf = f + table[i][0], nr = r + table[i][1];
                    if (!onBoard(nf, nr)) continue;
                    const int t = sqOf(nf, nr);
                    const int tp = p.board[t];
                    if (tp == EMPTY)
                        out.push_back({static_cast<uint8_t>(s), static_cast<uint8_t>(t), 0, F_NONE});
                    else if (tp * us < 0)
                        out.push_back({static_cast<uint8_t>(s), static_cast<uint8_t>(t), 0, F_NONE});
                }
                break;
            }
            default: {   // sliders
                const bool diagMover = (type == BISHOP || type == QUEEN);
                const bool orthoMover = (type == ROOK || type == QUEEN);
                if (diagMover) {
                    for (const auto& d : DIAG) {
                        int nf = f + d[0], nr = r + d[1];
                        while (onBoard(nf, nr)) {
                            const int t = sqOf(nf, nr), tp = p.board[t];
                            if (tp == EMPTY) out.push_back({static_cast<uint8_t>(s), static_cast<uint8_t>(t), 0, F_NONE});
                            else { if (tp * us < 0) out.push_back({static_cast<uint8_t>(s), static_cast<uint8_t>(t), 0, F_NONE}); break; }
                            nf += d[0]; nr += d[1];
                        }
                    }
                }
                if (orthoMover) {
                    for (const auto& d : ORTHO) {
                        int nf = f + d[0], nr = r + d[1];
                        while (onBoard(nf, nr)) {
                            const int t = sqOf(nf, nr), tp = p.board[t];
                            if (tp == EMPTY) out.push_back({static_cast<uint8_t>(s), static_cast<uint8_t>(t), 0, F_NONE});
                            else { if (tp * us < 0) out.push_back({static_cast<uint8_t>(s), static_cast<uint8_t>(t), 0, F_NONE}); break; }
                            nf += d[0]; nr += d[1];
                        }
                    }
                }
                break;
            }
        }
    }

    // Castling: rights present + path empty + king's path not attacked.
    if (us > 0) {
        if ((p.castling & CASTLE_WK)
            && p.board[4] == KING && p.board[7] == ROOK
            && p.board[5] == EMPTY && p.board[6] == EMPTY
            && !isAttacked(p, 4, them) && !isAttacked(p, 5, them) && !isAttacked(p, 6, them))
            out.push_back({4, 6, 0, F_CASTLE});
        if ((p.castling & CASTLE_WQ)
            && p.board[4] == KING && p.board[0] == ROOK
            && p.board[1] == EMPTY && p.board[2] == EMPTY && p.board[3] == EMPTY
            && !isAttacked(p, 4, them) && !isAttacked(p, 3, them) && !isAttacked(p, 2, them))
            out.push_back({4, 2, 0, F_CASTLE});
    } else {
        if ((p.castling & CASTLE_BK)
            && p.board[60] == -KING && p.board[63] == -ROOK
            && p.board[61] == EMPTY && p.board[62] == EMPTY
            && !isAttacked(p, 60, them) && !isAttacked(p, 61, them) && !isAttacked(p, 62, them))
            out.push_back({60, 62, 0, F_CASTLE});
        if ((p.castling & CASTLE_BQ)
            && p.board[60] == -KING && p.board[56] == -ROOK
            && p.board[57] == EMPTY && p.board[58] == EMPTY && p.board[59] == EMPTY
            && !isAttacked(p, 60, them) && !isAttacked(p, 59, them) && !isAttacked(p, 58, them))
            out.push_back({60, 58, 0, F_CASTLE});
    }

    return out;
}

// Fully legal moves: every pseudo-legal move is applied on a copy; the move is
// legal iff our king is not attacked afterwards. Correct by construction
// (handles pins, ep discoveries, illegal castling through check, etc.).
inline std::vector<Move> genLegalMoves(const Position& p) {
    std::vector<Move> pseudo = genPseudoMoves(p);
    std::vector<Move> out;
    out.reserve(pseudo.size());

    const int ksq = kingSq(p);
    if (ksq < 0) return out;

    for (const Move& m : pseudo) {
        Position child = p;
        applyMove(child, m);
        const int nk = (m.from == ksq) ? m.to : ksq;
        if (!isAttacked(child, nk, -p.side)) out.push_back(m);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Game-level wrapper: position + repetition history + fullmove counter
// ---------------------------------------------------------------------------
enum GameResult { ONGOING = 0, WHITE_WIN = 1, BLACK_WIN = 2, DRAW = 3 };

struct GameStatus {
    GameResult result = ONGOING;
    std::string reason;   // "checkmate", "stalemate", "fifty-move rule", ...
};

struct ChessGame {
    Position pos;
    std::vector<uint64_t> rep_history;   // hashes incl. the initial position
    int fullmove = 1;                    // starts at 1, increments after black moves

    ChessGame() { reset(); }
    explicit ChessGame(const std::string& fen) { setFen(fen); }

    void reset() { setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"); }

    void setFen(const std::string& fen);

    void make(const Move& m) {
        const bool blackMoved = (pos.side < 0);
        applyMove(pos, m);
        rep_history.push_back(pos.hash);
        if (blackMoved) ++fullmove;
    }

    std::vector<Move> legalMoves() const { return genLegalMoves(pos); }
    bool inCheck() const { return chess::inCheck(pos); }
    std::string fen() const;

    int repetitionCount() const {
        int c = 0;
        for (const uint64_t h : rep_history)
            if (h == pos.hash) ++c;
        return c;
    }

    GameStatus status() const;
};

// Insufficient material: K vs K, K+minor vs K, or KB vs KB with bishops on the
// same square color (all basic forced-draw cases; conservative otherwise).
inline bool insufficientMaterial(const Position& p) {
    int minorsW = 0, minorsB = 0, wBishopColor = -1, bBishopColor = -1;
    for (int s = 0; s < 64; ++s) {
        const int pc = p.board[s];
        if (pc == EMPTY || pc == KING || pc == -KING) continue;
        const int t = pc > 0 ? pc : -pc;
        if (t == PAWN || t == ROOK || t == QUEEN) return false;
        if (pc > 0) {
            ++minorsW;
            if (t == BISHOP) wBishopColor = (fileOf(s) + rankOf(s)) & 1;
        } else {
            ++minorsB;
            if (t == BISHOP) bBishopColor = (fileOf(s) + rankOf(s)) & 1;
        }
    }
    const int total = minorsW + minorsB;
    if (total <= 1) return true;
    if (minorsW == 1 && minorsB == 1 && wBishopColor >= 0 && wBishopColor == bBishopColor) return true;
    return false;
}

inline GameStatus ChessGame::status() const {
    const std::vector<Move> moves = genLegalMoves(pos);
    if (moves.empty()) {
        if (inCheck())
            return {pos.side > 0 ? BLACK_WIN : WHITE_WIN, "checkmate"};
        return {DRAW, "stalemate"};
    }
    if (pos.halfmove >= 100) return {DRAW, "fifty-move rule"};
    if (repetitionCount() >= 3) return {DRAW, "threefold repetition"};
    if (insufficientMaterial(pos)) return {DRAW, "insufficient material"};
    return {ONGOING, ""};
}

// ---------------------------------------------------------------------------
// FEN (parse + print). Missing counters default to 0 / 1. Castling rights are
// sanitized against actual piece placement so internal invariants always hold.
// ---------------------------------------------------------------------------
inline void ChessGame::setFen(const std::string& fen) {
    std::istringstream ss(fen);
    std::string placement, stm, castles, epStr, hmStr, fmStr;
    if (!(ss >> placement))
        throw std::runtime_error("setFen: empty FEN string");
    ss >> stm;
    ss >> castles;
    ss >> epStr;
    ss >> hmStr;
    ss >> fmStr;

    Position p{};
    int rank = 7, file = 0;
    for (const char c : placement) {
        if (c == '/') {
            --rank; file = 0;
            if (rank < 0) throw std::runtime_error("setFen: too many ranks in '" + fen + "'");
            continue;
        }
        if (c >= '1' && c <= '8') {
            file += c - '0';
            if (file > 8) throw std::runtime_error("setFen: rank overflow in '" + fen + "'");
            continue;
        }
        int piece;
        switch (c) {
            case 'P': piece = PAWN; break;
            case 'N': piece = KNIGHT; break;
            case 'B': piece = BISHOP; break;
            case 'R': piece = ROOK; break;
            case 'Q': piece = QUEEN; break;
            case 'K': piece = KING; break;
            case 'p': piece = -PAWN; break;
            case 'n': piece = -KNIGHT; break;
            case 'b': piece = -BISHOP; break;
            case 'r': piece = -ROOK; break;
            case 'q': piece = -QUEEN; break;
            case 'k': piece = -KING; break;
            default: throw std::runtime_error(std::string("setFen: bad char '") + c + "' in '" + fen + "'");
        }
        if (file > 7 || rank < 0)
            throw std::runtime_error("setFen: bad placement in '" + fen + "'");
        p.board[sqOf(file, rank)] = static_cast<int8_t>(piece);
        ++file;
    }

    p.side = (!stm.empty() && stm[0] == 'b') ? -1 : 1;

    if (castles.empty() || castles == "-") p.castling = 0;
    else {
        for (const char c : castles) {
            switch (c) {
                case 'K': p.castling |= CASTLE_WK; break;
                case 'Q': p.castling |= CASTLE_WQ; break;
                case 'k': p.castling |= CASTLE_BK; break;
                case 'q': p.castling |= CASTLE_BQ; break;
                default: break;   // ignore unknown tokens (e.g. Chess960 files)
            }
        }
    }
    // Sanitize against placement so applyMove's invariants hold for any input.
    if (!(p.board[4] == KING && p.board[7] == ROOK)) p.castling &= ~CASTLE_WK;
    if (!(p.board[4] == KING && p.board[0] == ROOK)) p.castling &= ~CASTLE_WQ;
    if (!(p.board[60] == -KING && p.board[63] == -ROOK)) p.castling &= ~CASTLE_BK;
    if (!(p.board[60] == -KING && p.board[56] == -ROOK)) p.castling &= ~CASTLE_BQ;

    p.ep = -1;
    if (!epStr.empty() && epStr != "-" && epStr.size() >= 2) {
        const int ef = epStr[0] - 'a', er = epStr[1] - '1';
        if (ef >= 0 && ef < 8 && er >= 0 && er < 8) p.ep = static_cast<int8_t>(sqOf(ef, er));
        else p.ep = -1;
    }

    try { p.halfmove = static_cast<int16_t>(std::max(0, std::stoi(hmStr))); } catch (...) { p.halfmove = 0; }
    try { fullmove = std::max(1, std::stoi(fmStr)); } catch (...) { fullmove = 1; }

    p.hash = computeHash(p);
    pos = p;
    rep_history.assign(1, p.hash);
}

inline std::string ChessGame::fen() const {
    std::string out;
    for (int r = 7; r >= 0; --r) {
        int run = 0;
        for (int f = 0; f < 8; ++f) {
            const int pc = pos.board[sqOf(f, r)];
            if (pc == EMPTY) { ++run; continue; }
            if (run) { out += static_cast<char>('0' + run); run = 0; }
            char c;
            switch (pc > 0 ? pc : -pc) {
                case PAWN: c = 'P'; break;
                case KNIGHT: c = 'N'; break;
                case BISHOP: c = 'B'; break;
                case ROOK: c = 'R'; break;
                case QUEEN: c = 'Q'; break;
                default: c = 'K'; break;
            }
            out += (pc < 0) ? static_cast<char>(c - 'A' + 'a') : c;
        }
        if (run) out += static_cast<char>('0' + run);
        if (r) out += '/';
    }
    out += pos.side > 0 ? " w " : " b ";
    if (!pos.castling) out += '-';
    else {
        if (pos.castling & CASTLE_WK) out += 'K';
        if (pos.castling & CASTLE_WQ) out += 'Q';
        if (pos.castling & CASTLE_BK) out += 'k';
        if (pos.castling & CASTLE_BQ) out += 'q';
    }
    out += ' ';
    if (pos.ep < 0) out += '-';
    else {
        out += static_cast<char>('a' + fileOf(pos.ep));
        out += static_cast<char>('1' + rankOf(pos.ep));
    }
    out += ' ' + std::to_string(pos.halfmove);
    out += ' ' + std::to_string(fullmove);
    return out;
}

// ---------------------------------------------------------------------------
// NN state encoding (THE contract between this engine and your trained net --
// see howto/createAI.md):
//   [   0 .. 767 ] 12 one-hot piece planes, plane*64 + square, value 1.0
//                  planes 0..5   = white P N B R Q K
//                  planes 6..11  = black P N B R Q K
//   [ 768 ]         side to move: +1 white, -1 black
//   [ 769..772 ]    castling rights one-hot: WK WQ BK BQ
//   [ 773..780 ]    en passant FILE one-hot (a..h), all zeros when no ep
// Total: 781 floats.
// ---------------------------------------------------------------------------
constexpr size_t STATE_SIZE = 781;

inline void getState(const Position& p, std::vector<float>& out) {
    out.assign(STATE_SIZE, 0.0f);
    for (int s = 0; s < 64; ++s) {
        const int pc = p.board[s];
        if (pc != EMPTY)
            out[static_cast<size_t>(zobrist::pieceIndex(pc)) * 64 + s] = 1.0f;
    }
    float* tail = out.data() + 768;
    tail[0] = static_cast<float>(p.side);
    if (p.castling & CASTLE_WK) tail[1] = 1.0f;
    if (p.castling & CASTLE_WQ) tail[2] = 1.0f;
    if (p.castling & CASTLE_BK) tail[3] = 1.0f;
    if (p.castling & CASTLE_BQ) tail[4] = 1.0f;
    if (p.ep >= 0) tail[5 + (p.ep & 7)] = 1.0f;
}

// ---------------------------------------------------------------------------
// UCI helpers
// ---------------------------------------------------------------------------
inline std::string moveToUci(const Move& m) {
    if (m == NO_MOVE) return "(none)";
    std::string s;
    s += static_cast<char>('a' + fileOf(m.from));
    s += static_cast<char>('1' + rankOf(m.from));
    s += static_cast<char>('a' + fileOf(m.to));
    s += static_cast<char>('1' + rankOf(m.to));
    if (m.promo) {
        static const char PROMO[6] = {'?', '?', 'n', 'b', 'r', 'q'};
        s += PROMO[m.promo];
    }
    return s;
}

// Parses tokens like "e2e4" / "e7e8q" against the given position. Returns false
// if malformed or not among the LEGAL moves.
inline bool parseUciMove(const std::string& str, const Position& p, Move& out) {
    if (str.size() < 4 || str.size() > 5) return false;
    auto coord = [](const char fc, const char rc) -> int {
        if (fc < 'a' || fc > 'h' || rc < '1' || rc > '8') return -1;
        return sqOf(fc - 'a', rc - '1');
    };
    const int from = coord(str[0], str[1]);
    const int to = coord(str[2], str[3]);
    if (from < 0 || to < 0 || from == to) return false;

    int promo = 0;
    if (str.size() == 5) {
        switch (str[4]) {
            case 'n': case 'N': promo = KNIGHT; break;
            case 'b': case 'B': promo = BISHOP; break;
            case 'r': case 'R': promo = ROOK; break;
            case 'q': case 'Q': promo = QUEEN; break;
            default: return false;
        }
    }
    for (const Move& m : genLegalMoves(p)) {
        if (m.from == from && m.to == to && m.promo == promo) {
            out = m;
            return true;
        }
    }
    return false;
}

// ASCII board for debugging / selftest output.
inline std::string boardString(const Position& p) {
    static const char GLYPH[7] = {'?', 'P', 'N', 'B', 'R', 'Q', 'K'};
    std::string out = "  +------------------------+\n";
    
    for (int r = 7; r >= 0; --r) {
        out += static_cast<char>('1' + r);
        out += " |";
        
        for (int f = 0; f < 8; ++f) {
            const int pc = p.board[sqOf(f, r)];
            
            // Checkerboard pattern logic
            bool isLightSquare = (r + f) % 2 != 0;
            
            // 24-bit ANSI Background colors (Light Gray vs Dark Gray)
            std::string bg = isLightSquare ? "\x1b[48;2;210;210;210m" 
                                           : "\x1b[48;2;90;90;90m";
            
            char g = ' '; // Empty squares use space so the background color fills the block cleanly
            std::string fg = "";
            
            if (pc != EMPTY) {
                g = GLYPH[pc > 0 ? pc : -pc];
                if (pc < 0) {
                    g = static_cast<char>(g - 'A' + 'a');
                    // Black pieces: Solid black text
                    fg = "\x1b[38;2;0;0;0m"; 
                } else {
                    // White pieces: Solid white text (with bold modifier)
                    fg = "\x1b[38;2;255;255;255m\x1b[1m"; 
                }
            }
            
            // Apply background, foreground, pad with space to match original spacing, add piece, then reset (\x1b[0m)
            out += bg + fg + ' ' + g + "\x1b[0m";
        }
        out += " |\n";
    }
    out += "    a b c d e f g h\n";
    return out;
}

}   // namespace chess
