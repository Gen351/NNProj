#pragma once

// Iterative-deepening negamax alpha-beta search with quiescence.
//
// The engine owns ALL move selection. Your network only scores positions:
// at every leaf the Evaluator is asked "who is better here?" (centipawns,
// White POV) and negamax turns those scores into a best move via minimax.
// See howto/createAI.md for the full mental model.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

#include "./chess_environment.h"
#include "./evaluator.h"
#include "./transposition_table.h"

namespace chess {

constexpr float MATE_SCORE = 30000.0f;
constexpr float SCORE_INF = 1e9f;
constexpr int MAX_PLY = 64;
constexpr int MAX_QPLY = 24;   // extra quiescence plies beyond the main search

// ---------------------------------------------------------------------------
// Transposition-table score normalization. All scores are side-to-move POV.
// Mate scores are stored relative to the STORING node's ply so a mate distance
// survives reuse from a differently-plied position; everything else is stored
// verbatim (truncated to centipawn ints).
// ---------------------------------------------------------------------------
inline int32_t ttStoreScore(const float s, const int ply) {
    if (std::abs(s) >= MATE_SCORE - static_cast<float>(MAX_PLY))
        return static_cast<int32_t>(s > 0 ? s + ply : s - ply);
    return static_cast<int32_t>(s);
}

inline float ttReadScore(const int32_t s, const int ply) {
    if (std::abs(s) >= MATE_SCORE - static_cast<float>(MAX_PLY))
        return static_cast<float>(s > 0 ? s - ply : s + ply);
    return static_cast<float>(s);
}

struct SearchLimits {
    long long wtime = -1;        // ms remaining, -1 = not given
    long long btime = -1;
    long long winc = 0;          // ms increment per move
    long long binc = 0;
    long long movestogo = -1;    // moves until next time control
    long long movetime = -1;     // fixed time per move
    long long nodesLimit = -1;
    int depthLimit = -1;
    bool infinite = false;       // search until stop() is called
};

struct SearchInfo {              // one line of progress (UCI "info")
    int depth = 0;
    long long nodes = 0;
    long long timeMs = 0;
    float scoreCp = 0.0f;
    bool mate = false;
    int mateIn = 0;              // signed: positive = side to move mates in N
    std::vector<Move> pv;
};

struct SearchResult {
    Move best = NO_MOVE;
    bool hasMove = false;
    float score = 0.0f;
    int depth = 0;
    long long nodes = 0;
    long long timeMs = 0;
    std::vector<Move> pv;
};

class Searcher {
public:
    // `shared` is an optional table shared across searchers (the UCI engine's
    // Lazy-SMP helpers and self_play all point at one table). When omitted the
    // search lazily owns a private default-size table, so every existing
    // caller keeps working unchanged.
    explicit Searcher(Evaluator& evaluator, TranspositionTable* shared = nullptr)
        : evaluator_(evaluator), sharedTable_(shared) {
        if (!sharedTable_) {
            ownTable_ = std::make_unique<TranspositionTable>();
            ownTable_->setSize(kDefaultTableMb);
        }
    }

    // Called from another thread to abort the running search ASAP.
    void stop() { stopFlag_.store(true); }

    // Draw contempt in centipawns (self-play anti-shuffle). It is a ROOT-side
    // penalty: a root move that would immediately draw the game -- creating a
    // threefold repetition (3rd occurrence), fifty-move, insufficient material,
    // or stalemate -- is compared as if it scored 2*drawContempt cp worse, so a
    // non-drawing alternative is preferred whenever it exists and isn't clearly
    // worse. Drawn positions inside the tree score 0.0f (the true game value).
    // 0 disables it; the UCI engine leaves it at 0.
    void setDrawContempt(const float cp) { drawContempt_ = cp; }

    std::function<void(const SearchInfo&)> onInfo;

    SearchResult think(const ChessGame& game, const SearchLimits& lim) {
        SearchResult out;
        nodes_ = 0;
        // NOTE: stopFlag_ is deliberately NOT reset here -- a stop() arriving
        // between thread spawn and think() entry must not be lost. Each UCI
        // "go" constructs a fresh Searcher whose flag starts cleared.
        start_ = clock::now();

        setupBudget(lim);
        maxDepth_ = (lim.depthLimit > 0)
            ? std::min(lim.depthLimit, MAX_PLY - 2)
            : MAX_PLY - 2;

        table().newSearch();   // entries from prior searches become stale

        repHist_ = game.rep_history;   // positions already played in the real game

        rootMoves_ = genLegalMoves(game.pos);
        out.hasMove = !rootMoves_.empty();
        if (!out.hasMove) return out;
        orderMoves(game.pos, rootMoves_, 0);

        // Root draw-avoidance: flag moves that would immediately end the game
        // in a draw, so the comparison below can penalize them. Kept as raw
        // flags (not scores) since the penalty is re-applied per iteration.
        std::vector<char> drawish;
        drawish.reserve(rootMoves_.size());
        for (const Move& m : rootMoves_) {
            Position child = game.pos;
            applyMove(child, m);
            drawish.push_back(drawishChild(child) ? 1 : 0);
        }

        Move bestMove = rootMoves_[0];
        float bestScore = 0.0f;
        std::vector<Move> bestPV(1, bestMove);
        int completedDepth = 0;

        for (int d = 1; d <= maxDepth_; ++d) {
            if (completedDepth > 0) {
                auto it = std::find(rootMoves_.begin(), rootMoves_.end(), bestMove);
                if (it != rootMoves_.begin() && it != rootMoves_.end())
                    std::rotate(rootMoves_.begin(), it, it + 1);   // search best first
            }

            float iterBest = -SCORE_INF;
            float iterCmp = -SCORE_INF;   // penalized comparator for selection
            Move iterMove = NO_MOVE;
            std::vector<Move> iterPV;
            float alpha = -SCORE_INF, beta = SCORE_INF;
            bool aborted = false;

            try {
                for (size_t i = 0; i < rootMoves_.size(); ++i) {
                    const Move& m = rootMoves_[i];
                    Position child = game.pos;
                    applyMove(child, m);
                    repHist_.push_back(child.hash);
                    const float sc = -negamax(child, d - 1, -beta, -alpha, 1);
                    repHist_.pop_back();

                    const float cmp = sc - (drawish[i] ? 2.0f * drawContempt_ : 0.0f);
                    if (cmp > iterCmp) {
                        iterCmp = cmp;
                        iterBest = sc;   // raw score, kept for reporting
                        iterMove = m;
                        iterPV.assign(1, m);
                        for (int k = 0; k < pvLen_[1]; ++k)
                            iterPV.push_back(pvTable_[1][k]);
                    }
                    if (iterCmp > alpha) alpha = iterCmp;
                }
            } catch (const AbortSearch&) {
                aborted = true;   // partial iteration: results are untrustworthy
            }

            if (aborted && completedDepth > 0) break;
            if (!aborted) {
                bestScore = iterBest;
                bestMove = iterMove;
                bestPV = iterPV.empty() ? std::vector<Move>(1, bestMove) : iterPV;
                completedDepth = d;
            }

            if (onInfo && completedDepth > 0) {
                SearchInfo inf;
                inf.depth = completedDepth;
                inf.nodes = nodes_;
                inf.timeMs = elapsedMs();
                finalizeScore(bestScore, inf);
                inf.pv = bestPV;
                onInfo(inf);
            }

            if (stopFlag_.load()) break;
            if (softMs_ >= 0 && elapsedMs() >= softMs_) break;
            if (std::abs(bestScore) >= MATE_SCORE - static_cast<float>(MAX_PLY)) break;
        }

        out.best = bestMove;
        out.score = bestScore;
        out.depth = completedDepth;
        out.nodes = nodes_;
        out.timeMs = elapsedMs();
        out.pv = bestPV;
        return out;
    }

private:
    struct AbortSearch {};   // thrown to unwind the tree on timeout / node cap

    static constexpr size_t kDefaultTableMb = 16;

    Evaluator& evaluator_;
    std::atomic<bool> stopFlag_{false};
    float drawContempt_ = 0.0f;

    TranspositionTable* sharedTable_ = nullptr;
    std::unique_ptr<TranspositionTable> ownTable_;
    TranspositionTable& table() { return sharedTable_ ? *sharedTable_ : *ownTable_; }

    using clock = std::chrono::steady_clock;
    clock::time_point start_{};
    long long nodes_ = 0;
    long long softMs_ = -1;      // stop STARTING new iterations past this
    long long hardMs_ = -1;      // hard abort inside the tree (-1 = none)
    long long nodeLimit_ = -1;
    int maxDepth_ = MAX_PLY - 2;

    std::vector<uint64_t> repHist_;       // game prefix + current search line
    std::vector<Move> rootMoves_;
    Move pvTable_[MAX_PLY][MAX_PLY]{};
    int pvLen_[MAX_PLY]{};

    long long elapsedMs() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   clock::now() - start_).count();
    }

    void setupBudget(const SearchLimits& lim) {
        softMs_ = hardMs_ = -1;
        nodeLimit_ = lim.nodesLimit > 0 ? lim.nodesLimit : -1;

        if (lim.movetime > 0) {
            softMs_ = hardMs_ = lim.movetime;
        } else if (!lim.infinite) {
            long long remain = -1, inc = 0;
            if (lim.wtime >= 0 || lim.btime >= 0) {
                remain = lim.wtime >= 0 ? lim.wtime : lim.btime;
                inc = lim.wtime >= 0 ? lim.winc : lim.binc;
            }
            if (remain >= 0) {
                const long long mts = lim.movestogo > 0
                    ? std::min<long long>(lim.movestogo, 60) : 35;
                if (remain <= 100) {
                    softMs_ = hardMs_ = 15;                 // panic mode
                } else {
                    const long long alloc = remain / mts + inc * 4 / 5;
                    softMs_ = std::max(15LL, std::min(alloc, remain / 2));
                    hardMs_ = std::max(softMs_,
                               std::min(std::max(softMs_ * 4, remain / 10), remain - 30));
                    hardMs_ = std::max(hardMs_, 20LL);
                }
            }
        }
        if (hardMs_ >= 0) hardMs_ = std::max(hardMs_, 15LL);
    }

    void checkBudget() {
        ++nodes_;
        if ((nodes_ & 2047) != 0) return;
        if (stopFlag_.load()) throw AbortSearch{};   // "stop" aborts mid-tree too
        if (nodeLimit_ > 0 && nodes_ >= nodeLimit_) throw AbortSearch{};
        if (hardMs_ > 0 && elapsedMs() >= hardMs_) throw AbortSearch{};
    }

    static void finalizeScore(const float score, SearchInfo& inf) {
        inf.scoreCp = score;
        inf.mate = std::abs(score) >= MATE_SCORE - static_cast<float>(MAX_PLY);
        if (inf.mate) {
            const int dist = static_cast<int>(MATE_SCORE - std::abs(score));
            inf.mateIn = (score > 0 ? 1 : -1) * ((dist + 1) / 2);
        } else {
            inf.mateIn = 0;
        }
    }

    // White POV -> side-to-move POV for negamax.
    float evalPov(const Position& p) {
        const float v = evaluator_.evaluate(p);
        return p.side > 0 ? v : -v;
    }

    // Threefold repetition WITHIN the searched line counts as a draw (the
    // FIDE rule, matching ChessGame::status()): the current position must
    // have appeared two times before at the same side-to-move parity -- a
    // mere second occurrence is NOT yet a draw.
    bool isRepetition(const Position& p) const {
        const int n = static_cast<int>(repHist_.size());
        const int window = p.halfmove + 1;
        int occurrences = 1;   // current position itself
        int examined = 0;
        for (int j = n - 2; j >= 0; j -= 2) {
            if (++examined > window) break;
            if (repHist_[static_cast<size_t>(j)] == p.hash) ++occurrences;
        }
        return occurrences >= 3;
    }

    // True if a root move producing this child position would immediately end
    // the GAME in a draw: the child becomes a threefold repetition (3rd
    // occurrence at the child's side-to-move parity), the fifty-move rule,
    // insufficient material, or stalemate. Checkmate is NOT a draw.
    bool drawishChild(const Position& child) const {
        if (child.halfmove >= 100) return true;
        if (insufficientMaterial(child)) return true;
        if (genLegalMoves(child).empty() && !chess::inCheck(child)) return true;

        const int n = static_cast<int>(repHist_.size());   // current pos already included
        const int window = child.halfmove + 1;
        int occurrences = 0;
        int examined = 0;
        for (int j = n - 2; j >= 0; j -= 2) {
            if (++examined > window) break;
            if (repHist_[static_cast<size_t>(j)] == child.hash) ++occurrences;
        }
        return occurrences >= 2;   // child is the 3rd occurrence
    }

    // MVV-LVA for captures + promotion bonus; quiets keep insertion order. The
    // transposition-table best move gets absolute priority -- it is the single
    // strongest pruning hint alpha-beta has.
    static int orderScore(const Position& p, const Move& m, const uint32_t ttMove) {
        if (ttMove != 0 && TranspositionTable::packMove(m) == ttMove) return 4000000;
        int sc = 0;
        const int victim = (m.flags & F_EP) ? PAWN
                          : (p.board[m.to] != EMPTY ? (p.board[m.to] > 0 ? p.board[m.to] : -p.board[m.to]) : 0);
        if (victim != EMPTY) {
            static constexpr int VAL[7] = {0, 100, 320, 330, 500, 900, 20000};
            const int attacker = p.board[m.from] > 0 ? p.board[m.from] : -p.board[m.from];
            sc += 1000000 + VAL[victim] * 16 - VAL[attacker];
        }
        if (m.promo) {
            static constexpr int VAL[7] = {0, 100, 320, 330, 500, 900, 20000};
            sc += 900000 + VAL[m.promo] * 16;
        }
        return sc;
    }

    static void orderMoves(const Position& p, std::vector<Move>& moves,
                           const uint32_t ttMove) {
        std::stable_sort(moves.begin(), moves.end(),
                         [&p, ttMove](const Move& a, const Move& b) {
                             return orderScore(p, a, ttMove) > orderScore(p, b, ttMove);
                         });
    }

    static bool isTactical(const Position& p, const Move& m) {
        if (m.promo) return true;
        if (m.flags & F_EP) return true;
        return p.board[m.to] != EMPTY;
    }

    float negamax(const Position& p, int depth, float alpha, float beta, const int ply) {
        checkBudget();
        pvLen_[ply] = 0;

        const float origAlpha = alpha;
        uint32_t ttMove = 0;

        if (ply > 0) {
            if (p.halfmove >= 100) return 0.0f;             // fifty-move draw
            if (isRepetition(p)) return 0.0f;               // repetition draw
            if (insufficientMaterial(p)) return 0.0f;
            if (ply >= MAX_PLY - 1) return evalPov(p);  // safety valve

            if (depth > 0) {
                // TT probe runs AFTER the draw checks so stored scores never
                // mask a repetition / fifty-move / material draw.
                TranspositionTable::TTEntry tte;
                if (table().probe(p.hash, tte)) {
                    const float ts = ttReadScore(tte.score, ply);
                    if (tte.depth >= static_cast<uint8_t>(depth)) {
                        if (tte.flag == TranspositionTable::BOUND_EXACT) return ts;
                        if (tte.flag == TranspositionTable::BOUND_LOWER && ts >= beta) return ts;
                        if (tte.flag == TranspositionTable::BOUND_UPPER && ts <= alpha) return ts;
                    }
                    ttMove = tte.move;
                    // A genuine static evaluation from an earlier visit raises
                    // alpha exactly as qsearch's stand-pat would, pruning lines
                    // that are clearly worse than the position's value. Interior
                    // entries carry EVAL_NONE (no extra eval calls), so this only
                    // fires when a qsearch actually scored this position.
                    if (tte.eval != TranspositionTable::EVAL_NONE)
                        alpha = std::max(alpha, static_cast<float>(tte.eval));
                }
            }
        }

        const int ksq = kingSq(p);
        const bool checked = ksq >= 0 && isAttacked(p, ksq, -p.side);

        if (depth <= 0) return qsearch(p, alpha, beta, ply, 0, checked);

        std::vector<Move> moves = genLegalMoves(p);
        if (moves.empty())
            return checked ? -(MATE_SCORE - static_cast<float>(ply)) : 0.0f;

        if (checked && ply < MAX_PLY - MAX_QPLY) ++depth;   // check extension

        orderMoves(p, moves, ttMove);

        float best = -SCORE_INF;
        Move bestM = NO_MOVE;
        const size_t baseLen = repHist_.size();

        for (const Move& m : moves) {
            Position child = p;
            applyMove(child, m);
            repHist_.push_back(child.hash);
            const float sc = -negamax(child, depth - 1, -beta, -alpha, ply + 1);
            repHist_.resize(baseLen);

            if (sc > best) {
                best = sc;
                bestM = m;
                if (sc > alpha) {
                    alpha = sc;
                    pvTable_[ply][0] = m;
                    pvLen_[ply] = pvLen_[ply + 1] + 1;
                    for (int i = 0; i < pvLen_[ply + 1]; ++i)
                        pvTable_[ply][i + 1] = pvTable_[ply + 1][i];
                }
            }
            if (alpha >= beta) break;
        }

        // Bind the result to the ORIGINAL window: fail-low when every move was
        // below it, fail-high when a cutoff occurred, exact otherwise.
        const uint8_t flag = (best <= origAlpha) ? TranspositionTable::BOUND_UPPER
                           : (best >= beta)      ? TranspositionTable::BOUND_LOWER
                           : TranspositionTable::BOUND_EXACT;
        table().store(p.hash, ttStoreScore(best, ply), TranspositionTable::EVAL_NONE,
                      TranspositionTable::packMove(bestM),
                      static_cast<uint8_t>(depth), flag);
        return best;
    }

    float qsearch(const Position& p, float alpha, float beta,
                  const int ply, const int qsPly, const bool checked) {
        checkBudget();

        const float origAlpha = alpha;
        uint32_t ttMove = 0;
        int32_t ttEval = TranspositionTable::EVAL_NONE;

        {
            TranspositionTable::TTEntry tte;
            if (table().probe(p.hash, tte)) {
                const float ts = ttReadScore(tte.score, ply);
                if (tte.flag == TranspositionTable::BOUND_EXACT) return ts;
                if (tte.flag == TranspositionTable::BOUND_LOWER && ts >= beta) return ts;
                if (tte.flag == TranspositionTable::BOUND_UPPER && ts <= alpha) return ts;
                ttMove = tte.move;
                ttEval = tte.eval;
            }
        }

        if (checked) {
            // In check we cannot stand pat -- search ALL evasions (bounded).
            if (qsPly >= MAX_QPLY || ply >= MAX_PLY - 1) return evalPov(p);
            std::vector<Move> moves = genLegalMoves(p);
            if (moves.empty()) return -(MATE_SCORE - static_cast<float>(ply));
            orderMoves(p, moves, ttMove);

            float best = -SCORE_INF;
            Move bestM = NO_MOVE;
            const size_t baseLen = repHist_.size();
            for (const Move& m : moves) {
                Position child = p;
                applyMove(child, m);
                repHist_.push_back(child.hash);
                const float sc = -qsearch(child, -beta, -alpha, ply + 1, qsPly + 1, false);
                repHist_.resize(baseLen);
                if (sc > best) { best = sc; bestM = m; }
                if (best > alpha) alpha = best;
                if (alpha >= beta) break;
            }

            const uint8_t flag = (best <= origAlpha) ? TranspositionTable::BOUND_UPPER
                               : (best >= beta)      ? TranspositionTable::BOUND_LOWER
                               : TranspositionTable::BOUND_EXACT;
            table().store(p.hash, ttStoreScore(best, ply), TranspositionTable::EVAL_NONE,
                          TranspositionTable::packMove(bestM), 0, flag);
            return best;
        }

        // Static evaluation -- reused from the TT when available, which is the
        // whole point of the eval cache: re-visits skip the net forward pass.
        const float standPat = (ttEval != TranspositionTable::EVAL_NONE)
                                   ? static_cast<float>(ttEval)
                                   : evalPov(p);
        if (standPat >= beta) {
            table().store(p.hash, ttStoreScore(standPat, ply), static_cast<int32_t>(standPat),
                          ttMove, 0, TranspositionTable::BOUND_LOWER);
            return standPat;
        }
        if (standPat > alpha) alpha = standPat;
        if (qsPly >= MAX_QPLY || ply >= MAX_PLY - 1) return standPat;

        std::vector<Move> moves = genLegalMoves(p);
        orderMoves(p, moves, ttMove);

        float best = standPat;
        Move bestM = NO_MOVE;
        const size_t baseLen = repHist_.size();
        for (const Move& m : moves) {
            if (!isTactical(p, m)) continue;   // captures and promotions only
            Position child = p;
            applyMove(child, m);
            repHist_.push_back(child.hash);
            const float sc = -qsearch(child, -beta, -alpha, ply + 1, qsPly + 1, false);
            repHist_.resize(baseLen);
            if (sc > best) { best = sc; bestM = m; }
            if (best > alpha) alpha = best;
            if (alpha >= beta) break;
        }

        const uint8_t flag = (best <= origAlpha) ? TranspositionTable::BOUND_UPPER
                           : (best >= beta)      ? TranspositionTable::BOUND_LOWER
                           : TranspositionTable::BOUND_EXACT;
        table().store(p.hash, ttStoreScore(best, ply), static_cast<int32_t>(standPat),
                      bestM == NO_MOVE ? ttMove : TranspositionTable::packMove(bestM), 0, flag);
        return best;
    }
};

}   // namespace chess
