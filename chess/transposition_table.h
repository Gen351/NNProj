#pragma once

// Transposition table for the chess search (see search.h).
//
// Stores (Zobrist hash -> {score, static eval, best move, depth, bound flag})
// so alpha-beta cutoffs, move ordering, and static-eval reuse survive across
// transpositions, iterations, ponder, and -- via a shared table -- Lazy-SMP
// helper threads (uci_main.cpp / self_play.cpp).
//
// Design notes:
//   * The FULL 64-bit Zobrist hash is kept in every entry, so probing is
//     collision-free (no 16-bit fingerprint risk).
//   * A "generation" (gen) is bumped per search via newSearch(); entries from
//     previous searches become stale and are replaced before same-search,
//     same-depth rivals (depth-preferred replacement, Stockfish-lite).
//   * SMP search: the key is std::atomic (relaxed). The remaining payload
//     fields are written/read non-atomically -- a torn entry can at worst
//     return one slightly stale score and is self-corrected on the next
//     store. This is the standard trade-off of single-table shared-memory
//     engines.
//   * Scores are stored as the node's side-to-move score and mate scores are
//     ply-normalized by the caller (see ttStoreScore/ttReadScore in search.h).
//   * Best moves are stored as the four bytes of chess::Move packed into one
//     uint32 (lossless -- no 16-bit ambiguity bugs) so ordering is exact.

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>

#include "./chess_environment.h"

namespace chess {

class TranspositionTable {
public:
    enum Bound : uint8_t {
        BOUND_EMPTY = 0,   // slot free
        BOUND_EXACT = 1,   // score == node value
        BOUND_LOWER = 2,   // score >= node value (fail-high / beta cutoff)
        BOUND_UPPER = 3,   // score <= node value (fail-low / alpha bound)
    };

    // Sentinel for "no static evaluation stored for this entry". A real eval
    // of 0 (equal material) is meaningful, so the marker must be distinct.
    static constexpr int32_t EVAL_NONE = INT32_MIN;

    struct TTEntry {
        std::atomic<uint64_t> key{0};
        int32_t score = 0;
        int32_t eval = EVAL_NONE;
        uint32_t move = 0;   // packed chess::Move (packMove / unpackMove)
        uint8_t depth = 0;
        uint8_t flag = BOUND_EMPTY;
        uint8_t gen = 0;
        uint8_t pad = 0;
    };
    static_assert(sizeof(TTEntry) == 24, "TTEntry must stay 24 bytes");

    TranspositionTable() = default;

    // (Re)allocate to ~mb megabytes (rounded down to a power-of-two entry
    // count) and clear. 4 MB minimum.
    void setSize(const size_t mb) {
        const size_t bytes = mb * 1024ULL * 1024ULL;
        size_t n = bytes / sizeof(TTEntry);
        if (n < 1024) n = 1024;
        size_t po2 = 1;
        while (po2 * 2 <= n) po2 *= 2;
        table_ = std::make_unique<TTEntry[]>(po2);
        count_ = po2;
        mask_ = po2 - 1;
        clear();
    }

    void clear() {
        gen_.store(1, std::memory_order_relaxed);
        if (table_) std::memset(table_.get(), 0, count_ * sizeof(TTEntry));
    }

    // Bumps the generation: entries written during older searches are now
    // stale and will be replaced first.
    void newSearch() {
        uint8_t g = gen_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (g == 0) gen_.store(1, std::memory_order_relaxed);   // wrap-around
    }

    // Full-key match: fills `out` and returns true.
    bool probe(const uint64_t key, TTEntry& out) const {
        if (!table_) return false;
        const TTEntry& e = table_[static_cast<size_t>(key & mask_)];
        const uint64_t k = e.key.load(std::memory_order_relaxed);
        if (k != key || e.flag == BOUND_EMPTY) return false;
        out.key = k;
        out.score = e.score;
        out.eval = e.eval;
        out.move = e.move;
        out.depth = e.depth;
        out.flag = e.flag;
        out.gen = e.gen;
        return true;
    }

    void store(const uint64_t key, const int32_t score, const int32_t eval,
               const uint32_t move, const uint8_t depth, const uint8_t flag) {
        if (!table_) return;
        TTEntry& e = table_[static_cast<size_t>(key & mask_)];
        const uint64_t oldKey = e.key.load(std::memory_order_relaxed);

        // Replacement: a strictly deeper entry from the SAME search for a
        // DIFFERENT position is kept; otherwise overwrite (empty slot, same
        // position, or any stale-generation slot).
        if (oldKey != key && e.flag != BOUND_EMPTY
            && e.gen == gen_.load(std::memory_order_relaxed) && e.depth > depth)
            return;

        e.key.store(key, std::memory_order_relaxed);
        e.score = score;
        e.eval = eval;
        e.move = move;
        e.depth = depth;
        e.flag = flag;
        e.gen = gen_.load(std::memory_order_relaxed);
    }

    size_t numEntries() const { return count_; }
    size_t usedBytes() const { return count_ * sizeof(TTEntry); }

    // -----------------------------------------------------------------------
    // Move packing: store the four bytes of chess::Move as one uint32 so the
    // stored "best move" round-trips losslessly (from|to<<8|promo<<16|flags<<24).
    // -----------------------------------------------------------------------
    static uint32_t packMove(const Move& m) {
        return static_cast<uint32_t>(m.from)
             | (static_cast<uint32_t>(m.to) << 8)
             | (static_cast<uint32_t>(static_cast<uint8_t>(m.promo)) << 16)
             | (static_cast<uint32_t>(m.flags) << 24);
    }

    static Move unpackMove(const uint32_t packed) {
        Move m;
        m.from = static_cast<uint8_t>(packed & 0xFF);
        m.to = static_cast<uint8_t>((packed >> 8) & 0xFF);
        m.promo = static_cast<int8_t>((packed >> 16) & 0xFF);
        m.flags = static_cast<uint8_t>((packed >> 24) & 0xFF);
        return m;
    }

private:
    std::unique_ptr<TTEntry[]> table_;
    size_t count_ = 0;
    uint64_t mask_ = 0;
    std::atomic<uint8_t> gen_{1};
};

}   // namespace chess