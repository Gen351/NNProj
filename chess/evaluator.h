#pragma once

// Evaluators: turn a Position into a score in CENTIPAWNS from WHITE's point of
// view (positive = white is better). The search converts to side-to-move POV.
//
// Two implementations:
//   MaterialPSTEvaluator -- classic handcrafted material + piece-square tables.
//                           Default so the engine plays real chess before you
//                           ever train anything.
//   NetEvaluator         -- wraps YOUR BackpropNet (.backprop_net file). Two
//                           output conventions are auto-detected:
//                             * legacy: the net's single output IS the score
//                               in centipawns directly;
//                             * win-prob: a net whose final layer is a TANH
//                               activation outputs a win-probability-ish value
//                               y in (-1,1); it is inverted back to centipawns
//                               via the same normalization the trainer used
//                               (see cpWinProb/winProbToCp below).
//                           See howto/createAI.md for the exact IO contract.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../nn_engine/backprop_net/backprop_net.h"
#include "../nn_engine/backprop_net/backprop_net_reader.h"

#include "./chess_environment.h"

namespace chess {

struct Evaluator {
    virtual ~Evaluator() = default;
    virtual float evaluate(const Position& p) = 0;
    virtual std::string name() const = 0;
};

// ---------------------------------------------------------------------------
// Win-probability label normalization (the "Option C" training convention).
//
// Raw centipawn labels are squashed onto (-1,1) with tanh(cp/S) before
// training. The net's final TANH activation layer outputs exactly this
// (-1,1) value, so the MSE gradient stays well-scaled, while the unstable,
// noisy centipawn tails of real datasets are compressed and the near-zero
// opening density stays informative. The same scale S is shared between the
// trainer (targets) and this evaluator (inversion), so no per-run knob can
// drift them out of sync and backprop_net is never touched.
//
//   cpWinProb(cp)  -> tanh(cp / kWinScaleCP)        in (-1,1), antisymmetric
//   winProbToCp(y) -> kWinScaleCP * atanh(y)        exact inverse within (-1,1)
//
// S=400 is chosen so a queen (900cp) maps to ~0.978, a +300cp edge to ~0.635
// and +100cp to ~0.245: near 0 the map is roughly linear (~1:1), so opening
// positions keep their ordering, while blowouts saturate instead of owning
// the MSE gradient.
// ---------------------------------------------------------------------------
constexpr float kWinScaleCP = 400.0f;   // centipawn scale of the tanh map

// Maps a centipawn score (already clamped to +-CLAMP) to the bounded (-1,1)
// target the net's final TANH layer is trained against.
inline float cpWinProb(const float cp) {
    return std::tanh(cp / kWinScaleCP);
}

// Inverts a bounded net output y in (-1,1) back to centipawns. Clamps |y|
// strictly below 1 so atanh stays finite even for a saturated net (tanh can
// only reach +-1 => guard against NaN from an exact +-1 or a malformed net).
inline float winProbToCp(const float y) {
    const float c = std::clamp(y, -0.999999f, 0.999999f);
    return kWinScaleCP * std::atanh(c);
}

// ---------------------------------------------------------------------------
// Material + piece-square tables ("Simplified Evaluation Function" values).
// Tables below are written visually (first row = rank 8); they are converted
// to a1-indexed lookups once at construction.
// ---------------------------------------------------------------------------
class MaterialPSTEvaluator final : public Evaluator {
public:
    MaterialPSTEvaluator() {
        buildTable(PAWN,
            { 0,  0,  0,  0,  0,  0,  0,  0,
             50, 50, 50, 50, 50, 50, 50, 50,
             10, 10, 20, 30, 30, 20, 10, 10,
              5,  5, 10, 25, 25, 10,  5,  5,
              0,  0,  0, 20, 20,  0,  0,  0,
              5, -5,-10,  0,  0,-10, -5,  5,
              5, 10, 10,-20,-20, 10, 10,  5,
              0,  0,  0,  0,  0,  0,  0,  0});
        buildTable(KNIGHT,
            {-50,-40,-30,-30,-30,-30,-40,-50,
             -40,-20,  0,  0,  0,  0,-20,-40,
             -30,  0, 10, 15, 15, 10,  0,-30,
             -30,  5, 15, 20, 20, 15,  5,-30,
             -30,  0, 15, 20, 20, 15,  0,-30,
             -30,  5, 10, 15, 15, 10,  5,-30,
             -40,-20,  0,  5,  5,  0,-20,-40,
             -50,-40,-30,-30,-30,-30,-40,-50});
        buildTable(BISHOP,
            {-20,-10,-10,-10,-10,-10,-10,-20,
             -10,  0,  0,  0,  0,  0,  0,-10,
             -10,  0,  5, 10, 10,  5,  0,-10,
             -10,  5,  5, 10, 10,  5,  5,-10,
             -10,  0, 10, 10, 10, 10,  0,-10,
             -10, 10, 10, 10, 10, 10, 10,-10,
             -10,  5,  0,  0,  0,  0,  5,-10,
             -20,-10,-10,-10,-10,-10,-10,-20});
        buildTable(ROOK,
            {  0,  0,  0,  0,  0,  0,  0,  0,
               5, 10, 10, 10, 10, 10, 10,  5,
              -5,  0,  0,  0,  0,  0,  0, -5,
              -5,  0,  0,  0,  0,  0,  0, -5,
              -5,  0,  0,  0,  0,  0,  0, -5,
              -5,  0,  0,  0,  0,  0,  0, -5,
              -5,  0,  0,  0,  0,  0,  0, -5,
               0,  0,  0,  5,  5,  0,  0,  0});
        buildTable(QUEEN,
            {-20,-10,-10, -5, -5,-10,-10,-20,
             -10,  0,  0,  0,  0,  0,  0,-10,
             -10,  0,  5,  5,  5,  5,  0,-10,
              -5,  0,  5,  5,  5,  5,  0, -5,
               0,  0,  5,  5,  5,  5,  0, -5,
             -10,  5,  5,  5,  5,  5,  0,-10,
             -10,  0,  5,  0,  0,  0,  0,-10,
             -20,-10,-10, -5, -5,-10,-10,-20});
        buildTable(KING,
            {-30,-40,-40,-50,-50,-40,-40,-30,
             -30,-40,-40,-50,-50,-40,-40,-30,
             -30,-40,-40,-50,-50,-40,-40,-30,
             -30,-40,-40,-50,-50,-40,-40,-30,
             -20,-30,-30,-40,-40,-30,-30,-20,
             -10,-20,-20,-20,-20,-20,-20,-10,
              20, 20,  0,  0,  0,  0, 20, 20,
              20, 30, 10,  0,  0, 10, 30, 20});
    }

    // White POV centipawns: sum(white) - sum(black).
    float evaluate(const Position& p) override {
        float score = 0.0f;
        for (int s = 0; s < 64; ++s) {
            const int pc = p.board[s];
            if (pc == EMPTY) continue;
            const int t = pc > 0 ? pc : -pc;
            const int bonus = MAT[t] + pst_[t][pc > 0 ? s : mirrorSq(s)];
            score += (pc > 0) ? bonus : -bonus;
        }
        return score;
    }

    std::string name() const override { return "material+pst"; }

private:
    static constexpr int MAT[7] = {0, 100, 320, 330, 500, 900, 0};
    int pst_[7][64]{};

    void buildTable(const int type, const std::array<int, 64>& visual) {
        for (int v = 0; v < 64; ++v) {
            const int file = v & 7, visRow = v >> 3;   // visRow 0 == rank 8
            pst_[type][sqOf(file, 7 - visRow)] = visual[v];
        }
    }
};

// ---------------------------------------------------------------------------
// Your neural network as an evaluator. Contract (see howto/createAI.md):
//   input  : exactly STATE_SIZE (781) floats from ChessGame getState()
//   output : ONE value. If the net's final layer is a TANH activation, that
//            value is a bounded win-probability in (-kWinScaleCP,kWinScaleCP)
//            and is inverted to centipawns; otherwise it is read directly as
//            centipawns (legacy linear head). White's point of view.
// A legacy Dense layer WITHOUT activation acts as an affine scaler, so you
// can end a TANH stack with e.g. weight 8000 / bias 0 to map [-1,1] -> cp.
// ---------------------------------------------------------------------------
class NetEvaluator final : public Evaluator {
public:
    explicit NetEvaluator(std::shared_ptr<BackpropNet::Net> net, std::string source)
        : net_(std::move(net)), source_(std::move(source)) {
        validate();
        winprob_ = hasTrailingTanh(*net_);
        buf_.reserve(STATE_SIZE);
    }

    float evaluate(const Position& p) override {
        getState(p, buf_);
        const std::vector<float> out = net_->predict(buf_);
        if (out.empty()) return 0.0f;
        float v = out[0];
        if (!std::isfinite(v)) return 0.0f;   // guard NaN/inf nets
        if (winprob_) v = winProbToCp(v);
        return std::clamp(v, -MAX_EVAL_CP, MAX_EVAL_CP);
    }

    std::string name() const override {
        return "net:" + source_ + (winprob_ ? " (win-prob)" : " (linear)");
    }

private:
    static constexpr float MAX_EVAL_CP = 29000.0f;

    std::shared_ptr<BackpropNet::Net> net_;
    std::string source_;
    bool winprob_ = false;
    std::vector<float> buf_;

    // True if the net's final layer is a TANH activation (win-prob convention).
    bool hasTrailingTanh(const BackpropNet::Net& net) const {
        for (auto it = net_.get()->layers.rbegin(); it != net_.get()->layers.rend(); ++it) {
            if ((*it)->get_type() == "DEN") break;     // weights come after activations
            if ((*it)->get_type() == "ACT") {
                const auto* a = static_cast<const BackpropNet::ActivationLayer*>(it->get());
                return a->get_activation() == BackpropNet::ActivationLayer::TANH;
            }
        }
        return false;
    }

    void validate() const {
        const BackpropNet::DenseLayer* first = nullptr;
        for (const auto& l : net_->layers) {
            if (l->get_type() == "DEN") {
                first = static_cast<const BackpropNet::DenseLayer*>(l.get());
                break;
            }
        }
        if (!first)
            throw std::runtime_error("NetEvaluator: '" + source_ + "' has no Dense (DEN) layers");

        if (first->get_weights().cols() != STATE_SIZE)
            throw std::runtime_error(
                "NetEvaluator: '" + source_ + "' input size mismatch: expected " +
                std::to_string(STATE_SIZE) + " (see howto/createAI.md), got " +
                std::to_string(first->get_weights().cols()));

        const BackpropNet::DenseLayer* last = nullptr;
        for (auto it = net_->layers.rbegin(); it != net_->layers.rend(); ++it) {
            if ((*it)->get_type() == "DEN") {
                last = static_cast<const BackpropNet::DenseLayer*>(it->get());
                break;
            }
        }
        if (!last || last->get_biases().size() != 1)
            throw std::runtime_error("NetEvaluator: '" + source_ +
                                     "' must end with a Dense layer producing exactly 1 output");
    }
};

// Loads a .backprop_net file and wraps it (throws on any validation failure).
inline std::shared_ptr<Evaluator> loadNetEvaluator(const std::string& path) {
    auto net = std::make_shared<BackpropNet::Net>(BackpropNetReader::load(path));
    return std::make_shared<NetEvaluator>(std::move(net), path);
}

inline std::shared_ptr<Evaluator> makePstEvaluator() {
    return std::make_shared<MaterialPSTEvaluator>();
}

}   // namespace chess
