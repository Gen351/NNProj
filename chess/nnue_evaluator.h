#pragma once

// NNUEEvaluator -- an NNUE-style accumulator accelerator for the SAME
// .backprop_net eval nets this project already uses (781 inputs: one-hot
// piece-square, +-1 side, castling/EP bits).
//
// The whole speedup is in the FIRST layer. The reference path (NetEvaluator)
// recomputes the full 781 x H1 matrix product on every eval. Here it becomes
// a sparse accumulator over the features that are actually active:
//
//       acc = b1 + sum over ACTIVE features of  W1[feature][*] * value
//
// A chess position touches only ~34 of the 781 features, so the first layer
// is ~34 column-adds of length H1 instead of 781 * H1 multiply-adds -- an
// order-of-magnitude drop before SIMD matters. Everything downstream
// (H1 -> ... -> 1) is small.
//
// Weights are lifted from the loaded net unchanged ("distillation by
// copying": identical feature semantics, so strength carries over).
//
// QUANTIZATION / AVX2 READY
// -------------------------
// The first-layer weights are stored FEATURE-MAJOR, i.e. the column for
// feature f is the contiguous slice w1[f*stride .. f*stride + H1), stride
// being H1 rounded UP to a multiple of 16 (rows are zero-padded, so SIMD has
// no tail handling). That is exactly the layout _mm256_add_epi16 wants: one
// 256-bit register holds 16 lanes of the accumulator.
//
//   * float mode   (quantizeFirstLayer = false): column adds over float w1f_
//   * int16 mode   (quantizeFirstLayer = true ): load-time quantization into
//     int16_t w1q_, and an accumulator that is either int16 (16 lanes per
//     _mm256_add_epi16) or int32 (8 lanes per _mm256_cvtepi16_epi32 +
//     _mm256_add_epi32), chosen ONCE per net so no position can overflow:
//     the accumulator width is int16 only if every hidden unit's worst-case
//     active-feature column sum (|bias| + the 38 largest |w1q| entries, 38 =
//     max simultaneously-active features: <=32 pieces + side + <=4 castling
//     + 1 ep) fits in int16. Integer adds then cannot wrap, so both widths
//     are bit-exact. The scalar loops below are the SIMD counterparts.
//
// SIMD: all intrinsics live behind `#if defined(__AVX2__)` (build with
// `-mavx2`); without the flag the exact same arithmetic runs scalar, byte
// for byte. The downstream float dense layers are AVX2/`-mfma`-vectorized
// too (reordered float rounding only, well inside the eval tolerances).
//
// Thread-safety: all weights are read-only after construction; working
// buffers are per-instance. Follow the NetEvaluator rule -- one instance per
// thread (see uci_main / self_play factory wiring).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "./evaluator.h"

namespace chess {

class NNUEEvaluator final : public Evaluator {
public:
    explicit NNUEEvaluator(std::shared_ptr<BackpropNet::Net> net,
                           std::string source,
                           const bool quantizeFirstLayer = false)
        : net_(std::move(net)), source_(std::move(source)),
          quantized_(quantizeFirstLayer) {
        validateEvalNet(*net_, source_);
        winprob_ = netHasTrailingTanh(*net_);
        extractPipeline();
        if (quantized_) quantizeFirstLayerWeights();
    }

    float evaluate(const Position& p) override {
        std::vector<float>& h = work_;
        h.assign(b1f_.begin(), b1f_.end());               // first-layer pre-act = b1

        if (quantized_) {
            if (int16AccOk_) {
                accI16_.assign(b1q16_.begin(), b1q16_.end());
                accumulateInt(p);
                for (size_t o = 0; o < h1_; ++o)
                    h[o] = static_cast<float>(accI16_[o]) * w1inv_;
            } else {
                accI_.assign(b1q_.begin(), b1q_.end());
                accumulateInt(p);
                for (size_t o = 0; o < h1_; ++o)
                    h[o] = static_cast<float>(accI_[o]) * w1inv_;
            }
        } else {
            accumulateFloat(p, h);
        }

        applyAct(h.data(), h1_, firstAct_);
        for (size_t k = 1; k < dens_.size(); ++k) {
            denseMul(dens_[k], h, out_);
            applyAct(out_.data(), out_.size(), actAfter_[k]);
            h.swap(out_);
        }

        float v = h[0];
        if (!std::isfinite(v)) v = 0.0f;
        if (winprob_) v = winProbToCp(v);
        return std::clamp(v, -MAX_EVAL_CP, MAX_EVAL_CP);
    }

    std::string name() const override {
        std::string s = "nnue-acc:" + source_;
        if (quantized_) {
            if (int16AccOk_) s += " (quant:int16-acc)";
            else s += " (quant:int32-acc)";
        }
        if (winprob_) s += " (win-prob)"; else s += " (linear)";
        return s;
    }

private:
    static constexpr float MAX_EVAL_CP = 29000.0f;

    struct DenseF {
        size_t cols = 0, rows = 0;
        std::vector<float> w;      // unit-major [rows][cols]
        std::vector<float> b;      // [rows]
    };

    // Activation ids: 0 = linear (no activation), 1..4 = SIGM/LEAK/RELU/TANH.
    std::shared_ptr<BackpropNet::Net> net_;
    std::string source_;
    bool winprob_ = false;
    bool quantized_ = false;
    bool int16AccOk_ = true;   // per-net: int16 accumulator cannot overflow

    size_t h1_ = 0;                                        // first hidden width (logical)
    size_t h1pad_ = 0;                                     // stride = h1_ rounded up to 16
    std::vector<float> w1f_;                               // feature-major [781*h1pad_]
    std::vector<int16_t> w1q_;                             // feature-major [781*h1pad_] (quant)
    std::vector<float> b1f_;                               // first-layer bias, [h1pad_]
    std::vector<int32_t> b1q_;                             // [h1pad_] (quant)
    std::vector<int16_t> b1q16_;                           // [h1pad_] (quant, int16-acc path)
    float w1scale_ = 0.0f;                                 // int16 quant scale (S)
    float w1inv_ = 0.0f;                                   // 1/S -> dequant factor
    int8_t firstAct_ = 0;                                  // activation after the first layer
    std::vector<DenseF> dens_;                             // all layers (dens_[0] = first)
    std::vector<int8_t> actAfter_;                         // activation after each layer
    std::vector<float> work_, out_;                        // per-instance scratch
    std::vector<int32_t> accI_;                            // int32-acc path accumulator
    std::vector<int16_t> accI16_;                          // int16-acc path accumulator

    // ---- Pipeline extraction -------------------------------------------------

    void extractPipeline() {
        for (const auto& l : net_->layers) {
            if (l->get_type() == "DEN") {
                const auto* d = static_cast<const BackpropNet::DenseLayer*>(l.get());
                DenseF df;
                df.cols = d->get_weights().cols();
                df.rows = d->get_weights().rows();
                df.w.assign(d->get_weights().data.begin(), d->get_weights().data.end());
                df.b = d->get_biases();
                dens_.push_back(std::move(df));
                actAfter_.push_back(0);
            } else if (l->get_type() == "ACT") {
                if (dens_.empty())
                    throw std::runtime_error("eval net '" + source_ +
                                             "' has an activation before any Dense layer");
                const auto* a = static_cast<const BackpropNet::ActivationLayer*>(l.get());
                switch (a->get_activation()) {
                    case BackpropNet::ActivationLayer::SIGM: actAfter_.back() = 1; break;
                    case BackpropNet::ActivationLayer::LEAK: actAfter_.back() = 2; break;
                    case BackpropNet::ActivationLayer::RELU: actAfter_.back() = 3; break;
                    case BackpropNet::ActivationLayer::TANH: actAfter_.back() = 4; break;
                    case BackpropNet::ActivationLayer::SOFT:
                        throw std::runtime_error("eval net '" + source_ +
                                                 "' uses SoftMax (unsupported in eval nets)");
                }
            }
        }
        if (dens_.empty())
            throw std::runtime_error("eval net '" + source_ + "' has no Dense layers");

        h1_ = dens_[0].rows;
        h1pad_ = (h1_ + 15) & ~static_cast<size_t>(15);   // SIMD row stride (16-aligned)
        firstAct_ = actAfter_[0];
        b1f_ = dens_[0].b;
        b1f_.resize(h1pad_, 0.0f);

        // Transpose the first layer to feature-major (AVX2-friendly): the column
        // for feature f is the contiguous slice w1f_[f*stride .. f*stride + h1_],
        // zero-padded up to the 16-lane stride so SIMD has no tail handling.
        w1f_.assign(dens_[0].cols * h1pad_, 0.0f);
        for (size_t f = 0; f < dens_[0].cols; ++f)
            for (size_t o = 0; o < h1_; ++o)
                w1f_[f * h1pad_ + o] = dens_[0].w[o * dens_[0].cols + f];

        // The first layer's unit-major copy is now only referenced through the
        // transposed/bias storage; free this duplicate (~800 KB at h1 = 256).
        dens_[0].w.clear();
        dens_[0].w.shrink_to_fit();
        dens_[0].b.clear();
        dens_[0].b.shrink_to_fit();

        // Dimension-consistency check from the first hidden layer onward.
        size_t in = h1_;
        for (size_t k = 1; k < dens_.size(); ++k) {
            if (dens_[k].cols != in)
                throw std::runtime_error("eval net '" + source_ +
                                         "' has disconnected layer sizes");
            in = dens_[k].rows;
        }
        if (in != 1)
            throw std::runtime_error("eval net '" + source_ +
                                     "' output size must be 1");
    }

    void quantizeFirstLayerWeights() {
        float maxAbs = 0.0f;
        for (const float w : b1f_) maxAbs = std::max(maxAbs, std::fabs(w));
        for (const float w : w1f_) maxAbs = std::max(maxAbs, std::fabs(w));
        if (maxAbs == 0.0f) maxAbs = 1.0f;

        w1scale_ = 32767.0f / maxAbs;
        w1inv_ = 1.0f / w1scale_;

        const size_t cols = dens_[0].cols;
        w1q_.resize(cols * h1pad_);
        for (size_t i = 0; i < w1f_.size(); ++i)
            w1q_[i] = static_cast<int16_t>(std::lround(w1f_[i] * w1scale_));
        b1q_.resize(h1pad_, 0);
        b1q16_.resize(h1pad_, 0);
        for (size_t o = 0; o < h1_; ++o) {
            const int32_t q = static_cast<int32_t>(std::lround(b1f_[o] * w1scale_));
            b1q_[o] = q;
            b1q16_[o] = static_cast<int16_t>(q);
        }

        // Adaptive accumulator width: the int16 accumulator is only safe if NO
        // legal position can overflow it. The most features ever active at once
        // is 38 (<=32 occupied squares + side + <=4 castling + 1 ep), so bound
        // every unit by |bias| plus its 38 largest |w1q| magnitudes.
        int16AccOk_ = true;
        const size_t maxActive = 38;
        std::vector<int16_t> mags(cols);
        std::vector<int16_t> tops(std::min(maxActive, cols));
        for (size_t o = 0; o < h1_ && int16AccOk_; ++o) {
            for (size_t f = 0; f < cols; ++f)
                mags[f] = static_cast<int16_t>(
                    std::min<int32_t>(32767, std::abs(w1q_[f * h1pad_ + o])));
            std::partial_sort(mags.begin(), mags.begin() + tops.size(), mags.end(),
                              [](const int16_t a, const int16_t b) { return a > b; });
            int64_t bound = std::llabs(static_cast<int64_t>(b1q_[o]));
            for (size_t k = 0; k < tops.size(); ++k) bound += mags[k];
            if (bound > 32767) int16AccOk_ = false;
        }
    }

    // ---- Sparse first-layer accumulation ------------------------------------

#if !defined(__AVX2__)

    inline void addFeatureColFloat(const size_t f, std::vector<float>& h) {
        const float* col = &w1f_[f * h1pad_];
        float* hd = h.data();
        for (size_t o = 0; o < h1_; ++o) hd[o] += col[o];
    }
    inline void subFeatureColFloat(const size_t f, std::vector<float>& h) {
        const float* col = &w1f_[f * h1pad_];
        float* hd = h.data();
        for (size_t o = 0; o < h1_; ++o) hd[o] -= col[o];
    }
    inline void addFeatureColInt16(const size_t f) {
        const int16_t* col = &w1q_[f * h1pad_];
        int16_t* a = accI16_.data();
        for (size_t o = 0; o < h1_; ++o) a[o] += col[o];
    }
    inline void subFeatureColInt16(const size_t f) {
        const int16_t* col = &w1q_[f * h1pad_];
        int16_t* a = accI16_.data();
        for (size_t o = 0; o < h1_; ++o) a[o] -= col[o];
    }
    inline void addFeatureColInt32(const size_t f) {
        const int16_t* col = &w1q_[f * h1pad_];
        int32_t* a = accI_.data();
        for (size_t o = 0; o < h1_; ++o) a[o] += col[o];
    }
    inline void subFeatureColInt32(const size_t f) {
        const int16_t* col = &w1q_[f * h1pad_];
        int32_t* a = accI_.data();
        for (size_t o = 0; o < h1_; ++o) a[o] -= col[o];
    }

#else   // __AVX2__

    // 16 lanes per register: load a +/-1 feature column and add it into the
    // int16 accumulator (exact -- the width was chosen at load so it cannot
    // overflow, see int16AccOk_).
    inline void addFeatureColInt16(const size_t f) {
        const int16_t* col = &w1q_[f * h1pad_];
        int16_t* a = accI16_.data();
        size_t o = 0;
        for (; o + 16 <= h1pad_; o += 16) {
            __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col + o));
            __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + o));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + o), _mm256_add_epi16(b, c));
        }
        for (; o < h1_; ++o) a[o] += col[o];   // dead when h1_ is 16-aligned
    }
    inline void subFeatureColInt16(const size_t f) {
        const int16_t* col = &w1q_[f * h1pad_];
        int16_t* a = accI16_.data();
        size_t o = 0;
        for (; o + 16 <= h1pad_; o += 16) {
            __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col + o));
            __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + o));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + o), _mm256_sub_epi16(b, c));
        }
        for (; o < h1_; ++o) a[o] -= col[o];
    }

    // 8 lanes per register: widen the 16 int16 column lanes to int32 (two
    // cvtepi16_epi32 per 128-bit half) and add into the int32 accumulator.
    inline void addFeatureColInt32(const size_t f) {
        const int16_t* col = &w1q_[f * h1pad_];
        int32_t* a = accI_.data();
        size_t o = 0;
        for (; o + 16 <= h1pad_; o += 16) {
            __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col + o));
            __m256i lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(c));
            __m256i hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(c, 1));
            __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + o));
            __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + o + 8));
            a0 = _mm256_add_epi32(a0, lo);
            a1 = _mm256_add_epi32(a1, hi);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + o), a0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + o + 8), a1);
        }
        for (; o < h1_; ++o) a[o] += col[o];   // dead when h1_ is 16-aligned
    }
    inline void subFeatureColInt32(const size_t f) {
        const int16_t* col = &w1q_[f * h1pad_];
        int32_t* a = accI_.data();
        size_t o = 0;
        for (; o + 16 <= h1pad_; o += 16) {
            __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col + o));
            __m256i lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(c));
            __m256i hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(c, 1));
            __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + o));
            __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + o + 8));
            a0 = _mm256_sub_epi32(a0, lo);
            a1 = _mm256_sub_epi32(a1, hi);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + o), a0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + o + 8), a1);
        }
        for (; o < h1_; ++o) a[o] -= col[o];
    }

    // 8 float lanes per register (float reference path, padded rows harmless).
    inline void addFeatureColFloat(const size_t f, std::vector<float>& h) {
        const float* col = &w1f_[f * h1pad_];
        float* hd = h.data();
        size_t o = 0;
        for (; o + 8 <= h1pad_; o += 8) {
            __m256 a = _mm256_loadu_ps(hd + o);
            __m256 c = _mm256_loadu_ps(col + o);
            _mm256_storeu_ps(hd + o, _mm256_add_ps(a, c));
        }
        for (; o < h1_; ++o) hd[o] += col[o];
    }
    inline void subFeatureColFloat(const size_t f, std::vector<float>& h) {
        const float* col = &w1f_[f * h1pad_];
        float* hd = h.data();
        size_t o = 0;
        for (; o + 8 <= h1pad_; o += 8) {
            __m256 a = _mm256_loadu_ps(hd + o);
            __m256 c = _mm256_loadu_ps(col + o);
            _mm256_storeu_ps(hd + o, _mm256_sub_ps(a, c));
        }
        for (; o < h1_; ++o) hd[o] -= col[o];
    }

#endif   // __AVX2__

    inline void addFeatureColInt(const size_t f) {
        if (int16AccOk_) addFeatureColInt16(f); else addFeatureColInt32(f);
    }
    inline void subFeatureColInt(const size_t f) {
        if (int16AccOk_) subFeatureColInt16(f); else subFeatureColInt32(f);
    }

    // Adds the contributions of every ACTIVE feature to `h` (float mode).
    // Exactly mirrors getState() in chess_environment.h.
    void accumulateFloat(const Position& p, std::vector<float>& h) {
        for (int s = 0; s < 64; ++s) {
            const int pc = p.board[s];
            if (pc != EMPTY)
                addFeatureColFloat(static_cast<size_t>(zobrist::pieceIndex(pc)) * 64 + s, h);
        }
        if (p.side < 0) subFeatureColFloat(768, h); else addFeatureColFloat(768, h);
        if (p.castling & CASTLE_WK) addFeatureColFloat(769, h);
        if (p.castling & CASTLE_WQ) addFeatureColFloat(770, h);
        if (p.castling & CASTLE_BK) addFeatureColFloat(771, h);
        if (p.castling & CASTLE_BQ) addFeatureColFloat(772, h);
        if (p.ep >= 0) addFeatureColFloat(773 + static_cast<size_t>(p.ep & 7), h);
    }

    // Same, but into the int32 accumulator (quant mode).
    void accumulateInt(const Position& p) {
        for (int s = 0; s < 64; ++s) {
            const int pc = p.board[s];
            if (pc != EMPTY)
                addFeatureColInt(static_cast<size_t>(zobrist::pieceIndex(pc)) * 64 + s);
        }
        if (p.side < 0) subFeatureColInt(768); else addFeatureColInt(768);
        if (p.castling & CASTLE_WK) addFeatureColInt(769);
        if (p.castling & CASTLE_WQ) addFeatureColInt(770);
        if (p.castling & CASTLE_BK) addFeatureColInt(771);
        if (p.castling & CASTLE_BQ) addFeatureColInt(772);
        if (p.ep >= 0) addFeatureColInt(773 + static_cast<size_t>(p.ep & 7));
    }

    // ---- Downstream dense + activations (float, small) ----------------------

    void denseMul(const DenseF& df, const std::vector<float>& src,
                  std::vector<float>& dst) {
        dst.assign(df.rows, 0.0f);
        for (size_t o = 0; o < df.rows; ++o) {
            float s = df.b[o];
            const float* row = &df.w[o * df.cols];
#if defined(__AVX2__)
            size_t j = 0;
            __m256 acc = _mm256_setzero_ps();
            for (; j + 8 <= df.cols; j += 8) {
                __m256 a = _mm256_loadu_ps(src.data() + j);
                __m256 r = _mm256_loadu_ps(row + j);
#if defined(__FMA__)
                acc = _mm256_fmadd_ps(a, r, acc);
#else
                acc = _mm256_add_ps(acc, _mm256_mul_ps(a, r));
#endif
            }
            float t[8];
            _mm256_storeu_ps(t, acc);
            s += (t[0] + t[1] + t[2] + t[3]) + (t[4] + t[5] + t[6] + t[7]);
            for (; j < df.cols; ++j) s += src[j] * row[j];
#else
            for (size_t j = 0; j < df.cols; ++j) s += src[j] * row[j];
#endif
            dst[o] = s;
        }
    }

    // Matches BackpropNet::ActivationLayer math exactly (SIGM/LEAK/RELU/TANH).
    inline void applyAct(float* v, const size_t n, const int act) {
        if (act == 0) return;
        const float alpha = 0.01f;
        for (size_t i = 0; i < n; ++i) {
            float& x = v[i];
            switch (act) {
                case 1: x = 1.0f / (1.0f + std::exp(-x)); break;
                case 2: x = x * (x > 0) + alpha * x * (x <= 0); break;
                case 3: x = (x > 0) * x; break;
                case 4: x = std::tanh(x); break;
            }
        }
    }
};

inline std::shared_ptr<Evaluator> makeNnueEvaluator(std::shared_ptr<BackpropNet::Net> net,
                                                    const std::string& source,
                                                    const bool quantizeFirstLayer = false) {
    return std::make_shared<NNUEEvaluator>(std::move(net), source, quantizeFirstLayer);
}

}   // namespace chess