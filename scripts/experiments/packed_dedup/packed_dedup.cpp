// Experiment-only exact integer-weight mask IoU, packed one bit per cell.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <vector>
#if defined(__SSE2__)
#include <emmintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

static uint64_t pack_contiguous(const uint8_t* p, int64_t count) {
    uint64_t word = 0;
    int64_t k = 0;
#if defined(__SSE2__)
    for (; k + 16 <= count; k += 16) {
        const auto bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + k));
        const auto zero = _mm_cmpeq_epi8(bytes, _mm_setzero_si128());
        word |= uint64_t(uint16_t(~_mm_movemask_epi8(zero))) << k;
    }
#elif defined(__aarch64__)
    // AArch64 equivalent of movemask: each nonzero byte contributes its bit
    // weight. Each eight-byte half sums to at most 255, so u8 reduction is exact.
    const uint8_t bit_weights[16] = {1, 2, 4, 8, 16, 32, 64, 128,
                                   1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x16_t weights = vld1q_u8(bit_weights);
    for (; k + 16 <= count; k += 16) {
        const uint8x16_t nonzero = vcgtq_u8(vld1q_u8(p + k), vdupq_n_u8(0));
        const uint8x16_t bits = vandq_u8(nonzero, weights);
        const uint16_t packed = uint16_t(vaddv_u8(vget_low_u8(bits)))
                              | (uint16_t(vaddv_u8(vget_high_u8(bits))) << 8);
        word |= uint64_t(packed) << k;
    }
#endif
    for (; k < count; ++k) word |= uint64_t(p[k] != 0) << k;
    return word;
}

extern "C" int packed_dedup(
    const uint8_t* masks, int64_t n, int64_t pixels,
    const int64_t* indices, const int64_t* lengths, const int64_t* weights,
    int64_t groups, const int64_t* order, double threshold, int want_keeper, int unweighted,
    uint8_t* keep, int64_t* keeper, double* best_iou) {
    try {
        const int64_t words = (pixels + 63) / 64;
        int64_t covered = 0;
        for (int64_t g = 0; g < groups; ++g) covered += lengths[g];
        const uint64_t base_weight = covered == pixels && groups ? weights[0] : 0;
        struct Correction { int64_t word; uint64_t mask, weight; };
        std::vector<Correction> corrections;
        if (indices) {
            int64_t start = 0;
            for (int64_t g = 0; g < groups; ++g) {
                const uint64_t extra = weights[g] - base_weight;
                if (extra) {
                    int64_t previous = -1;
                    for (int64_t k = 0; k < lengths[g]; ++k) {
                        const int64_t cell = indices[start + k], word = cell / 64;
                        if (word != previous) {
                            corrections.push_back({word, 0, extra});
                            previous = word;
                        }
                        corrections.back().mask |= uint64_t(1) << (cell % 64);
                    }
                }
                start += lengths[g];
            }
        }
        std::vector<uint64_t> packed(n * words, 0), area(n, 0);
        for (int64_t q = 0; q < n; ++q) {
            const uint8_t* source = masks + q * pixels;
            uint64_t* target = packed.data() + q * words;
            for (int64_t w = 0; w < words; ++w) {
                const uint64_t word = pack_contiguous(source + 64*w, std::min<int64_t>(64, pixels - 64*w));
                target[w] = word;
                area[q] += uint64_t(__builtin_popcountll(word)) * base_weight;
            }
            for (const auto& c : corrections)
                area[q] += uint64_t(__builtin_popcountll(target[c.word] & c.mask)) * c.weight;
        }
        std::vector<int64_t> retained;
        for (int64_t oi = 0; oi < n; ++oi) {
            const int64_t q = order[oi];
            bool suppress = false;
            double maximum = -1;
            int64_t winner = -1;
            for (const int64_t r : retained) {
                // Integer-weight IoU cannot exceed min(area)/max(area). Do not
                // use this for float32 reference arithmetic or keeper reporting.
                if (!want_keeper && !unweighted) {
                    const uint64_t largest = std::max(area[q], area[r]);
                    const double bound = largest ? double(std::min(area[q], area[r])) / double(largest) : 0.;
                    if (bound <= threshold) continue;
                }
                uint64_t intersection = 0;
                const uint64_t* left = packed.data() + q * words;
                const uint64_t* right = packed.data() + r * words;
                if (base_weight) {
                    uint64_t count = 0;
                    for (int64_t w = 0; w < words; ++w)
                        count += __builtin_popcountll(left[w] & right[w]);
                    intersection = count * base_weight;
                }
                for (const auto& c : corrections)
                    intersection += uint64_t(__builtin_popcountll(left[c.word] & right[c.word] & c.mask)) * c.weight;
                const uint64_t uni = area[q] + area[r] - intersection;
                double overlap = uni ? double(intersection) / double(uni) : 0.;
                if (unweighted) {
                    const float float_union = (float(area[q]) + float(area[r])) - float(intersection);
                    overlap = float_union > 0 ? double(float(intersection) / float_union) : 0.;
                }
                if (overlap > maximum) { maximum = overlap; winner = r; }
                if (overlap > threshold) {
                    suppress = true;
                    if (!want_keeper) break;
                }
            }
            if (suppress) {
                if (want_keeper) { keeper[q] = winner; best_iou[q] = maximum; }
            } else {
                keep[q] = 1;
                retained.push_back(q);
            }
        }
        return 0;
    } catch (const std::exception&) { return 1; }
}
