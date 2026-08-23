#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace gpf {

template<std::size_t N, typename Scalar = double>
struct BBox
{
    static_assert(std::is_floating_point_v<Scalar>);

    std::array<Scalar, N> lo;
    std::array<Scalar, N> hi;

    BBox()
    {
        lo.fill(std::numeric_limits<Scalar>::max());
        hi.fill(std::numeric_limits<Scalar>::lowest());
    }

    // Compiler-generated copy/move constructors are sufficient:
    // BBox only holds fixed-size std::arrays (inline storage, no indirection),
    // so moving is identical to copying — there is no heap pointer to steal.
    BBox(std::array<Scalar, N> lo, std::array<Scalar, N> hi) noexcept
      : lo(lo)
      , hi(hi)
    {
    }

    std::array<Scalar, N>& min_bound() { return lo; }
    const std::array<Scalar, N>& min_bound() const { return lo; }
    std::array<Scalar, N>& max_bound() { return hi; }
    const std::array<Scalar, N>& max_bound() const { return hi; }

    Scalar min_coord(std::size_t i) const { return lo[i]; }
    Scalar max_coord(std::size_t i) const { return hi[i]; }

    std::size_t longest_axis() const
    {
        std::size_t best = 0;
        Scalar best_len = hi[0] - lo[0];
        for (std::size_t i = 1; i < N; ++i) {
            Scalar len = hi[i] - lo[i];
            if (len > best_len) {
                best = i;
                best_len = len;
            }
        }
        return best;
    }

    BBox& operator+=(const BBox& o) noexcept
    {
        for (std::size_t i = 0; i < N; ++i) {
            lo[i] = std::min(lo[i], o.lo[i]);
            hi[i] = std::max(hi[i], o.hi[i]);
        }
        return *this;
    }

    BBox& add(const Scalar* pt) noexcept
    {
        for (std::size_t i = 0; i < N; ++i) {
            lo[i] = std::min(lo[i], pt[i]);
            hi[i] = std::max(hi[i], pt[i]);
        }
        return *this;
    }

    [[nodiscard]] bool intersects(const BBox& o) const noexcept
    {
        for (std::size_t i = 0; i < N; ++i) {
            if (lo[i] > o.hi[i] || hi[i] < o.lo[i])
                return false;
        }
        return true;
    }
};

using BBox2 = BBox<2>;
using BBox3 = BBox<3>;

} // namespace gpf
