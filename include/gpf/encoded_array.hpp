#pragma once

#include <cstddef>
#include <iterator>
#include <ranges>
#include <utility>

namespace gpf::detail {

class EncodedArrayCursor
{
  public:
    using value_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using iterator_concept = std::forward_iterator_tag;

    constexpr EncodedArrayCursor() noexcept = default;

    constexpr EncodedArrayCursor(std::size_t encoded_p, std::size_t encoded_q) noexcept
      : p(encoded_p)
      , q(encoded_q)
    {
    }

    [[nodiscard]] constexpr value_type operator*() const noexcept { return p / q - 2; }

    constexpr EncodedArrayCursor& operator++() noexcept
    {
        const auto previous_p = std::exchange(p, q);
        q = previous_p % q;
        return *this;
    }

    constexpr EncodedArrayCursor operator++(int) noexcept
    {
        auto previous = *this;
        ++*this;
        return previous;
    }

    friend constexpr bool operator==(const EncodedArrayCursor&, const EncodedArrayCursor&) noexcept = default;

    friend constexpr bool operator==(const EncodedArrayCursor& cursor, std::default_sentinel_t) noexcept
    {
        return cursor.p == 1 && cursor.q == 0;
    }

  private:
    std::size_t p = 1;
    std::size_t q = 0;
};

} // namespace gpf::detail

namespace gpf {

struct EncodedArray
{
    std::size_t p = 1;
    std::size_t q = 0;

    constexpr void push_inner(std::size_t m) noexcept { q = std::exchange(p, m * p + q); }

    constexpr void push(std::size_t val) noexcept { push_inner(val + 2); }

    [[nodiscard]] constexpr bool empty() const noexcept { return p == 1 && q == 0; }

    [[nodiscard]] constexpr auto range() const noexcept
    {
        return std::ranges::subrange(detail::EncodedArrayCursor{ p, q }, std::default_sentinel);
    }
};

} // namespace gpf
