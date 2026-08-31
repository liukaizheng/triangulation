#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <vector>

#include <gpf/encoded_array.hpp>

using EncodedRange = decltype(gpf::EncodedArray{}.range());
using EncodedIterator = std::ranges::iterator_t<EncodedRange>;
using EncodedSentinel = std::ranges::sentinel_t<EncodedRange>;

static_assert(!std::ranges::range<gpf::EncodedArray>);
static_assert(std::same_as<EncodedRange, std::ranges::subrange<EncodedIterator, EncodedSentinel>>);
static_assert(std::forward_iterator<EncodedIterator>);
static_assert(std::sentinel_for<EncodedSentinel, EncodedIterator>);
static_assert(std::ranges::view<EncodedRange>);
static_assert(std::ranges::forward_range<EncodedRange>);
static_assert(std::ranges::forward_range<const EncodedRange>);
static_assert(std::ranges::borrowed_range<EncodedRange>);
static_assert(std::ranges::borrowed_range<const EncodedRange>);
static_assert(std::ranges::viewable_range<EncodedRange>);

namespace {

constexpr gpf::EncodedArray
make_encoded_array()
{
    gpf::EncodedArray values;
    values.push(4);
    values.push(0);
    values.push(9);
    return values;
}

constexpr bool
encoded_array_constexpr_round_trip()
{
    const auto values = make_encoded_array();

    const auto encoded_p = values.p;
    const auto encoded_q = values.q;
    constexpr std::array<std::size_t, 3> expected{ 9, 0, 4 };

    auto expected_iterator = expected.begin();
    for (const auto value : values.range()) {
        if (expected_iterator == expected.end() || value != *expected_iterator) {
            return false;
        }
        ++expected_iterator;
    }

    return expected_iterator == expected.end() && values.p == encoded_p && values.q == encoded_q;
}

static_assert(encoded_array_constexpr_round_trip());

} // namespace

void
test_encoded_array_iteration()
{
    gpf::EncodedArray empty;
    assert(empty.empty());

    const auto empty_range = empty.range();
    assert(std::ranges::empty(empty_range));
    assert(empty_range.begin() == empty_range.end());

    std::size_t empty_count = 0;
    for (const auto value : empty.range()) {
        static_cast<void>(value);
        ++empty_count;
    }
    assert(empty_count == 0);

    gpf::EncodedArray single;
    single.push(0);
    assert(!single.empty());
    constexpr std::array<std::size_t, 1> single_expected{ 0 };
    assert(std::ranges::equal(single.range(), single_expected));

    auto values = make_encoded_array();

    assert(values.p == 149);
    assert(values.q == 13);

    const auto encoded_p = values.p;
    const auto encoded_q = values.q;
    const std::vector<std::size_t> expected{ 9, 0, 4 };
    assert(std::ranges::equal(values.range(), expected));

    std::vector<std::size_t> ranges_decoded;
    std::ranges::copy(values.range(), std::back_inserter(ranges_decoded));
    assert(ranges_decoded == expected);

    const auto incremented_values = values.range() | std::views::transform([](const auto value) { return value + 1; });
    const std::vector<std::size_t> incremented_expected{ 10, 1, 5 };
    assert(std::ranges::equal(incremented_values, incremented_expected));

    auto found_in_temporary = std::ranges::find(make_encoded_array().range(), 0);
    assert(found_in_temporary != std::default_sentinel);
    assert(*found_in_temporary == 0);

    const auto& const_values = values;
    std::vector<std::size_t> const_decoded;
    for (const auto value : const_values.range()) {
        const_decoded.push_back(value);
    }
    assert(const_decoded == expected);

    const auto decoded_range = values.range();
    auto original = decoded_range.begin();
    auto copy = original;
    assert(original == copy);
    ++copy;
    assert(original != copy);
    assert(*original == 9);
    assert(*copy == 0);

    auto iterator = decoded_range.begin();
    auto previous = iterator++;
    assert(*previous == 9);
    assert(*iterator == 0);

    auto& incremented = ++iterator;
    assert(&incremented == &iterator);
    assert(*incremented == 4);
    ++iterator;
    assert(iterator == decoded_range.end());

    assert(values.p == encoded_p);
    assert(values.q == encoded_q);
}
