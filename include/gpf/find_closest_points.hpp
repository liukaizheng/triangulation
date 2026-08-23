#pragma once

#include "bbox.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Dense>

namespace gpf {

namespace ranges = std::ranges;

template<std::size_t N, typename Scalar>
struct BoundingSphere
{
    static_assert(std::is_floating_point_v<Scalar>);

    std::span<const Scalar, N> c;
    Scalar r2{ std::numeric_limits<Scalar>::max() };
};

template<std::size_t N, typename Scalar>
struct Interaction
{
    static_assert(std::is_floating_point_v<Scalar>);

    Scalar d{ std::numeric_limits<Scalar>::max() };
    std::size_t primitive_index{ std::numeric_limits<std::size_t>::max() };
    std::size_t node_index{ std::numeric_limits<std::size_t>::max() };
    std::array<Scalar, N> p{};
    std::array<Scalar, 2> uv{};
};

namespace bvh {
inline constexpr std::size_t kMaxBvhDepth = 64;

template<std::size_t N, typename Scalar>
using Vec = Eigen::Vector<Scalar, static_cast<int>(N)>;

template<std::size_t N, typename Scalar>
[[nodiscard]] Scalar
bbox_surface_area(const BBox<N, Scalar>& bbox)
{
    constexpr Scalar kMinSurfaceExtent = static_cast<Scalar>(1e-5);
    const Vec<N, Scalar> extent =
      (Vec<N, Scalar>::Map(bbox.max_bound().data()) - Vec<N, Scalar>::Map(bbox.min_bound().data()))
        .cwiseMax(kMinSurfaceExtent);
    return Scalar{ 2 } * Vec<N, Scalar>::Constant(extent.prod()).cwiseQuotient(extent).sum();
}

struct Leaf
{
    std::size_t start;
    std::size_t end;
};

struct Interior
{
    std::array<std::size_t, 2> children;
};

using NodeBase = std::variant<Leaf, Interior>;

template<std::size_t N, typename Scalar>
struct Node
{
    NodeBase base;
    BBox<N, Scalar> bbox;
};

template<std::size_t N, typename Scalar>
[[nodiscard]] auto
calc_triangle_bboxes_and_centroids(const std::span<const Scalar> points, const std::span<const std::size_t> triangles)
{
    assert(points.size() % N == 0);
    assert(triangles.size() % 3 == 0);

    const std::size_t n_points = points.size() / N;

    std::vector<BBox<N, Scalar>> bboxes;
    std::vector<std::array<Scalar, N>> centroids;
    bboxes.reserve(triangles.size() / 3);
    centroids.reserve(triangles.size() / 3);

    for (std::size_t i = 0; i < triangles.size(); i += 3) {
        const auto* tri = &triangles[i];
        const auto* pa = &points[tri[0] * N];
        const auto* pb = &points[tri[1] * N];
        const auto* pc = &points[tri[2] * N];

        BBox<N, Scalar> bbox;
        bbox.add(pa).add(pb).add(pc);
        bboxes.push_back(std::move(bbox));

        std::array<Scalar, N> centroid{};
        Vec<N, Scalar>::Map(centroid.data()) =
          (Vec<N, Scalar>::Map(pa) + Vec<N, Scalar>::Map(pb) + Vec<N, Scalar>::Map(pc)) / Scalar{ 3 };
        centroids.push_back(std::move(centroid));
    }
    return std::make_pair(std::move(bboxes), std::move(centroids));
}

template<std::size_t N, typename Scalar>
[[nodiscard]] auto
compute_split_info(const BBox<N, Scalar>& node_bbox,
                   const std::span<const BBox<N, Scalar>> bboxes,
                   const std::span<const std::array<Scalar, N>> centroids,
                   const std::span<const std::size_t> triangle_indices,
                   const std::size_t start,
                   const std::size_t end,
                   const std::size_t n_leaves,
                   const bool is_deep)
{
    assert(bboxes.size() == centroids.size());
    assert(start < end);
    assert(end <= triangle_indices.size());

    constexpr std::size_t kNumBuckets = 8;
    constexpr Scalar kMinExtent = static_cast<Scalar>(1e-6);
    struct Bucket
    {
        BBox<N, Scalar> bbox;
        std::size_t count{};
    };

    Scalar split_cost = std::numeric_limits<Scalar>::max();
    std::size_t split_dim = N;
    Scalar split_val{};

    for (std::size_t dim = 0; dim < N; ++dim) {
        const Scalar extent = node_bbox.max_coord(dim) - node_bbox.min_coord(dim);
        if (extent < kMinExtent) {
            continue;
        }

        const Scalar bucket_width = extent / static_cast<Scalar>(kNumBuckets);
        std::array<Bucket, kNumBuckets> buckets{};
        std::array<Bucket, kNumBuckets> right_buckets{};

        for (std::size_t i = start; i < end; ++i) {
            const std::size_t triangle_idx = triangle_indices[i];
            const Scalar bucket_offset = (centroids[triangle_idx][dim] - node_bbox.min_coord(dim)) / bucket_width;
            const std::size_t bucket_idx =
              static_cast<std::size_t>(std::clamp(bucket_offset, Scalar{ 0 }, static_cast<Scalar>(kNumBuckets - 1)));
            auto& bucket = buckets[bucket_idx];
            if (bucket.count == 0) {
                bucket.bbox = bboxes[triangle_idx];
            } else {
                bucket.bbox += bboxes[triangle_idx];
            }
            ++bucket.count;
        }

        BBox<N, Scalar> right_bbox;
        std::size_t right_count = 0;
        for (std::size_t bucket_idx = kNumBuckets - 1; bucket_idx > 0; --bucket_idx) {
            const auto& bucket = buckets[bucket_idx];
            if (bucket.count != 0) {
                if (right_count == 0) {
                    right_bbox = bucket.bbox;
                } else {
                    right_bbox += bucket.bbox;
                }
                right_count += bucket.count;
            }
            right_buckets[bucket_idx].bbox = right_bbox;
            right_buckets[bucket_idx].count = right_count;
        }

        BBox<N, Scalar> left_bbox;
        std::size_t left_count = 0;
        for (std::size_t bucket_idx = 1; bucket_idx < kNumBuckets; ++bucket_idx) {
            const auto& bucket = buckets[bucket_idx - 1];
            if (bucket.count != 0) {
                if (left_count == 0) {
                    left_bbox = bucket.bbox;
                } else {
                    left_bbox += bucket.bbox;
                }
                left_count += bucket.count;
            }

            const auto& right_bucket = right_buckets[bucket_idx];
            if (left_count == 0 || right_bucket.count == 0) {
                continue;
            }
            // Near the leaves, align at least one side to the batch width to reduce padded lanes.
            if (is_deep && left_count % n_leaves != 0 && right_bucket.count % n_leaves != 0) {
                continue;
            }

            const Scalar cost = static_cast<Scalar>(left_count) * bbox_surface_area(left_bbox) +
                                static_cast<Scalar>(right_bucket.count) * bbox_surface_area(right_bucket.bbox);
            if (cost < split_cost) {
                split_cost = cost;
                split_dim = dim;
                split_val = node_bbox.min_coord(dim) + static_cast<Scalar>(bucket_idx) * bucket_width;
            }
        }
    }

    if (split_dim == N) {
        // Degenerate bucket distributions fall back to the midpoint of the longest centroid extent.
        const auto& first_centroid = centroids[triangle_indices[start]];
        BBox<N, Scalar> centroid_bbox{ first_centroid, first_centroid };
        for (std::size_t i = start + 1; i < end; ++i) {
            const std::size_t triangle_idx = triangle_indices[i];
            centroid_bbox.add(centroids[triangle_idx].data());
        }
        split_dim = centroid_bbox.longest_axis();
        split_val = (centroid_bbox.min_coord(split_dim) + centroid_bbox.max_coord(split_dim)) * Scalar{ 0.5 };
    }

    return std::make_pair(split_dim, split_val);
}

template<std::size_t N, typename Scalar>
[[nodiscard]] std::size_t
perform_split(const std::size_t start,
              const std::size_t end,
              const std::size_t dim,
              const Scalar val,
              const std::span<const std::array<Scalar, N>> centroids,
              const std::span<std::size_t> triangle_indices)
{
    assert(start < end);
    assert(end <= triangle_indices.size());
    assert(dim < N);

    std::size_t mid_idx = start;
    for (std::size_t i = start; i < end; ++i) {
        if (centroids[triangle_indices[i]][dim] < val) {
            std::swap(triangle_indices[i], triangle_indices[mid_idx]);
            ++mid_idx;
        }
    }

    if (mid_idx == start || mid_idx == end) {
        mid_idx = start + (end - start) / 2;
    }

    return mid_idx;
}

template<std::size_t N, typename Scalar>
std::size_t
build_bvh_recursive(const std::span<const Scalar> points,
                    const std::span<const std::size_t> triangles,
                    const std::span<const BBox<N, Scalar>> bboxes,
                    const std::span<const std::array<Scalar, N>> centroids,
                    std::vector<std::size_t>& triangle_indices,
                    std::vector<Node<N, Scalar>>& nodes,
                    const std::size_t start,
                    const std::size_t end,
                    std::size_t depth,
                    const std::size_t guess_depth)
{
    const auto node_idx = nodes.size();
    BBox<N, Scalar> bbox;
    for (std::size_t i{ start }; i < end; i++) {
        bbox += bboxes[triangle_indices[i]];
    }
    // Four primitives fit one downstream wide leaf node.
    constexpr std::size_t kNLeaves{ 4 };
    if (end - start <= kNLeaves || depth + 1 == kMaxBvhDepth) {
        nodes.emplace_back(Leaf{ start, end }, std::move(bbox));
    } else {
        nodes.emplace_back(Interior{}, std::move(bbox));
        const auto is_deep = depth * 3 > guess_depth * 2;
        const auto [split_dim, split_val] = compute_split_info<N, Scalar>(
          nodes[node_idx].bbox, bboxes, centroids, triangle_indices, start, end, kNLeaves, is_deep);
        const auto split_idx = perform_split<N, Scalar>(start, end, split_dim, split_val, centroids, triangle_indices);
        const auto left_node_idx = build_bvh_recursive(
          points, triangles, bboxes, centroids, triangle_indices, nodes, start, split_idx, depth + 1, guess_depth);
        const auto right_node_idx = build_bvh_recursive(
          points, triangles, bboxes, centroids, triangle_indices, nodes, split_idx, end, depth + 1, guess_depth);
        nodes[node_idx].base = Interior{ left_node_idx, right_node_idx };
    }

    return node_idx;
}

template<std::size_t N, typename Scalar>
struct BVHTree
{
    static_assert(std::is_floating_point_v<Scalar>);

    std::span<const Scalar> points;
    std::span<const std::size_t> triangles;
    std::vector<Node<N, Scalar>> nodes;
    std::vector<std::size_t> triangle_indices;
};

template<std::size_t N, typename Scalar>
BVHTree<N, Scalar>
build_bvh(const std::span<const Scalar> points, const std::span<const std::size_t> triangles)
{
    const auto [bboxes, centroids] = calc_triangle_bboxes_and_centroids<N, Scalar>(points, triangles);
    auto triangle_indices = ranges::iota_view{ std::size_t{ 0 }, triangles.size() / 3 } | ranges::to<std::vector>();
    std::vector<Node<N, Scalar>> nodes;
    const std::size_t guess_depth = std::log2(triangle_indices.size());
    build_bvh_recursive<N, Scalar>(
      points, triangles, bboxes, centroids, triangle_indices, nodes, 0, bboxes.size(), 0, guess_depth);
    return BVHTree<N, Scalar>{ points, triangles, std::move(nodes), std::move(triangle_indices) };
}

} // namespace bvh

namespace mbvh {

// Interior child slots and batched triangle lanes intentionally share one width.
inline constexpr std::size_t kNodeWidth = 4;
inline constexpr std::size_t kInvalidNodeIndex = std::numeric_limits<std::size_t>::max();

template<std::size_t N, std::size_t M, typename Scalar>
using Mat = Eigen::Matrix<Scalar, N, M, Eigen::RowMajor>;

template<typename Scalar>
using WideVec = Eigen::Vector<Scalar, static_cast<int>(kNodeWidth)>;

template<typename Scalar>
using WideMaskValues = Eigen::Array<Scalar, static_cast<int>(kNodeWidth), 1>;

// Typed comparisons produce exact zero/one values, so public Eigen bitwise operators preserve mask semantics.
template<typename Scalar>
class WideMask
{
    static_assert(std::is_floating_point_v<Scalar>);

  public:
    EIGEN_ALWAYS_INLINE WideMask()
      : values_(WideMaskValues<Scalar>::Zero())
    {
    }

    [[nodiscard]] EIGEN_ALWAYS_INLINE static WideMask less_equal(const WideVec<Scalar>& lhs, const WideVec<Scalar>& rhs)
    {
        return WideMask{ lhs.cwiseTypedLessOrEqual(rhs).array().eval() };
    }

    [[nodiscard]] EIGEN_ALWAYS_INLINE static WideMask greater_equal(const WideVec<Scalar>& lhs,
                                                                    const WideVec<Scalar>& rhs)
    {
        return WideMask{ lhs.cwiseTypedGreaterOrEqual(rhs).array().eval() };
    }

    [[nodiscard]] EIGEN_ALWAYS_INLINE WideVec<Scalar> select(const WideVec<Scalar>& if_true,
                                                             const WideVec<Scalar>& if_false) const
    {
        return values_.select(if_true.array(), if_false.array()).matrix();
    }

    [[nodiscard]] EIGEN_ALWAYS_INLINE bool any() const { return values_.maxCoeff() != Scalar{ 0 }; }

    [[nodiscard]] EIGEN_ALWAYS_INLINE bool all() const { return values_.minCoeff() != Scalar{ 0 }; }

    [[nodiscard]] EIGEN_ALWAYS_INLINE friend WideMask operator&(const WideMask& lhs, const WideMask& rhs)
    {
        return WideMask{ (lhs.values_ & rhs.values_).eval() };
    }

    [[nodiscard]] EIGEN_ALWAYS_INLINE friend WideMask operator|(const WideMask& lhs, const WideMask& rhs)
    {
        return WideMask{ (lhs.values_ | rhs.values_).eval() };
    }

    [[nodiscard]] EIGEN_ALWAYS_INLINE friend WideMask operator~(const WideMask& mask)
    {
        return WideMask{ (mask.values_ ^ WideMaskValues<Scalar>::Ones()).eval() };
    }

  private:
    explicit EIGEN_ALWAYS_INLINE WideMask(WideMaskValues<Scalar> values)
      : values_(std::move(values))
    {
    }

    WideMaskValues<Scalar> values_;
};

template<std::size_t N, typename Scalar>
struct LeafNode
{
    static_assert(std::is_floating_point_v<Scalar>);

    using MatX = Eigen::Matrix<Scalar, kNodeWidth, N>;
    using WeightMat = Eigen::Matrix<Scalar, kNodeWidth, 2>;

    // Rows are independent triangle lanes. Cache edges, but form AX/BX/CX at query time to avoid cancellation from
    // expanded dot products.
    MatX TA;
    MatX TB;
    MatX TC;
    MatX AB;
    MatX AC;
    std::size_t start{};
    std::size_t end{};

    EIGEN_ALWAYS_INLINE std::pair<MatX, WeightMat> closest_point_on_triangle(const Scalar* pt_data) const;
};

struct Leaf
{
    std::size_t start{};
    std::size_t end{};
};

template<std::size_t N, typename Scalar>
struct Interior
{
    static_assert(std::is_floating_point_v<Scalar>);

    std::array<std::size_t, kNodeWidth> children;
    std::size_t n_children{};
    std::array<WideVec<Scalar>, N> box_min;
    std::array<WideVec<Scalar>, N> box_max;

    Interior()
    {
        children.fill(kInvalidNodeIndex);
        for (auto& bound : box_min) {
            bound.setConstant(std::numeric_limits<Scalar>::max());
        }
        for (auto& bound : box_max) {
            bound.setConstant(std::numeric_limits<Scalar>::lowest());
        }
    }
};

template<std::size_t N, typename Scalar>
using Node = std::variant<Leaf, Interior<N, Scalar>>;

template<std::size_t N, typename Scalar>
struct MBVHTree
{
    static_assert(std::is_floating_point_v<Scalar>);

    std::span<const Scalar> points;
    std::span<const std::size_t> triangles;
    std::vector<Node<N, Scalar>> nodes;
    std::vector<LeafNode<N, Scalar>> node_leaves;
    std::vector<std::size_t> triangle_indices;
    std::size_t max_depth{};

    MBVHTree() = default;

    explicit MBVHTree(const bvh::BVHTree<N, Scalar>& binary_bvh) { initialize(binary_bvh); }

    [[nodiscard]] bool find_closest_point_from_node(BoundingSphere<N, Scalar>& sphere,
                                                    Interaction<N, Scalar>& interaction,
                                                    int node_start_index) const;

    void initialize(const bvh::BVHTree<N, Scalar>& binary_bvh)
    {
        points = binary_bvh.points;
        triangles = binary_bvh.triangles;
        triangle_indices = binary_bvh.triangle_indices;
        nodes.clear();
        node_leaves.clear();
        max_depth = 0;

        if (binary_bvh.nodes.empty()) {
            return;
        }

        nodes.reserve(binary_bvh.nodes.size());
        node_leaves.reserve(binary_bvh.nodes.size());
        static_cast<void>(collapse_bvh(binary_bvh, 0, 0));
    }

  private:
    [[nodiscard]] Leaf populate_leaf_node(const bvh::Leaf& bvh_leaf)
    {
        assert(bvh_leaf.start < bvh_leaf.end);
        assert(bvh_leaf.end <= triangle_indices.size());
        assert(points.size() % N == 0);
        assert(triangles.size() % 3 == 0);

        const std::size_t n_points = points.size() / N;
        const std::size_t leaf_start = node_leaves.size();
        for (std::size_t i = bvh_leaf.start; i < bvh_leaf.end;) {
            auto& leaf_node = node_leaves.emplace_back();
            leaf_node.start = i;
            const std::size_t n_references = std::min(kNodeWidth, bvh_leaf.end - i);
            leaf_node.end = i + n_references;

            leaf_node.TA.setZero();
            leaf_node.TB.setZero();
            leaf_node.TC.setZero();
            for (std::size_t p = 0; p < n_references; ++p) {
                const std::size_t idx = triangle_indices[i + p];
                assert(idx < triangles.size() / 3);

                const auto* tri = &triangles[idx * 3];
                assert(tri[0] < n_points);
                assert(tri[1] < n_points);
                assert(tri[2] < n_points);
                leaf_node.TA.row(p) = bvh::Vec<N, Scalar>::Map(&points[tri[0] * N]).transpose();
                leaf_node.TB.row(p) = bvh::Vec<N, Scalar>::Map(&points[tri[1] * N]).transpose();
                leaf_node.TC.row(p) = bvh::Vec<N, Scalar>::Map(&points[tri[2] * N]).transpose();
            }
            leaf_node.AB = leaf_node.TB - leaf_node.TA;
            leaf_node.AC = leaf_node.TC - leaf_node.TA;
            i = leaf_node.end;
        }
        return Leaf{ leaf_start, node_leaves.size() };
    }

    [[nodiscard]] std::size_t collapse_bvh(const bvh::BVHTree<N, Scalar>& binary_bvh,
                                           const std::size_t bvh_node_idx,
                                           const std::size_t depth)
    {
        assert(bvh_node_idx < binary_bvh.nodes.size());
        max_depth = std::max(max_depth, depth);

        const auto& bvh_node = binary_bvh.nodes[bvh_node_idx];
        const std::size_t mbvh_node_idx = nodes.size();
        // Collapse runs only during construction, keeping std::visit out of the traversal hot path.
        return std::visit(
          [&](const auto& node_base) -> std::size_t {
              using NodeBaseT = std::remove_cvref_t<decltype(node_base)>;
              if constexpr (std::is_same_v<NodeBaseT, bvh::Leaf>) {
                  nodes.emplace_back(populate_leaf_node(node_base));
              } else {
                  static_assert(std::is_same_v<NodeBaseT, bvh::Interior>);
                  nodes.emplace_back(Interior<N, Scalar>{});

                  std::array<std::size_t, kNodeWidth> nodes_to_collapse;
                  nodes_to_collapse.fill(kInvalidNodeIndex);
                  nodes_to_collapse[0] = node_base.children[0];
                  nodes_to_collapse[1] = node_base.children[1];
                  std::size_t n_nodes_to_collapse = 2;

                  // Greedily open the largest interior frontier node until the wide node is full.
                  while (n_nodes_to_collapse < kNodeWidth) {
                      Scalar max_surface_area = std::numeric_limits<Scalar>::lowest();
                      std::size_t max_idx = n_nodes_to_collapse;

                      for (std::size_t i = 0; i < n_nodes_to_collapse; ++i) {
                          assert(nodes_to_collapse[i] < binary_bvh.nodes.size());
                          const auto& candidate = binary_bvh.nodes[nodes_to_collapse[i]];
                          std::visit(
                            [&](const auto& candidate_base) {
                                using CandidateBaseT = std::remove_cvref_t<decltype(candidate_base)>;
                                if constexpr (std::is_same_v<CandidateBaseT, bvh::Interior>) {
                                    const Scalar surface_area = bvh::bbox_surface_area(candidate.bbox);
                                    if (max_surface_area < surface_area) {
                                        max_surface_area = surface_area;
                                        max_idx = i;
                                    }
                                }
                            },
                            candidate.base);
                      }

                      if (max_idx == n_nodes_to_collapse) {
                          break;
                      }

                      const auto& candidate = binary_bvh.nodes[nodes_to_collapse[max_idx]];
                      std::visit(
                        [&](const auto& candidate_base) {
                            using CandidateBaseT = std::remove_cvref_t<decltype(candidate_base)>;
                            if constexpr (std::is_same_v<CandidateBaseT, bvh::Interior>) {
                                nodes_to_collapse[max_idx] = candidate_base.children[0];
                                nodes_to_collapse[n_nodes_to_collapse] = candidate_base.children[1];
                            } else {
                                assert(false);
                            }
                        },
                        candidate.base);
                      ++n_nodes_to_collapse;
                  }

                  // Binary nodes use preorder indices, so sorting restores their stable construction order.
                  std::sort(nodes_to_collapse.begin(), nodes_to_collapse.begin() + n_nodes_to_collapse);
                  std::visit(
                    [&](auto& mbvh_node_base) {
                        using MbvhNodeBaseT = std::remove_cvref_t<decltype(mbvh_node_base)>;
                        if constexpr (std::is_same_v<MbvhNodeBaseT, Interior<N, Scalar>>) {
                            mbvh_node_base.n_children = n_nodes_to_collapse;
                        } else {
                            assert(false);
                        }
                    },
                    nodes[mbvh_node_idx]);

                  for (std::size_t i = 0; i < n_nodes_to_collapse; ++i) {
                      const std::size_t child_bvh_node_idx = nodes_to_collapse[i];
                      assert(child_bvh_node_idx < binary_bvh.nodes.size());
                      const auto& child_bvh_node = binary_bvh.nodes[child_bvh_node_idx];
                      const std::size_t child_mbvh_node_idx = collapse_bvh(binary_bvh, child_bvh_node_idx, depth + 1);
                      // Recursion may reallocate nodes, so reacquire the parent by index.
                      std::visit(
                        [&](auto& mbvh_node_base) {
                            using MbvhNodeBaseT = std::remove_cvref_t<decltype(mbvh_node_base)>;
                            if constexpr (std::is_same_v<MbvhNodeBaseT, Interior<N, Scalar>>) {
                                for (std::size_t dim = 0; dim < N; ++dim) {
                                    mbvh_node_base.box_min[dim][i] = child_bvh_node.bbox.min_coord(dim);
                                    mbvh_node_base.box_max[dim][i] = child_bvh_node.bbox.max_coord(dim);
                                }
                                mbvh_node_base.children[i] = child_mbvh_node_idx;
                            } else {
                                assert(false);
                            }
                        },
                        nodes[mbvh_node_idx]);
                  }
              }

              return mbvh_node_idx;
          },
          bvh_node.base);
    }
};

template<typename Scalar, typename... Conditions>
EIGEN_ALWAYS_INLINE WideMask<Scalar>
where_all(WideMask<Scalar> first, const Conditions&... conditions)
{
    ((first = first & conditions), ...);
    return first;
}

template<std::size_t N, typename Scalar>
EIGEN_ALWAYS_INLINE std::pair<Eigen::Matrix<Scalar, kNodeWidth, N>, Eigen::Matrix<Scalar, kNodeWidth, 2>>
LeafNode<N, Scalar>::closest_point_on_triangle(const Scalar* pt_data) const
{
    using WideMaskT = WideMask<Scalar>;
    using WideVecT = WideVec<Scalar>;

    const auto x = Eigen::Matrix<Scalar, 1, N>::Map(pt_data).eval();
    const MatX AX = x.template replicate<kNodeWidth, 1>() - TA;
    auto dot = [](const MatX& lhs, const MatX& rhs) -> WideVecT {
        return (lhs.array() * rhs.array()).rowwise().sum().matrix();
    };
    const WideVecT d1 = dot(AB, AX);
    const WideVecT d2 = dot(AC, AX);

    const WideVecT zero = WideVecT::Zero();
    const WideVecT one = WideVecT::Ones();
    WideVecT ta_vec = zero;
    WideVecT tb_vec = zero;
    WideVecT tc_vec = zero;
    WideMaskT activated;
    // Resolve Voronoi regions in precedence order; activated lanes cannot be overwritten by later regions.
    auto resolve = [&activated, &ta_vec, &tb_vec, &tc_vec](const WideMaskT& condition,
                                                           const WideVecT& ta_candidate,
                                                           const WideVecT& tb_candidate,
                                                           const WideVecT& tc_candidate) {
        ta_vec = condition.select(ta_candidate, ta_vec);
        tb_vec = condition.select(tb_candidate, tb_vec);
        tc_vec = condition.select(tc_candidate, tc_vec);
        activated = activated | condition;
    };
    auto make_result = [&]() {
        const MatX tb = tb_vec.template replicate<1, N>();
        const MatX tc = tc_vec.template replicate<1, N>();
        MatX P = (TA.array() + AB.array() * tb.array() + AC.array() * tc.array()).matrix();
        WeightMat W;
        // Store A and B barycentric weights; the C weight is implicit as one minus their sum.
        W.col(0) = ta_vec;
        W.col(1) = tb_vec;
        return std::make_pair(std::move(P), std::move(W));
    };

    resolve(where_all(WideMaskT::less_equal(d1, zero), WideMaskT::less_equal(d2, zero)), one, zero, zero);
    if (activated.all()) {
        return make_result();
    }

    const MatX BX = x.template replicate<kNodeWidth, 1>() - TB;
    const WideVecT d3 = dot(AB, BX);
    const WideVecT d4 = dot(AC, BX);
    resolve(where_all(~activated, WideMaskT::greater_equal(d3, zero), WideMaskT::less_equal(d4, d3)), zero, one, zero);
    if (activated.all()) {
        return make_result();
    }

    const MatX CX = x.template replicate<kNodeWidth, 1>() - TC;
    const WideVecT d5 = dot(AB, CX);
    const WideVecT d6 = dot(AC, CX);

    resolve(where_all(~activated, WideMaskT::greater_equal(d6, zero), WideMaskT::less_equal(d5, d6)), zero, zero, one);
    if (activated.all()) {
        return make_result();
    }

    auto vc = (d1.cwiseProduct(d4).eval() - d3.cwiseProduct(d2).eval()).eval();
    const WideMaskT mask_ab = where_all(
      ~activated, WideMaskT::less_equal(vc, zero), WideMaskT::greater_equal(d1, zero), WideMaskT::less_equal(d3, zero));
    if (mask_ab.any()) {
        const WideVecT candidate_tb = (d1.array() / (d1 - d3).eval().array()).matrix();
        const WideVecT candidate_ta = one - candidate_tb;
        resolve(mask_ab, candidate_ta, candidate_tb, zero);
        if (activated.all()) {
            return make_result();
        }
    }

    auto vb = (d5.cwiseProduct(d2).eval() - d1.cwiseProduct(d6).eval()).eval();
    const WideMaskT mask_ac = where_all(
      ~activated, WideMaskT::less_equal(vb, zero), WideMaskT::greater_equal(d2, zero), WideMaskT::less_equal(d6, zero));
    if (mask_ac.any()) {
        const WideVecT candidate_tc = (d2.array() / (d2 - d6).eval().array()).matrix();
        const WideVecT candidate_ta = one - candidate_tc;
        resolve(mask_ac, candidate_ta, zero, candidate_tc);
        if (activated.all()) {
            return make_result();
        }
    }

    auto va = (d3.cwiseProduct(d6).eval() - d5.cwiseProduct(d4).eval()).eval();
    auto d34 = (d4 - d3).eval();
    auto d56 = (d6 - d5).eval();
    const WideMaskT mask_bc = where_all(~activated,
                                        WideMaskT::less_equal(va, zero),
                                        WideMaskT::greater_equal(d34, zero),
                                        WideMaskT::less_equal(d56, zero));
    if (mask_bc.any()) {
        const WideVecT candidate_tc = (d34.array() / (d34 - d56).eval().array()).matrix();
        const WideVecT candidate_tb = one - candidate_tc;
        resolve(mask_bc, zero, candidate_tb, candidate_tc);
        if (activated.all()) {
            return make_result();
        }
    }

    auto s = (va + vb + vc).eval();
    const WideVecT candidate_ta = (va.array() / s.array()).matrix();
    const WideVecT candidate_tb = (vb.array() / s.array()).matrix();
    const WideVecT candidate_tc = one - candidate_ta - candidate_tb;
    resolve(~activated, candidate_ta, candidate_tb, candidate_tc);
    return make_result();
}

template<std::size_t N, typename Scalar>
bool
MBVHTree<N, Scalar>::find_closest_point_from_node(BoundingSphere<N, Scalar>& sphere,
                                                  Interaction<N, Scalar>& interaction,
                                                  const int node_start_index) const
{
    if (nodes.empty()) {
        return false;
    }

    assert(node_start_index >= 0);
    assert(static_cast<std::size_t>(node_start_index) < nodes.size());
    assert(sphere.r2 >= Scalar{ 0 });
    assert(points.size() % N == 0);
    assert(triangles.size() % 3 == 0);

    struct TraversalEntry
    {
        std::size_t node;
        Scalar distance;
    };

    // Each depth-first pop can add at most kNodeWidth - 1 live entries at one level.
    constexpr std::size_t kMaxTraversalStackSize = 1 + (kNodeWidth - 1) * bvh::kMaxBvhDepth;
    std::array<TraversalEntry, kMaxTraversalStackSize> stack;
    stack[0] = TraversalEntry{ static_cast<std::size_t>(node_start_index), std::numeric_limits<Scalar>::lowest() };
    std::size_t stack_size = 1;
    bool found = false;

    const bvh::Vec<N, Scalar> query = bvh::Vec<N, Scalar>::Map(sphere.c.data());
    typename LeafNode<N, Scalar>::MatX Q = query.transpose().template replicate<kNodeWidth, 1>();
    while (stack_size != 0) {
        const TraversalEntry entry = stack[--stack_size];
        if (entry.distance > sphere.r2) {
            continue;
        }

        assert(entry.node < nodes.size());
        const auto& node = nodes[entry.node];
        // Direct variant probes avoid the indirect dispatch observed with libc++ std::visit in this hot loop.
        if (const auto* leaf = std::get_if<Leaf>(&node)) {
            assert(leaf->start < leaf->end);
            assert(leaf->end <= node_leaves.size());
            for (std::size_t leaf_node_idx = leaf->start; leaf_node_idx < leaf->end; ++leaf_node_idx) {
                const auto& leaf_node = node_leaves[leaf_node_idx];
                assert(leaf_node.start <= leaf_node.end);
                assert(leaf_node.end <= triangle_indices.size());
                const std::size_t n_references = leaf_node.end - leaf_node.start;
                assert(n_references != 0);
                assert(n_references <= kNodeWidth);

                const auto [P, W] = leaf_node.closest_point_on_triangle(query.data());
                const auto SD = (P - Q).rowwise().squaredNorm().eval();
                Eigen::Index mxri = 0;
                // Ignore zero-padded lanes that have no triangle reference.
                for (Eigen::Index lane = 1; lane < static_cast<Eigen::Index>(n_references); ++lane) {
                    if (SD[lane] <= SD[mxri]) {
                        mxri = lane;
                    }
                }
                if (const Scalar squared_distance = SD(mxri); squared_distance <= sphere.r2) {
                    const std::size_t reference_idx = leaf_node.start + static_cast<std::size_t>(mxri);
                    const std::size_t triangle_idx = triangle_indices[reference_idx];
                    assert(triangle_idx < triangles.size() / 3);
                    sphere.r2 = squared_distance;
                    interaction.d = std::sqrt(squared_distance);
                    interaction.primitive_index = triangle_idx;
                    interaction.node_index = entry.node;
                    bvh::Vec<N, Scalar>::Map(interaction.p.data()) = P.row(mxri).transpose();
                    bvh::Vec<2, Scalar>::Map(interaction.uv.data()) = W.row(mxri).transpose();
                    found = true;
                }
            }
            continue;
        }

        const auto& interior = std::get<Interior<N, Scalar>>(node);
        assert(interior.n_children <= kNodeWidth);

        // Bound the squared query distance to every point in each child box.
        WideVec<Scalar> d2_min = WideVec<Scalar>::Zero();
        WideVec<Scalar> d2_max = WideVec<Scalar>::Zero();
        for (std::size_t dim = 0; dim < N; ++dim) {
            const WideVec<Scalar> center = WideVec<Scalar>::Constant(sphere.c[dim]);
            const WideVec<Scalar> u = interior.box_min[dim] - center;
            const WideVec<Scalar> v = center - interior.box_max[dim];
            d2_min.array() += u.cwiseMax(v).cwiseMax(Scalar{ 0 }).array().square();
            d2_max.array() += u.cwiseMin(v).array().square();
        }

        std::array<std::size_t, kNodeWidth> order{};
        for (std::size_t lane = 0; lane < kNodeWidth; ++lane) {
            order[lane] = lane;
        }
        // Push children far-to-near so the LIFO stack visits the nearest child first.
        auto swap_if_closer = [&d2_min, &order](const std::size_t lhs, const std::size_t rhs) {
            if (d2_min[order[lhs]] < d2_min[order[rhs]]) {
                std::swap(order[lhs], order[rhs]);
            }
        };
        swap_if_closer(0, 1);
        swap_if_closer(2, 3);
        swap_if_closer(0, 2);
        swap_if_closer(1, 3);
        swap_if_closer(1, 2);

        const Scalar traversal_radius = sphere.r2;
        for (const std::size_t lane : order) {
            if (lane >= interior.n_children || d2_min[lane] > traversal_radius) {
                continue;
            }

            assert(interior.children[lane] != kInvalidNodeIndex);
            assert(interior.children[lane] < nodes.size());
            // A nonempty child has a primitive vertex in its box, making d2_max a valid global upper bound.
            sphere.r2 = std::min(sphere.r2, d2_max[lane]);
            assert(stack_size < stack.size());
            stack[stack_size++] = TraversalEntry{ interior.children[lane], d2_min[lane] };
        }
    }

    return found;
}

} // namespace mbvh
} // namespace gpf
