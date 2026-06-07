#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <expected>
#include <functional>
#include <limits>
#include <numbers>
#include <numeric>
#include <queue>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <CGAL/AABB_traits_2.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_triangle_primitive_2.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <predicates/predicates.hpp>

#include <gpf/ids.hpp>
#include <gpf/mesh.hpp>
#include <gpf/mesh_property.hpp>
#include <gpf/triangulation.hpp>

namespace gpf {
enum class WalkOnMeshSurfaceFailure
{
    BoundaryReached,
    IterationLimitExceeded,
    DegenerateDirection,
    InvalidPath,
    DegenerateStep
};

// Result pair: face id, then barycentric coords ordered by mesh.face(fid).halfedges() from vertices.
using WalkOnMeshSurfaceResult =
  std::expected<std::vector<std::pair<gpf::FaceId, std::array<double, 3>>>, WalkOnMeshSurfaceFailure>;

namespace detail {
namespace views = std::views;
namespace ranges = std::ranges;
using Vector2d = Eigen::Vector2d;
using Vector3d = Eigen::Vector3d;
inline constexpr double BARY_EPS = 1e-3;

template<std::size_t N>
using VectorNd = Eigen::Vector<double, static_cast<int>(N)>;

// Complex division: treat 2D vectors as complex numbers (x + yi)
inline Vector2d
complex_div(const Vector2d& a, const Vector2d& b)
{
    double denom = b.squaredNorm();
    return Vector2d{ (a.x() * b.x() + a.y() * b.y()) / denom, (a.y() * b.x() - a.x() * b.y()) / denom };
}
inline std::vector<double>
compute_bary_coordinates(const std::span<const double> points)
{
    auto pa = Vector2d::Map(points.data());
    auto pb = Vector2d::Map(points.data() + 2);
    auto pc = Vector2d::Map(points.data() + 4);
    const auto v0 = (pb - pa).eval();
    const auto v1 = (pc - pa).eval();
    const auto d00 = v0.squaredNorm();
    const auto d01 = v0.dot(v1);
    const auto d11 = v1.squaredNorm();
    const auto denom = d00 * d11 - d01 * d01;
    std::vector<double> bary_coords((points.size() / 2 - 3) * 3);
    std::size_t idx = 0;
    for (std::size_t i = 6; i < points.size(); i += 2) {
        const auto v2 = Vector2d::Map(points.data() + i) - pa;
        const auto d20 = v2.dot(v0);
        const auto d21 = v2.dot(v1);
        const auto b1 = (d11 * d20 - d01 * d21) / denom;
        const auto b2 = (d00 * d21 - d01 * d20) / denom;
        const auto b0 = 1.0 - b1 - b2;
        bary_coords[idx++] = b0;
        bary_coords[idx++] = b1;
        bary_coords[idx++] = b2;
    }
    return bary_coords;
}

template<std::size_t N>
struct FaceCoords;

template<>
struct FaceCoords<2>
{
    FaceCoords() noexcept = default;

    std::array<double, 2> uv(const std::span<const double, 2> pt) const noexcept { return { pt[0], pt[1] }; }
};

template<>
struct FaceCoords<3>
{
    std::array<double, 9> data;
    FaceCoords() noexcept = default;
    FaceCoords(std::array<double, 9>&& data) noexcept
      : data(std::move(data))
    {
    }
    FaceCoords(const auto& mesh, const gpf::FaceId fid) noexcept
    {
        auto face = mesh.face(fid);
        auto face_vertices =
          face.halfedges() | views::transform([](auto he) { return he.to().id; }) | ranges::to<std::vector>();
        const auto pa = Eigen::Vector3d::Map(mesh.vertex_prop(face_vertices[0]).pt.data());
        const auto pb = Eigen::Vector3d::Map(mesh.vertex_prop(face_vertices[1]).pt.data());
        const auto pc = Eigen::Vector3d::Map(mesh.vertex_prop(face_vertices[2]).pt.data());

        auto o = Vector3d::Map(data.data());
        auto x = Vector3d::Map(data.data() + 3);
        auto y = Vector3d::Map(data.data() + 6);
        o = pa;
        x = (pb - o).normalized();
        auto z = x.cross((pc - o)).normalized();
        y = z.cross(x).normalized();
    }

    std::array<double, 2> uv(const std::span<const double, 3> pt) const noexcept
    {
        auto o = Vector3d::Map(data.data());
        auto x = Vector3d::Map(data.data() + 3);
        auto y = Vector3d::Map(data.data() + 6);
        auto v = (Eigen::Vector3d::Map(pt.data()) - o).eval();
        return { v.dot(x), v.dot(y) };
    }
};

template<std::size_t N, typename Mesh>
FaceCoords<N>
make_face_coords(Mesh& mesh, const gpf::FaceId fid) noexcept
{
    if constexpr (N == 3) {
        return FaceCoords<N>(mesh, fid);
    } else {
        return FaceCoords<N>{};
    }
}

template<std::size_t N>
struct FaceInfo
{
    FaceCoords<N> ccs;
    std::vector<std::size_t> point_indices;
};

inline bool
normalize_barycentric(std::span<double, 3> bary, const double eps) noexcept
{
    bool changed = false;
    for (double& val : bary) {
        if (val < eps) [[unlikely]] {
            val = 0.0;
            changed = true;
        }
    }
    if (!changed) {
        return false;
    }
    const double sum = ranges::fold_left(bary, 0.0, std::plus{});
    for (double& value : bary) {
        value /= sum;
    }
    return true;
}

template<std::size_t N, typename Mesh>
auto
identify_points(Mesh& mesh,
                const gpf::FaceId fid,
                std::vector<std::array<double, N>>& all_points,
                std::unordered_map<gpf::EdgeId, std::vector<std::size_t>>& edge_to_points_map,
                std::vector<std::size_t>& face_point_indices,
                std::vector<gpf::VertexId>& point_vertices,
                const double sq_eps)
{
    auto face = mesh.face(fid);
    auto ccs = make_face_coords<N>(mesh, fid);

    std::size_t idx = 0;
    for (std::size_t i = 0; i < face_point_indices.size(); i++) {
        const auto pid = face_point_indices[i];
        auto pt = VectorNd<N>::Map(all_points[pid].data());
        bool finished = false;
        for (const auto he : face.halfedges()) {
            auto v = he.to();
            if ((pt - VectorNd<N>::Map(v.prop().pt.data())).squaredNorm() < sq_eps) {
                point_vertices[pid] = v.id;
                all_points[pid] = v.prop().pt;
                finished = true;
                break;
            }
        }
        if (finished) {
            continue;
        }

        for (const auto he : face.halfedges()) {
            auto pa = VectorNd<N>::Map(he.from().prop().pt.data());
            auto pb = VectorNd<N>::Map(he.to().prop().pt.data());
            auto vab = (pb - pa).eval();
            auto sq_len = vab.squaredNorm();
            auto t = std::clamp(vab.dot(pt - pa) / sq_len, 0.0, 1.0);
            auto mid_pt = ((1.0 - t) * pa + t * pb).eval();
            if ((mid_pt - pt).squaredNorm() < sq_eps) {
                pt = mid_pt;
                edge_to_points_map[he.edge().id].emplace_back(pid);
                finished = true;
                break;
            }
        }
        if (!finished) {
            if (idx != i) {
                face_point_indices[idx] = pid;
            }
            idx += 1;
        }
    }
    face_point_indices.resize(idx);
    return ccs;
}

template<std::size_t N, typename Mesh, typename PointAccessor>
void
split_edge_by_points(Mesh& mesh,
                     const gpf::EdgeId eid,
                     PointAccessor&& get_point,
                     const std::vector<std::size_t>& edge_point_indices,
                     std::vector<gpf::VertexId>& point_vertices,
                     const double eps,
                     std::unordered_map<gpf::EdgeId, gpf::EdgeId>* edge_parent_map = nullptr)
{
    // Find the root parent for this edge
    auto get_root_parent = [&edge_parent_map](gpf::EdgeId e) {
        gpf::EdgeId root = e;
        if (edge_parent_map) {
            auto it = edge_parent_map->find(e);
            if (it != edge_parent_map->end()) {
                root = it->second;
            }
        }
        return root;
    };

    if (edge_point_indices.size() == 1) {
        const auto root_parent = get_root_parent(eid);
        const auto new_vid = mesh.split_edge(eid);
        const auto pid = edge_point_indices[0];
        mesh.vertex_prop(new_vid).pt = get_point(pid);
        point_vertices[pid] = new_vid;
        if (edge_parent_map) {
            // eid is reused for one sub-edge, new edge is the other
            (*edge_parent_map)[eid] = root_parent;
            auto new_eid = mesh.vertex(new_vid).halfedge().edge().id;
            (*edge_parent_map)[new_eid] = root_parent;
        }
    } else {
        auto curr_hid = mesh.edge(eid).halfedge().id;
        const auto [va, vb] = mesh.he_vertices(curr_hid);
        const auto pa = mesh.vertex_prop(va).pt;
        const auto pb = mesh.vertex_prop(vb).pt;
        const auto pa_ref = VectorNd<N>::Map(pa.data());
        const auto pb_ref = VectorNd<N>::Map(pb.data());
        auto vab = (pb_ref - pa_ref).eval();
        auto sq_edge_len = vab.squaredNorm();
        std::vector<std::size_t> indices(edge_point_indices.size());
        std::iota(indices.begin(), indices.end(), 0);
        const auto distances = edge_point_indices |
                               std::views::transform([pa_ref, sq_edge_len, &vab, &get_point](const auto pid) {
                                   auto pt = get_point(pid);
                                   return std::max((VectorNd<N>::Map(pt.data()) - pa_ref).dot(vab), 0.0) / sq_edge_len;
                               }) |
                               std::ranges::to<std::vector>();

        std::sort(indices.begin(), indices.end(), [&distances](auto i, auto j) { return distances[i] < distances[j]; });
        const auto root_parent = get_root_parent(eid);
        std::size_t j = 0;
        gpf::EdgeId curr_eid = eid;
        for (std::size_t i = 0; i < indices.size(); i++) {
            const auto pid = edge_point_indices[indices[i]];
            if (i == 0 || (distances[indices[i]] - distances[indices[j]]) > eps) {
                const auto new_vid = mesh.split_edge(curr_eid);
                auto new_v = mesh.vertex(new_vid);
                new_v.prop().pt = get_point(pid);
                point_vertices[pid] = new_vid;
                if (edge_parent_map) {
                    auto new_eid = new_v.halfedge().edge().id;
                    (*edge_parent_map)[new_eid] = root_parent;
                }
                curr_eid = new_v.halfedge().edge().id;
                j = i;
            } else {
                point_vertices[pid] = point_vertices[edge_point_indices[indices[j]]];
            }
        }
        if (edge_parent_map) {
            (*edge_parent_map)[eid] = root_parent;
        }
    }
}

template<std::size_t N, typename Mesh>
void
triangulate_on_face(Mesh& mesh,
                    const gpf::FaceId fid,
                    const std::span<const std::array<double, N>> all_points,
                    const FaceCoords<N>& ccs,
                    const std::vector<std::size_t>& point_indices,
                    const std::vector<gpf::VertexId>& cross_edge_vertices,
                    std::vector<gpf::VertexId>& point_vertices,
                    std::unordered_map<gpf::FaceId, gpf::FaceId>* face_parent_map = nullptr)
{
    auto face_vertices = mesh.face(fid).halfedges() | views::transform([](const auto& he) { return he.to().id; }) |
                         ranges::to<std::vector>();

    std::unordered_map<gpf::VertexId, std::size_t> vertex_indices;
    for (auto vid : cross_edge_vertices) {
        vertex_indices.emplace(vid, gpf::kInvalidIndex);
    }
    for (std::size_t i = 0; i < face_vertices.size(); i++) {
        const auto vid = face_vertices[i];
        auto it = vertex_indices.find(vid);
        if (it != vertex_indices.end()) {
            it->second = i;
        }
    }
    const auto n_old_face_vertices = face_vertices.size();
    const auto n_old_vertices = mesh.n_vertices_capacity();
    mesh.new_vertices(point_indices.size());
    face_vertices.append_range(ranges::iota_view{ n_old_vertices, mesh.n_vertices_capacity() } |
                               views::transform([](const auto idx) { return gpf::VertexId{ idx }; }));
    if (face_vertices.size() == 3) {
        return;
    }
    for (auto [pid, vid] : views::zip(
           point_indices, ranges::drop_view{ face_vertices, static_cast<std::ptrdiff_t>(n_old_face_vertices) })) {
        mesh.vertex_prop(vid).pt = all_points[pid];
        point_vertices[pid] = vid;
    }

    std::vector<double> points(face_vertices.size() * 2);
    std::size_t idx = 0;
    for (auto vid : face_vertices) {
        const auto& pt = mesh.vertex_prop(vid).pt;
        Vector2d::Map(points.data() + idx) = Vector2d::Map(ccs.uv(pt).data());
        idx += 2;
    }

    std::vector<std::size_t> segments;
    segments.reserve(n_old_face_vertices * 2 + cross_edge_vertices.size());
    for (std::size_t i = 0; i < n_old_face_vertices; i++) {
        segments.push_back(i);
        segments.push_back((i + 1) % n_old_face_vertices);
    }

    segments.append_range(cross_edge_vertices |
                          views::transform([&vertex_indices](const auto vid) { return vertex_indices[vid]; }));

    const auto triangle_indices = gpf::triangulate_polygon(points, segments, n_old_face_vertices, true);
    auto triangles = triangle_indices |
                     views::transform([&face_vertices](const auto idx) { return face_vertices[idx]; }) |
                     ranges::to<std::vector>();
    assert(!triangles.empty());

    const auto n_faces_before = mesh.n_faces_capacity();
    mesh.split_face_into_triangles(fid, triangles);

    if (face_parent_map) {
        // Find the root parent (in case fid was already a sub-triangle)
        gpf::FaceId root_parent = fid;
        auto it = face_parent_map->find(fid);
        if (it != face_parent_map->end()) {
            root_parent = it->second;
        }

        // The original face fid is reused for one of the sub-triangles
        (*face_parent_map)[fid] = root_parent;
        // New faces are created starting from n_faces_before
        for (std::size_t i = n_faces_before; i < mesh.n_faces_capacity(); ++i) {
            (*face_parent_map)[gpf::FaceId{ i }] = root_parent;
        }
    }
}

template<std::size_t N, typename VP, typename HP, typename EP, typename FP>
auto
project_points_on_mesh(std::vector<std::array<double, N>>& points,
                       gpf::ManifoldMesh<VP, HP, EP, FP>& mesh,
                       const double eps,
                       std::unordered_map<gpf::FaceId, gpf::FaceId>* face_parent_map = nullptr,
                       std::unordered_map<gpf::EdgeId, gpf::EdgeId>* edge_parent_map = nullptr)
{
    static_assert(N == 2 || N == 3);

    using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Point = std::conditional_t<N == 2, Kernel::Point_2, Kernel::Point_3>;
    using Triangle = std::conditional_t<N == 2, Kernel::Triangle_2, Kernel::Triangle_3>;
    using TreeIterator = std::vector<Triangle>::const_iterator;
    using TreePrimitive = std::conditional_t<N == 2,
                                             CGAL::AABB_triangle_primitive_2<Kernel, TreeIterator>,
                                             CGAL::AABB_triangle_primitive_3<Kernel, TreeIterator>>;
    using TreeTraits = std::
      conditional_t<N == 2, CGAL::AABB_traits_2<Kernel, TreePrimitive>, CGAL::AABB_traits_3<Kernel, TreePrimitive>>;
    using Tree = CGAL::AABB_tree<TreeTraits>;

    auto to_kernel_point = [](const std::array<double, N>& pt) {
        if constexpr (N == 2) {
            return Point(pt[0], pt[1]);
        } else {
            return Point(pt[0], pt[1], pt[2]);
        }
    };

    std::vector<Triangle> triangles;
    std::vector<gpf::FaceId> face_ids;
    triangles.reserve(mesh.n_faces());
    face_ids.reserve(mesh.n_faces());

    for (auto face : mesh.faces()) {
        auto he = face.halfedge();
        const auto& pa = he.to().prop().pt;
        he = he.next();
        const auto& pb = he.to().prop().pt;
        he = he.next();
        const auto& pc = he.to().prop().pt;
        triangles.emplace_back(to_kernel_point(pa), to_kernel_point(pb), to_kernel_point(pc));
        face_ids.emplace_back(face.id);
    }

    Tree tree(triangles.begin(), triangles.end());
    tree.accelerate_distance_queries();
    std::unordered_map<gpf::FaceId, FaceInfo<N>> face_info_map;
    for (std::size_t pid = 0; pid < points.size(); pid++) {
        auto closest_ret = tree.closest_point_and_primitive(to_kernel_point(points[pid]));
        auto fid = face_ids[std::distance(triangles.cbegin(), closest_ret.second)];
        face_info_map[fid].point_indices.emplace_back(pid);
    }

    std::vector<gpf::VertexId> point_vertices(points.size(), gpf::VertexId{});
    std::unordered_map<gpf::EdgeId, std::vector<std::size_t>> edge_to_points_map;
    for (auto& [fid, info] : face_info_map) {
        info.ccs =
          identify_points<N>(mesh, fid, points, edge_to_points_map, info.point_indices, point_vertices, eps * eps);
    }

    // add ccs for all triangles sharing this edge to prepare triangulation
    for (const auto eid : edge_to_points_map | views::keys) {
        for (const auto he : mesh.edge(eid).halfedges()) {
            const auto fid = he.face().id;
            if (fid.valid() && face_info_map.find(fid) == face_info_map.end()) {
                face_info_map.emplace(fid, FaceInfo<N>{ .ccs = make_face_coords<N>(mesh, fid) });
            }
        }
    }

    for (const auto& [eid, point_indices] : edge_to_points_map) {
        split_edge_by_points<N>(
          mesh,
          eid,
          [&points](std::size_t pid) { return points[pid]; },
          point_indices,
          point_vertices,
          eps,
          edge_parent_map);
    }

    for (const auto& [fid, info] : face_info_map) {
        triangulate_on_face<N>(mesh,
                               fid,
                               std::span<const std::array<double, N>>{ points },
                               info.ccs,
                               info.point_indices,
                               {},
                               point_vertices,
                               face_parent_map);
    }
    return point_vertices;
}

struct VertexProp
{
    double angle_sum = 0.0;
};

struct EdgeProp
{
    double len = 0.0;
    bool locked = false;
    bool is_origin = true;
};

struct HalfedgeProp
{
    std::array<double, 2> vector;
    double angle = 0.0;
    double signpost_angle = 0.0;
    gpf::HalfedgeId path_prev{};
    gpf::HalfedgeId path_next{};

    void unconnect()
    {
        path_prev = {};
        path_next = {};
    }
};

using AuxiliaryMesh = gpf::ManifoldMesh<VertexProp, HalfedgeProp, EdgeProp, gpf::Empty>;

template<typename Mesh, typename IsLocked, typename GetEdgeLength>
inline std::vector<gpf::HalfedgeId>
shortest_patch_by_dijksta(const Mesh& mesh,
                          const gpf::VertexId start_vid,
                          const gpf::VertexId end_vid,
                          IsLocked&& is_locked,
                          GetEdgeLength&& get_edge_length)
{
    for (const auto he : mesh.vertex(start_vid).outgoing_halfedges()) {
        if (he.to().id == end_vid && !is_locked(he.edge())) {
            return { he.id };
        }
    }

    using WeightedHalfedge = std::pair<double, gpf::HalfedgeId>;
    std::unordered_map<gpf::VertexId, gpf::HalfedgeId> incomindg_halfedges_map;
    std::priority_queue<WeightedHalfedge, std::vector<WeightedHalfedge>, std::greater<WeightedHalfedge>> pq;
    auto vertex_used = [start_vid, &incomindg_halfedges_map](const gpf::VertexId vid) {
        return vid == start_vid || incomindg_halfedges_map.find(vid) != incomindg_halfedges_map.end();
    };

    auto enqueue_vertex_neighbors = [&mesh, &is_locked, &get_edge_length, &vertex_used, &pq](const gpf::VertexId vid,
                                                                                             double dist) {
        for (const auto he : mesh.vertex(vid).outgoing_halfedges()) {
            if (is_locked(he.edge())) {
                continue;
            }
            auto vb = he.to().id;
            if (!vertex_used(vb)) {
                auto len = get_edge_length(he.edge());
                pq.emplace(dist + len, he.id);
            }
        }
    };

    enqueue_vertex_neighbors(start_vid, 0.0);
    while (!pq.empty()) {
        auto [curr_dist, hid] = pq.top();
        pq.pop();
        auto vid = mesh.he_to(hid);
        if (vertex_used(vid)) {
            continue;
        }
        if (vid == end_vid) {
            std::vector<gpf::HalfedgeId> path;
            do {
                path.emplace_back(hid);
                auto vid = mesh.he_from(hid);
                if (vid == start_vid) {
                    break;
                }
                assert(incomindg_halfedges_map.find(vid) != incomindg_halfedges_map.end());
                hid = incomindg_halfedges_map[vid];
            } while (true);
            ranges::reverse(path);
            return path;
        }
        incomindg_halfedges_map.emplace(vid, hid);
        enqueue_vertex_neighbors(vid, curr_dist);
    }
    return {};
}

struct FlipGeodesic
{
    enum class TurnDirection
    {
        Left,
        Right
    };
    struct Wedge
    {
        double angle;
        std::array<gpf::VertexId, 3> vertices;
        gpf::HalfedgeId hid;
        TurnDirection dir;

        auto operator<=>(const Wedge& other) const noexcept { return angle <=> other.angle; }
        bool operator==(const Wedge& other) const noexcept = default;
    };

    std::vector<gpf::HalfedgeId> perform(std::vector<gpf::HalfedgeId>&& raw_path);

    void init_wedge_queue(const std::vector<gpf::HalfedgeId>& raw_path);
    void add_wedge(const gpf::HalfedgeId ha, const gpf::HalfedgeId hb);
    void shorten_locally();
    bool flip(const gpf::HalfedgeId hid) const;
    void replace_path(std::vector<gpf::HalfedgeId>&& new_path,
                      const gpf::HalfedgeId path_prev_prev_hid,
                      const gpf::HalfedgeId path_next_next_hid);

    AuxiliaryMesh* mesh;
    std::priority_queue<Wedge, std::vector<Wedge>, std::greater<Wedge>> pq{};
    gpf::VertexId path_start_vid{};
    gpf::VertexId path_end_vid{};
    gpf::HalfedgeId path_start_hid{};
};

inline std::vector<gpf::HalfedgeId>
FlipGeodesic::perform(std::vector<gpf::HalfedgeId>&& raw_path)
{
    if (raw_path.size() == 1) {
        assert(!mesh->halfedge(raw_path.back()).edge().prop().locked);
        assert(mesh->halfedge(raw_path.back()).edge().prop().is_origin);
        return raw_path;
    }
    init_wedge_queue(raw_path);
    while (!pq.empty()) {
        shorten_locally();
    }
    std::vector<gpf::HalfedgeId> result;
    result.reserve(raw_path.size());
    auto curr_hid = path_start_hid;
    while (true) {
        result.emplace_back(curr_hid);
        mesh->halfedge(curr_hid).edge().prop().locked = true;
        if (mesh->he_to(curr_hid) == path_end_vid) {
            break;
        }

        curr_hid = mesh->halfedge_prop(curr_hid).path_next;
        assert(curr_hid.valid());
    }
    return result;
}

inline void
FlipGeodesic::init_wedge_queue(const std::vector<gpf::HalfedgeId>& raw_path)
{
    path_start_vid = mesh->he_from(raw_path.front());
    path_start_hid = raw_path.front();
    path_end_vid = mesh->he_to(raw_path.back());
    for (std::size_t i = 0; i + 1 < raw_path.size(); i++) {
        auto h1 = raw_path[i];
        auto h2 = raw_path[i + 1];
        mesh->halfedge_prop(h1).path_next = h2;
        mesh->halfedge_prop(h2).path_prev = h1;
        add_wedge(h1, h2);
    }
}

inline void
FlipGeodesic::add_wedge(const gpf::HalfedgeId h1, const gpf::HalfedgeId h2)
{
    auto ha = mesh->halfedge(h1);
    auto hb = mesh->halfedge(h2);
    auto vb = ha.to();
    auto angle_sum = vb.prop().angle_sum;
    auto angle_in = ha.twin().prop().signpost_angle;
    auto angle_out = hb.prop().signpost_angle;
    auto is_boundary = mesh->v_is_boundary(vb.id);
    double right_angle = std::numeric_limits<double>::infinity();
    double left_angle = std::numeric_limits<double>::infinity();
    if (angle_in < angle_out) {
        right_angle = angle_out - angle_in;
    } else if (!is_boundary) {
        right_angle = angle_sum - angle_in + angle_out;
    }

    if (angle_out < angle_in) {
        left_angle = angle_in - angle_out;
    } else if (!is_boundary) {
        left_angle = angle_sum - angle_out + angle_in;
    }

    auto va = ha.from().id;
    auto vc = hb.to().id;
    constexpr double EPS_ANGLE = 1e-3;
    if (left_angle < std::numbers::pi - EPS_ANGLE) {
        pq.push(Wedge{ .angle = left_angle, .vertices{ va, vb.id, vc }, .hid = h1, .dir = TurnDirection::Left });
    }
    if (right_angle < std::numbers::pi - EPS_ANGLE) {
        pq.push(Wedge{ .angle = right_angle, .vertices{ va, vb.id, vc }, .hid = h1, .dir = TurnDirection::Right });
    }
}

inline void
FlipGeodesic::shorten_locally()
{
    auto [angle, vertices, path_prev_hid, dir] = pq.top();
    auto path_next_hid = mesh->halfedge_prop(path_prev_hid).path_next;
    pq.pop();
    if (!path_next_hid.valid() || mesh->he_from(path_prev_hid) != vertices[0] ||
        mesh->he_to(path_prev_hid) != vertices[1] || mesh->he_to(path_next_hid) != vertices[2]) {
        return;
    }

    const auto path_prev_prev_hid = mesh->halfedge_prop(path_prev_hid).path_prev;
    const auto path_next_next_hid = mesh->halfedge_prop(path_next_hid).path_next;

    auto [prev_hid, next_hid] = dir == TurnDirection::Left
                                  ? std::pair{ path_prev_hid, path_next_hid }
                                  : std::pair{ mesh->he_twin(path_next_hid), mesh->he_twin(path_prev_hid) };
    auto curr_he = mesh->halfedge(prev_hid).next();
    while (curr_he.id != next_hid) {
        if (curr_he.twin().id == prev_hid) {
            curr_he = curr_he.twin().next();
            continue;
        }
        if (flip(curr_he.id)) {
            curr_he = curr_he.next().twin();
        } else {
            curr_he = curr_he.twin().next();
        }
    }

    std::vector<gpf::HalfedgeId> new_path;
    curr_he = mesh->halfedge(prev_hid).next();
    while (true) {
        new_path.emplace_back(curr_he.next().twin().id);
        if (curr_he.id == next_hid) {
            break;
        }
        curr_he = curr_he.twin().next();
    }
    if (dir == TurnDirection::Right) {
        ranges::reverse(new_path);
        for (auto& hid : new_path) {
            hid = mesh->he_twin(hid);
        }
    }
    mesh->halfedge_prop(path_prev_hid).unconnect();
    mesh->halfedge_prop(path_next_hid).unconnect();
    replace_path(std::move(new_path), path_prev_prev_hid, path_next_next_hid);
}

inline bool
FlipGeodesic::flip(const gpf::HalfedgeId hid) const
{
    if (mesh->halfedge(hid).edge().prop().locked) {
        return false;
    }

    auto hac = hid;
    auto hca = mesh->he_twin(hac);
    auto hcd = mesh->he_next(hac);
    auto hda = mesh->he_next(hcd);
    auto hab = mesh->he_next(hca);
    auto hbc = mesh->he_next(hab);

    auto lab = mesh->halfedge(hab).edge().prop().len;
    auto lbc = mesh->halfedge(hbc).edge().prop().len;
    auto lcd = mesh->halfedge(hcd).edge().prop().len;
    auto lda = mesh->halfedge(hda).edge().prop().len;
    auto lca = mesh->halfedge(hca).edge().prop().len;

    Eigen::Vector2d pa{ lca, 0.0 };
    auto pb = triangle_apex_from_base_lengths(lca, lab, lbc, false);
    auto pd = triangle_apex_from_base_lengths(lca, lcd, lda, true);

    auto left_area = pd.cross(pb); // pc = {0.0, 0.0}
    auto right_area = (pb - pa).cross(pd - pa);
    constexpr double TRIANGLE_TEST_EPS = 1e-6;
    auto area_sum = left_area + right_area;
    if (left_area / area_sum < TRIANGLE_TEST_EPS || right_area / area_sum < TRIANGLE_TEST_EPS) {
        return false;
    }

    mesh->flip(hid);
    auto he_ab = mesh->halfedge(hab);
    auto he_bc = mesh->halfedge(hbc);
    auto he_cd = mesh->halfedge(hcd);
    auto he_da = mesh->halfedge(hda);
    auto he_bd = mesh->halfedge(hid);
    auto he_db = he_bd.twin();
    assert(he_bd.next().id == he_da.id);

    he_bd.edge().prop().len = (pd - pb).norm();
    gpf::update_corner_angles_on_face(he_bd.face());
    gpf::update_corner_angles_on_face(he_db.face());

    // edge bd counld't be boundary
    he_bd.prop().signpost_angle =
      std::fmod(he_bc.prop().signpost_angle + he_bc.prop().angle, he_ab.to().prop().angle_sum);
    gpf::update_halfedge_vector(he_bd);
    he_db.prop().signpost_angle =
      std::fmod(he_da.prop().signpost_angle + he_da.prop().angle, he_cd.to().prop().angle_sum);
    gpf::update_halfedge_vector(he_db);

    // auto _a1 = std::fmod(he_ab.twin().prop().signpost_angle - he_bd.prop().angle, he_bd.from().prop().angle_sum);
    // auto _a2 = he_bd.prop().signpost_angle - _a1;
    assert(std::abs(he_db.prop().signpost_angle -
                    std::fmod(he_cd.twin().prop().signpost_angle - he_db.prop().angle + he_db.from().prop().angle_sum,
                              he_db.from().prop().angle_sum)) < 1e-8);
    assert(std::abs(he_bd.prop().signpost_angle -
                    std::fmod(he_ab.twin().prop().signpost_angle - he_bd.prop().angle + he_bd.from().prop().angle_sum,
                              he_bd.from().prop().angle_sum)) < 1e-8);
    assert(std::abs(ranges::fold_left(he_bd.from().outgoing_halfedges() |
                                        views::transform([](auto he) { return he.prop().angle; }),
                                      0.0,
                                      std::plus{}) -
                    he_bd.from().prop().angle_sum) < 1e-8);
    assert(std::abs(ranges::fold_left(he_db.from().outgoing_halfedges() |
                                        views::transform([](auto he) { return he.prop().angle; }),
                                      0.0,
                                      std::plus{}) -
                    he_db.from().prop().angle_sum) < 1e-8);
    return true;
}

inline void
FlipGeodesic::replace_path(std::vector<gpf::HalfedgeId>&& new_path,
                           const gpf::HalfedgeId path_prev_prev_hid,
                           const gpf::HalfedgeId path_next_next_hid)
{
    auto prev_he = mesh->halfedge(path_prev_prev_hid);
    auto curr_he = mesh->halfedge(new_path.front());
    curr_he.prop().path_prev = prev_he.id;
    if (prev_he.id.valid()) {
        prev_he.prop().path_next = curr_he.id;
        add_wedge(prev_he.id, curr_he.id);
    } else if (mesh->he_from(curr_he.id) == path_start_vid) {
        path_start_hid = curr_he.id;
    }

    for (std::size_t i = 1; i < new_path.size(); ++i) {
        prev_he = curr_he;
        curr_he = mesh->halfedge(new_path[i]);
        curr_he.prop().path_prev = prev_he.id;
        prev_he.prop().path_next = curr_he.id;
        add_wedge(prev_he.id, curr_he.id);
    }
    curr_he.prop().path_next = path_next_next_hid;
    if (path_next_next_hid.valid()) {
        mesh->halfedge_prop(path_next_next_hid).path_prev = curr_he.id;
        add_wedge(curr_he.id, path_next_next_hid);
    }
}

template<std::size_t N>
struct EdgePoint
{
    gpf::EdgeId eid;
    double t;
    std::array<double, N> pt;
};

template<std::size_t N, typename Mesh>
[[nodiscard]] EdgePoint<N>
make_edge_point(const Mesh& mesh, const double left_ori, const double right_ori, const gpf::HalfedgeId hab)
{
    const auto s = left_ori + right_ori;
    const auto tb = left_ori / s;
    const auto ta = 1.0 - tb;
    EdgePoint<N> edge_point;
    auto he_ab = mesh.halfedge(hab);
    auto pa = VectorNd<N>::Map(he_ab.from().prop().pt.data());
    auto pb = VectorNd<N>::Map(he_ab.to().prop().pt.data());
    VectorNd<N>::Map(edge_point.pt.data()) = pa * tb + pb * ta;
    auto e_ab = he_ab.edge();
    if (e_ab.halfedge().id == he_ab.id) {
        edge_point.t = ta;
    } else {
        edge_point.t = tb;
    }
    edge_point.eid = e_ab.id;
    return edge_point;
}

template<std::size_t N, typename Mesh>
struct TracePolyline
{
    using Anchor = std::variant<gpf::VertexId, std::size_t>;

    void trace_from_vertex(gpf::HalfedgeId start_hid);
    void trace_from_edge(gpf::HalfedgeId hab, const double* dir, const gpf::VertexId end_vid);
    void add_intersection_point(double left_ori, double right_ori, gpf::HalfedgeId hab);
    std::span<const double> origin_signpost_angles;
    std::span<const double> origin_edge_lengths;
    std::vector<EdgePoint<N>>& edge_points;
    Mesh* mesh;
    AuxiliaryMesh* aux_mesh;
    std::vector<Anchor> path;
    std::vector<gpf::FaceId> path_on_face_vec;
};

template<std::size_t N, typename Mesh>
void
TracePolyline<N, Mesh>::trace_from_vertex(gpf::HalfedgeId start_hid)
{
    auto start_vid = aux_mesh->he_from(start_hid);
    assert(std::get<gpf::VertexId>(path.back()) == start_vid);
    auto end_vid = aux_mesh->he_to(start_hid);
    for (auto he : mesh->vertex(start_vid).outgoing_halfedges()) {
        if (he.to().id == end_vid) {
            path.emplace_back(end_vid);
            path_on_face_vec.emplace_back();
            return;
        }
    }

    if constexpr (N == 2) {
        auto vc = mesh->vertex(start_vid);
        auto prev_he = vc.halfedge().prev().twin();
        const auto first_hid = prev_he.id;
        const auto& pc = vc.prop().pt;
        auto pa = prev_he.to().prop().pt;
        const auto& pd = mesh->vertex(end_vid).prop().pt;
        auto right_ori = predicates::orient2d(pc.data(), pa.data(), pd.data());
        while (true) {
            auto curr_he = prev_he.prev().twin();

            const auto& pb = curr_he.to().prop().pt;
            auto left_ori = predicates::orient2d(pc.data(), pd.data(), pb.data());
            if (right_ori >= 0.0 && left_ori > 0.0) {
                add_intersection_point(left_ori, right_ori, prev_he.next().id);
                const Vector2d dir = Vector2d::Map(pd.data()) - Vector2d::Map(pc.data());
                trace_from_edge(prev_he.next().twin().id, dir.data(), end_vid);
                return;
            } else {
                prev_he = curr_he;
                if (prev_he.id == first_hid) {
                    // never arrive here
                    break;
                }
                right_ori = -left_ori;
                pa = pb;
            }
        }
    } else {
        auto get_orientations = [this](gpf::HalfedgeId hca, gpf::HalfedgeId hcb, double angle) {
            const auto hab = mesh->he_next(hca);
            auto lab = origin_edge_lengths[mesh->he_edge(hab).idx];
            auto lbc = origin_edge_lengths[mesh->he_edge(hcb).idx];
            auto lca = origin_edge_lengths[mesh->he_edge(hca).idx];

            Vector2d pa{ lab, 0.0 };
            constexpr std::array<double, 2> pb{ { 0.0, 0.0 } };
            Vector2d pc = triangle_apex_from_base_lengths(lab, lbc, lca, true);
            std::array<double, 2> dir_arr;
            auto dir = Vector2d::Map(dir_arr.data());
            dir = Eigen::Rotation2D<double>(angle) * (pa - pc).normalized();
            auto pd = (pc + dir).eval();
            auto left_ori = predicates::orient2d(pc.data(), pd.data(), pb.data());
            auto right_ori = predicates::orient2d(pc.data(), pa.data(), pd.data());
            assert(left_ori > 0.0 && right_ori > 0.0);
            return std::make_tuple(left_ori, right_ori, std::move(dir_arr));
        };
        const auto signpost_angle = aux_mesh->halfedge(start_hid).prop().signpost_angle;
        auto prev_he = mesh->vertex(start_vid).halfedge().prev().twin();
        const auto first_hid = prev_he.id;
        auto angle_sum = aux_mesh->vertex(start_vid).prop().angle_sum;
        while (true) {
            auto curr_he = prev_he.prev().twin();
            auto in_angle = origin_signpost_angles[prev_he.id.idx];
            auto out_angle = origin_signpost_angles[curr_he.id.idx];
            if (out_angle < in_angle) {
                out_angle += angle_sum;
            }
            auto curr_angle = signpost_angle;
            if (curr_angle < in_angle) {
                curr_angle += angle_sum;
            }
            if (curr_angle >= in_angle && curr_angle < out_angle) {
                auto [left_ori, right_ori, dir] = get_orientations(prev_he.id, curr_he.id, curr_angle - in_angle);
                add_intersection_point(left_ori, right_ori, prev_he.next().id);
                trace_from_edge(prev_he.next().twin().id, dir.data(), end_vid);
                return;
            } else {
                prev_he = curr_he;
                if (prev_he.id == first_hid) {
                    // never arrive here
                    break;
                }
            }
        }
    }
}

template<std::size_t N, typename Mesh>
void
TracePolyline<N, Mesh>::trace_from_edge(gpf::HalfedgeId hab, const double* dir_data, const gpf::VertexId end_vid)
{
    using Vec = Eigen::Matrix<double, N, 1>;
    Eigen::Vector2d dir = Eigen::Vector2d::Map(dir_data);
    while (true) {
        auto he_ab = mesh->halfedge(hab);
        auto he_bc = he_ab.next();
        if (he_bc.to().id == end_vid) {
            path.emplace_back(end_vid);
            path_on_face_vec.emplace_back(he_bc.face().id);
            return;
        }
        auto trace_next = [this, &dir, &hab, &he_bc](
                            const auto& mid_pt, const auto& pa, const auto& pb, const auto& pc, const auto& pd) {
            auto vc_ori = predicates::orient2d(mid_pt.data(), pd.data(), pc.data());
            if (vc_ori > 0.0) {
                auto right_ori = predicates::orient2d(mid_pt.data(), pb.data(), pd.data());
                assert(right_ori > 0.0);
                add_intersection_point(vc_ori, right_ori, he_bc.id);
                if constexpr (N == 3) {
                    dir = complex_div(dir, pb - pc).normalized();
                }
                hab = he_bc.twin().id;
            } else {
                auto right_ori = -vc_ori;
                auto left_ori = predicates::orient2d(mid_pt.data(), pd.data(), pa.data());
                assert(left_ori > 0.0);
                auto he_ca = he_bc.next();
                add_intersection_point(left_ori, right_ori, he_bc.next().id);
                if constexpr (N == 3) {
                    dir = complex_div(dir, pc - pa).normalized();
                }
                hab = he_ca.twin().id;
            }
        };
        if constexpr (N == 2) {
            auto mid_pt = Vec::Map(edge_points.back().pt.data());
            auto pa = Vec::Map(he_ab.from().prop().pt.data());
            auto pb = Vec::Map(he_ab.to().prop().pt.data());
            auto pc = Vec::Map(he_bc.to().prop().pt.data());
            auto pd = Vec::Map(mesh->vertex(end_vid).prop().pt.data());
            trace_next(mid_pt, pa, pb, pc, pd);
        } else {
            auto he_bc = he_ab.next();
            auto he_ca = he_bc.next();
            auto lab = origin_edge_lengths[he_ab.edge().id.idx];
            auto lbc = origin_edge_lengths[he_bc.edge().id.idx];
            auto lca = origin_edge_lengths[he_ca.edge().id.idx];

            Vector2d pa{ 0.0, 0.0 };
            Vector2d pb{ lab, 0.0 };
            Vector2d pc = triangle_apex_from_base_lengths(lab, lbc, lca, false);
            double t = edge_points.back().t;
            if (he_ab.edge().halfedge().id != he_ab.id) {
                t = 1.0 - t;
            }
            Vector2d mid_pt{ pa * (1.0 - t) + pb * t };
            Vector2d pd = mid_pt + dir;
            trace_next(mid_pt, pa, pb, pc, pd);
        }
    }
}

template<std::size_t N, typename Mesh>
void
TracePolyline<N, Mesh>::add_intersection_point(double left_ori, double right_ori, gpf::HalfedgeId hab)
{
    auto he_ab = mesh->halfedge(hab);
    auto edge_point = make_edge_point<N>(*mesh, left_ori, right_ori, hab);
    auto pid = this->edge_points.size();
    this->edge_points.push_back(std::move(edge_point));
    this->path.emplace_back(pid);
    this->path_on_face_vec.emplace_back(he_ab.face().id);
}

[[nodiscard]] inline Vector2d
local_edge_point(const double left_ori, const double right_ori, const Vector2d& pa, const Vector2d& pb)
{
    const auto s = left_ori + right_ori;
    const auto tb = left_ori / s;
    const auto ta = 1.0 - tb;
    return pa * tb + pb * ta;
}

[[nodiscard]] inline std::array<double, 3>
interpolate_barycentric(const std::array<double, 3>& from, const std::array<double, 3>& to, const double t) noexcept
{
    std::array<double, 3> bary{};
    for (std::size_t i = 0; i < bary.size(); ++i) {
        bary[i] = from[i] + (to[i] - from[i]) * t;
    }
    return bary;
}

template<typename Mesh>
struct WalkOnMeshSurface
{
    std::size_t sample_idx = 0;
    std::span<const double> lengths;

    [[nodiscard]] WalkOnMeshSurfaceResult operator()(const Mesh& mesh,
                                                     gpf::FaceId start_fid,
                                                     std::span<const double, 3> start_pt,
                                                     std::span<const double, 3> direction,
                                                     double eps);

    [[nodiscard]] std::expected<void, WalkOnMeshSurfaceFailure> walk_from_edge(
      const Mesh& mesh,
      HalfedgeId cross_hid,
      Vector2d curr_dir,
      EdgePoint<3> entry,
      double traveled,
      std::vector<std::pair<gpf::FaceId, std::array<double, 3>>>& result,
      double eps);
};

template<typename Mesh>
[[nodiscard]] std::expected<void, WalkOnMeshSurfaceFailure>
WalkOnMeshSurface<Mesh>::walk_from_edge(const Mesh& mesh,
                                        HalfedgeId cross_hid,
                                        Vector2d curr_dir,
                                        EdgePoint<3> entry,
                                        double traveled,
                                        std::vector<std::pair<gpf::FaceId, std::array<double, 3>>>& result,
                                        const double eps)
{
    const std::size_t max_iters = mesh.n_faces() + lengths.size() + 8;
    for (std::size_t iter = 0; iter < max_iters; ++iter) {
        if (sample_idx == lengths.size()) {
            return {};
        }

        const auto fid = mesh.he_face(cross_hid);
        if (!fid.valid()) {
            return std::unexpected(WalkOnMeshSurfaceFailure::BoundaryReached);
        }
        const std::size_t cross_idx = [&] {
            const auto indexed_halfedges =
              views::zip(mesh.face(fid).halfedges(), ranges::iota_view{ std::size_t{ 0 }, std::size_t{ 3 } });
            const auto cross_iter = ranges::find_if(
              indexed_halfedges, [cross_hid](auto&& item) { return std::get<0>(item).id == cross_hid; });
            assert(cross_iter != ranges::end(indexed_halfedges));
            return std::get<1>(*cross_iter);
        }();

        const auto he_ab = mesh.halfedge(cross_hid);
        const auto he_bc = he_ab.next();
        const auto he_ca = he_bc.next();
        assert(he_ca.next().id == he_ab.id);

        const Vector3d va3 = Vector3d::Map(he_ab.from().prop().pt.data());
        const Vector3d vb3 = Vector3d::Map(he_ab.to().prop().pt.data());
        const Vector3d vc3 = Vector3d::Map(he_bc.to().prop().pt.data());
        const double lab = (vb3 - va3).norm();
        const double lbc = (vc3 - vb3).norm();
        const double lca = (va3 - vc3).norm();
        const Vector2d pa2{ 0.0, 0.0 };
        const Vector2d pb2{ lab, 0.0 };
        const Vector2d pc2 = triangle_apex_from_base_lengths(lab, lbc, lca, false);

        double t_in = entry.t;
        if (he_ab.edge().halfedge().id != he_ab.id) {
            t_in = 1.0 - t_in;
        }
        std::array<double, 3> entry_barycentric{ 0.0, 0.0, 0.0 };
        entry_barycentric[cross_idx] = 1.0 - t_in;
        entry_barycentric[(cross_idx + 1) % 3] = t_in;
        const Vector2d mid_pt = pa2 * (1.0 - t_in) + pb2 * t_in;
        const double ray_scale = std::max(lengths.back() - traveled, 1.0);
        const Vector2d pd = mid_pt + curr_dir * ray_scale;
        const double vc_ori = predicates::orient2d(mid_pt.data(), pd.data(), pc2.data());

        EdgePoint<3> exit;
        std::array<double, 3> exit_barycentric{ 0.0, 0.0, 0.0 };
        Vector2d exit_2d;
        Vector2d base_2d;
        HalfedgeId exit_hid;
        if (vc_ori > 0.0) {
            const double right = std::max(predicates::orient2d(mid_pt.data(), pb2.data(), pd.data()), 0.0);
            exit = make_edge_point<3>(mesh, vc_ori, right, he_bc.id);
            double t = exit.t;
            if (mesh.e_halfedge(exit.eid) == he_bc.id) {
                t = 1.0 - t;
            }
            exit_barycentric[(cross_idx + 1) % 3] = t;
            exit_barycentric[(cross_idx + 2) % 3] = 1.0 - t;
            exit_2d = local_edge_point(vc_ori, right, pb2, pc2);
            base_2d = pb2 - pc2;
            exit_hid = he_bc.id;
        } else {
            const double right = std::max(-vc_ori, 0.0);
            const double left = std::max(predicates::orient2d(mid_pt.data(), pd.data(), pa2.data()), 0.0);
            exit = make_edge_point<3>(mesh, left, right, he_ca.id);
            double t = exit.t;
            if (mesh.e_halfedge(exit.eid) == he_ca.id) {
                t = 1.0 - t;
            }
            exit_barycentric[(cross_idx + 2) % 3] = t;
            exit_barycentric[cross_idx] = 1.0 - t;
            exit_2d = local_edge_point(left, right, pc2, pa2);
            base_2d = pc2 - pa2;
            exit_hid = he_ca.id;
        }

        const double seg_len = (exit_2d - mid_pt).norm();
        while (sample_idx < lengths.size() && lengths[sample_idx] <= traveled + seg_len + eps) {
            const double s = lengths[sample_idx] - traveled;
            const double t = (seg_len > 0.0) ? (s / seg_len) : 0.0;
            auto barycentric_coords = interpolate_barycentric(entry_barycentric, exit_barycentric, t);
            normalize_barycentric(barycentric_coords, BARY_EPS);
            result.emplace_back(fid, barycentric_coords);
            ++sample_idx;
        }
        if (sample_idx == lengths.size()) {
            return {};
        }

        traveled += seg_len;
        cross_hid = mesh.he_twin(exit_hid);
        entry = std::move(exit);
        curr_dir = complex_div(curr_dir, base_2d).normalized();
    }

    return std::unexpected(WalkOnMeshSurfaceFailure::IterationLimitExceeded);
}

template<typename Mesh>
[[nodiscard]] WalkOnMeshSurfaceResult
WalkOnMeshSurface<Mesh>::operator()(const Mesh& mesh,
                                    const gpf::FaceId start_fid,
                                    std::span<const double, 3> start_pt,
                                    std::span<const double, 3> direction,
                                    const double eps)
{
    static_assert(gpf::mesh_position_dim_v<Mesh> == 3);

    sample_idx = 0;
    std::vector<std::pair<gpf::FaceId, std::array<double, 3>>> result;
    result.reserve(lengths.size() + 1);
    if (lengths.empty()) {
        return result;
    }

    assert(std::abs(Eigen::Vector3d::Map(direction.data()).norm() - 1.0) < 1e-6);

    const auto face = mesh.face(start_fid);
    std::array<gpf::HalfedgeId, 3> halfedges{};
    std::array<gpf::VertexId, 3> vertices{};
    std::size_t n_halfedges = 0;
    for (const auto he : face.halfedges()) {
        halfedges[n_halfedges] = he.id;
        vertices[n_halfedges] = he.from().id;
        ++n_halfedges;
    }
    assert(n_halfedges == halfedges.size());

    Eigen::Vector2d dir{};
    std::array<double, 6> local_points{};
    ranges::fill(local_points, 0.0);
    Eigen::Vector2d start_proj{};
    const Eigen::Vector3d tri_a3 = Eigen::Vector3d::Map(mesh.vertex_prop(vertices[0]).pt.data());
    const Eigen::Vector3d tri_b3 = Eigen::Vector3d::Map(mesh.vertex_prop(vertices[1]).pt.data());
    const Eigen::Vector3d tri_c3 = Eigen::Vector3d::Map(mesh.vertex_prop(vertices[2]).pt.data());
    std::array<double, 3> start_barycentric{};
    {
        Eigen::Vector3d xaxis = tri_b3 - tri_a3;
        const auto lab = xaxis.norm();
        xaxis /= lab;
        auto vac = (tri_c3 - tri_a3).eval();
        const auto lac = vac.norm();
        vac /= lac;
        const auto normal = xaxis.cross(vac).normalized().eval();
        const Eigen::Vector3d yaxis = normal.cross(xaxis);

        const auto v = Eigen::Vector3d::Map(direction.data());
        dir[0] = v.dot(xaxis);
        dir[1] = v.dot(yaxis);

        local_points[2] = lab;
        const auto lbc = (tri_c3 - tri_b3).norm();
        Eigen::Vector2d::Map(local_points.data() + 4) = triangle_apex_from_base_lengths(lab, lbc, lac, false);

        auto start_vec = (Eigen::Vector3d::Map(start_pt.data()) - tri_a3).eval();
        start_proj[0] = start_vec.dot(xaxis);
        start_proj[1] = start_vec.dot(yaxis);

        std::vector<double> bary_points;
        bary_points.reserve(8);
        bary_points.append_range(local_points);
        bary_points.append_range(start_proj);
        auto start_bary = compute_bary_coordinates(bary_points);
        start_barycentric = { start_bary[0], start_bary[1], start_bary[2] };
        if (normalize_barycentric(std::span<double, 3>{ start_barycentric.data(), 3 }, BARY_EPS)) {
            start_proj = start_barycentric[0] * Eigen::Vector2d::Map(&local_points[0]) +
                         start_barycentric[1] * Eigen::Vector2d::Map(&local_points[2]) +
                         start_barycentric[2] * Eigen::Vector2d::Map(&local_points[4]);
        }
    }
    result.emplace_back(start_fid, start_barycentric);

    const Eigen::Vector2d ray_ref = start_proj + dir;
    auto right_ori = predicates::orient2d(ray_ref.data(), start_proj.data(), &local_points[0]);
    auto left_ori = predicates::orient2d(ray_ref.data(), start_proj.data(), &local_points[2]);
    std::size_t idx{ 0 };
    {
        while (left_ori <= 0.0 && right_ori <= 0.0) {
            idx = (idx + 4) % 6;
            left_ori = right_ori;
            right_ori = predicates::orient2d(ray_ref.data(), start_proj.data(), &local_points[idx]);
        }

        while (right_ori < 0.0 || left_ori > 0.0) {
            idx = (idx + 2) % 6;
            right_ori = left_ori;
            left_ori = predicates::orient2d(ray_ref.data(), start_proj.data(), &local_points[(idx + 2) % 6]);
        }
    }
    idx >>= 1;
    assert(right_ori > 0.0);
    assert(left_ori <= 0.0);

    const auto cross_hid_start = halfedges[idx];
    const Eigen::Vector2d from_pt = Eigen::Vector2d::Map(&local_points[idx << 1]);
    const Eigen::Vector2d to_pt = Eigen::Vector2d::Map(&local_points[((idx + 1) % 3) << 1]);

    const auto start_edge_point = make_edge_point<3>(mesh, -left_ori, right_ori, cross_hid_start);
    double t_along_exit = start_edge_point.t;
    if (mesh.e_halfedge(start_edge_point.eid) != cross_hid_start) {
        t_along_exit = 1.0 - t_along_exit;
    }
    std::array<double, 3> exit_barycentric_start{ 0.0, 0.0, 0.0 };
    exit_barycentric_start[idx] = 1.0 - t_along_exit;
    exit_barycentric_start[(idx + 1) % 3] = t_along_exit;
    const Eigen::Vector2d exit_2d_start = from_pt * (1.0 - t_along_exit) + to_pt * t_along_exit;
    const double seg_len_start = (exit_2d_start - start_proj).norm();

    while (sample_idx < lengths.size() && lengths[sample_idx] <= seg_len_start + eps) {
        const double s = lengths[sample_idx];
        const double t = (seg_len_start > 0.0) ? (s / seg_len_start) : 0.0;
        auto barycentric_coords = interpolate_barycentric(start_barycentric, exit_barycentric_start, t);
        normalize_barycentric(barycentric_coords, BARY_EPS);
        result.emplace_back(start_fid, std::move(barycentric_coords));
        ++sample_idx;
    }
    if (sample_idx == lengths.size()) {
        return result;
    }

    const Eigen::Vector2d next_dir = complex_div(dir, from_pt - to_pt).normalized();
    auto inner =
      walk_from_edge(mesh, mesh.he_twin(cross_hid_start), next_dir, start_edge_point, seg_len_start, result, eps);
    if (!inner) {
        return std::unexpected(inner.error());
    }
    return result;
}

} // namespace detail

template<typename VP, typename HP, typename EP, typename FP>
[[nodiscard]] WalkOnMeshSurfaceResult
walk_on_mesh_surface(const ManifoldMesh<VP, HP, EP, FP>& mesh,
                     const gpf::FaceId start_fid,
                     std::span<const double, 3> start_pt,
                     std::span<const double, 3> direction,
                     std::span<const double> lengths,
                     const double eps = 1e-6)
{
    using Mesh = gpf::ManifoldMesh<VP, HP, EP, FP>;
    detail::WalkOnMeshSurface<Mesh> walk{ .sample_idx = 0, .lengths{ lengths } };
    return walk(mesh, start_fid, start_pt, direction, eps);
}

template<std::size_t N, typename VP, typename HP, typename EP, typename FP>
auto
project_polylines_on_mesh(std::vector<std::array<double, N>>& points,
                          const std::vector<std::vector<std::size_t>>& polylines,
                          gpf::ManifoldMesh<VP, HP, EP, FP>& mesh,
                          std::unordered_map<gpf::FaceId, gpf::FaceId>* face_parent_map = nullptr,
                          std::unordered_map<gpf::EdgeId, gpf::EdgeId>* edge_parent_map = nullptr)
{
    using Mesh = gpf::ManifoldMesh<VP, HP, EP, FP>;

    constexpr double EPS = 1e-3;
    auto point_vertices = detail::project_points_on_mesh<N>(points, mesh, EPS, face_parent_map, edge_parent_map);
    mesh.update_vertex_halfedges();

    detail::AuxiliaryMesh aux_mesh;
    aux_mesh.copy_from(mesh);

    gpf::update_edge_lengths<N>(aux_mesh,
                                [&mesh](auto v) { return std::span<const double, N>{ mesh.vertex_prop(v.id).pt }; });
    gpf::update_corner_angles(aux_mesh);
    gpf::update_vertex_angle_sums(aux_mesh);
    gpf::update_halfedge_signpost_angles(aux_mesh);
    gpf::update_halfedge_vectors(aux_mesh);

    std::vector<double> origin_signpost_angles(mesh.n_halfedges_capacity());
    for (auto he : aux_mesh.halfedges()) {
        origin_signpost_angles[he.id.idx] = he.prop().signpost_angle;
    }
    std::vector<double> origin_edge_lengths(mesh.n_edges_capacity());
    for (auto edge : aux_mesh.edges()) {
        origin_edge_lengths[edge.id.idx] = edge.prop().len;
    }

    detail::FlipGeodesic flip_geodesic{ .mesh = &aux_mesh };
    std::vector<detail::EdgePoint<N>> edge_points;
    std::vector<std::vector<typename detail::TracePolyline<N, Mesh>::Anchor>> polyline_paths;
    std::vector<std::vector<gpf::FaceId>> path_segment_faces;
    polyline_paths.reserve(polylines.size());
    for (const auto polyline : polylines) {
        detail::TracePolyline<N, Mesh> trace{ .origin_signpost_angles{ origin_signpost_angles },
                                              .origin_edge_lengths{ origin_edge_lengths },
                                              .edge_points{ edge_points },
                                              .mesh{ &mesh },
                                              .aux_mesh{ &aux_mesh } };

        trace.path.push_back(point_vertices[polyline.front()]);
        for (std::size_t i = 0; i + 1 < polyline.size(); i++) {
            auto va = point_vertices[polyline[i]];
            auto vb = point_vertices[polyline[i + 1]];
            if (va == vb) {
                continue;
            }
            auto local_path = flip_geodesic.perform(detail::shortest_patch_by_dijksta(
              aux_mesh, va, vb, [](auto e) { return false; }, [](auto e) { return e.prop().len; }));
            for (const auto hid : std::move(local_path)) {
                trace.trace_from_vertex(hid);
            }
        }
        polyline_paths.push_back(std::move(trace.path));
        path_segment_faces.push_back(std::move(trace.path_on_face_vec));
    }

    std::unordered_map<gpf::EdgeId, std::vector<std::size_t>> edge_to_points_map;
    std::vector<gpf::VertexId> edge_point_vertices(edge_points.size(), gpf::VertexId{});
    for (std::size_t pid = 0; pid < edge_points.size(); pid++) {
        const auto& point = edge_points[pid];
        if (point.t < EPS) {
            edge_point_vertices[pid] = mesh.edge(point.eid).halfedge().from().id;
        } else if (point.t > 1.0 - EPS) {
            edge_point_vertices[pid] = mesh.edge(point.eid).halfedge().to().id;
        } else {
            edge_to_points_map[point.eid].push_back(pid);
        }
    }

    std::unordered_map<gpf::FaceId,
                       std::pair<detail::FaceCoords<N>, std::vector<typename detail::TracePolyline<N, Mesh>::Anchor>>>
      face_to_ccs_map;
    for (const auto& [path, faces] : std::views::zip(polyline_paths, path_segment_faces)) {
        for (std::size_t i = 0; i < faces.size(); i++) {
            const auto fid = faces[i];
            if (!fid.valid()) {
                continue;
            }
            const auto va = path[i];
            const auto vb = path[i + 1];
            auto iter = face_to_ccs_map.find(fid);
            if (iter != face_to_ccs_map.end()) {
                iter->second.second.push_back(va);
                iter->second.second.push_back(vb);
            } else {
                face_to_ccs_map.emplace(
                  fid,
                  std::make_pair(detail::make_face_coords<N>(mesh, fid),
                                 std::vector<typename detail::TracePolyline<N, Mesh>::Anchor>{ va, vb }));
            }
        }
    }

    for (const auto& [eid, point_indices] : edge_to_points_map) {
        detail::split_edge_by_points<N>(
          mesh,
          eid,
          [&edge_points](std::size_t pid) { return edge_points[pid].pt; },
          point_indices,
          edge_point_vertices,
          EPS,
          edge_parent_map);
    }

    auto get_vertex_id = [&edge_point_vertices](auto anchor) {
        return std::visit(
          [&edge_point_vertices](auto&& arg) -> gpf::VertexId {
              using T = std::decay_t<decltype(arg)>;
              if constexpr (std::is_same_v<T, gpf::VertexId>) {
                  return arg;
              } else {
                  return edge_point_vertices[arg];
              }
          },
          anchor);
    };

    for (const auto& [fid, ccs_and_segments] : face_to_ccs_map) {
        const auto& ccs = ccs_and_segments.first;
        const auto& segments = ccs_and_segments.second;
        auto segment_vertices = segments | std::views::transform(get_vertex_id) | std::ranges::to<std::vector>();
        detail::triangulate_on_face<N>(mesh,
                                       fid,
                                       std::span<const std::array<double, N>>{},
                                       ccs,
                                       {},
                                       segment_vertices,
                                       edge_point_vertices,
                                       face_parent_map);
    }

    return std::make_pair(std::move(point_vertices),
                          std::move(polyline_paths) | std::views::transform([&get_vertex_id, &mesh](auto&& path) {
                              std::vector<gpf::HalfedgeId> halfedges;
                              halfedges.reserve(path.size() - 1);
                              for (std::size_t i = 0; i + 1 < path.size(); ++i) {
                                  const auto va = get_vertex_id(path[i]);
                                  const auto vb = get_vertex_id(path[i + 1]);
                                  if (va == vb) {
                                      continue;
                                  }
                                  auto hid = mesh.he_from_vertices(va, vb);
                                  if (hid.valid()) {
                                      halfedges.emplace_back(hid);
                                  } else {
                                      halfedges.append_range(detail::shortest_patch_by_dijksta(
                                        mesh,
                                        va,
                                        vb,
                                        [](auto e) { return false; },
                                        [](auto e) {
                                            auto [v1, v2] = e.vertices();
                                            const auto p1 = std::span<const double, N>(v1.prop().pt);
                                            const auto p2 = std::span<const double, N>(v2.prop().pt);
                                            return std::sqrt(gpf::squared_distance(p1, p2));
                                        }));
                                  }
                              }
                              return halfedges;
                          }) |
                            std::ranges::to<std::vector>());
}
} // namespace gpf
