#include <array>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <random>
#include <tuple>
#include <vector>

#include "gpf/ids.hpp"
#include "gpf/triangulation.hpp"

#include <gpf/mesh.hpp>
#include <gpf/mesh_property.hpp>
#include <gpf/mesh_upkeep.hpp>
#include <predicates/predicates.hpp>

namespace {
struct VertexProp
{
    std::array<double, 2> pt;
};

struct EdgeProp
{
    double len = 0.0;
    std::size_t parent{ gpf::kInvalidIndex };
    bool need_update{ false };
};

using Mesh = gpf::ManifoldMesh<VertexProp, gpf::Empty, EdgeProp>;

void
write_off(const char* path, const Mesh& mesh)
{
    std::ofstream out(path);
    assert(out);
    out << "OFF\n";
    out << mesh.n_vertices_capacity() << ' ' << mesh.n_faces() << " 0\n";
    out << std::setprecision(17);
    for (std::size_t i{ 0 }; i < mesh.n_vertices_capacity(); ++i) {
        const auto& pt = mesh.vertex(gpf::VertexId{ i }).prop().pt;
        out << pt[0] << ' ' << pt[1] << " 0\n";
    }
    for (const auto f : mesh.faces()) {
        std::vector<std::size_t> face_vertices;
        for (const auto he : f.halfedges()) {
            const auto vertex_index = he.from().id.idx;
            face_vertices.push_back(vertex_index);
        }
        out << face_vertices.size();
        for (const auto vertex_index : face_vertices) {
            out << ' ' << vertex_index;
        }
        out << '\n';
    }
}

Mesh
make_mesh(const std::vector<std::array<double, 2>>& points, std::vector<std::vector<std::size_t>> faces)
{
    auto mesh = Mesh::new_in(std::move(faces));
    for (auto v : mesh.vertices()) {
        v.prop().pt = points[v.id.idx];
    }
    gpf::update_edge_lengths<2>(mesh);
    return mesh;
}

std::optional<std::tuple<gpf::VertexId, gpf::VertexId, bool>>
check_edge(const Mesh::Edge& edge)
{
    const auto [va, vb] = edge.vertices();
    const auto va_is_boundary = !va.halfedge().face().id.valid();
    const auto vb_is_boundary = !vb.halfedge().face().id.valid();
    if (!va_is_boundary && !vb_is_boundary) {
        if (va.id < vb.id) {
            return std::make_tuple(va.id, vb.id, true);
        } else {
            return std::make_tuple(vb.id, va.id, true);
        }
    }

    if (va_is_boundary && !vb_is_boundary) {
        return std::make_tuple(va.id, vb.id, false);
    }
    if (vb_is_boundary && !va_is_boundary) {
        return std::make_tuple(vb.id, va.id, false);
    }
    return {};
}

void
assert_all_faces_ccw(Mesh& mesh)
{
    for (auto face : mesh.faces()) {
        if (mesh.face_is_deleted(face.id)) {
            continue;
        }
        auto ha = face.halfedge();
        auto hb = ha.next();
        auto hc = hb.next();
        const auto& pa = ha.to().prop().pt;
        const auto& pb = hb.to().prop().pt;
        const auto& pc = hc.to().prop().pt;
        assert(predicates::orient2d(pa.data(), pb.data(), pc.data()) > 0.0);
    }
}
} // namespace

// Collapsing R-S in either direction would flip a triangle, so the edge must be kept.
void
test_collapse_short_edges_2d_skips_flip()
{
    const std::vector<std::array<double, 2>> points{
        { 0.0, 0.0 },    // 0: R (interior)
        { 0.05, 0.0 },   // 1: S (boundary)
        { 0.025, 0.5 },  // 2: p
        { 0.03, 1.0 },   // 3: q
        { -1.0, 0.0 },   // 4: T
        { 0.025, -0.5 }, // 5: m
        { 0.02, -1.0 },  // 6: n
    };
    auto mesh = make_mesh(points, { { 0, 1, 2 }, { 0, 2, 3 }, { 0, 3, 4 }, { 0, 4, 5 }, { 0, 5, 1 }, { 1, 5, 6 } });
    assert_all_faces_ccw(mesh);

    const bool collapsed = gpf::collapse_short_edges(mesh, 0.1, check_edge);

    assert(!collapsed);
    assert(mesh.n_faces() == 6);
    assert(mesh.n_vertices() == 7);
    assert(mesh.e_from_vertices(gpf::VertexId{ 0 }, gpf::VertexId{ 1 }).valid());
    assert_all_faces_ccw(mesh);
}

// No surrounding triangle flips: the short edge collapses normally.
void
test_collapse_short_edges_2d_collapses()
{
    const std::vector<std::array<double, 2>> points{
        { 0.0, 0.0 },  // 0: R (interior)
        { 0.05, 0.0 }, // 1: S (interior)
        { 0.0, 1.0 },  // 2: U
        { -1.0, 0.0 }, // 3: T
        { 0.0, -1.0 }, // 4: D
        { 1.0, 0.0 },  // 5: W
    };
    auto mesh = make_mesh(points, { { 0, 1, 2 }, { 0, 2, 3 }, { 0, 3, 4 }, { 0, 4, 1 }, { 1, 5, 2 }, { 1, 4, 5 } });
    assert_all_faces_ccw(mesh);

    const bool collapsed = gpf::collapse_short_edges(mesh, 0.1, check_edge);

    assert(collapsed);
    assert(mesh.n_faces() == 4);
    assert(mesh.n_vertices() == 5);
    assert(mesh.vertex_is_deleted(gpf::VertexId{ 0 }) != mesh.vertex_is_deleted(gpf::VertexId{ 1 }));
    assert_all_faces_ccw(mesh);
}

// Removing S would flip (S, m, n), but removing R is safe: the collapse direction must swap.
void
test_collapse_short_edges_2d_swaps_direction()
{
    const std::vector<std::array<double, 2>> points{
        { 0.0, 0.0 },    // 0: R (interior)
        { 0.05, 0.0 },   // 1: S (boundary)
        { 0.0, 1.0 },    // 2: U
        { -1.0, 0.0 },   // 3: T
        { 0.025, -0.5 }, // 4: m
        { 0.02, -1.0 },  // 5: n
    };
    auto mesh = make_mesh(points, { { 0, 1, 2 }, { 0, 2, 3 }, { 0, 3, 4 }, { 0, 4, 1 }, { 1, 4, 5 } });
    assert_all_faces_ccw(mesh);

    const bool collapsed = gpf::collapse_short_edges(mesh, 0.1, check_edge);

    assert(collapsed);
    assert(mesh.n_faces() == 3);
    assert(mesh.n_vertices() == 5);
    assert(mesh.vertex_is_deleted(gpf::VertexId{ 0 }));
    assert(!mesh.vertex_is_deleted(gpf::VertexId{ 1 }));
    assert(mesh.vertex_prop(gpf::VertexId{ 1 }).pt[0] == 0.05);
    assert(mesh.vertex_prop(gpf::VertexId{ 1 }).pt[1] == 0.0);
    assert_all_faces_ccw(mesh);
}

void
test_collapse_1000_points()
{

    const int N = 10000;
    std::vector<double> points;
    points.reserve(N * 2);

    std::mt19937 rng(42); // Seed for reproducibility
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < N; ++i) {
        points.push_back(dist(rng));
        points.push_back(dist(rng));
    }

    auto triangles_data = gpf::triangulate_points(points, true);
    std::vector<std::vector<std::size_t>> triangles;
    triangles.reserve(triangles_data.size() / 3);
    for (std::size_t i = 0; i < triangles_data.size(); i += 3) {
        triangles.push_back({ triangles_data[i], triangles_data[i + 1], triangles_data[i + 2] });
    }
    auto mesh = Mesh::new_in(triangles);
    for (auto v : mesh.vertices()) {
        const auto id = v.id.idx;
        v.prop().pt = { points[id * 2], points[id * 2 + 1] };
    }
    gpf::update_edge_lengths<2>(mesh);
    write_off("collapse_1000_points_before.off", mesh);
    const bool collapsed = gpf::collapse_short_edges(mesh, 0.1, check_edge);
    for (const auto he : mesh.halfedges()) {
        if (!he.face().id.valid()) {
            assert(he.from().halfedge().id == he.id);
        }
    }
    (void)collapsed;
    write_off("collapse_1000_points_collapsed.off", mesh);
}

auto
collapse_mesh_edges(Mesh& mesh, const std::vector<std::size_t>& vertex_on_edge, const double tol)
{
    return gpf::collapse_short_edges(
      mesh, tol, [&vertex_on_edge](auto edge) -> std::optional<std::tuple<gpf::VertexId, gpf::VertexId, bool>> {
          const auto [v1, v2] = edge.vertices();
          const auto va_is_boundary = !v1.halfedge().face().id.valid();
          const auto vb_is_boundary = !v2.halfedge().face().id.valid();
          auto va = v1.id;
          auto vb = v2.id;
          if (!va_is_boundary && !vb_is_boundary) {
              if (va < vb) {
                  return std::make_tuple(va, vb, true);
              } else {
                  return std::make_tuple(vb, va, true);
              }
          }

          if (va_is_boundary && !vb_is_boundary) {
              return std::make_tuple(va, vb, false);
          }
          if (vb_is_boundary && !va_is_boundary) {
              return std::make_tuple(vb, va, false);
          }

          // va_is_boundary && vb_is_boundary
          if (va.idx >= std::size_t{ 3 } && vb.idx >= std::size_t{ 3 }) {
              if (vertex_on_edge[va.idx] != vertex_on_edge[vb.idx]) {
                  return {};
              }
              if (va < vb) {
                  return std::make_tuple(va, vb, true);
              } else {
                  return std::make_tuple(vb, va, true);
              }
          }
          if (va.idx < std::size_t{ 3 } && vb.idx >= std::size_t{ 3 }) {
              return std::make_tuple(va, vb, false);
          }
          if (vb.idx < std::size_t{ 3 } && va.idx >= std::size_t{ 3 }) {
              return std::make_tuple(vb, va, false);
          }
          return {};
      });
}

void
upkeep_mesh(Mesh& mesh, const std::vector<std::size_t>& vertex_on_edge, const double tol)
{
    gpf::update_edge_lengths<2>(mesh);
    for (int _{ 0 }; _ < 3; _++) {
        if (!collapse_mesh_edges(mesh, vertex_on_edge, tol)) {
            return;
        }
        if (!gpf::collapse_slivers_on_longest_edge(mesh, tol)) {
            return;
        }
    }
}

void
test_collapse_on_triangle()
{
    std::vector<double> points{ -1.0, 0.0, 1.0, 0.0, 0.0, std::sqrt(3.0) };
    std::vector<std::size_t> vertex_on_edge{ 0, 1, 2 };
    std::vector<std::size_t> segments;
    auto pa = Eigen::Vector2d::Map(points.data()).eval();
    auto pb = Eigen::Vector2d::Map(points.data() + 2).eval();
    auto pc = Eigen::Vector2d::Map(points.data() + 4).eval();

    const std::size_t n_on_boundary = 50;
    const auto add_points_on_boundary =
      [n_on_boundary, &points, &segments, &vertex_on_edge](auto& p1, auto& p2, std::size_t idx) {
          const auto s = 1.0 / n_on_boundary;
          auto prev_idx = idx;
          for (std::size_t i = 1; i < n_on_boundary; i++) {
              auto t = i * s;
              Eigen::Vector2d pt = (1.0 - t) * p1 + t * p2;
              segments.push_back(prev_idx);
              auto curr_idx = points.size() >> 1;
              segments.push_back(curr_idx);
              prev_idx = curr_idx;
              points.append_range(pt);
              vertex_on_edge.push_back(idx);
          }
          segments.push_back(prev_idx);
          segments.push_back((idx + 1) % 3);
      };
    add_points_on_boundary(pa, pb, 0);
    add_points_on_boundary(pb, pc, 1);
    add_points_on_boundary(pc, pa, 2);

    std::size_t N = 1000;
    std::mt19937 rng(42); // Seed for reproducibility
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (std::size_t i = 0; i < N; i++) {
        auto t1 = dist(rng);
        auto t2 = dist(rng);
        auto t3 = dist(rng);
        const auto sum = t1 + t2 + t3;
        Eigen::Vector2d pt = t1 / sum * pa + t2 / sum * pb + t3 / sum * pc;
        points.append_range(pt);
    }
    vertex_on_edge.resize(points.size() >> 1, gpf::kInvalidIndex);
    const auto triangles_data = gpf::triangulate_polygon(points, segments, true);
    std::vector<std::vector<std::size_t>> triangles;
    triangles.reserve(triangles_data.size() / 3);
    for (std::size_t i = 0; i < triangles_data.size(); i += 3) {
        triangles.push_back({ triangles_data[i], triangles_data[i + 1], triangles_data[i + 2] });
    }
    auto mesh = Mesh::new_in(triangles);
    for (auto v : mesh.vertices()) {
        const auto i = v.id.idx;
        v.prop().pt = { points[i * 2], points[i * 2 + 1] };
    }
    {
        auto first_hid = mesh.vertex(gpf::VertexId{ 0 }).halfedge().prev().id;
        std::size_t parent{ 0 };
        std::size_t count{ 0 };
        auto curr_he = mesh.halfedge(first_hid);
        while (true) {
            curr_he.edge().prop().parent = parent;
            count += 1;
            curr_he = curr_he.prev();
            if (curr_he.id == first_hid) {
                break;
            }
            if (count == n_on_boundary) {
                count = 0;
                parent += 1;
            }
        }
    }
    write_off("123_origin.off", mesh);

    gpf::update_edge_lengths<2>(mesh);
    // upkeep_mesh(mesh, vertex_on_edge, 0.1);
    // write_off("123_upkeep.off", mesh);

    gpf::collapse_points_on_boundary(mesh, mesh.vertex(gpf::VertexId{ 0 }).halfedge().id, 0.1);
    std::vector<gpf::HalfedgeId> boundary_halfedges;
    for (const auto he : mesh.halfedges()) {
        if (!he.twin().face().id.valid()) {
            boundary_halfedges.push_back(he.id);
        }
    }
    for (const auto hid : boundary_halfedges) {
        auto he = mesh.halfedge(hid);
        if (he.from().halfedge().face().id.valid()) {
            const auto a = 2;
        }
        if (he.to().halfedge().face().id.valid()) {
            const auto a = 2;
        }
    }
    std::vector<gpf::HalfedgeId> boundary_halfedges1;
    auto first_hid = mesh.vertex(gpf::VertexId{ 0 }).halfedge().id;
    auto curr_hid = first_hid;
    while (true) {
        boundary_halfedges1.push_back(curr_hid);
        auto next_hid = mesh.he_next(curr_hid);
        if (next_hid == first_hid) {
            break;
        }
        curr_hid = next_hid;
    }

    write_off("123_collapsed.off", mesh);
    const auto a = 2;
}
