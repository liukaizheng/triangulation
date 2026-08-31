#include "read_off.hpp"

#include <gpf/exp_map.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

struct ExpMapVertexProp
{
    std::array<double, 3> pt{};
    double angle_sum = 0.0;
};

struct HalfedgeProp
{
    double angle = 0.0;
    double signpost_angle = 0.0;
    std::array<double, 2> vector{};
};

struct EdgeProp
{
    double len = 0.0;
};

using ExpMapMesh = gpf::ManifoldMesh<ExpMapVertexProp, HalfedgeProp, EdgeProp, gpf::Empty>;

} // namespace

void
test_exp_map()
{
    const auto data = read_off("sphere.off");
    assert(!data.vertices.empty());
    assert(!data.faces.empty());

    auto mesh = ExpMapMesh::new_in(data.faces);
    assert(mesh.n_vertices() == data.vertices.size());
    assert(mesh.n_faces() == data.faces.size());
    for (auto vertex : mesh.vertices()) {
        assert(vertex.id.idx < data.vertices.size());
        vertex.prop().pt = data.vertices[vertex.id.idx];
    }

    const std::size_t n_vertices_before = mesh.n_vertices();
    const std::size_t n_vertices_capacity_before = mesh.n_vertices_capacity();
    const std::array<double, 3> center_pt{ -0.23, 0.26, 0.93 };
    const auto result = gpf::exp_map(center_pt, mesh, 1.0);

    assert(result.center_vertex.valid());
    assert(result.center_vertex.idx == n_vertices_capacity_before);
    assert(result.center_vertex.idx < mesh.n_vertices_capacity());
    assert(mesh.vertex(result.center_vertex).data().valid());
    assert(mesh.n_vertices() == n_vertices_before + 1);
    assert(mesh.n_vertices_capacity() == n_vertices_capacity_before + 1);

    assert(!result.vertex_ids.empty());
    assert(!result.uvs.empty());
    assert(result.vertex_ids.size() == result.uvs.size());
    for (std::size_t i = 0; i < result.vertex_ids.size(); ++i) {
        const auto vertex_id = result.vertex_ids[i];
        assert(vertex_id.valid());
        assert(vertex_id.idx < mesh.n_vertices_capacity());
        assert(mesh.vertex(vertex_id).data().valid());
        assert(std::isfinite(result.uvs[i][0]));
        assert(std::isfinite(result.uvs[i][1]));
    }

    const auto center_vertex = mesh.vertex(result.center_vertex);
    std::vector<gpf::VertexId> one_ring_vertices{ result.center_vertex };
    std::vector<gpf::HalfedgeId> one_ring_halfedges;
    std::vector<gpf::EdgeId> one_ring_edges;
    std::vector<gpf::FaceId> one_ring_faces;

    auto append_unique = [](auto& ids, const auto id) {
        if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
            ids.push_back(id);
        }
    };
    for (const auto halfedge : center_vertex.outgoing_halfedges()) {
        const auto face_id = halfedge.face().id;
        if (face_id.valid()) {
            append_unique(one_ring_faces, face_id);
        }
        const auto twin_face_id = halfedge.twin().face().id;
        if (twin_face_id.valid()) {
            append_unique(one_ring_faces, twin_face_id);
        }
    }
    for (const auto face_id : one_ring_faces) {
        for (const auto halfedge : mesh.face(face_id).halfedges()) {
            append_unique(one_ring_vertices, halfedge.from().id);
            append_unique(one_ring_vertices, halfedge.to().id);
            append_unique(one_ring_halfedges, halfedge.id);
            append_unique(one_ring_edges, halfedge.edge().id);
        }
    }

    assert(one_ring_vertices.size() > 1);
    assert(!one_ring_halfedges.empty());
    assert(!one_ring_edges.empty());
    assert(!one_ring_faces.empty());
    assert(one_ring_halfedges.size() == one_ring_faces.size() * 3);

    auto contains = [](const auto& ids, const auto id) { return std::find(ids.begin(), ids.end(), id) != ids.end(); };
    auto assert_halfedge_properties = [](const auto halfedge) {
        const double edge_len = halfedge.edge().prop().len;
        const auto& vector = halfedge.prop().vector;
        assert(std::isfinite(halfedge.prop().angle));
        assert(std::isfinite(halfedge.prop().signpost_angle));
        assert(std::isfinite(vector[0]));
        assert(std::isfinite(vector[1]));
        const double vector_len = std::hypot(vector[0], vector[1]);
        const double tolerance = 1e-12 * std::max(1.0, edge_len);
        assert(std::abs(vector_len - edge_len) <= tolerance);
    };

    for (const auto edge_id : one_ring_edges) {
        const double edge_len = mesh.edge(edge_id).prop().len;
        assert(std::isfinite(edge_len));
        assert(edge_len > 0.0);
    }
    for (const auto halfedge_id : one_ring_halfedges) {
        assert_halfedge_properties(mesh.halfedge(halfedge_id));
    }
    for (const auto vertex_id : one_ring_vertices) {
        const double angle_sum = mesh.vertex(vertex_id).prop().angle_sum;
        assert(std::isfinite(angle_sum));
        assert(angle_sum > 0.0);
    }

    const std::array<double, 2> zero_vector{};
    bool found_adjacent_out_of_ring_halfedge = false;
    for (const auto halfedge : mesh.halfedges()) {
        if (contains(one_ring_halfedges, halfedge.id)) {
            continue;
        }

        assert(halfedge.prop().angle == 0.0);
        assert(halfedge.prop().signpost_angle == 0.0);
        assert(halfedge.prop().vector == zero_vector);
        if (contains(one_ring_vertices, halfedge.from().id)) {
            found_adjacent_out_of_ring_halfedge = true;
        }
    }
    assert(found_adjacent_out_of_ring_halfedge);

    bool found_out_of_ring_edge = false;
    for (const auto edge : mesh.edges()) {
        if (contains(one_ring_edges, edge.id)) {
            continue;
        }

        assert(edge.prop().len == 0.0);
        found_out_of_ring_edge = true;
    }
    assert(found_out_of_ring_edge);

    for (const auto vertex : mesh.vertices()) {
        if (!contains(one_ring_vertices, vertex.id)) {
            assert(vertex.prop().angle_sum == 0.0);
        }
    }
}
