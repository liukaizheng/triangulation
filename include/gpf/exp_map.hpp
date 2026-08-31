#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <queue>
#include <span>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <gpf/ids.hpp>
#include <gpf/manifold_mesh.hpp>
#include <gpf/mesh_property.hpp>
#include <gpf/project_polylines_on_mesh.hpp>

namespace gpf {

/// Raw normal coordinates for the active vertices reached by the exponential-map propagation.
/// vertex_ids[i] and uvs[i] refer to the same vertex, and entries are ordered by ascending VertexId.
struct ExpMapResult
{
    gpf::VertexId center_vertex{};
    std::vector<gpf::VertexId> vertex_ids;
    std::vector<std::array<double, 2>> uvs;
};

namespace detail {

template<typename Mesh>
void
update_exp_map_properties_around_vertex(Mesh& mesh, const VertexId center_vid)
{
    auto center_vertex = mesh.vertex(center_vid);
    for (const auto he : center_vertex.incoming_halfedges()) {
        update_edge_length<3>(he.edge());
    }

    for (const auto he : center_vertex.incoming_halfedges()) {
        update_corner_angles_on_face(he.face());
    }
    update_vertex_angle_sum(center_vertex);
    update_halfedge_signpost_angles_at_vertex(center_vertex);

    for (auto he : center_vertex.incoming_halfedges()) {
        const auto prev = he.prev().twin();
        he.prop().signpost_angle =
          std::fmod(prev.prop().signpost_angle + prev.prop().angle, he.from().prop().angle_sum);
    }

    for (const auto he : center_vertex.incoming_halfedges()) {
        update_halfedge_vector(he);
        update_halfedge_vector(he.twin());
    }
}

} // namespace detail

/// Computes raw discrete exponential-map coordinates within a pseudo-geodesic radius.
/// The center is first projected onto the mesh and may be inserted as a new vertex, mutating mesh topology.
/// Connectivity handles and required mesh-derived properties are recomputed only in the projected center's one-ring.
/// Properties outside that one-ring are left unchanged. Returned coordinates remain in mesh units and are not
/// transformed into decal or texture space.
template<typename VP, typename HP, typename EP, typename FP>
    requires HasPositionProperty<VertexHandle<ManifoldMesh<VP, HP, EP, FP>, false>, 3> &&
             HasAngleSumProperty<VertexHandle<ManifoldMesh<VP, HP, EP, FP>, false>> &&
             HasAngleProperty<HalfedgeHandle<ManifoldMesh<VP, HP, EP, FP>, false>> &&
             HasSignpostAngleProperty<HalfedgeHandle<ManifoldMesh<VP, HP, EP, FP>, false>> &&
             HasVectorProperty<HalfedgeHandle<ManifoldMesh<VP, HP, EP, FP>, false>> &&
             HasLengthProperty<EdgeHandle<ManifoldMesh<VP, HP, EP, FP>, false>>
[[nodiscard]] ExpMapResult
exp_map(const std::span<const double, 3> center_pt,
        gpf::ManifoldMesh<VP, HP, EP, FP>& mesh,
        const double max_pseudo_geodesic_dist)
{
    using Mesh = gpf::ManifoldMesh<VP, HP, EP, FP>;
    static_assert(gpf::mesh_position_dim_v<Mesh> == 3);

    // Projection reuses a nearby vertex or retriangulates the containing face around a newly inserted center.
    std::vector<std::array<double, 3>> center_points{ { center_pt[0], center_pt[1], center_pt[2] } };
    const std::vector<gpf::VertexId> projected_vertices = detail::project_points_on_mesh<3>(center_points, mesh, 1e-3);

    const gpf::VertexId center_vertex = projected_vertices.front();
    detail::update_exp_map_properties_around_vertex(mesh, center_vertex);

    ExpMapResult result;
    result.center_vertex = center_vertex;

    // Mesh IDs can contain gaps, so propagation state is indexed by storage capacity rather than active count.
    const std::size_t n_vertices = mesh.n_vertices_capacity();
    const double infinity = std::numeric_limits<double>::infinity();
    std::vector<double> distances(n_vertices, infinity);
    std::vector<std::array<double, 2>> uvs(n_vertices, { 0.0, 0.0 });
    std::vector<bool> settled(n_vertices, false);

    using QueueEntry = std::tuple<double, VertexId, HalfedgeId, std::array<double, 2>, std::array<double, 2>>;
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> pq;
    distances[center_vertex.idx] = 0.0;
    settled[center_vertex.idx] = true;

    for (const auto he : mesh.vertex(center_vertex).outgoing_halfedges()) {
        const auto vid = he.to().id;
        distances[vid.idx] = he.edge().prop().len;
        uvs[vid.idx] = he.prop().vector;
        settled[vid.idx] = true;
    }

    auto enqueue = [&mesh, &uvs, &settled, &pq](
                     const auto vb_id, const HalfedgeId hab_id, const std::array<double, 2>& vab_data) noexcept {
        const auto vob{ Eigen::Vector2d::Map(uvs[vb_id.idx].data()) };
        const auto vab = Eigen::Vector2d::Map(vab_data.data()).normalized().eval();
        const auto hba_id = mesh.he_twin(hab_id);
        const auto& local_vba = mesh.halfedge_prop(hba_id).vector;
        Eigen::Vector2d local_vab{ -local_vba[0], -local_vba[1] };
        local_vab.normalize();
        Eigen::Matrix2d mat = Eigen::Matrix2d{ { vab[0], -vab[1] }, { vab[1], vab[0] } } *
                              Eigen::Matrix2d{ { local_vab[0], local_vab[1] }, { -local_vab[1], local_vab[0] } };
        mat.col(0).normalize();
        mat(0, 1) = -mat(1, 0);
        mat(1, 1) = mat(0, 0);

        for (const auto hbc : mesh.vertex(vb_id).outgoing_halfedges()) {
            const auto vc_id = hbc.to().id;
            if (settled[vc_id.idx]) {
                continue;
            }

            std::array<double, 2> vbc_data{};
            auto vbc{ Eigen::Vector2d::Map(vbc_data.data()) };
            vbc = mat * Eigen::Vector2d::Map(hbc.prop().vector.data());
            std::array<double, 2> voc_data{};
            auto voc{ Eigen::Vector2d::Map(voc_data.data()) };
            voc = vob + vbc;
            const auto len = voc.norm();

            pq.emplace(len, vc_id, hbc.id, std::move(vbc_data), std::move(voc_data));
        }
    };

    for (const auto he : mesh.vertex(center_vertex).outgoing_halfedges()) {
        const auto vid = he.to().id;
        if (settled[vid.idx]) {
            continue;
        }
        enqueue(vid, he.id, he.prop().vector);
    }

    while (!pq.empty()) {
        const auto [len, vb, hab, vab, vob] = pq.top();
        pq.pop();
        if (settled[vb.idx]) {
            continue;
        }
        settled[vb.idx] = true;
        distances[vb.idx] = len;
        uvs[vb.idx] = vob;
        if (len < max_pseudo_geodesic_dist) {
            enqueue(vb, hab, vab);
        }
    }

    // mesh.vertices() yields active vertices in ascending ID order, establishing the public result ordering.
    result.vertex_ids.reserve(mesh.n_vertices());
    result.uvs.reserve(mesh.n_vertices());
    for (const auto vertex : mesh.vertices()) {
        const gpf::VertexId vertex_id = vertex.id;
        assert(vertex_id.idx < n_vertices);
        if (vertex_id.idx >= n_vertices || !std::isfinite(distances[vertex_id.idx])) [[unlikely]] {
            continue;
        }
        result.vertex_ids.push_back(vertex_id);
        result.uvs.push_back(uvs[vertex_id.idx]);
    }
    return result;
}

} // namespace gpf
