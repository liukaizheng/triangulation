#pragma once

#include <algorithm>
#include <concepts>
#include <functional>
#include <optional>
#include <queue>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include <predicates/predicates.hpp>

#include <gpf/detail.hpp>
#include <gpf/manifold_mesh.hpp>
#include <gpf/mesh.hpp>
#include <gpf/mesh_property.hpp>

namespace gpf {
namespace detail {
using CollapseShortEdgeCheckResultT = std::optional<std::tuple<VertexId, VertexId, bool>>;

template<typename CheckEdge, typename Mesh>
concept CollapseShortEdgeChecker = requires(CheckEdge check_edge) {
    { check_edge(std::declval<typename Mesh::Edge>()) } -> std::same_as<CollapseShortEdgeCheckResultT>;
};

bool
collapse_would_flip(const auto& mesh,
                    const double* pb,
                    VertexId survive_vid,
                    VertexId remove_vid,
                    const std::vector<FaceId>* to_remove_faces)
{
    auto remove_vertex = mesh.vertex(remove_vid);
    for (const auto he : remove_vertex.incoming_halfedges()) {
        if (!he.face().id.valid() ||
            (to_remove_faces && std::ranges::find(*to_remove_faces, he.face().id) != to_remove_faces->end())) {
            continue;
        }
        auto va = he.from();
        auto vc = he.next().to();
        if (va.id == survive_vid || vc.id == survive_vid) {
            continue;
        }
        if (predicates::orient2d(va.prop().pt.data(), pb, vc.prop().pt.data()) <= 0.0) {
            return false;
        }
    }
    return true;
}
} // namespace detail

template<typename Mesh>
bool
collapse_short_edges(Mesh& mesh, const double tol, bool skip_non_manifold_check)
{
    bool collapsed = false;
    std::priority_queue<std::pair<double, EdgeId>,
                        std::vector<std::pair<double, EdgeId>>,
                        std::greater<std::pair<double, EdgeId>>>
      queue;
    for (auto edge : mesh.edges()) {
        if (edge.prop().len < tol) {
            queue.emplace(edge.prop().len, edge.id);
        }
    }

    std::unordered_set<VertexId> vb_oppo_vertices;
    while (!queue.empty()) {
        auto [len, eid] = queue.top();
        queue.pop();
        if (mesh.edge_is_deleted(eid) || mesh.edge_prop(eid).len != len) {
            continue;
        }
        auto [va, vb] = mesh.e_vertices(eid);
        if (mesh.v_is_boundary(va)) {
            if (mesh.v_is_boundary(vb)) {
                continue;
            } else {
                std::swap(va, vb);
            }
        }
        if (!skip_non_manifold_check) {
            vb_oppo_vertices.clear();
            for (auto e : mesh.vertex(vb).edges()) {
                auto [v1, v2] = mesh.e_vertices(e.id);
                if (v1 == vb) {
                    vb_oppo_vertices.emplace(v2);
                } else {
                    assert(v2 == vb);
                    vb_oppo_vertices.emplace(v1);
                }
            }
        }

        if (!skip_non_manifold_check &&
            std::ranges::any_of(mesh.vertex(va).edges(), [&mesh, &vb_oppo_vertices, va, eid](auto e) {
                auto [v1, v2] = mesh.e_vertices(e.id);
                const VertexId va_oppo = v1 == va ? v2 : v1;
                if (vb_oppo_vertices.contains(va_oppo)) {
                    for (const auto he : mesh.edge(eid).halfedges()) {
                        if (he.next().to().id == va_oppo) {
                            return false;
                        }
                    }
                    return true;
                }
                return false;
            })) {
            continue;
        }

        for (auto e : mesh.vertex(vb).edges()) {
            e.prop().need_update = true;
        }

        mesh.collapse_edge(eid, va, vb);
        collapsed = true;
        for (auto e : mesh.vertex(va).edges()) {
            auto& ep = e.prop();
            if (ep.need_update) {
                update_edge_length<mesh_position_dim_v<Mesh>>(e);
                if (ep.len < tol) {
                    queue.emplace(ep.len, e.id);
                }
                ep.need_update = false;
            }
        }
    }
    return collapsed;
}

template<typename VP, typename HP, typename EP, typename FP>
    requires(mesh_position_dim_v<ManifoldMesh<VP, HP, EP, FP>> == 2)
[[nodiscard]] bool
collapse_short_edges(ManifoldMesh<VP, HP, EP, FP>& mesh,
                     const double tol,
                     detail::CollapseShortEdgeChecker<ManifoldMesh<VP, HP, EP, FP>> auto&& check_edge)
{
    bool collapsed = false;
    std::priority_queue<std::pair<double, EdgeId>,
                        std::vector<std::pair<double, EdgeId>>,
                        std::greater<std::pair<double, EdgeId>>>
      queue;

    for (auto edge : mesh.edges()) {
        if (edge.prop().len < tol) {
            queue.emplace(edge.prop().len, edge.id);
        }
    }

    while (!queue.empty()) {
        auto [len, eid] = queue.top();
        queue.pop();
        if (mesh.edge_is_deleted(eid) || mesh.edge_prop(eid).len != len) {
            continue;
        }
        auto check_ret = check_edge(mesh.edge(eid));
        if (!check_ret.has_value()) {
            continue;
        }

        auto [va, vb, can_swap] = *check_ret;
        if (!detail::collapse_would_flip(mesh, mesh.vertex_prop(va).pt.data(), va, vb, nullptr)) {
            if (can_swap && detail::collapse_would_flip(mesh, mesh.vertex_prop(vb).pt.data(), vb, va, nullptr)) {
                std::swap(va, vb);
            } else {
                continue;
            }
        }

        for (auto e : mesh.vertex(vb).edges()) {
            e.prop().need_update = true;
        }

        mesh.collapse_edge(eid, va, vb);
        collapsed = true;
        for (auto e : mesh.vertex(va).edges()) {
            auto& ep = e.prop();
            if (ep.need_update) {
                update_edge_length<2>(e);
                if (ep.len < tol) {
                    queue.emplace(ep.len, e.id);
                }
                ep.need_update = false;
            }
        }
    }
    return collapsed;
}

/// Collapses or moves near-collinear interior vertices adjacent to the boundary loop containing start_hid.
///
/// start_hid must be on that boundary loop. Boundary edges must have a valid parent value grouping the original
/// polyline segment they belong to; edges with kInvalidIndex parent are not valid input for this routine. Edge
/// lengths must already be initialized. Edge properties must expose len, parent, and need_update; this function
/// updates len/parent for modified boundary edges and uses need_update as temporary scratch state.
template<typename VP, typename HP, typename EP, typename FP>
    requires(mesh_position_dim_v<ManifoldMesh<VP, HP, EP, FP>> == 2)
void
collapse_points_on_boundary(ManifoldMesh<VP, HP, EP, FP>& mesh, gpf::HalfedgeId start_hid, const double tol)
{
    std::priority_queue<std::tuple<double, double, VertexId>,
                        std::vector<std::tuple<double, double, VertexId>>,
                        std::greater<std::tuple<double, double, VertexId>>>
      queue;
    std::vector<std::vector<std::tuple<double, double, HalfedgeId, HalfedgeId, std::vector<FaceId>>>>
      vertex_collapse_data(mesh.n_vertices_capacity());
    auto enqueue = [&queue, &vertex_collapse_data, tol](const auto& mesh, const auto hab_id) {
        const auto hab = mesh.halfedge(hab_id);
        auto hbc = hab.next();
        const auto vc_id = hbc.to().id;
        if (mesh.v_is_boundary(vc_id)) {
            return;
        }

        if (std::ranges::any_of(vertex_collapse_data[vc_id.idx], [fid = hab.face().id](const auto& tuple) {
                return std::ranges::find(std::get<4>(tuple), fid) != std::get<4>(tuple).end();
            })) {
            return;
        }

        auto hca = hbc.next();
        std::vector<gpf::FaceId> to_remove_faces;
        const auto parent = hab.edge().prop().parent;
        gpf::HalfedgeId left_boundary_hid{};
        gpf::HalfedgeId right_boundary_hid{};
        double bottom_len{ 0.0 };
        {
            auto curr_he = hca.twin();
            while (true) {
                auto bottom_edge = curr_he.prev().edge();
                if (bottom_edge.prop().parent != parent) {
                    break;
                }
                to_remove_faces.push_back(curr_he.face().id);
                curr_he = curr_he.next().twin();
                bottom_len += bottom_edge.prop().len;
            }
            left_boundary_hid = curr_he.id;
        }
        std::ranges::reverse(to_remove_faces);
        to_remove_faces.push_back(hab.face().id);
        bottom_len += hab.edge().prop().len;
        {
            auto curr_he = hbc.twin();
            while (true) {
                auto bottom_edge = curr_he.next().edge();
                if (bottom_edge.prop().parent != parent) {
                    break;
                }
                to_remove_faces.push_back(curr_he.face().id);
                curr_he = curr_he.prev().twin();
                bottom_len += bottom_edge.prop().len;
            }
            right_boundary_hid = curr_he.id;
        }

        auto left_len = mesh.halfedge(left_boundary_hid).edge().prop().len;
        auto right_len = mesh.halfedge(right_boundary_hid).edge().prop().len;
        if (bottom_len <= left_len || bottom_len <= right_len) {
            return;
        }

        auto diff = left_len + right_len - bottom_len;
        if (2.0 * tol <= diff) {
            return;
        }

        vertex_collapse_data[vc_id.idx].emplace_back(
          diff, bottom_len, left_boundary_hid, right_boundary_hid, std::move(to_remove_faces));
        queue.emplace(diff, bottom_len, vc_id);
    };
    auto curr_he = mesh.halfedge(start_hid);
    while (true) {
        enqueue(mesh, curr_he.twin().id);
        curr_he = curr_he.prev();
        if (curr_he.id == start_hid) {
            break;
        }
    }

    while (!queue.empty()) {
        auto [diff, lab, vc] = queue.top();
        queue.pop();
        if (mesh.vertex_is_deleted(vc) || mesh.v_is_boundary(vc)) {
            continue;
        }
        auto entry = std::ranges::find_if(vertex_collapse_data[vc.idx], [diff, lab](const auto& tuple) {
            return std::get<0>(tuple) == diff && std::get<1>(tuple) == lab;
        });
        assert(entry != vertex_collapse_data[vc.idx].end());

        auto [_, __, hac, hcb, to_remove_faces] = *entry;

        auto lbc = mesh.halfedge(hcb).edge().prop().len;
        auto lca = mesh.halfedge(hac).edge().prop().len;
        if (lbc + lca - lab != diff) {
            continue;
        }
        auto pc = triangle_apex_from_base_lengths(lab, lbc, lca, false);
        if (pc[1] >= tol) {
            continue;
        }
        const auto va = mesh.halfedge(hac).from().id;
        const auto vb = mesh.halfedge(hcb).to().id;
        const auto parent = mesh.halfedge(hac).twin().next().edge().prop().parent;
        assert(parent != gpf::kInvalidIndex);

        bool collapse_left = pc[0] < tol;
        bool collapse_right = pc[0] + tol > lab;
        if (collapse_left || collapse_right) {
            gpf::VertexId survive_vid{}, remove_vid{};
            gpf::EdgeId collapse_eid{};
            gpf::HalfedgeId new_bottom_hid{};
            if (collapse_left) {
                survive_vid = va;
                collapse_eid = mesh.he_edge(hac);
                new_bottom_hid = hcb;
            } else {
                survive_vid = vb;
                collapse_eid = mesh.he_edge(hcb);
                new_bottom_hid = hac;
            }
            if (detail::collapse_would_flip(
                  mesh, mesh.vertex_prop(survive_vid).pt.data(), survive_vid, vc, &to_remove_faces)) {
                for (auto e : mesh.vertex(vc).edges()) {
                    e.prop().need_update = true;
                }
                if (to_remove_faces.size() > std::size_t{ 1 }) {
                    for (const auto fid : to_remove_faces) {
                        mesh.remove_face(fid);
                    }
                } else {
                    new_bottom_hid = mesh.halfedge(hac).twin().next().id;
                }
                mesh.collapse_edge(collapse_eid, survive_vid, vc);
                assert(mesh.halfedge(new_bottom_hid).face().id.valid());
                assert(!mesh.halfedge(new_bottom_hid).twin().face().id.valid());

                for (auto e : mesh.vertex(survive_vid).edges()) {
                    if (e.prop().need_update) {
                        update_edge_length<2>(e);
                        e.prop().need_update = false;
                    }
                }
                mesh.halfedge(new_bottom_hid).edge().prop().parent = parent;
                enqueue(mesh, new_bottom_hid);
            }
        } else {
            std::array<double, 2> new_pt{};
            double s = pc[0] / lab;
            Eigen::Vector2d::Map(new_pt.data()) = (1.0 - s) * Eigen::Vector2d::Map(mesh.vertex_prop(va).pt.data()) +
                                                  s * Eigen::Vector2d::Map(mesh.vertex_prop(vb).pt.data());
            if (detail::collapse_would_flip(mesh, new_pt.data(), {}, vc, &to_remove_faces)) {
                for (auto e : mesh.vertex(vc).edges()) {
                    e.prop().need_update = true;
                }
                mesh.vertex(vc).prop().pt = new_pt;
                for (const auto fid : to_remove_faces) {
                    mesh.remove_face(fid);
                }

                assert(!mesh.halfedge(hac).twin().face().id.valid());
                assert(!mesh.halfedge(hcb).twin().face().id.valid());
                assert(mesh.halfedge(hcb).twin().next().id == mesh.he_twin(hac));
                mesh.halfedge(hac).edge().prop().parent = parent;
                mesh.halfedge(hcb).edge().prop().parent = parent;

                for (auto e : mesh.vertex(vc).edges()) {
                    if (e.prop().need_update) {
                        update_edge_length<2>(e);
                        e.prop().need_update = false;
                    }
                }
                enqueue(mesh, hac);
                enqueue(mesh, hcb);
            }
        }
    }
}

template<typename Mesh>
bool
collapse_slivers_on_longest_edge(Mesh& mesh, const double tol)
{
    bool collapsed = false;
    std::priority_queue<std::pair<double, HalfedgeId>,
                        std::vector<std::pair<double, HalfedgeId>>,
                        std::greater<std::pair<double, HalfedgeId>>>
      queue;
    std::array<HalfedgeId, 3> tri_halfedges;
    std::array<double, 3> tri_edge_lengths;
    std::unordered_set<std::pair<VertexId, VertexId>, detail::PairHash> flipped_edges;
    auto metric_pair = [&mesh, &tri_halfedges, &tri_edge_lengths, &queue, tol](const gpf::FaceId fid) {
        auto face = mesh.face(fid);
        if (mesh.face_is_deleted(face.id)) {
            return std::make_pair(0.0, HalfedgeId{});
        }
        auto ha = face.halfedge();
        auto hb = ha.next();
        auto hc = hb.next();
        tri_halfedges[0] = ha.id;
        tri_halfedges[1] = hb.id;
        tri_halfedges[2] = hc.id;

        tri_edge_lengths[0] = ha.edge().prop().len;
        tri_edge_lengths[1] = hb.edge().prop().len;
        tri_edge_lengths[2] = hc.edge().prop().len;
        if (tri_edge_lengths[0] < tol || tri_edge_lengths[1] < tol || tri_edge_lengths[2] < tol) {
            return std::make_pair(0.0, HalfedgeId{});
        }
        std::size_t max_idx = 0;
        for (std::size_t i = 1; i < 3; ++i) {
            if (tri_edge_lengths[i] > tri_edge_lengths[max_idx]) {
                max_idx = i;
            }
        }
        const auto max_he_twin = mesh.he_twin(tri_halfedges[max_idx]);
        if (!max_he_twin.valid() || !mesh.he_face(max_he_twin).valid()) {
            return std::make_pair(0.0, HalfedgeId{});
        }

        auto diff =
          tri_edge_lengths[(max_idx + 1) % 3] + tri_edge_lengths[(max_idx + 2) % 3] - tri_edge_lengths[max_idx];
        return std::make_pair(diff, tri_halfedges[max_idx]);
    };

    auto enqueue = [&metric_pair, &queue, &mesh, &flipped_edges, tol](const gpf::FaceId fid) {
        auto pair = metric_pair(fid);
        if (pair.second.valid() && pair.first < 2.0 * tol) {
            auto [va, vb] = mesh.he_vertices(pair.second);
            auto verts = detail::ordered_pair(va, vb);
            if (!flipped_edges.contains(verts)) {
                queue.push(std::move(pair));
            }
        }
    };

    for (auto face : mesh.faces()) {
        enqueue(face.id);
    }

    while (!queue.empty()) {
        const auto old_pair = queue.top();
        queue.pop();
        const auto curr_pair = metric_pair(mesh.he_face(old_pair.second));
        if (curr_pair != old_pair) {
            continue;
        }
        auto [va, vb] = mesh.he_vertices(curr_pair.second);
        auto vert_pair = detail::ordered_pair(va, vb);
        if (flipped_edges.contains(vert_pair)) {
            continue;
        }
        auto hac = curr_pair.second;

        auto hca = mesh.he_twin(hac);
        auto hcd = mesh.he_next(hac);
        auto hda = mesh.he_next(hcd);
        auto hab = mesh.he_next(hca);
        auto hbc = mesh.he_next(hab);

        auto lab = mesh.halfedge(hab).edge().prop().len;
        auto lbc = mesh.halfedge(hbc).edge().prop().len;
        auto lcd = mesh.halfedge(hcd).edge().prop().len;
        auto lda = mesh.halfedge(hda).edge().prop().len;
        auto lca = mesh.halfedge(hca).edge().prop().len;

        auto pd = triangle_apex_from_base_lengths(lca, lcd, lda, true);
        if (std::abs(pd[1]) > tol) {
            continue;
        }
        Eigen::Vector2d pa{ lca, 0.0 };
        auto pb = triangle_apex_from_base_lengths(lca, lab, lbc, false);
        auto bottom_area = -pd[1] * lca;

        auto left_area = pd.cross(pb); // pc = {0.0, 0.0}
        auto right_area = (pb - pa).cross(pd - pa);
        constexpr double TRIANGLE_TEST_EPS = 1e-3;
        auto area_sum = bottom_area + pb[1] * lca;
        if (std::min(left_area, right_area) <= std::min(bottom_area, TRIANGLE_TEST_EPS * area_sum)) {
            continue;
        }

        collapsed = true;
        mesh.collapse_triangle_on_edge(hac);
        auto [new_va, new_vb] = mesh.he_vertices(hac);
        flipped_edges.emplace(detail::ordered_pair(new_va, new_vb));
        update_edge_length<mesh_position_dim_v<Mesh>>(mesh.halfedge(hca).edge());
        enqueue(mesh.he_face(hac));
        enqueue(mesh.he_face(hca));
    }
    return collapsed;
}

}
