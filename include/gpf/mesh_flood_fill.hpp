#pragma once

#include <queue>
#include <span>
#include <vector>

#include <gpf/mesh.hpp>

namespace gpf {

template<typename VP, typename HP, typename EP, typename FP>
void
flood_fill_faces(const ManifoldMesh<VP, HP, EP, FP>& mesh, auto&& flag_face, auto&& is_inner_halfedge)
{
    std::vector<bool> visited(mesh.n_faces_capacity(), false);
    for (const auto he : mesh.halfedges()) {
        auto face = he.face();
        if (!face.id.valid() || visited[face.id.idx] || !is_inner_halfedge(he)) {
            continue;
        }

        visited[face.id.idx] = true;
        auto twin_face = he.twin().face();
        if (twin_face.id.valid()) {
            visited[twin_face.id.idx] = true;
        }

        std::queue<gpf::FaceId> queue;
        queue.push(face.id);
        flag_face(face);

        while (!queue.empty()) {
            auto curr = queue.front();
            queue.pop();
            for (auto h : mesh.face(curr).halfedges()) {
                if (is_inner_halfedge(h)) {
                    continue;
                }
                auto f = h.twin().face();
                if (!f.id.valid() || visited[f.id.idx]) {
                    continue;
                }

                visited[f.id.idx] = true;
                queue.push(f.id);
                flag_face(f);
            }
        }
    }
}

template<typename VP, typename HP, typename EP, typename FP>
std::vector<gpf::FaceId>
surround_faces_by_halfedges(const ManifoldMesh<VP, HP, EP, FP>& mesh, const std::span<const gpf::HalfedgeId> halfedges)
{
    std::vector<bool> he_is_boundary(mesh.n_halfedges_capacity(), false);
    for (const auto hid : halfedges) {
        auto twin_hid = mesh.he_twin(hid);
        if (he_is_boundary[twin_hid.idx]) {
            he_is_boundary[twin_hid.idx] = false;
        } else {
            he_is_boundary[hid.idx] = true;
        }
    }

    std::vector<gpf::FaceId> result;
    flood_fill_faces(
      mesh,
      [&result](auto face) { result.push_back(face.id); },
      [&he_is_boundary](auto he) { return he_is_boundary[he.id.idx]; });
    return result;
}

} // namespace gpf
