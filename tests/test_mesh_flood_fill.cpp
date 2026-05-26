#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <vector>

#include <gpf/mesh_flood_fill.hpp>

namespace {

using Mesh = gpf::ManifoldMesh<gpf::Empty, gpf::Empty, gpf::Empty, gpf::Empty>;

Mesh
make_tetrahedron_mesh()
{
    return Mesh::new_in(std::vector<std::array<std::size_t, 3>>{
      { 0, 1, 2 },
      { 0, 3, 1 },
      { 1, 3, 2 },
      { 2, 3, 0 },
    });
}

std::vector<gpf::HalfedgeId>
face_halfedges(const Mesh& mesh, const gpf::FaceId fid)
{
    std::vector<gpf::HalfedgeId> result;
    for (const auto he : mesh.face(fid).halfedges()) {
        result.push_back(he.id);
    }
    return result;
}

void
append_face_halfedges(const Mesh& mesh, const gpf::FaceId fid, std::vector<gpf::HalfedgeId>& halfedges)
{
    for (const auto he : mesh.face(fid).halfedges()) {
        halfedges.push_back(he.id);
    }
}

std::vector<std::size_t>
sorted_face_indices(const std::vector<gpf::FaceId>& faces)
{
    std::vector<std::size_t> result;
    result.reserve(faces.size());
    for (const auto fid : faces) {
        result.push_back(fid.idx);
    }
    std::ranges::sort(result);
    return result;
}

} // namespace

void
test_mesh_flood_fill_surround_single_face()
{
    const auto mesh = make_tetrahedron_mesh();
    const auto halfedges = face_halfedges(mesh, gpf::FaceId{ 0 });

    const auto faces = gpf::surround_faces_by_halfedges(mesh, halfedges);
    const auto indices = sorted_face_indices(faces);

    assert((indices == std::vector<std::size_t>{ 0 }));
}

void
test_mesh_flood_fill_surround_two_faces()
{
    const auto mesh = make_tetrahedron_mesh();
    std::vector<gpf::HalfedgeId> halfedges;
    append_face_halfedges(mesh, gpf::FaceId{ 0 }, halfedges);
    append_face_halfedges(mesh, gpf::FaceId{ 1 }, halfedges);

    const auto faces = gpf::surround_faces_by_halfedges(mesh, halfedges);
    const auto indices = sorted_face_indices(faces);

    assert((indices == std::vector<std::size_t>{ 0, 1 }));
}
