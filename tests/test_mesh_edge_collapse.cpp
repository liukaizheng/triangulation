#include "read_off.hpp"
#include <algorithm>
#include <array>
#include <fstream>
#include <gpf/detail.hpp>
#include <gpf/ids.hpp>
#include <gpf/mesh_property.hpp>
#include <gpf/mesh_upkeep.hpp>
#include <gpf/project_polylines_on_mesh.hpp>
#include <gpf/surface_mesh.hpp>
#include <iomanip>
#include <queue>
#include <utility>
#include <vector>

#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <Eigen/Dense>

template<typename Mesh>
void
write_off(const std::string& path, const std::vector<std::array<double, 3>>& vertices, const Mesh& mesh)
{
    std::ofstream out(path);
    std::size_t n_triangles = 0;
    for (const auto& f : mesh.faces()) {
        std::vector<std::size_t> face_vertices;
        for (const auto he : f.halfedges()) {
            face_vertices.push_back(he.from().id.idx);
        }
        if (face_vertices.size() >= 3)
            n_triangles += face_vertices.size() - 2;
    }
    out << "OFF\n";
    out << vertices.size() << ' ' << n_triangles << " 0\n";
    out << std::setprecision(17);
    for (const auto& v : vertices) {
        out << v[0] << ' ' << v[1] << ' ' << v[2] << '\n';
    }
    for (const auto& f : mesh.faces()) {
        std::vector<std::size_t> face_vertices;
        for (const auto he : f.halfedges()) {
            face_vertices.push_back(he.from().id.idx);
        }
        if (face_vertices[0] == 8232 && face_vertices[1] == 1254 && face_vertices[2] == 8192) {
            const auto a = 2;
        }
        for (std::size_t i = 1; i + 1 < face_vertices.size(); ++i) {
            out << "3 " << face_vertices[0] << ' ' << face_vertices[i] << ' ' << face_vertices[i + 1] << '\n';
        }
    }
}

template<typename Mesh>
void
write_boundary_points_off(const std::string& path, const Mesh& mesh)
{
    std::vector<std::array<double, 3>> boundary_pts;
    for (const auto& v : mesh.vertices()) {
        if (v.prop().on_boundary) {
            boundary_pts.push_back(v.prop().pt);
        }
    }
    std::ofstream out(path);
    out << "OFF\n";
    out << boundary_pts.size() << " 0 0\n";
    out << std::setprecision(17);
    for (const auto& pt : boundary_pts) {
        out << pt[0] << ' ' << pt[1] << ' ' << pt[2] << '\n';
    }
}

template<typename Mesh>
void
write_non_manifold_edges_obj(const std::string& path,
                             const std::vector<std::array<double, 3>>& vertices,
                             const Mesh& mesh)
{
    std::ofstream out(path);
    out << std::setprecision(17);
    for (const auto& v : vertices) {
        out << "v " << v[0] << ' ' << v[1] << ' ' << v[2] << '\n';
    }
    for (const auto& e : mesh.edges()) {
        std::size_t face_count = 0;
        for (const auto he : e.halfedges()) {
            if (he.face().id.valid()) {
                ++face_count;
            }
        }
        if (face_count > 2) {
            auto [va, vb] = e.vertices();
            out << "l " << (va.id.idx + 1) << ' ' << (vb.id.idx + 1) << '\n';
        }
    }
}

template<typename Mesh>
void
split_long_edges(Mesh& mesh, const double max_edge_len)
{
    using namespace gpf;
    std::priority_queue<std::pair<double, EdgeId>> queue;
    for (const auto e : mesh.edges()) {
        auto edge_len = e.prop().len;
        if (edge_len > max_edge_len) {
            queue.emplace(edge_len, e.id);
        }
    }
    while (!queue.empty()) {
        auto [edge_len, eid] = queue.top();
        queue.pop();
        if (edge_len != mesh.edge_prop(eid).len) {
            continue;
        }

        auto [va, vb] = mesh.e_vertices(eid);
        const auto new_vid = mesh.split_edge(eid);
        Eigen::Vector3d::Map(mesh.vertex_prop(new_vid).pt.data()) =
          (Eigen::Vector3d::Map(mesh.vertex_prop(va).pt.data()) +
           Eigen::Vector3d::Map(mesh.vertex_prop(vb).pt.data())) *
          0.5;

        auto new_v_he = mesh.vertex(new_vid).halfedge();
        const auto half_edge_len = edge_len * 0.5;
        new_v_he.edge().prop().len = half_edge_len;
        new_v_he.prev().edge().prop().len = half_edge_len;

        for (auto he : mesh.edge(eid).halfedges()) {
            VertexId vc{};
            if (he.to().id == new_vid) {
                vc = he.next().next().to().id;
            } else {
                assert(he.to().id == va || he.to().id == vb);
                vc = he.next().to().id;
            }
            mesh.split_face(he.face().id, new_vid, vc);
        }

        for (auto e : mesh.vertex(new_vid).edges()) {
            update_edge_length<3>(e);
            auto len = e.prop().len;
            if (len > max_edge_len) {
                queue.emplace(len, e.id);
            }
        }
    }
}

template<typename Mesh>
void
identify_boundary_points(const OffData& boundary_mesh, Mesh& mesh)
{
    using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Point_3 = Kernel::Point_3;
    using Triangle_3 = Kernel::Triangle_3;
    using TreeIterator = std::vector<Triangle_3>::const_iterator;
    using TreePrimitive = CGAL::AABB_triangle_primitive_3<Kernel, TreeIterator>;
    using TreeTraits = CGAL::AABB_traits_3<Kernel, TreePrimitive>;
    using Tree = CGAL::AABB_tree<TreeTraits>;

    std::vector<Triangle_3> triangles;
    triangles.reserve(mesh.n_faces());

    std::array<Point_3, 3> tri;
    for (const auto& indices : boundary_mesh.faces) {
        for (std::size_t i = 0; i < std::size_t{ 3 }; ++i) {
            const auto& p = boundary_mesh.vertices[indices[i]];
            tri[i] = Point_3(p[0], p[1], p[2]);
        }
        triangles.push_back(Triangle_3(tri[0], tri[1], tri[2]));
    }

    Tree tree(triangles.begin(), triangles.end());
    tree.accelerate_distance_queries();
    for (auto v : mesh.vertices()) {
        const auto& pt = v.prop().pt;
        Point_3 p(pt[0], pt[1], pt[2]);

        auto closest_ret = tree.closest_point_and_primitive(p);
        auto sq_dist = (closest_ret.first - p).squared_length();
        if (sq_dist < 1e-8) {
            v.prop().on_boundary = true;
        } else {
            const auto a = 2;
        }
    }
}

template<typename Mesh>
void
smooth_faces(const std::size_t n_smooths, Mesh& mesh)
{
    using Vec3 = Eigen::Vector3d;
    std::vector<std::array<double, 3>> cache_points(mesh.n_vertices_capacity());
    std::vector<gpf::VertexId> patch_interior_vertices;
    for (const auto v : mesh.vertices()) {
        if (!v.prop().on_boundary) {
            patch_interior_vertices.emplace_back(v.id);
        }
    }
    for (std::size_t _ = 0; _ < n_smooths; _++) {
        for (const auto va : patch_interior_vertices) {
            if (va.idx == 194) {
                const auto a = 2;
            }
            auto pt = Vec3::Map(cache_points[va.idx].data());
            pt.setZero();
            auto count = 0;
            for (const auto edge : mesh.vertex(va).edges()) {
                auto he = edge.halfedge();
                if (he.to().id == va) {
                    auto vb = he.from().id;
                    pt += Vec3::Map(he.from().prop().pt.data());
                } else {
                    auto vb = he.to().id;
                    pt += Vec3::Map(he.to().prop().pt.data());
                }
                count += 1;
            }
            pt /= static_cast<double>(count);
            const auto a = 2;
        }
        for (const auto vid : patch_interior_vertices) {
            mesh.vertex_prop(vid).pt = cache_points[vid.idx];
        }
    }
}
void
test_mesh_edge_collapse1()
{
    using namespace gpf;
    auto data = read_off("clip_material_1.off");
    struct VertexProp
    {
        std::array<double, 3> pt;
        bool on_boundary{ false };
    };
    struct EdgeProp
    {
        double len;
        bool need_update{ false };
    };
    using Mesh = gpf::SurfaceMesh<VertexProp, Empty, EdgeProp>;
    auto mesh = Mesh::new_in(data.faces);
    for (auto v : mesh.vertices()) {
        v.prop().pt = data.vertices[v.id.idx];
    }
    gpf::update_edge_lengths<3>(mesh);
    {
        auto eid = mesh.e_from_vertices(gpf::VertexId{ 1180 }, gpf::VertexId{ 1182 });
        auto len = mesh.edge_prop(eid).len;
        const auto a = 2;
    }

    const double tol = 0.04;
    gpf::collapse_short_edges(mesh, tol, false);
    // write_off("edge_collapsed.off", data.vertices, mesh);
    // write_non_manifold_edges_obj("non_manifold.obj", data.vertices, mesh);
    // collapse_degenerate_triangles(data.vertices, mesh, tol);
    collapse_slivers_on_longest_edge(mesh, tol);
    const auto n_queue_cap = static_cast<std::size_t>(mesh.n_edges() * 0.4);
    std::priority_queue<double> pq;
    for (const auto e : mesh.edges()) {
        const auto len = e.prop().len;
        if (pq.size() < n_queue_cap) {
            pq.emplace(len);
        } else if (len < pq.top()) {
            pq.pop();
            pq.emplace(len);
        }
    }
    const auto max_edge_len = std::max(0.5, pq.top());
    split_long_edges(mesh, max_edge_len);

    auto base_mesh_data = read_off("bunny_stamped.off");
    identify_boundary_points(base_mesh_data, mesh);
    write_boundary_points_off("boundary_points.off", mesh);
    smooth_faces(100, mesh);
    data.vertices.resize(mesh.n_vertices_capacity());
    for (std::size_t i = 0; i < mesh.n_vertices_capacity(); i++) {
        data.vertices[i] = mesh.vertex_prop(VertexId{ i }).pt;
    }
    write_off("material_1_smooth.off", data.vertices, mesh);

    const auto a = 2;
}
