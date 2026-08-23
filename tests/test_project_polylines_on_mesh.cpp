#include <gpf/ids.hpp>
#include <gpf/manifold_mesh.hpp>
#include <gpf/project_polylines_on_mesh.hpp>
#include <gpf/triangulation.hpp>

#include <cassert>
#include <fstream>
#include <ostream>
#include <print>
#include <random>
#include <ranges>

namespace {
struct VertexProp
{
    std::array<double, 2> pt;
};

struct VertexProp3d
{
    std::array<double, 3> pt;
};

using Mesh3d = gpf::ManifoldMesh<VertexProp3d, gpf::Empty, gpf::Empty, gpf::Empty>;

void
write_obj(const std::string& name, const auto& mesh)
{
    std::ofstream out(name);
    for (const auto v : mesh.vertices()) {
        std::println(out, "v {} {} 0", v.prop().pt[0], v.prop().pt[1]);
    }
    for (const auto f : mesh.faces()) {
        std::print(out, "f");
        for (const auto he : f.halfedges()) {
            std::print(out, " {}", he.to().id.idx + 1);
        }
        std::println(out);
    }
    out.close();
}

std::array<Eigen::Vector3d, 3>
face_points(const Mesh3d& mesh, const gpf::FaceId fid)
{
    std::array<Eigen::Vector3d, 3> points;
    std::size_t idx = 0;
    for (const auto he : mesh.face(fid).halfedges()) {
        points[idx++] = Eigen::Vector3d::Map(mesh.vertex_prop(he.from().id).pt.data());
    }
    assert(idx == 3);
    return points;
}

std::array<double, 3>
barycentric_coordinates(const Mesh3d& mesh, const gpf::FaceId fid, const Eigen::Vector3d& pt)
{
    const auto points = face_points(mesh, fid);
    Eigen::Matrix<double, 3, 2> basis;
    basis.col(0) = points[1] - points[0];
    basis.col(1) = points[2] - points[0];
    const Eigen::Vector2d uv = (basis.transpose() * basis).ldlt().solve(basis.transpose() * (pt - points[0]));
    return { 1.0 - uv.x() - uv.y(), uv.x(), uv.y() };
}

Eigen::Vector3d
point_from_barycentric(const Mesh3d& mesh, const auto& point)
{
    const auto points = face_points(mesh, point.first);
    Eigen::Vector3d pt = Eigen::Vector3d::Zero();
    for (std::size_t i = 0; i < points.size(); ++i) {
        pt += point.second[i] * points[i];
    }
    return pt;
}

void
assert_valid_barycentric(const auto& point)
{
    double sum = 0.0;
    for (const double coord : point.second) {
        assert(coord > -1e-9);
        assert(coord < 1.0 + 1e-9);
        sum += coord;
    }
    assert(std::abs(sum - 1.0) < 1e-9);
}

bool
is_close(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const double eps = 1e-9)
{
    return (a - b).norm() < eps;
}
} // namespace

void
test_project_polylines_on_mesh_2d_points()
{
    using Mesh = gpf::ManifoldMesh<VertexProp, gpf::Empty, gpf::Empty, gpf::Empty>;

    Mesh mesh = Mesh::new_in(std::vector<std::vector<std::size_t>>{ { 0, 1, 2 } });
    mesh.vertex_prop(gpf::VertexId{ 0 }).pt = { 0.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ 1 }).pt = { 1.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ 2 }).pt = { 0.0, 1.0 };

    std::vector<std::array<double, 2>> points{ { 0.75, 0.0 }, { 0.1, 0.8 } };
    const std::size_t N = 1000;

    std::mt19937 rng(42); // Seed for reproducibility
    std::uniform_real_distribution<double> dist(0.0, 1000.0);

    for (int i = 0; i < N; ++i) {
        const auto a = dist(rng);
        const auto b = dist(rng);
        const auto c = dist(rng);
        const auto sum = a + b + c;
        const auto t1 = a / sum;
        const auto t2 = b / sum;

        points.emplace_back(std::array<double, 2>{ t1, t2 });
    }
    const std::vector<std::vector<std::size_t>> polylines{ { 0, 1 } };

    const auto [point_vertices, paths] = gpf::project_polylines_on_mesh<2>(points, polylines, mesh, 1e-3);
    write_obj("project_mesh.obj", mesh);
    assert(paths.size() == 1);
    assert(!paths.front().empty());

    auto is_close = [](double a, double b) { return std::abs(a - b) < 1e-9; };
    for (std::size_t i = 0; i + 1 < paths.front().size(); ++i) {
        assert(mesh.he_to(paths.front()[i]) == mesh.he_from(paths.front()[i + 1]));
    }

    const auto first_he = mesh.halfedge(paths.front().front());
    const auto last_he = mesh.halfedge(paths.front().back());
    const auto pa = first_he.from().prop().pt;
    const auto pb = last_he.to().prop().pt;
    const bool forward = is_close(pa[0], 0.75) && is_close(pa[1], 0.0) && is_close(pb[0], 0.1) && is_close(pb[1], 0.8);
    const bool backward = is_close(pa[0], 0.1) && is_close(pa[1], 0.8) && is_close(pb[0], 0.75) && is_close(pb[1], 0.0);
    assert(forward || backward);

    assert(is_close(points[0][0], 0.75) && is_close(points[0][1], 0.0));
    assert(is_close(points[1][0], 0.1) && is_close(points[1][1], 0.8));
}

void
test_prepare_projected_points_with_mbvh()
{
    Mesh3d mesh = Mesh3d::new_in(std::vector<std::vector<std::size_t>>{ { 0, 1, 2 }, { 2, 1, 3 } });
    mesh.vertex_prop(gpf::VertexId{ 0 }).pt = { 0.0, 0.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ 1 }).pt = { 1.0, 0.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ 2 }).pt = { 0.0, 1.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ 3 }).pt = { 1.0, 1.0, 0.0 };

    std::vector<std::array<double, 3>> points{
        { 0.25, 0.25, 2.0 },
        { -0.25, -0.25, 1.0 },
        { 0.4, -0.25, 1.0 },
        { 0.75, 0.75, -2.0 },
    };
    const auto [face_info_map, point_vertices, edge_to_points_map] =
      gpf::detail::prepare_projected_points<3>(points, mesh, 1e-6);

    assert(is_close(Eigen::Vector3d::Map(points[0].data()), Eigen::Vector3d{ 0.25, 0.25, 0.0 }));
    assert(is_close(Eigen::Vector3d::Map(points[1].data()), Eigen::Vector3d{ 0.0, 0.0, 0.0 }));
    assert(is_close(Eigen::Vector3d::Map(points[2].data()), Eigen::Vector3d{ 0.4, 0.0, 0.0 }));
    assert(is_close(Eigen::Vector3d::Map(points[3].data()), Eigen::Vector3d{ 0.75, 0.75, 0.0 }));

    assert(!point_vertices[0].valid());
    assert(point_vertices[1] == gpf::VertexId{ 0 });
    assert(!point_vertices[2].valid());
    assert(!point_vertices[3].valid());

    const auto edge_id = mesh.he_edge(mesh.he_from_vertices(gpf::VertexId{ 0 }, gpf::VertexId{ 1 }));
    assert(edge_to_points_map.at(edge_id) == std::vector<std::size_t>{ 2 });

    assert(face_info_map.at(gpf::FaceId{ 0 }).point_indices == std::vector<std::size_t>{ 0 });
    assert(face_info_map.at(gpf::FaceId{ 1 }).point_indices == std::vector<std::size_t>{ 3 });
    for (const auto& face_info : face_info_map | std::views::values) {
        assert(face_info.barycentric_coordinates.size() == face_info.point_indices.size());
        for (const auto& barycentric : face_info.barycentric_coordinates) {
            assert(std::abs(barycentric[0] + barycentric[1] + barycentric[2] - 1.0) < 1e-14);
        }
    }
}

void
test_walk_on_mesh_surface()
{
    auto make_single_triangle = [] {
        Mesh3d mesh = Mesh3d::new_in(std::vector<std::vector<std::size_t>>{ { 0, 1, 2 }, { 2, 1, 3 }, { 0, 2, 3 } });
        mesh.vertex_prop(gpf::VertexId{ 0 }).pt = { -1.0, 0.0, 0.0 };
        mesh.vertex_prop(gpf::VertexId{ 1 }).pt = { 1.0, 0.0, 0.0 };
        mesh.vertex_prop(gpf::VertexId{ 2 }).pt = { 0.0, 1.0, 0.0 };
        mesh.vertex_prop(gpf::VertexId{ 3 }).pt = { 1e-6, 2.0, 0.0 };
        return mesh;
    };

    {
        auto mesh = make_single_triangle();
        const gpf::FaceId fid{ 0 };
        const std::array<double, 3> start_pt{ 0.0, 0.5, 0.0 };
        const std::array<double, 3> direction{ 0.0, 1.0, 0.0 };
        const std::array<double, 4> lengths{ 0.0, 0.1, 0.25, 0.4 };

        const auto result =
          gpf::walk_on_mesh_surface(mesh, fid, start_pt, direction, std::span<const double>{ lengths });
        assert(result.has_value());
        assert(result->size() == lengths.size() + 1);
        const auto& points = *result;
        assert(points[0].first == fid);
        const Eigen::Vector3d expected_start = Eigen::Vector3d::Map(start_pt.data());
        assert(is_close(point_from_barycentric(mesh, points[0]), expected_start, 1e-12));
        const auto expected_barycentric = barycentric_coordinates(mesh, fid, expected_start);
        for (std::size_t i = 0; i < expected_barycentric.size(); ++i) {
            assert(std::abs(points[0].second[i] - expected_barycentric[i]) < 1e-12);
        }
        for (const auto& point : points) {
            assert_valid_barycentric(point);
        }
        for (std::size_t i = 0; i < lengths.size(); ++i) {
            const Eigen::Vector3d expected_pt = expected_start + Eigen::Vector3d::Map(direction.data()) * lengths[i];
            assert(is_close(point_from_barycentric(mesh, points[i + 1]), expected_pt, 1e-12));
        }
    }
    {
        std::vector<double> points{ 0.0, 0.0, 1.0, 0.0, 0.0, 1.0 };
        const std::size_t N = 1000;

        std::mt19937 rng(42); // Seed for reproducibility
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for (int i = 0; i < N; ++i) {
            const auto a = dist(rng);
            const auto b = dist(rng);
            const auto c = dist(rng);
            const auto sum = a + b + c;
            const auto t1 = a / sum;
            const auto t2 = b / sum;

            points.append_range(std::array<double, 2>{ t1, t2 });
        }
        auto triangles = gpf::triangulate_points(points, true);
        assert(triangles.size() % 3 == 0);
        std::vector<std::array<std::size_t, 3>> triangle_faces;
        triangle_faces.reserve(triangles.size() / 3);
        for (std::size_t i = 0; i < triangles.size(); i += 3) {
            triangle_faces.push_back({ triangles[i], triangles[i + 1], triangles[i + 2] });
        }
        Mesh3d mesh = Mesh3d::new_in(triangle_faces);
        for (auto [v, i] : std::views::zip(mesh.vertices(), std::views::iota(std::size_t{ 0 }, mesh.n_vertices()))) {
            v.prop().pt = std::array{ points[2 * i], points[2 * i + 1], 0.0 };
        }

        write_obj("123.obj", mesh);

        const Eigen::Vector3d start_pt(0.75, 0.0005, 0.0);
        const Eigen::Vector3d end_pt(0.1, 0.8, 0.0);
        const Eigen::Vector3d diff = end_pt - start_pt;
        const double total = diff.norm();
        const Eigen::Vector3d direction = diff / total;
        const std::array<double, 5> lengths{ 0.0, total * 0.25, total * 0.5, total * 0.75, total };

        auto ret = gpf::walk_on_mesh_surface(mesh,
                                             gpf::FaceId{ 1325 },
                                             std::span<const double, 3>{ start_pt.data(), 3 },
                                             std::span<const double, 3>{ direction.data(), 3 },
                                             std::span<const double>{ lengths });
        assert(ret.has_value());
        assert(ret->size() == lengths.size() + 1);
        assert_valid_barycentric(ret->front());
        const auto walked_start = point_from_barycentric(mesh, ret->front());
        for (std::size_t i = 0; i < lengths.size(); ++i) {
            const auto& point = (*ret)[i + 1];
            assert_valid_barycentric(point);
            const auto expected_pt = (walked_start + direction * lengths[i]).eval();
            assert(is_close(point_from_barycentric(mesh, point), expected_pt, 1e-7));
        }
    }
}
