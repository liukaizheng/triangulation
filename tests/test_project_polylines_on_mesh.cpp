#include <gpf/manifold_mesh.hpp>
#include <gpf/project_polylines_on_mesh.hpp>

#include <cassert>
#include <print>
#include <random>

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
        points[idx++] = Eigen::Vector3d::Map(mesh.vertex_prop(he.to().id).pt.data());
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

    const auto [point_vertices, paths] = gpf::project_polylines_on_mesh<2>(points, polylines, mesh);
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
test_walk_on_mesh_surface()
{
    auto make_single_triangle = [] {
        Mesh3d mesh = Mesh3d::new_in(std::vector<std::vector<std::size_t>>{ { 0, 1, 2 } });
        mesh.vertex_prop(gpf::VertexId{ 0 }).pt = { -1.0, 0.0, 0.0 };
        mesh.vertex_prop(gpf::VertexId{ 1 }).pt = { 1.0, 0.0, 0.0 };
        mesh.vertex_prop(gpf::VertexId{ 2 }).pt = { 0.0, 1.0, 0.0 };
        return mesh;
    };

    {
        auto mesh = make_single_triangle();
        const gpf::FaceId fid{ 0 };
        std::array<double, 3> start_pt{ 0.0, 0.5, 0.0 };
        const std::array<double, 3> direction{ 0.0, 1.0, 0.0 };

        const auto result = gpf::walk_on_mesh_surface(mesh, fid, start_pt, direction);
    }
}
