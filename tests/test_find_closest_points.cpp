#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <print>
#include <random>
#include <variant>
#include <vector>

#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <Eigen/Core>

#include <gpf/find_closest_points.hpp>

#include "read_off.hpp"

void
test_build_bvh()
{
    using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Point = Kernel::Point_3;
    using Triangle = Kernel::Triangle_3;
    using TreeIterator = std::vector<Triangle>::const_iterator;
    using TreePrimitive = CGAL::AABB_triangle_primitive_3<Kernel, TreeIterator>;
    using TreeTraits = CGAL::AABB_traits_3<Kernel, TreePrimitive>;
    using Tree = CGAL::AABB_tree<TreeTraits>;

    const auto data = read_off("dragon.off");
    const auto points = std::span<const double>{ &data.vertices[0][0], data.vertices.size() * 3 };
    std::vector<std::size_t> triangles;
    triangles.reserve(data.faces.size() * 3);
    triangles.append_range(data.faces | std::views::join);

    const auto bvh = gpf::bvh::build_bvh<3, double>(points, triangles);
    const gpf::mbvh::MBVHTree<3, double> mbvh{ bvh };

    std::vector<Triangle> cgal_triangles;
    cgal_triangles.reserve(triangles.size() / 3);
    for (std::size_t i = 0; i < triangles.size(); i += 3) {
        std::array<Point, 3> triangle_points;
        for (std::size_t j = 0; j < triangle_points.size(); ++j) {
            const auto* point = points.data() + triangles[i + j] * 3;
            triangle_points[j] = Point{ point[0], point[1], point[2] };
        }
        cgal_triangles.emplace_back(triangle_points[0], triangle_points[1], triangle_points[2]);
    }
    Tree cgal_tree(cgal_triangles.cbegin(), cgal_triangles.cend());
    cgal_tree.accelerate_distance_queries();

    const auto& bbox = bvh.nodes.front().bbox;
    std::array<std::uniform_real_distribution<double>, 3> distributions{
        std::uniform_real_distribution<double>{ bbox.min_coord(0), bbox.max_coord(0) },
        std::uniform_real_distribution<double>{ bbox.min_coord(1), bbox.max_coord(1) },
        std::uniform_real_distribution<double>{ bbox.min_coord(2), bbox.max_coord(2) },
    };
    std::mt19937 rng{ 42 };
    constexpr std::size_t kNumQueries = 1000000;
    constexpr double kTolerance = 1e-12;
    std::vector<double> all_points;
    all_points.reserve(kNumQueries * 3);
    for (std::size_t i = 0; i < kNumQueries; ++i) {
        all_points.append_range(std::array{ distributions[0](rng), distributions[1](rng), distributions[2](rng) });
    }
    std::cout << "p0: [" << all_points[0] << ", " << all_points[1] << ", " << all_points[2] << "]" << std::endl;
    std::vector<double> closest_points;
    closest_points.reserve(kNumQueries * 3);
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n_success = 0;
    for (std::size_t i = 0; i < kNumQueries; ++i) {
        const std::array<double, 3> query{ all_points[i * 3], all_points[i * 3 + 1], all_points[i * 3 + 2] };
        gpf::BoundingSphere<3, double> sphere{ std::span<const double, 3>{ all_points.data() + i * 3, 3 } };
        gpf::Interaction<3, double> interaction;
        if (mbvh.find_closest_point_from_node(sphere, interaction, 0)) {
            n_success += 1;
            closest_points.append_range(interaction.p);
        }
    }
    std::cout << "MBVH time: "
              << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count()
              << "ms" << std::endl;
    if (n_success != kNumQueries) {
        std::cout << "MBVH failed: " << n_success << " of " << kNumQueries << " queries succeeded" << std::endl;
    }
    start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < kNumQueries; ++i) {
        const std::array<double, 3> query{ all_points[i * 3], all_points[i * 3 + 1], all_points[i * 3 + 2] };
        const auto closest_ret = cgal_tree.closest_point_and_primitive(Point{ query[0], query[1], query[2] });
        const auto fid = std::distance(cgal_triangles.cbegin(), closest_ret.second);
        assert(fid >= 0);
        assert(static_cast<std::size_t>(fid) < cgal_triangles.size());
        const double* cp = closest_points.data() + i * 3;
        for (std::size_t dim = 0; dim < query.size(); ++dim) {
            if (std::abs(cp[dim] - closest_ret.first[dim]) > kTolerance) {
                std::cout << "MBVH failed: " << i << "th query: " << cp[dim] << " vs " << closest_ret.first[dim]
                          << std::endl;
            }
        }
    }
    std::cout << "CGAL time: "
              << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count()
              << "ms" << std::endl;
}
