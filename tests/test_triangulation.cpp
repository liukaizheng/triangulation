#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <gpf/triangulation.hpp>

void
write_obj(const std::string& name, const std::vector<double>& points, const std::vector<std::size_t>& triangles)
{
    std::ofstream out(name);
    for (std::size_t i = 0; i < points.size(); i += 2) {
        out << "v " << points[i] << " " << points[i + 1] << " 0\n";
    }
    for (std::size_t i = 0; i < triangles.size(); i += 3) {
        out << "f " << triangles[i] + 1 << " " << triangles[i + 1] + 1 << " " << triangles[i + 2] + 1 << "\n";
    }
    out.close();
}

void
test_degenerate()
{
    std::vector<double> points{ 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0 };
    std::vector<std::size_t> segments{ 0, 1, 1, 2, 2, 3, 3, 0 };
    auto triangles = gpf::triangulate_polygon(points, segments, true);
    assert(triangles.size() == 0);
}

void
test_triangulate_bug2()
{
    std::vector<double> points{ 0.0,
                                0.0,
                                0.14763533445827418,
                                9.1838131789240771E-16,
                                0.28476798756884003,
                                6.9193097214003481E-15,
                                1.2257306612987737,
                                -4.2993281473632734E-17,
                                0.22469289424128275,
                                0.068038519204616399 };
    auto triangles = gpf::triangulate_points(points, true);
    assert(triangles.size() == 15);
}

void
test_triangulate_bug1()
{
    std::vector<double> points{ 0.43336197097855234,
                                0.43336197097855234,
                                0.32502147823393557,
                                0.32502147823393557,
                                0.0,
                                0.0,
                                0.65004295646789778,
                                0.0,
                                1.0395797536276792,
                                0.0,
                                1.0395797536276792,
                                0.43336197097858076,
                                1.0395797536276792,
                                1.0395797536276916 };
    auto triangles =
      gpf::triangulate_polygon(points, std::vector<std::size_t>{ 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 0 }, 6);
    assert(triangles.size() == 15);
}

void
test_triangulate_10000_random_points()
{
    const int N = 10000;
    std::vector<double> points;
    points.reserve(N * 2);

    std::mt19937 rng(42); // Seed for reproducibility
    std::uniform_real_distribution<double> dist(0.0, 1000.0);

    for (int i = 0; i < N; ++i) {
        points.push_back(dist(rng));
        points.push_back(dist(rng));
    }

    auto start = std::chrono::high_resolution_clock::now();

    auto triangles = gpf::triangulate_points(points, true);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Triangulation of " << N << " random points took " << duration.count() << " ms\n";
    std::cout << "Generated " << triangles.size() / 3 << " triangles\n";
    write_obj("123.obj", points, triangles);
}

void
test_triangulate_points_simple()
{
    // Test: Triangulate 3 points (simple triangle)
    std::vector<double> points = {
        0.0, 0.0, // point 0
        1.0, 0.0, // point 1
        0.5, 1.0, // point 2
    };

    auto triangles = gpf::triangulate_points(points, true);

    // Should produce exactly 1 triangle with 3 indices
    assert(triangles.size() == 3);

    // Verify all indices are valid (0, 1, or 2)
    for (std::size_t idx : triangles) {
        assert(idx < 3);
    }
}

void
test_triangulate_points_square()
{
    // Test: Triangulate 4 points (square)
    std::vector<double> points = {
        -0.37532790381475534, -0.5701866788379338,  -0.3746855466249469,  -0.5701664269221419, -0.37421247631072985,
        -0.5692436605107906,  -0.37385918384824846, -0.5691439523641123,  -0.3710912705614987, -0.5629674306723101,
        -0.37098555925350085, -0.5609913087007848,  -0.37096292038201095, -0.5597273368099648,
    };

    auto triangles = gpf::triangulate_points(points, true);
    assert(triangles.size() == 18);
}

void
test_triangulate_points_pentagon()
{
    // Test: Triangulate 5 points (regular pentagon-ish)
    std::vector<double> points = {
        0.5, 0.0, // point 0
        1.0, 0.4, // point 1
        0.8, 1.0, // point 2
        0.2, 1.0, // point 3
        0.0, 0.4, // point 4
    };

    auto triangles = gpf::triangulate_points(points, true);

    // Should produce 3 triangles with 9 indices
    assert(triangles.size() == 9);

    // Verify all indices are valid (0-4)
    for (std::size_t idx : triangles) {
        assert(idx < 5);
    }
}

void
test_alternate_axes()
{
    // Test: Verify alternate_axes produces a valid ordering
    std::vector<double> points = {
        0.0, 0.0, 1.0, 0.0, 0.5, 1.0, 0.0, 1.0, 1.0, 1.0,
    };

    std::vector<gpf::VertexId> indices;
    for (std::size_t i = 0; i < 5; ++i) {
        indices.emplace_back(gpf::VertexId{ i });
    }

    gpf::triangulation::alternate_axes(points.data(), std::span<gpf::VertexId>(indices), true);

    // Verify all original indices are present
    std::vector<bool> found(5, false);
    for (const auto& vid : indices) {
        assert(vid.idx < 5);
        found[vid.idx] = true;
    }
    for (bool b : found) {
        assert(b);
    }
}

void
test_cdt_with_intersections()
{
    const int N = 60;
    std::vector<double> points;
    points.reserve(N * 2);

    std::mt19937 rng(42); // Seed for reproducibility
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < N; ++i) {
        points.push_back(dist(rng));
        points.push_back(dist(rng));
    }

    std::vector<std::size_t> segments;
    segments.reserve(N * (N - 1) / 2);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            segments.push_back(i);
            segments.push_back(j);
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    auto [new_points, triangles] = gpf::triangulate_with_new_points(points, segments, true);
    points.append_range(std::move(new_points));

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Triangulation of " << N << " random points took " << duration.count() << " ms\n";
    std::cout << "Generated " << triangles.size() / 3 << " triangles\n";
    write_obj("124.obj", points, triangles);
}
