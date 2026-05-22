#include <cstdlib>
#include <iostream>

void
test_triangulate_bug1();
void
test_triangulate_bug2();
void
test_triangulate_points_simple();
void
test_triangulate_points_square();
void
test_triangulate_points_pentagon();
void
test_triangulate_10000_random_points();
void
test_cdt_with_intersections();
void
test_property_edge_length_updates();
void
test_project_polylines_on_mesh_2d_points();
void
test_walk_on_mesh_surface();
void
test_mesh_edge_collapse1();
void
test_orthtree_quadtree();
void
test_orthtree_octree();
void
test_orthtree_traversal();

int
main()
{
    test_project_polylines_on_mesh_2d_points();
    test_walk_on_mesh_surface();
    test_mesh_edge_collapse1();
    test_triangulate_bug2();
    test_triangulate_bug1();
    test_cdt_with_intersections();
    test_triangulate_10000_random_points();
    test_triangulate_points_simple();
    test_triangulate_points_square();
    test_triangulate_points_pentagon();
    test_property_edge_length_updates();
    test_orthtree_quadtree();
    test_orthtree_octree();
    test_orthtree_traversal();

    std::cout << "gpf_algorithm_tests: OK\n";
    return EXIT_SUCCESS;
}
