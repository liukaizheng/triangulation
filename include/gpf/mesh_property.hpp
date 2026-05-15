#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <numbers>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

#include <gpf/handles.hpp>

#include <Eigen/Dense>

namespace gpf {

inline double
triangle_area(const double lab, const double lbc, const double lca) noexcept
{
    const double s = (lab + lbc + lca) / 2.0;
    return std::sqrt(std::max(0.0, s * (s - lab) * (s - lbc) * (s - lca)));
}

inline auto
triangle_apex_from_base_lengths(const double lab, const double lbc, const double lca, bool reversed) noexcept
{
    const auto h = triangle_area(lab, lbc, lca) / lab * 2.0;
    if (reversed) {
        auto w = (lbc * lbc + lab * lab - lca * lca) / lab * 0.5;
        return Eigen::Vector2d{ w, -h };
    } else {
        const auto w = (lca * lca + lab * lab - lbc * lbc) / lab * 0.5;
        return Eigen::Vector2d{ w, h };
    }
}

template<typename Mesh>
inline constexpr std::size_t mesh_position_dim_v =
  std::tuple_size_v<std::remove_cvref_t<decltype(std::declval<Mesh&>().vertex_prop(VertexId{ 0 }).pt)>>;

// Concept for vertex with position property that can be viewed as span<const double, N>
template<typename V, std::size_t N>
concept HasPositionProperty = requires(V v) {
    { std::span<const double, N>{ v.prop().pt } };
};

// Concept for edge with squared length property
template<typename E>
concept HasSquaredLengthProperty = requires(E e) {
    { e.prop().square_len } -> std::convertible_to<double>;
};

// Concept for edge with length property
template<typename E>
concept HasLengthProperty = requires(E e) {
    { e.prop().len } -> std::convertible_to<double>;
};

// Concept for halfedge with angle property
template<typename H>
concept HasAngleProperty = requires(H h) {
    { h.prop().angle } -> std::convertible_to<double>;
};

template<typename H>
concept HasSignpostAngleProperty = requires(H h) {
    { h.prop().signpost_angle } -> std::convertible_to<double>;
};

template<typename H>
concept HasVectorProperty = requires(H h) {
    { h.prop().vector } -> std::convertible_to<std::span<double, 2>>;
};

// Concept for vertex with angle sum property
template<typename V>
concept HasAngleSumProperty = requires(V v) {
    { v.prop().angle_sum } -> std::convertible_to<double>;
};

template<std::size_t N, typename Mesh>
    requires HasPositionProperty<VertexHandle<Mesh, false>, N>
[[nodiscard]] std::span<const double, N>
position_span(VertexHandle<Mesh, false> v)
{
    return std::span<const double, N>{ v.prop().pt };
}

template<std::size_t N>
[[nodiscard]] double
squared_distance(std::span<const double, N> a, std::span<const double, N> b)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

template<std::size_t N, class F, class Arg>
concept ReturnsPositionSpan = requires(F f, Arg arg) {
    { f(arg) } -> std::same_as<std::span<const double, N>>;
};

template<std::size_t N, class Mesh, class PositionSpan>
    requires ReturnsPositionSpan<N, PositionSpan, VertexHandle<Mesh, false>>
void
update_edge_length_squared(gpf::EdgeHandle<Mesh, false> edge, PositionSpan pos_span)
{
    const auto [va, vb] = edge.vertices();
    const auto pa = pos_span(va);
    const auto pb = pos_span(vb);
    edge.prop().square_len = squared_distance(pa, pb);
}

template<std::size_t N, class Mesh>
    requires HasPositionProperty<VertexHandle<Mesh, false>, N> && HasSquaredLengthProperty<EdgeHandle<Mesh, false>>
void
update_edge_length_squared(gpf::EdgeHandle<Mesh, false> edge)
{
    update_edge_length_squared<N>(edge, position_span<N, Mesh>);
}

template<std::size_t N, class Mesh, class PositionSpan>
    requires ReturnsPositionSpan<N, PositionSpan, VertexHandle<Mesh, false>>
void
update_edge_lengths_squared(Mesh& mesh, PositionSpan pos_span)
{
    for (auto e : mesh.edges()) {
        update_edge_length_squared<N>(e, pos_span);
    }
}

template<std::size_t N, class Mesh>
    requires HasPositionProperty<VertexHandle<Mesh, false>, N> && HasSquaredLengthProperty<EdgeHandle<Mesh, false>>
void
update_edge_lengths_squared(Mesh& mesh)
{
    update_edge_lengths_squared<N>(mesh, position_span<N, Mesh>);
}

template<std::size_t N, class Mesh, class PositionSpan>
    requires ReturnsPositionSpan<N, PositionSpan, VertexHandle<Mesh, false>>
void
update_edge_length(gpf::EdgeHandle<Mesh, false> edge, PositionSpan pos_span)
{
    const auto [va, vb] = edge.vertices();
    const auto pa = pos_span(va);
    const auto pb = pos_span(vb);
    edge.prop().len = std::sqrt(squared_distance(pa, pb));
}

template<std::size_t N, class Mesh>
    requires HasPositionProperty<VertexHandle<Mesh, false>, N> && HasLengthProperty<EdgeHandle<Mesh, false>>
void
update_edge_length(gpf::EdgeHandle<Mesh, false> edge)
{
    update_edge_length<N>(edge, position_span<N, Mesh>);
}

template<std::size_t N, class Mesh, class PositionSpan>
    requires ReturnsPositionSpan<N, PositionSpan, VertexHandle<Mesh, false>>
void
update_edge_lengths(Mesh& mesh, PositionSpan pos_span)
{
    for (auto e : mesh.edges()) {
        update_edge_length<N>(e, pos_span);
    }
}

template<std::size_t N, class Mesh>
    requires HasPositionProperty<VertexHandle<Mesh, false>, N> && HasLengthProperty<EdgeHandle<Mesh, false>>
void
update_edge_lengths(Mesh& mesh)
{
    update_edge_lengths<N>(mesh, position_span<N, Mesh>);
}

template<class Mesh>
    requires HasLengthProperty<EdgeHandle<Mesh, false>> && HasAngleProperty<HalfedgeHandle<Mesh, false>>
void
update_corner_angle(gpf::HalfedgeHandle<Mesh, false> hab,
                    gpf::HalfedgeHandle<Mesh, false> hbc,
                    gpf::HalfedgeHandle<Mesh, false> hca)
{
    auto lab = hab.edge().prop().len;
    auto lbc = hbc.edge().prop().len;
    auto lca = hca.edge().prop().len;
    auto q = (lab * lab + lbc * lbc - lca * lca) / (2.0 * lab * lbc);
    hbc.prop().angle = std::acos(std::max(-1.0, std::min(q, 1.0)));
}

template<class Mesh>
    requires HasLengthProperty<EdgeHandle<Mesh, false>> && HasAngleProperty<HalfedgeHandle<Mesh, false>>
void
update_corner_angles_on_face(gpf::FaceHandle<Mesh, false> face)
{
    auto ha = face.halfedge();
    auto hb = ha.next();
    auto hc = hb.next();
    assert(hc.next().id == ha.id);
    update_corner_angle(ha, hb, hc);
    update_corner_angle(hb, hc, ha);
    update_corner_angle(hc, ha, hb);
}

template<class Mesh>
    requires HasLengthProperty<EdgeHandle<Mesh, false>> && HasAngleProperty<HalfedgeHandle<Mesh, false>>
void
update_corner_angles(Mesh& mesh)
{
    for (auto face : mesh.faces()) {
        update_corner_angles_on_face(face);
    }
}

template<class Mesh>
    requires HasAngleProperty<HalfedgeHandle<Mesh, false>> && HasAngleSumProperty<VertexHandle<Mesh, false>>
void
update_vertex_angle_sum(gpf::VertexHandle<Mesh, false> vertex)
{
    double sum = 0.0;
    for (auto he : vertex.incoming_halfedges()) {
        sum += he.twin().prop().angle;
    }
    vertex.prop().angle_sum = sum;
}

template<class Mesh>
    requires HasAngleProperty<HalfedgeHandle<Mesh, false>> && HasAngleSumProperty<VertexHandle<Mesh, false>>
void
update_vertex_angle_sums(Mesh& mesh)
{
    for (auto vertex : mesh.vertices()) {
        update_vertex_angle_sum(vertex);
    }
}

template<class Mesh>
    requires HasSignpostAngleProperty<HalfedgeHandle<Mesh, false>> && HasAngleProperty<HalfedgeHandle<Mesh, false>> &&
             HasAngleSumProperty<VertexHandle<Mesh, false>>
void
update_halfedge_signpost_angles_at_vertex(VertexHandle<Mesh, false> vertex)
{
    auto sum = 0.0;
    for (auto he_twin : vertex.incoming_halfedges()) {
        auto he = he_twin.twin();
        auto& prop = he.prop();
        prop.signpost_angle = sum;
        sum += prop.angle;
    }
}

template<class Mesh>
    requires HasSignpostAngleProperty<HalfedgeHandle<Mesh, false>> && HasAngleProperty<HalfedgeHandle<Mesh, false>> &&
             HasAngleSumProperty<VertexHandle<Mesh, false>>
void
update_halfedge_signpost_angles(Mesh& mesh)
{
    for (const auto vertex : mesh.vertices()) {
        update_halfedge_signpost_angles_at_vertex(vertex);
    }
}

template<class Mesh>
    requires HasSignpostAngleProperty<HalfedgeHandle<Mesh, false>> && HasLengthProperty<EdgeHandle<Mesh, false>> &&
             HasAngleSumProperty<VertexHandle<Mesh, false>> && HasVectorProperty<HalfedgeHandle<Mesh, false>>
void
update_halfedge_vector(HalfedgeHandle<Mesh, false> halfedge)
{
    constexpr auto PI_2 = std::numbers::pi * 2;
    auto& he_prop = halfedge.prop();
    const auto angle = he_prop.signpost_angle / halfedge.from().prop().angle_sum * PI_2;
    std::span<double, 2> vec = he_prop.vector;
    auto edge_len = halfedge.edge().prop().len;
    vec[0] = edge_len * std::cos(angle);
    vec[1] = edge_len * std::sin(angle);
}

template<class Mesh>
void
update_halfedge_vectors(Mesh& mesh)
{
    for (auto he : mesh.halfedges()) {
        if (!mesh.he_is_boundary(he.id)) {
            update_halfedge_vector(he);
        }
    }
}

} // namespace gpf
