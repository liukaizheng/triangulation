# Closest Point on a Triangle

The triangle calculation in
[`include/gpf/find_closest_points.hpp`](include/gpf/find_closest_points.hpp#L499) solves the constrained minimization

$$
\min_{\alpha,\beta,\gamma}
\left\|X-(\alpha A+\beta B+\gamma C)\right\|^2
$$

subject to

$$
\alpha+\beta+\gamma=1,
\qquad
\alpha,\beta,\gamma\ge 0.
$$

Here, $X$ is the query point and $A$, $B$, and $C$ are the triangle vertices. A triangle has seven
closest-point regions: three vertex regions, three edge regions, and the face-interior region. The implementation tests
these regions directly.

## Dot-product setup

Define the two edges leaving $A$:

$$
e_0=B-A,
\qquad
e_1=C-A.
$$

The implementation computes

$$
\begin{aligned}
d_1 &= e_0\cdot(X-A), & d_2 &= e_1\cdot(X-A),\\
d_3 &= e_0\cdot(X-B), & d_4 &= e_1\cdot(X-B),\\
d_5 &= e_0\cdot(X-C), & d_6 &= e_1\cdot(X-C).
\end{aligned}
$$

These dot products determine whether moving away from a vertex along either adjacent edge decreases the distance to
$X$.

## Vertex regions

At a vertex, the only allowed movements into the triangle are combinations of the two incident edge directions. A
vertex is closest exactly when neither of those movements can decrease the distance to $X$.

### Why the test works at vertex A

Every point $P$ in the triangle can be written relative to $A$ as

$$
P=A+s(B-A)+t(C-A)=A+s e_0+t e_1,
$$

where

$$
s\ge0,
\qquad
t\ge0,
\qquad
s+t\le1.
$$

Let $u=P-A=s e_0+t e_1$. Compare the squared distance from $P$ to $X$ with the squared distance from $A$ to $X$:

$$
\begin{aligned}
\|P-X\|^2-\|A-X\|^2
&=\|(A-X)+u\|^2-\|A-X\|^2\\
&=\|u\|^2+2u\cdot(A-X)\\
&=\|u\|^2-2u\cdot(X-A)\\
&=\|u\|^2-2(sd_1+td_2).
\end{aligned}
$$

Suppose that

$$
d_1\le 0,
\qquad
d_2\le 0.
$$

Because $s,t\ge0$, this implies

$$
sd_1+td_2\le0.
$$

Therefore

$$
\|P-X\|^2-\|A-X\|^2
=\underbrace{\|u\|^2}_{\ge0}
-2\underbrace{(sd_1+td_2)}_{\le0}
\ge0.
$$

This inequality holds for every $P$ in the triangle, so no point in the triangle is closer to $X$ than $A$. Hence $A$
is the closest point.

The converse explains why both conditions are necessary. If $d_1>0$, move a small distance from $A$ toward $B$:

$$
P_\varepsilon=A+\varepsilon e_0.
$$

Then

$$
\|P_\varepsilon-X\|^2-\|A-X\|^2
=\varepsilon^2\|e_0\|^2-2\varepsilon d_1.
$$

For sufficiently small positive $\varepsilon$, the negative linear term dominates the positive quadratic term, so
$P_\varepsilon$ is closer than $A$. Similarly, if $d_2>0$, moving slightly toward $C$ reduces the distance. Thus $A$
is the closest point exactly when $d_1\le0$ and $d_2\le0$.

### Applying the same test at B and C

For any vertex $V$ with adjacent vertices $U$ and $W$, the vertex-region condition is

$$
(U-V)\cdot(X-V)\le0,
\qquad
(W-V)\cdot(X-V)\le0.
$$

These inequalities say that $X-V$ points behind both edge directions available from $V$. The implementation expresses
the tests at all three vertices using the previously defined $d_i$ values:

| Vertex | Directions into the triangle | Direction dot products | Conditions |
| --- | --- | --- | --- |
| $A$ | $B-A=e_0$, $C-A=e_1$ | $d_1$, $d_2$ | $d_1\le0$, $d_2\le0$ |
| $B$ | $A-B=-e_0$, $C-B=e_1-e_0$ | $-d_3$, $d_4-d_3$ | $d_3\ge0$, $d_4\le d_3$ |
| $C$ | $A-C=-e_1$, $B-C=e_0-e_1$ | $-d_6$, $d_5-d_6$ | $d_6\ge0$, $d_5\le d_6$ |

For example, the two conditions at $B$ come from

$$
\begin{aligned}
(A-B)\cdot(X-B)&=-d_3\le0 &&\Longleftrightarrow d_3\ge0,\\
(C-B)\cdot(X-B)&=d_4-d_3\le0 &&\Longleftrightarrow d_4\le d_3.
\end{aligned}
$$

Geometrically, draw through a vertex one line perpendicular to each incident edge. These lines bound an outward wedge.
When the projection of $X$ onto the triangle's plane lies in that wedge, $X$ is behind both incident edges and the
corner is closest. If either condition fails, moving a short distance along the corresponding edge direction decreases
the distance, so the corner cannot be closest. Components of $X$ perpendicular to the triangle's plane do not affect
these tests because their dot product with every triangle edge is zero.

## Edge regions

Consider edge $AB$. A point on its supporting line can be written as

$$
P=A+t(B-A).
$$

The orthogonal projection of $X$ onto that line has parameter

$$
t=\frac{(B-A)\cdot(X-A)}{\|B-A\|^2}.
$$

Because

$$
d_1-d_3=\|B-A\|^2,
$$

the implementation evaluates it as

$$
t=\frac{d_1}{d_1-d_3}.
$$

The conditions $d_1\ge0$ and $d_3\le0$ ensure $0\le t\le1$, so the projected point is on the segment rather than
beyond an endpoint. The three edge projections are

$$
\begin{aligned}
AB:\quad&P=A+\frac{d_1}{d_1-d_3}(B-A),\\
AC:\quad&P=A+\frac{d_2}{d_2-d_6}(C-A),\\
BC:\quad&P=B+
\frac{d_4-d_3}{(d_4-d_3)+(d_5-d_6)}(C-B).
\end{aligned}
$$

## Selecting the correct edge or face

The implementation calculates

$$
\begin{aligned}
v_a &= d_3d_6-d_5d_4,\\
v_b &= d_5d_2-d_1d_6,\\
v_c &= d_1d_4-d_3d_2.
\end{aligned}
$$

For a nondegenerate triangle, these quantities are proportional to the barycentric coordinates of the orthogonal
projection of $X$ onto the triangle's plane:

$$
\alpha=\frac{v_a}{S},
\qquad
\beta=\frac{v_b}{S},
\qquad
\gamma=\frac{v_c}{S},
\qquad
S=v_a+v_b+v_c.
$$

Their signs identify which triangle boundary is closest:

- $v_c\le0$: the projection is outside across edge $AB$.
- $v_b\le0$: the projection is outside across edge $AC$.
- $v_a\le0$: the projection is outside across edge $BC$.
- $v_a$, $v_b$, and $v_c$ all positive: the projection is inside the triangle face.

The implementation combines these sign checks with the segment-range checks described above. Vertex tests handle cases
where the projected point lies beyond an edge endpoint.

## Face-interior region

If none of the vertex or edge tests succeeds, the orthogonal projection lies inside the triangle. Its barycentric
coordinates are

$$
\beta=\frac{v_b}{v_a+v_b+v_c},
\qquad
\gamma=\frac{v_c}{v_a+v_b+v_c},
\qquad
\alpha=1-\beta-\gamma.
$$

The closest point is therefore

$$
P=A+\beta(B-A)+\gamma(C-A)
$$

or, equivalently,

$$
P=\alpha A+\beta B+\gamma C.
$$

## Stored barycentric coordinates

`interaction.uv` stores the first two barycentric weights:

$$
uv=(\alpha,\beta),
\qquad
\gamma=1-\alpha-\beta.
$$

Consequently, the three vertices have these values:

| Point | `uv` |
| --- | --- |
| $A$ | $(1,0)$ |
| $B$ | $(0,1)$ |
| $C$ | $(0,0)$ |

After finding $P$ for a triangle, the implementation computes

$$
\|X-P\|^2
$$

and retains the result only when it improves the current closest distance. The surrounding bounding-volume hierarchy
eliminates groups of triangles whose bounding boxes cannot contain a closer point; it does not change the per-triangle
calculation.

Because the algorithm uses dot products rather than a three-dimensional cross product, it works for a triangle embedded
in any $N$-dimensional Euclidean space. It assumes that the triangle is nondegenerate; for a degenerate triangle,
$v_a+v_b+v_c$ can be zero and the triangle should instead be treated as one or more line segments.
