use crate::*;
use uv::DVec3;

// Extend a vector with some element(s)
// ```
// extend![..foo, 4, 5, 6]
// ```
#[doc(hidden)]
#[macro_export]
macro_rules! extend {
    (..$v:expr, $($new:expr),*) => {{
        let mut tmp = $v.clone();
        $(
        tmp.push($new);
        )*
        tmp
    }}
}

#[inline]
pub(crate) fn _to_vadd(positions: &PointsSlice, v: &Vector) -> Points {
    positions.par_iter().map(|p| *p + *v).collect()
}

#[inline]
pub(crate) fn vadd(positions: &mut Points, v: &Vector) {
    positions.par_iter_mut().for_each(|p| *p += *v);
}

#[inline]
pub(crate) fn centroid(positions: &PointsSlice) -> Point {
    positions
        .iter()
        .fold(Point::zero(), |accumulate, point| accumulate + *point)
        //.into_par_iter()
        //.cloned()
        //.reduce(|| Point::zero(), |accumulate, point| accumulate + point);
        / positions.len() as Float
}

#[inline]
pub(crate) fn centroid_ref(positions: &PointsRefSlice) -> Point {
    positions
        .iter()
        .fold(Point::zero(), |accumulate, point| accumulate + **point)
        / positions.len() as Float
}

// Centroid projected onto the spherical surface that passes to the average of
// the given positions with the center at the origin.
#[inline]
pub(crate) fn _centroid_spherical_ref(
    positions: &PointsRefSlice,
    spherical: Float,
) -> Point {
    let point: Point = positions
        .iter()
        .fold(Point::zero(), |sum, point| sum + **point)
        / positions.len() as Float;

    if spherical != 0.0 {
        let avg_mag =
            positions.iter().fold(0.0, |sum, point| sum + point.mag())
                / positions.len() as Float;

        point * ((1.0 - spherical) + spherical * (point.mag() / avg_mag))
    } else {
        point
    }
}

/// Return the ordered edges containing v
pub(crate) fn vertex_edges(v: VertexKey, edges: &EdgesSlice) -> Edges {
    edges
        .iter()
        .filter_map(|edge| {
            if edge[0] == v || edge[1] == v {
                Some(*edge)
            } else {
                None
            }
        })
        .collect()
}

#[inline]
pub(crate) fn ordered_vertex_edges_recurse(
    v: VertexKey,
    vfaces: &FacesSlice,
    face: &FaceSlice,
    k: usize,
) -> Edges {
    if k < vfaces.len() {
        let i = index_of(&v, face).unwrap();
        /*match index_of(&v, face) {
            Some(i) => i,
            None => return vec![],
        };*/
        let j = (i + face.len() - 1) % face.len();
        let edge = [v, face[j]];
        let nface = face_with_edge(&edge, vfaces);
        let mut result = vec![edge];
        result.extend(ordered_vertex_edges_recurse(v, vfaces, &nface, k + 1));
        result
    } else {
        vec![]
    }
}

#[inline]
pub(crate) fn ordered_vertex_edges(v: VertexKey, vfaces: &FacesSlice) -> Edges {
    if vfaces.is_empty() {
        vec![]
    } else {
        let face = &vfaces[0];
        let i = index_of(&v, face).unwrap();
        let j = (i + face.len() - 1) % face.len();
        let edge = [v, face[j]];
        let nface = face_with_edge(&edge, vfaces);
        let mut result = vec![edge];
        result.extend(ordered_vertex_edges_recurse(v, vfaces, &nface, 1));
        result
    }
}

#[inline]
pub(crate) fn positions_to_faces(
    positions: &PointsSlice,
    face_index: &FacesSlice,
) -> Faces {
    positions
        .par_iter()
        .enumerate()
        .map(|vertex| {
            // Each old vertex creates a new face.
            ordered_vertex_faces(
                vertex.0 as VertexKey,
                &vertex_faces(vertex.0 as VertexKey, face_index),
            )
            .iter()
            .map(|original_face|
                // With vertex faces in left-hand order.
                index_of(original_face, face_index).unwrap() as VertexKey)
            .collect()
        })
        .collect()
}

#[inline]
pub(crate) fn distinct_edge(edge: &Edge) -> Edge {
    if edge[0] < edge[1] {
        *edge
    } else {
        [edge[1], edge[0]]
    }
}

#[inline]
pub(crate) fn distinct_face_edges(face: &FaceSlice) -> Edges {
    face.iter()
        .circular_tuple_windows::<(_, _)>()
        .map(|t| {
            if t.0 < t.1 {
                [*t.0, *t.1]
            } else {
                [*t.1, *t.0]
            }
        })
        .collect()
}

#[inline]
pub(crate) fn _to_centroid_positions(positions: &PointsSlice) -> Points {
    _to_vadd(positions, &-centroid(positions))
}

#[inline]
pub(crate) fn center_on_centroid(positions: &mut Points) {
    vadd(positions, &-centroid(positions));
}

#[inline]
pub(crate) fn vnorm(positions: &PointsSlice) -> Vec<Float> {
    positions.par_iter().map(|v| v.mag()).collect()
}
// Was: average_norm
#[inline]
pub(crate) fn _average_magnitude(positions: &PointsSlice) -> Float {
    vnorm(positions).par_iter().sum::<Float>() / positions.len() as Float
}

#[inline]
pub(crate) fn max_magnitude(positions: &PointsSlice) -> Float {
    vnorm(positions)
        .into_par_iter()
        .reduce(|| Float::NAN, Float::max)
}

/// Returns a [`Faces`] of faces
/// containing `vertex_number`.
#[inline]
pub(crate) fn vertex_faces(
    vertex_number: VertexKey,
    face_index: &FacesSlice,
) -> Faces {
    face_index
        .par_iter()
        .filter(|face| face.contains(&vertex_number))
        .cloned()
        .collect()
}

/// Returns a [`Vec`] of anticlockwise
/// ordered edges.
pub(crate) fn _ordered_face_edges_(face: &FaceSlice) -> Edges {
    face.iter()
        .circular_tuple_windows::<(_, _)>()
        .map(|edge| [*edge.0, *edge.1])
        .collect()
}

/// Returns a [`Vec`] of anticlockwise
/// ordered edges.
#[inline]
pub(crate) fn ordered_face_edges(face: &FaceSlice) -> Edges {
    (0..face.len())
        .map(|i| [face[i], face[(i + 1) % face.len()]])
        .collect()
}

#[inline]
pub(crate) fn face_with_edge(edge: &Edge, faces: &FacesSlice) -> Face {
    let result = faces
        .par_iter()
        .filter(|face| ordered_face_edges(face).contains(edge))
        .flatten()
        .cloned()
        .collect();
    result
}

#[inline]
pub(crate) fn index_of<T: PartialEq>(element: &T, list: &[T]) -> Option<usize> {
    list.iter().position(|e| *e == *element)
}

/// Used internally by [`ordered_vertex_faces()`].
#[inline]
pub(crate) fn ordered_vertex_faces_recurse(
    v: VertexKey,
    face_index: &FacesSlice,
    cface: &FaceSlice,
    k: VertexKey,
) -> Faces {
    if (k as usize) < face_index.len() {
        let i = index_of(&v, cface).unwrap() as i32;
        let j = ((i - 1 + cface.len() as i32) % cface.len() as i32) as usize;
        let edge = [v, cface[j]];
        let mut nfaces = vec![face_with_edge(&edge, face_index)];
        nfaces.extend(ordered_vertex_faces_recurse(
            v,
            face_index,
            &nfaces[0],
            k + 1,
        ));
        nfaces
    } else {
        Faces::new()
    }
}

#[inline]
pub(crate) fn ordered_vertex_faces(
    vertex_number: VertexKey,
    face_index: &FacesSlice,
) -> Faces {
    let mut result = vec![face_index[0].clone()];
    result.extend(ordered_vertex_faces_recurse(
        vertex_number,
        face_index,
        &face_index[0],
        1,
    ));

    result
}

#[inline]
pub(crate) fn edge_length(edge: &Edge, positions: &PointsSlice) -> Float {
    let edge = vec![edge[0], edge[1]];
    let positions = index_as_positions(&edge, positions);
    (*positions[0] - *positions[1]).mag()
}

#[inline]
pub(crate) fn _edge_lengths(
    edges: &_EdgeSlice,
    positions: &PointsSlice,
) -> Vec<Float> {
    edges
        .par_iter()
        .map(|edge| edge_length(edge, positions))
        .collect()
}

#[inline]
pub(crate) fn face_edges(
    face: &FaceSlice,
    positions: &PointsSlice,
) -> Vec<Float> {
    ordered_face_edges(face)
        .par_iter()
        .map(|edge| edge_length(edge, positions))
        .collect()
}

#[inline]
pub(crate) fn _circumscribed_resize(positions: &mut Points, radius: Float) {
    center_on_centroid(positions);
    let average = _average_magnitude(positions);

    positions
        .par_iter_mut()
        .for_each(|v| *v *= radius / average);
}

pub(crate) fn max_resize(positions: &mut Points, radius: Float) {
    center_on_centroid(positions);
    let max = max_magnitude(positions);

    positions.par_iter_mut().for_each(|v| *v *= radius / max);
}

#[inline]
pub(crate) fn _project_on_sphere(positions: &mut Points, radius: Float) {
    positions
        .par_iter_mut()
        .for_each(|point| *point = radius * point.normalized());
}

#[inline]
pub(crate) fn face_irregularity(
    face: &FaceSlice,
    positions: &PointsSlice,
) -> Float {
    let lengths = face_edges(face, positions);
    // The largest value in lengths or NaN (0./0.) otherwise.
    lengths.par_iter().cloned().reduce(|| Float::NAN, Float::max)
        // divide by the smallest value in lengths or NaN (0./0.) otherwise.
        / lengths.par_iter().cloned().reduce(|| Float::NAN, Float::min)
}

#[inline]
pub(crate) fn index_as_positions<'a>(
    f: &[VertexKey],
    positions: &'a PointsSlice,
) -> Vec<&'a Point> {
    f.par_iter()
        .map(|index| &positions[*index as usize])
        .collect()
}

#[inline]
pub(crate) fn _planar_area_ref(positions: &PointsRefSlice) -> Float {
    let sum = positions
        .iter()
        .circular_tuple_windows::<(_, _)>()
        //.take(positions.len() -1)
        .fold(Vector::zero(), |sum, position| {
            sum + position.0.cross(**position.1)
        });

    average_normal_ref(positions).unwrap().dot(sum).abs() * 0.5
}

#[inline]
fn _sig_figs(val: Float) -> [u8; 4] {
    val.to_ne_bytes()
}

/*
/// Congruence signature for assigning same colors to congruent faces
#[inline]
pub(crate) fn face_signature(positions: &PointsRefSlice, sensitivity: Float) -> Vec<u8> {
    let cross_array = positions
        .iter()
        .circular_tuple_windows::<(_, _, _)>()
        .map(|position|{
            position.0.sub(**position.1).cross(position.1.sub(**position.2)).mag()
        })
        .collect::<Vec<_>>()
        .sort_by(|a, b| a - b);


    let mut cross_array_reversed = cross_array.clone();
    cross_array_reversed.reverse();

    cross_array
        .iter()
        .map(|x| sig_figs(x, sensitivity))
        .chain(cross_array_reversed
            .iter()
            .map(|x| sig_figs(x, sensitivity))
        )
        .collect()
}*/

#[inline]
pub(crate) fn orthogonal(v0: &Vector, v1: &Vector, v2: &Vector) -> Vector {
    (*v1 - *v0).cross(*v2 - *v1)
}

#[inline]
pub(crate) fn _are_collinear(v0: &Point, v1: &Point, v2: &Point) -> bool {
    orthogonal(v0, v1, v2).mag_sq() < EPSILON
}

#[inline]
pub(crate) fn tangent(v0: &Vector, v1: &Vector) -> Vector {
    let distance = *v1 - *v0;
    *v0 - *v1 * (distance * *v0) / distance.mag_sq()
}

#[inline]
pub(crate) fn edge_distance(v1: &Vector, v2: &Vector) -> Float {
    tangent(v1, v2).mag()
}

#[inline]
pub(crate) fn average_edge_distance(positions: &PointsRefSlice) -> Float {
    positions
        .iter()
        .circular_tuple_windows::<(_, _)>()
        .fold(0.0, |sum, edge_point| {
            sum + edge_distance(edge_point.0, edge_point.1)
        })
        / positions.len() as Float
}

/// Computes the (normalized) normal of a set of an (ordered) set of positions,
///
/// Tries to do the right thing if the face
/// is non-planar or degenerate.
#[inline]
pub(crate) fn average_normal_ref(positions: &PointsRefSlice) -> Option<Normal> {
    let mut considered_edges = 0;

    let normal = positions
        .iter()
        .circular_tuple_windows::<(_, _, _)>()
        //.take(positions.len() -1)
        .fold(Vector::zero(), |normal, corner| {
            let ortho_normal = orthogonal(corner.0, corner.1, corner.2);
            let mag_sq = ortho_normal.mag_sq();
            // Filter out collinear edge pairs.
            if mag_sq < EPSILON as _ {
                normal
            } else {
                // Subtract normalized ortho_normal.
                considered_edges += 1;
                normal - ortho_normal / mag_sq.sqrt()
            }
        });

    if considered_edges != 0 {
        Some(normal / considered_edges as f32)
    } else {
        // Degenerate/zero size face.
        //None

        // We just return the normalized vector
        // from the origin to the center of the face.
        Some(centroid_ref(positions).normalized())
    }
}

#[inline]
pub(crate) fn angle_between(
    u: &Vector,
    v: &Vector,
    normal: Option<&Vector>,
) -> Float {
    // Protection against inaccurate computation.
    let x = u.normalized().dot(v.normalized());
    let y = if x <= -1.0 {
        -1.0
    } else if x >= 1.0 {
        1.0
    } else {
        x
    };

    let angle = y.acos();

    match normal {
        None => angle,
        Some(normal) => normal.dot(*u * *v).signum() * angle,
    }
}

#[inline]
pub(crate) fn minimal_edge_length(
    face: &FaceSlice,
    positions: &PointsSlice,
) -> Float {
    face_edges(face, positions)
        .into_iter()
        .fold(Float::NAN, Float::min)
}

#[inline]
pub(crate) fn _orthogonal_f64(v0: &Point, v1: &Point, v2: &Point) -> DVec3 {
    (DVec3::new(v1.x as _, v1.y as _, v1.z as _)
        - DVec3::new(v0.x as _, v0.y as _, v0.z as _))
    .cross(
        DVec3::new(v2.x as _, v2.y as _, v2.z as _)
            - DVec3::new(v1.x as _, v1.y as _, v1.z as _),
    )
}

#[inline]
pub(crate) fn _are_collinear_f64(v0: &Point, v1: &Point, v2: &Point) -> bool {
    _orthogonal_f64(v0, v1, v2).mag_sq() < EPSILON as _
}

#[inline]
pub(crate) fn _face_normal_f64(positions: &PointsRefSlice) -> Option<Normal> {
    let mut considered_edges = 0;

    let normal = positions.iter().circular_tuple_windows::<(_, _, _)>().fold(
        DVec3::zero(),
        |normal, corner| {
            considered_edges += 1;
            let ortho_normal = _orthogonal_f64(corner.0, corner.1, corner.2);
            let mag_sq = ortho_normal.mag_sq();
            // Filter out collinear edge pairs.
            if mag_sq < EPSILON as _ {
                normal
            } else {
                // Subtract normalized ortho_normal.
                normal - ortho_normal / mag_sq.sqrt()
            }
        },
    );

    if considered_edges != 0 {
        let n = normal / considered_edges as f64;
        Some(Vector::new(n.x as _, n.y as _, n.z as _))
    } else {
        // Total degenerate or zero size face.
        // We just return the normalized vector
        // from the origin to the center of the face.
        //Some(centroid_ref(positions).normalized())

        // FIXME: this branch should return None.
        // We need a method to cleanup geometry
        // of degenrate faces/edges instead.
        None
    }
}

#[inline]
pub(crate) fn vertex_ids_edge_ref_ref<'a>(
    entries: &[(&'a Edge, Point)],
    offset: VertexKey,
) -> Vec<(&'a Edge, VertexKey)> {
    entries
        .par_iter()
        .enumerate()
        // FIXME swap with next line once rustfmt is fixed.
        //.map(|i| (i.1.0, i.0 + offset))
        .map(|i| (entries[i.0].0, i.0 as VertexKey + offset))
        .collect()
}

#[inline]
pub(crate) fn vertex_ids_ref_ref<'a>(
    entries: &[(&'a FaceSlice, Point)],
    offset: VertexKey,
) -> Vec<(&'a FaceSlice, VertexKey)> {
    entries
        .par_iter()
        .enumerate()
        // FIXME swap with next line once rustfmt is fixed.
        //.map(|i| (i.1.0, i.0 + offset))
        .map(|i| (entries[i.0].0, i.0 as VertexKey + offset))
        .collect()
}

#[allow(clippy::needless_lifetimes)]
#[inline]
pub(crate) fn vertex_ids_ref<'a>(
    entries: &'a [(Face, Point)],
    offset: VertexKey,
) -> Vec<(&'a FaceSlice, VertexKey)> {
    entries
        .par_iter()
        .enumerate()
        // FIXME swap with next line once rustfmt is fixed.
        //.map(|i| (i.1.0, i.0 + offset))
        .map(|i| (entries[i.0].0.as_slice(), i.0 as VertexKey + offset))
        .collect()
}

#[inline]
pub(crate) fn _vertex_ids(
    entries: &[(Face, Point)],
    offset: VertexKey,
) -> Vec<(Face, VertexKey)> {
    entries
        .par_iter()
        .enumerate()
        // FIXME swap with next line once rustfmt is fixed.
        //.map(|i| (i.1.0, i.0 + offset))
        .map(|i| (entries[i.0].0.clone(), i.0 as VertexKey + offset))
        .collect()
}

#[inline]
pub(crate) fn vertex(
    key: &FaceSlice,
    entries: &[(&FaceSlice, VertexKey)],
) -> Option<VertexKey> {
    entries
        .par_iter()
        .find_first(|f| key == f.0)
        .map(|entry| entry.1)
}

#[inline]
pub(crate) fn vertex_point<'a>(
    key: &FaceSlice,
    entries: &'a [(&FaceSlice, Point)],
) -> Option<&'a Point> {
    entries
        .par_iter()
        .find_first(|f| key == f.0)
        .map(|entry| &entry.1)
}

#[inline]
pub(crate) fn vertex_edge(
    key: &Edge,
    entries: &[(&Edge, VertexKey)],
) -> Option<VertexKey> {
    entries
        .par_iter()
        .find_first(|f| key == f.0)
        .map(|entry| entry.1)
}

#[inline]
pub(crate) fn vertex_edge_point<'a>(
    key: &Edge,
    entries: &'a [(&Edge, Point)],
) -> Option<&'a Point> {
    entries
        .par_iter()
        .find_first(|f| key == f.0)
        .map(|entry| &entry.1)
}

#[inline]
pub(crate) fn vertex_values_as_ref<T>(entries: &[(T, Point)]) -> Vec<&Point> {
    entries.iter().map(|e| &e.1).collect()
}

pub(crate) fn vertex_values<T>(entries: &[(T, Point)]) -> Points {
    entries.iter().map(|e| e.1).collect()
}

#[inline]
pub(crate) fn face_arity_matches(
    face: &FaceSlice,
    face_arity_mask: Option<&[usize]>,
) -> bool {
    face_arity_mask.map_or_else(
        || true,
        |face_arity_mask| face_arity_mask.contains(&face.len()),
    )
}

pub(crate) fn is_face_selected(
    face: &Face,
    index: usize,
    positions: &Points,
    face_arity_mask: Option<&[usize]>,
    face_index_mask: Option<&[FaceKey]>,
    regular_faces_only: Option<bool>,
) -> bool {
    face_arity_mask.map_or_else(
        || true,
        |face_arity_mask| face_arity_mask.contains(&face.len()),
    ) && face_index_mask.map_or_else(
        || true,
        |face_index_mask| face_index_mask.contains(&(index as _)),
    ) && (!regular_faces_only.unwrap_or(false)
        || ((face_irregularity(face, positions) - 1.0).abs() < 0.1))
}

#[inline]
pub(crate) fn face_centers(
    face_index: &FacesSlice,
    positions: &PointsSlice,
) -> Points {
    face_index
        .iter()
        .map(|face| centroid_ref(&index_as_positions(face, positions)))
        .collect()
}

#[inline]
pub(crate) fn reciprocal(vector: &Vector) -> Vector {
    *vector / vector.mag_sq()
}

#[inline]
pub(crate) fn reciprocate_face_centers(
    face_index: &FacesSlice,
    positions: &PointsSlice,
) -> Points {
    face_centers(face_index, positions)
        .iter()
        .map(reciprocal)
        .collect()
}

#[inline]
pub(crate) fn reciprocate_faces(
    face_index: &FacesSlice,
    positions: &PointsSlice,
) -> Points {
    face_index
        .iter()
        .map(|face| {
            let face_positions = index_as_positions(face, positions);
            let centroid = centroid_ref(&face_positions);
            let normal = average_normal_ref(&face_positions).unwrap();
            let c_dot_n = centroid.dot(normal);
            let edge_distance = average_edge_distance(&face_positions);
            reciprocal(&(normal * c_dot_n)) * (1.0 + edge_distance) * 0.5
        })
        .collect()
}

#[inline]
pub(crate) fn _distinct_edges(faces: &FacesSlice) -> Edges {
    faces
        .iter()
        .flat_map(|face| {
            face.iter()
                // Grab two index entries.
                .circular_tuple_windows::<(_, _)>()
                .filter(|t| t.0 < t.1)
                // Create an edge from them.
                .map(|t| [*t.0, *t.1])
                .collect::<Vec<_>>()
        })
        .unique()
        .collect()
}
