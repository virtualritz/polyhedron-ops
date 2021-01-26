use crate::*;
use ultraviolet::DVec3;

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

pub(crate) fn format_vec<T: Display>(vector: &[T]) -> String {
    if vector.is_empty() {
        String::new()
    } else {
        let mut string = String::with_capacity(vector.len() * 2);
        if 1 == vector.len() {
            write!(&mut string, "{}", vector[0]).unwrap();
        } else {
            string.push('[');
            write!(&mut string, "{}", vector[0]).unwrap();
            for i in vector.get(1..).unwrap() {
                write!(&mut string, ",{}", i).unwrap();
            }
            string.push(']');
        }
        string
    }
}

#[inline]
pub(crate) fn _to_vadd(points: &PointsSlice, v: &Vector) -> Points {
    points.par_iter().map(|p| *p + *v).collect()
}

#[inline]
pub(crate) fn vadd(points: &mut Points, v: &Vector) {
    points.par_iter_mut().for_each(|p| *p += *v);
}

#[inline]
pub(crate) fn centroid(points: &PointsSlice) -> Point {
    points
        .iter()
        .fold(Point::zero(), |accumulate, point| accumulate + *point)
        //.into_par_iter()
        //.cloned()
        //.reduce(|| Point::zero(), |accumulate, point| accumulate + point);
        / points.len() as Float
}

#[inline]
pub(crate) fn centroid_ref(points: &PointRefSlice) -> Point {
    points
        .iter()
        .fold(Point::zero(), |accumulate, point| accumulate + **point)
        / points.len() as Float
}

// Centroid projected onto the spherical surface that passes to the average of
// the given points with the center at the origin.
#[inline]
pub(crate) fn _centroid_spherical_ref(points: &PointRefSlice, spherical: Float) -> Point {
    let point: Point = points
        .iter()
        .fold(Point::zero(), |sum, point| sum + **point)
        / points.len() as Float;

    if spherical != 0.0 {
        let avg_mag =
            points.iter().fold(0.0, |sum, point| sum + point.mag()) / points.len() as Float;

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
    v: u32,
    vfaces: &FacesSlice,
    face: &FaceSlice,
    k: usize,
) -> Edges {
    if k < vfaces.len() {
        let i = index_of(&v, face).unwrap();
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
pub(crate) fn ordered_vertex_edges(v: u32, vfaces: &FacesSlice) -> Edges {
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
pub(crate) fn points_to_faces(points: &PointsSlice, face_index: &FacesSlice) -> Faces {
    points
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
        .cycle()
        .tuple_windows::<(_, _)>()
        .map(|t| {
            if t.0 < t.1 {
                [*t.0, *t.1]
            } else {
                [*t.1, *t.0]
            }
        })
        .take(face.len())
        .collect()
}

#[inline]
pub(crate) fn _to_centroid_points(points: &PointsSlice) -> Points {
    _to_vadd(points, &-centroid(points))
}

#[inline]
pub(crate) fn center_on_centroid(points: &mut Points) {
    vadd(points, &-centroid(points));
}

#[inline]
pub(crate) fn vnorm(points: &PointsSlice) -> Vec<Float> {
    points
        .par_iter()
        .map(|v| Normal::new(v.x, v.y, v.z).mag())
        .collect()
}
// Was: average_norm
#[inline]
pub(crate) fn _average_magnitude(points: &PointsSlice) -> Float {
    vnorm(points).par_iter().sum::<Float>() / points.len() as Float
}

#[inline]
pub(crate) fn max_magnitude(points: &PointsSlice) -> Float {
    vnorm(points)
        .into_par_iter()
        .reduce(|| Float::NAN, Float::max)
}

/// Returns a [`Faces`] of faces
/// containing `vertex_number`.
#[inline]
pub(crate) fn vertex_faces(vertex_number: VertexKey, face_index: &FacesSlice) -> Faces {
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
        .cycle()
        .tuple_windows::<(_, _)>()
        .map(|edge| [*edge.0, *edge.1])
        .take(face.len())
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
        let i = index_of(&v, &cface).unwrap() as i32;
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
pub(crate) fn ordered_vertex_faces(vertex_number: VertexKey, face_index: &FacesSlice) -> Faces {
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
pub(crate) fn edge_length(edge: &Edge, points: &PointsSlice) -> Float {
    let edge = vec![edge[0], edge[1]];
    let points = index_as_points(&edge, points);
    (*points[0] - *points[1]).mag()
}

#[inline]
pub(crate) fn _edge_lengths(edges: &_EdgeSlice, points: &PointsSlice) -> Vec<Float> {
    edges
        .par_iter()
        .map(|edge| edge_length(edge, points))
        .collect()
}

#[inline]
pub(crate) fn face_edges(face: &FaceSlice, points: &PointsSlice) -> Vec<Float> {
    ordered_face_edges(face)
        .par_iter()
        .map(|edge| edge_length(edge, points))
        .collect()
}

#[inline]
pub(crate) fn _circumscribed_resize(points: &mut Points, radius: Float) {
    center_on_centroid(points);
    let average = _average_magnitude(points);

    points.par_iter_mut().for_each(|v| *v *= radius / average);
}

pub(crate) fn max_resize(points: &mut Points, radius: Float) {
    center_on_centroid(points);
    let max = max_magnitude(points);

    points.par_iter_mut().for_each(|v| *v *= radius / max);
}

#[inline]
pub(crate) fn _project_on_sphere(points: &mut Points, radius: Float) {
    points
        .par_iter_mut()
        .for_each(|point| *point = radius * point.normalized());
}

#[inline]
pub(crate) fn face_irregularity(face: &FaceSlice, points: &PointsSlice) -> Float {
    let lengths = face_edges(face, points);
    // The largest value in lengths or NaN (0./0.) otherwise.
    lengths.par_iter().cloned().reduce(|| Float::NAN, Float::max)
        // divide by the smallest value in lengths or NaN (0./0.) otherwise.
        / lengths.par_iter().cloned().reduce(|| Float::NAN, Float::min)
}

#[inline]
pub(crate) fn index_as_points<'a>(f: &[VertexKey], points: &'a PointsSlice) -> Vec<&'a Point> {
    f.par_iter().map(|index| &points[*index as usize]).collect()
}

#[inline]
pub(crate) fn orthogonal(v0: &Point, v1: &Point, v2: &Point) -> Vector {
    (*v1 - *v0).cross(*v2 - *v1)
}

#[inline]
pub(crate) fn _are_collinear(v0: &Point, v1: &Point, v2: &Point) -> bool {
    orthogonal(v0, v1, v2).mag_sq() < EPSILON
}

/// Computes the normal of a face.
/// Tries to do the right thing if the face
/// is non-planar or degenerate.
#[inline]
pub(crate) fn face_normal(points: &PointRefSlice) -> Option<Normal> {
    let mut considered_edges = 0;

    let normal = points
        .iter()
        .cycle()
        .tuple_windows::<(_, _, _)>()
        .take(points.len())
        .fold(Vector::zero(), |normal, corner| {
            considered_edges += 1;
            let ortho_normal = orthogonal(&corner.0, &corner.1, &corner.2);
            let mag_sq = ortho_normal.mag_sq();
            // Filter out collinear edge pairs.
            if mag_sq < EPSILON as _ {
                normal
            } else {
                // Subtract normalized ortho_normal.
                normal - ortho_normal / mag_sq.sqrt()
            }
        });

    if considered_edges != 0 {
        Some(normal / considered_edges as f32)
    } else {
        // Degenerate/zero size face.
        None

        // We just return the normalized vector
        // from the origin to the center of the face.
        //Some(centroid_ref(points).normalized())
    }
}

#[inline]
pub(crate) fn _orthogonal_f64(v0: &Point, v1: &Point, v2: &Point) -> DVec3 {
    (DVec3::new(v1.x as _, v1.y as _, v1.z as _) - DVec3::new(v0.x as _, v0.y as _, v0.z as _))
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
pub(crate) fn _face_normal_f64(points: &PointRefSlice) -> Option<Normal> {
    let mut considered_edges = 0;

    let normal = points
        .iter()
        .cycle()
        .tuple_windows::<(_, _, _)>()
        .take(points.len())
        .fold(DVec3::zero(), |normal, corner| {
            considered_edges += 1;
            let ortho_normal = _orthogonal_f64(&corner.0, &corner.1, &corner.2);
            let mag_sq = ortho_normal.mag_sq();
            // Filter out collinear edge pairs.
            if mag_sq < EPSILON as _ {
                normal
            } else {
                // Subtract normalized ortho_normal.
                normal - ortho_normal / mag_sq.sqrt()
            }
        });

    if considered_edges != 0 {
        let n = normal / considered_edges as f64;
        Some(Vector::new(n.x as _, n.y as _, n.z as _))
    } else {
        // Total degenerate or zero size face.
        // We just return the normalized vector
        // from the origin to the center of the face.
        //Some(centroid_ref(points).normalized())

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
pub(crate) fn _vertex_ids(entries: &[(Face, Point)], offset: VertexKey) -> Vec<(Face, VertexKey)> {
    entries
        .par_iter()
        .enumerate()
        // FIXME swap with next line once rustfmt is fixed.
        //.map(|i| (i.1.0, i.0 + offset))
        .map(|i| (entries[i.0].0.clone(), i.0 as VertexKey + offset))
        .collect()
}

#[inline]
pub(crate) fn vertex(key: &FaceSlice, entries: &[(&FaceSlice, VertexKey)]) -> Option<VertexKey> {
    match entries.par_iter().find_first(|f| key == f.0) {
        Some(entry) => Some(entry.1),
        None => None,
    }
}

#[inline]
pub(crate) fn vertex_point<'a>(
    key: &FaceSlice,
    entries: &'a [(&FaceSlice, Point)],
) -> Option<&'a Point> {
    match entries.par_iter().find_first(|f| key == f.0) {
        Some(entry) => Some(&entry.1),
        None => None,
    }
}

#[inline]
pub(crate) fn vertex_edge(key: &Edge, entries: &[(&Edge, VertexKey)]) -> Option<VertexKey> {
    match entries.par_iter().find_first(|f| key == f.0) {
        Some(entry) => Some(entry.1),
        None => None,
    }
}

#[inline]
pub(crate) fn vertex_edge_point<'a>(
    key: &Edge,
    entries: &'a [(&Edge, Point)],
) -> Option<&'a Point> {
    match entries.par_iter().find_first(|f| key == f.0) {
        Some(entry) => Some(&entry.1),
        None => None,
    }
}

#[inline]
pub(crate) fn vertex_values_as_ref<T>(entries: &[(T, Point)]) -> Vec<&Point> {
    entries.iter().map(|e| &e.1).collect()
}

pub(crate) fn vertex_values<T>(entries: &[(T, Point)]) -> Points {
    entries.iter().map(|e| e.1).collect()
}

#[inline]
pub(crate) fn selected_face(face: &FaceSlice, face_arity: Option<&Vec<usize>>) -> bool {
    match face_arity {
        None => true,
        Some(arity) => arity.contains(&face.len()),
    }
}

#[inline]
pub(crate) fn distinct_edges(faces: &FacesSlice) -> Edges {
    faces
        .iter()
        .flat_map(|face| {
            face.iter()
                .cycle()
                // Grab two index entries.
                .tuple_windows::<(_, _)>()
                .filter(|t| t.0 < t.1)
                // Create an edge from them.
                .map(|t| [*t.0, *t.1])
                .take(face.len())
                .collect::<Vec<_>>()
        })
        .unique()
        .collect()
}

#[inline]
pub(crate) fn face_centers(face_index: &FacesSlice, points: &PointsSlice) -> Points {
    face_index
        .iter()
        .map(|face| centroid_ref(&index_as_points(face, points)))
        .collect()
}

#[inline]
pub(crate) fn reciprocate_face_centers(face_index: &FacesSlice, points: &PointsSlice) -> Points {
    face_centers(face_index, points)
        .iter()
        .map(|center| *center / center.mag_sq())
        .collect()
}
