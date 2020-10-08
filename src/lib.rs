//use itertools::Itertools;
use itertools::Itertools;
use nsi;
use ultraviolet;

use std::{
    error::Error,
    fmt::{Display, Write},
    fs::File,
    io::Write as IoWrite,
    iter::Iterator,
    path::{Path, PathBuf},
};

use rayon::prelude::*;

pub type Float = f32;
pub type Index = u32;
pub type Face = Vec<Index>;
pub type FaceIndex = Vec<Face>;
pub type FaceSet = Vec<Index>;
// We treat an Edge as a Face with arity 2 to avoid copying in certain
// cases.
pub type Edge = Face;
pub type EdgeIndex = Vec<Edge>;

//type FlatPoints = Vec<Float>;
pub type Point = ultraviolet::vec::Vec3; //  cgmath::Point3<Float>;
pub type Vector = ultraviolet::vec::Vec3;
pub type Normal = Vector;
pub type Points = Vec<Point>;
pub type PointsRef<'a> = Vec<&'a Point>;

#[allow(dead_code)]
pub type Normals = Vec<Normal>;

pub enum NormalType {
    Smooth(Float),
    Flat,
}

pub mod prelude {
    //! Re-exports commonly used types and traits.
    //!
    //! Importing the contents of this module is recommended.

    //pub use crate::NormalType;

    pub use crate::*;
}

fn format_vec<T: Display>(vector: &Vec<T>) -> String {
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
fn _to_vadd(points: &Points, v: &Vector) -> Points {
    points.par_iter().map(|p| p.clone() + *v).collect()
}

#[inline]
fn vadd(points: &mut Points, v: &Vector) {
    points.par_iter_mut().for_each(|p| *p += *v);
}

#[inline]
fn centroid(points: &Points) -> Point {
    let total_displacement = points
        .into_par_iter()
        .cloned()
        .reduce(|| Point::zero(), |accumulate, point| accumulate + point);

    total_displacement / points.len() as Float
}

#[inline]
fn ordered_vertex_edges_recurse(
    v: u32,
    vfaces: &FaceIndex,
    face: &Face,
    k: usize,
) -> EdgeIndex {
    if k < vfaces.len() {
        let i = index_of(&v, face).unwrap();
        let j = (i + face.len() - 1) % face.len();
        let edge = vec![v, face[j]];
        let nface = face_with_edge(&edge, vfaces);
        let mut result = vec![edge];
        result.extend(ordered_vertex_edges_recurse(v, vfaces, &nface, k + 1));
        result
    } else {
        vec![]
    }
}

#[inline]
fn ordered_vertex_edges(v: u32, vfaces: &FaceIndex) -> EdgeIndex {
    if vfaces.is_empty() {
        vec![]
    } else {
        let face = &vfaces[0];
        let i = index_of(&v, face).unwrap();
        let j = (i + face.len() - 1) % face.len();
        let edge = vec![v, face[j]];
        let nface = face_with_edge(&edge, vfaces);
        let mut result = vec![edge];
        result.extend(ordered_vertex_edges_recurse(v, vfaces, &nface, 1));
        result
    }
}

#[inline]
fn distinct_edge(edge: &Edge) -> Edge {
    if edge[0] < edge[1] {
        edge.clone()
    } else {
        let mut e = edge.clone();
        e.reverse();
        e
    }
}

#[inline]
fn distinct_face_edges(face: &Face) -> EdgeIndex {
    face.iter()
        .cycle()
        .tuple_windows::<(_, _)>()
        .map(|t| {
            if t.0 < t.1 {
                vec![*t.0, *t.1]
            } else {
                vec![*t.1, *t.0]
            }
        })
        .take(face.len())
        .collect()
}

#[inline]
fn _to_centroid_points(points: &Points) -> Points {
    _to_vadd(points, &-centroid(points))
}

#[inline]
fn center_on_centroid(points: &mut Points) {
    vadd(points, &-centroid(points));
}

#[inline]
fn centroid_ref<'a>(points: &'a PointsRef) -> Point {
    let total_displacement = points
        .into_iter()
        //.cloned()
        .fold(Point::zero(), |accumulate, point| accumulate + **point);

    total_displacement / points.len() as Float
}

#[inline]
fn vnorm(points: &Points) -> Vec<Float> {
    points
        .par_iter()
        .map(|v| Normal::new(v.x, v.y, v.z).mag())
        .collect()
}
// Was: average_norm
#[inline]
fn _average_magnitude(points: &Points) -> Float {
    vnorm(points).par_iter().sum::<Float>() / points.len() as Float
}

#[inline]
fn max_magnitude(points: &Points) -> Float {
    vnorm(points)
        .into_par_iter()
        .reduce(|| Float::NAN, Float::max)
}

/// Returns a [`FaceIndex`] of faces
/// containing `vertex_number`.
#[inline]
fn vertex_faces(vertex_number: Index, face_index: &FaceIndex) -> FaceIndex {
    face_index
        .par_iter()
        .filter(|face| face.contains(&vertex_number))
        .cloned()
        .collect()
}

/// Returns a [`Vec`] of anticlockwise
/// ordered edges.
fn _ordered_face_edges_(face: &Face) -> EdgeIndex {
    face.iter()
        .cycle()
        .tuple_windows::<(_, _)>()
        .map(|edge| vec![*edge.0, *edge.1])
        .take(face.len())
        .collect()
}

/// Returns a [`Vec`] of anticlockwise
/// ordered edges.
#[inline]
fn ordered_face_edges(face: &Face) -> EdgeIndex {
    (0..face.len())
        .map(|i| vec![face[i], face[(i + 1) % face.len()]])
        .collect()
}

#[inline]
fn face_with_edge(edge: &Edge, faces: &FaceIndex) -> Face {
    let result = faces
        .par_iter()
        .filter(|face| ordered_face_edges(face).contains(edge))
        .flatten()
        .cloned()
        .collect();
    result
}

#[inline]
fn index_of<T: PartialEq>(element: &T, list: &Vec<T>) -> Option<usize> {
    list.iter().position(|e| *e == *element)
}

/// Used internally by [`ordered_vertex_faces()`].
#[inline]
fn ordered_vertex_faces_recurse(
    v: Index,
    face_index: &FaceIndex,
    cface: &Face,
    k: Index,
) -> FaceIndex {
    if (k as usize) < face_index.len() {
        let i = index_of(&v, &cface).unwrap() as i32;
        let j = ((i - 1 + cface.len() as i32) % cface.len() as i32) as usize;
        let edge = vec![v, cface[j]];
        let mut nfaces = vec![face_with_edge(&edge, face_index)];
        nfaces.extend(ordered_vertex_faces_recurse(
            v,
            face_index,
            &nfaces[0],
            k + 1,
        ));
        nfaces
    } else {
        FaceIndex::new()
    }
}

#[inline]
fn ordered_vertex_faces(
    vertex_number: Index,
    face_index: &FaceIndex,
) -> FaceIndex {
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
fn edge_length(edge: &Edge, points: &Points) -> Float {
    let edge = vec![edge[0], edge[1]];
    let points = as_points(&edge, points);
    (*points[0] - *points[1]).mag()
}

#[inline]
fn _edge_lengths(edges: &EdgeIndex, points: &Points) -> Vec<Float> {
    edges
        .par_iter()
        .map(|edge| edge_length(edge, points))
        .collect()
}

#[inline]
fn face_edges(face: &Face, points: &Points) -> Vec<Float> {
    ordered_face_edges(face)
        .par_iter()
        .map(|edge| edge_length(edge, points))
        .collect()
}

#[inline]
fn _circumscribed_resize(points: &mut Points, radius: Float) {
    center_on_centroid(points);
    let average = _average_magnitude(points);

    points.par_iter_mut().for_each(|v| *v *= radius / average);
}

fn max_resize(points: &mut Points, radius: Float) {
    center_on_centroid(points);
    let max = max_magnitude(points);

    points.par_iter_mut().for_each(|v| *v *= radius / max);
}

#[inline]
fn _project_on_sphere(points: &mut Points, radius: Float) {
    points
        .par_iter_mut()
        .for_each(|point| *point = radius * point.normalized());
}

#[inline]
fn face_irregularity(face: &Face, points: &Points) -> Float {
    let lengths = face_edges(face, points);
    // The largest value in lengths or NaN (0./0.) otherwise.
    lengths.par_iter().cloned().reduce(|| Float::NAN, Float::max)
        // divide by the smallest value in lengths or NaN (0./0.) otherwise.
        / lengths.par_iter().cloned().reduce(|| Float::NAN, Float::min)
}

#[inline]
fn as_points<'a>(f: &[Index], points: &'a Points) -> PointsRef<'a> {
    f.par_iter().map(|index| &points[*index as usize]).collect()
}

#[inline]
fn orthogonal(v0: &Point, v1: &Point, v2: &Point) -> Vector {
    (*v1 - *v0).cross(*v2 - *v1)
}

fn are_collinear(v0: &Point, v1: &Point, v2: &Point) -> bool {
    (v0.x * (v1.y - v2.y) + v1.x * (v2.y - v0.y) + v2.x * (v0.y - v1.y)).abs()
        < 0.0001
}

/// Computes the normal of a face.
/// Tries to do the right thing if the face
/// is non-planar or degenerate.
#[inline]
fn face_normal(points: &PointsRef) -> Option<Vector> {
    let mut normal = Vector::zero();
    let mut num_considered_edges = 0;

    points
        .iter()
        .cycle()
        .tuple_windows::<(_, _, _)>()
        .take(points.len())
        // Filter out collinear edge pairs
        .filter(|corner| !are_collinear(&corner.0, &corner.1, &corner.2))
        .for_each(|corner| {
            num_considered_edges += 1;
            normal -= orthogonal(&corner.0, &corner.1, &corner.2).normalized();
        });

    if 0 != num_considered_edges {
        normal /= num_considered_edges as f32;
        Some(normal)
    } else {
        None
    }
}

#[inline]
fn vertex_ids_ref_ref<'a>(
    entries: &Vec<(&'a Face, Point)>,
    offset: Index,
) -> Vec<(&'a Face, Index)> {
    entries
        .par_iter()
        .enumerate()
        // FIXME swap with next line once rustfmt is fixed.
        //.map(|i| (i.1.0, i.0 + offset))
        .map(|i| (entries[i.0].0, i.0 as Index + offset))
        .collect()
}

#[inline]
fn vertex_ids_ref<'a>(
    entries: &'a Vec<(Face, Point)>,
    offset: Index,
) -> Vec<(&'a Face, Index)> {
    entries
        .par_iter()
        .enumerate()
        // FIXME swap with next line once rustfmt is fixed.
        //.map(|i| (i.1.0, i.0 + offset))
        .map(|i| (&entries[i.0].0, i.0 as Index + offset))
        .collect()
}

#[inline]
fn _vertex_ids(
    entries: &Vec<(Face, Point)>,
    offset: Index,
) -> Vec<(Face, Index)> {
    entries
        .par_iter()
        .enumerate()
        // FIXME swap with next line once rustfmt is fixed.
        //.map(|i| (i.1.0, i.0 + offset))
        .map(|i| (entries[i.0].0.clone(), i.0 as Index + offset))
        .collect()
}

#[inline]
fn vertex(key: &Face, entries: &Vec<(&Face, Index)>) -> Option<Index> {
    match entries.par_iter().find_first(|f| key == f.0) {
        Some(entry) => Some(entry.1),
        None => None,
    }
}

#[inline]
fn vertex_values_as_ref<'a, T>(entries: &'a Vec<(T, Point)>) -> PointsRef<'a> {
    entries.iter().map(|e| &e.1).collect()
}

fn vertex_values<T>(entries: &Vec<(T, Point)>) -> Points {
    entries.iter().map(|e| e.1).collect()
}

#[inline]
fn selected_face(face: &Face, face_arity: Option<&Vec<usize>>) -> bool {
    match face_arity {
        None => true,
        Some(arity) => arity.contains(&face.len()),
    }
}

#[inline]
fn distinct_edges(faces: &FaceIndex) -> EdgeIndex {
    faces
        .par_iter()
        .flat_map(|face| {
            face.iter()
                .cycle()
                // Grab two index entries.
                .tuple_windows::<(_, _)>()
                .filter(|t| t.0 < t.1)
                // Create an edge from them.
                .map(|t| vec![*t.0, *t.1])
                .take(face.len())
                .collect::<Vec<_>>()
        })
        .collect::<EdgeIndex>()
        .into_iter()
        .unique()
        .collect()
}

/// Extend a vector with some element(s)
/// ```
/// extend![..foo, 4, 5, 6]
/// ```
macro_rules! extend {
    (..$v:expr, $($new:expr),*) => {{
        let mut tmp = $v.clone();
        $(
        tmp.push($new);
        )*
        tmp
    }}
}

#[derive(Clone, Debug)]
pub struct Polyhedron {
    points: Points,
    //face_arity: Vec<index>,
    face_index: FaceIndex,
    // This stores a FaceSet for each
    // set of faces belonging to the
    // same operations.
    face_set_index: Vec<FaceSet>,
    name: String,
}

impl Polyhedron {
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            face_index: Vec::new(),
            face_set_index: Vec::new(),
            name: String::new(),
        }
    }

    #[inline]
    fn points_to_faces(mesh: &Self) -> FaceIndex {
        mesh.points
            .par_iter()
            .enumerate()
            .map(|vertex| {
                // Each old vertex creates a new face.
                ordered_vertex_faces(
                    vertex.0 as Index,
                    &vertex_faces(vertex.0 as Index, &mesh.face_index),
                )
                .iter()
                .map(|original_face|
                    // With vertex faces in left-hand order.
                    index_of(original_face, &mesh.face_index).unwrap() as Index)
                .collect()
            })
            .collect()
    }

    /// Appends indices for newly added faces
    /// as a new FaceSet to the FaceSetIndex.
    fn append_new_face_set(&mut self, size: usize) {
        self.face_set_index
            .append(&mut vec![((self.face_index.len() as u32)
                ..((self.face_index.len() + size) as u32))
                .collect()]);
    }

    /// Creates degree-4 vertices. It is also called
    /// [rectification](https://en.wikipedia.org/wiki/Rectification_(geometry)),
    /// or the [medial graph](https://en.wikipedia.org/wiki/Medial_graph) in graph theory.
    #[inline]
    pub fn ambo<'a>(&'a mut self, change_name: bool) -> &'a mut Self {
        let edges = distinct_edges(&self.face_index);

        let points: Vec<(&Vec<u32>, Point)> = edges
            .par_iter()
            .map(|edge| {
                let edge_points = as_points(edge, &self.points);
                (edge, 0.5 * (*edge_points[0] + *edge_points[1]))
            })
            .collect();

        let new_ids = vertex_ids_ref_ref(&points, 0);

        let mut face_index: FaceIndex = self
            .face_index
            .par_iter()
            .map(|face| {
                let edges = distinct_face_edges(face);
                let result = edges
                    .iter()
                    .filter_map(|edge| vertex(edge, &new_ids))
                    .collect::<Vec<_>>();
                result
            })
            .collect::<Vec<_>>();

        let mut new_face_index: FaceIndex = self
            .points
            // Each old vertex creates a new face ...
            .par_iter()
            .enumerate()
            .map(|polygon_vertex| {
                let vertex_number = polygon_vertex.0 as Index;
                ordered_vertex_edges(
                    vertex_number,
                    &vertex_faces(vertex_number, &self.face_index),
                )
                .iter()
                .map(|ve| {
                    vertex(&distinct_edge(ve), &new_ids).unwrap() as Index
                })
                .collect::<Vec<_>>()
            })
            .collect();

        face_index.append(&mut new_face_index);

        self.append_new_face_set(face_index.len());

        self.face_index = face_index;
        self.points = vertex_values(&points);

        if change_name {
            self.name = format!("a{}", self.name);
        }

        self
    }

    pub fn bevel<'a>(
        &'a mut self,
        height: Option<Float>,
        vertex_degree: Option<Vec<usize>>,
        regular: bool,
        change_name: bool,
    ) -> &'a mut Self {
        self.truncate(height.clone(), vertex_degree.clone(), regular, false);
        self.ambo(false);

        if change_name {
            let mut params = String::new();
            if let Some(height) = height {
                write!(&mut params, "{}", height).unwrap();
            }
            if let Some(vertex_degree) = vertex_degree {
                write!(&mut params, ",{}", format_vec(&vertex_degree)).unwrap();
            }
            if regular {
                params.push_str(",{t}");
            }
            self.name = format!("b{}{}", params, self.name);
        }

        self
    }

    pub fn chamfer<'a>(
        &'a mut self,
        ratio: Option<Float>,
        change_name: bool,
    ) -> &'a mut Self {
        let ratio_ = match ratio {
            Some(r) => r,
            None => 1. / 2.,
        };

        let new_points = self
            .face_index
            .par_iter()
            .flat_map(|face| {
                let face_points = as_points(face, &self.points);
                let centroid = centroid_ref(&face_points);
                // println!("{:?}", ep);
                let mut result = Vec::new();
                face.iter().enumerate().for_each(|face_point| {
                    let j = face_point.0;
                    let mut new_face = face.clone();
                    new_face.push(face[j]);
                    result.push((
                        new_face,
                        *face_points[j] + ratio_ * (centroid - *face_points[j]),
                    ))
                });
                result
            })
            .collect::<Vec<_>>();

        let new_ids = vertex_ids_ref(&new_points, self.points_len() as Index);

        let mut face_index: FaceIndex = self
            .face_index
            .iter()
            .map(|face| {
                let mut new_face = Vec::with_capacity(face.len());
                face.iter().for_each(|vertex_key| {
                    let mut face_key = face.clone();
                    face_key.push(*vertex_key);
                    new_face.push(vertex(&face_key, &new_ids).unwrap());
                });
                new_face
            })
            .collect();

        face_index.extend(
            self.face_index
                .par_iter()
                .flat_map(|face| {
                    (0..face.len())
                        .filter(|j| face[*j] < face[(*j + 1) % face.len()])
                        .map(|j| {
                            let a: u32 = face[j];
                            let b: u32 = face[(j + 1) % face.len()];
                            let opposite_face =
                                face_with_edge(&vec![b, a], &self.face_index);

                            vec![
                                a,
                                vertex(&extend![..opposite_face, a], &new_ids)
                                    .unwrap(),
                                vertex(&extend![..opposite_face, b], &new_ids)
                                    .unwrap(),
                                b,
                                vertex(&extend![..face, b], &new_ids).unwrap(),
                                vertex(&extend![..face, a], &new_ids).unwrap(),
                            ]
                        })
                        .collect::<FaceIndex>()
                })
                .collect::<FaceIndex>(),
        );

        self.append_new_face_set(face_index.len());

        self.face_index = face_index;
        self.points.par_iter_mut().for_each(|point| {
            *point = (1.5 * ratio_) * *point;
        });
        self.points.extend(vertex_values(&new_points));

        if change_name {
            let mut params = String::new();
            if let Some(ratio) = ratio {
                write!(&mut params, "{}", ratio).unwrap();
            }
            self.name = format!("c{}{}", params, self.name);
        }

        self
    }

    pub fn dual<'a>(&'a mut self, change_name: bool) -> &'a mut Self {
        let new_points = self
            .face_index
            .par_iter()
            .map(|face| centroid_ref(&as_points(face, &self.points)))
            .collect();

        // FIXME: FaceSetIndex
        self.face_index = Self::points_to_faces(self);
        self.points = new_points;

        if change_name {
            self.name = format!("d{}", self.name);
        }

        self
    }

    pub fn expand<'a>(&'a mut self, change_name: bool) -> &'a mut Self {
        self.ambo(false);
        self.ambo(false);

        if change_name {
            self.name = format!("e{}", self.name);
        }

        self
    }

    pub fn gyro<'a>(
        &'a mut self,
        ratio: Option<f32>,
        height: Option<f32>,
        change_name: bool,
    ) -> &'a mut Self {
        let ratio_ = match ratio {
            Some(r) => r,
            None => 1. / 3.,
        };
        let height_ = match height {
            Some(h) => h,
            None => 0.,
        };

        // retain original points, add face centroids and directed
        // edge points each N-face becomes N pentagons
        let mut new_points: Vec<(&Face, Point)> = self
            .face_index
            .par_iter()
            .map(|face| {
                let fp = as_points(face, &self.points);
                (
                    face,
                    centroid_ref(&fp).normalized()
                        + face_normal(&fp).unwrap() * height_,
                )
            })
            .collect::<Vec<_>>();

        let edges = self.edges();

        let reversed_edges: EdgeIndex = edges
            .par_iter()
            .map(|edge| vec![edge[1], edge[0]])
            .collect();

        let new_points2 = edges
            .par_iter()
            .enumerate()
            .flat_map(|edge| {
                let edge_points = as_points(edge.1, &self.points);
                // println!("{:?}", ep);
                vec![
                    (
                        edge.1,
                        *edge_points[0]
                            + ratio_ * (*edge_points[1] - *edge_points[0]),
                    ),
                    (
                        &reversed_edges[edge.0],
                        *edge_points[1]
                            + ratio_ * (*edge_points[0] - *edge_points[1]),
                    ),
                ]
            })
            .collect::<Vec<_>>();

        new_points.extend(new_points2);
        //  2 points per edge

        let new_ids =
            vertex_ids_ref_ref(&new_points, self.points_len() as Index);

        self.points.extend(vertex_values_as_ref(&new_points));

        self.face_index = self
            .face_index
            .par_iter()
            .flat_map(|face| {
                (0..face.len())
                    .map(|j| {
                        let a = face[j];
                        let b = face[(j + 1) % face.len()];
                        let z = face[(j + face.len() - 1) % face.len()];
                        let eab = vertex(&vec![a, b], &new_ids).unwrap();
                        let eza = vertex(&vec![z, a], &new_ids).unwrap();
                        let eaz = vertex(&vec![a, z], &new_ids).unwrap();
                        let centroid = vertex(face, &new_ids).unwrap();
                        vec![a, eab, centroid, eza, eaz]
                    })
                    .collect::<FaceIndex>()
            })
            .collect();

        if change_name {
            let mut params = String::new();
            if let Some(ratio) = ratio {
                write!(&mut params, "{}", ratio).unwrap();
            }
            if let Some(height) = height {
                write!(&mut params, ",{}", height).unwrap();
            }
            self.name = format!("g{}{}", params, self.name);
        }

        self
    }

    pub fn join<'a>(&'a mut self, change_name: bool) -> &'a mut Self {
        self.dual(false);
        self.ambo(false);
        self.dual(false);

        if change_name {
            self.name = format!("j{}", self.name);
        }

        self
    }

    /// Splits each face into triangles, one for each edge,
    /// which extend to the face centroid. Existimg points
    /// are retained.
    /// # Arguments
    /// * `height` - An offset to add to the face centroid point along the
    ///              face normal.
    /// * `face_arity` - Only facs matching the arities given will be
    ///                  affected.
    /// * `regular` - Only faces whose edges are 90% the same length,
    ///               within the same face, are affected.
    pub fn kis<'a>(
        &'a mut self,
        height: Option<Float>,
        face_arity: Option<Vec<usize>>,
        regular: bool,
        change_name: bool,
    ) -> &'a mut Self {
        let height_ = match height {
            Some(h) => h,
            None => 0.,
        };

        let new_points: Vec<(&Face, Point)> = self
            .face_index
            .par_iter()
            .filter(|face| {
                selected_face(face, face_arity.as_ref()) && !regular
                    || ((face_irregularity(face, &self.points) - 1.0).abs()
                        < 0.1)
            })
            .map(|face| {
                let fp = as_points(face, &self.points);
                (
                    face,
                    centroid_ref(&fp) + face_normal(&fp).unwrap() * height_,
                )
            })
            .collect();

        let new_ids =
            vertex_ids_ref_ref(&new_points, self.points.len() as Index);

        self.points.extend(vertex_values_as_ref(&new_points));

        self.face_index = self
            .face_index
            .par_iter()
            .flat_map(|face: &Face| match vertex(face, &new_ids) {
                Some(centroid) => (0..face.len())
                    .map(|j| {
                        vec![
                            face[j],
                            face[(j + 1) % face.len()],
                            centroid as Index,
                        ]
                    })
                    .collect(),
                None => vec![face.clone()],
            })
            .collect();

        if change_name {
            let mut params = String::new();
            if let Some(height) = height {
                write!(&mut params, "{}", height).unwrap();
            }
            if let Some(face_arity) = face_arity {
                write!(&mut params, ",{}", format_vec(&face_arity)).unwrap();
            }
            self.name = format!("k{}{}", params, self.name);
        }

        self
    }

    fn _inset(&self) {
        self._loft()
    }

    fn _loft(&self) {
        // FIXME
    }

    pub fn medial<'a>(
        &'a mut self,
        height: Option<Float>,
        vertex_degree: Option<Vec<usize>>,
        regular: bool,
        change_name: bool,
    ) -> &'a mut Self {
        self.dual(false);
        self.truncate(height.clone(), vertex_degree.clone(), regular, false);
        self.ambo(false);

        if change_name {
            let mut params = String::new();
            if let Some(height) = height {
                write!(&mut params, "{}", height).unwrap();
            }
            if let Some(vertex_degree) = vertex_degree {
                write!(&mut params, ",{}", format_vec(&vertex_degree)).unwrap();
            }
            if regular {
                params.push_str(",{t}");
            }
            self.name = format!("M{}{}", params, self.name);
        }

        self
    }

    pub fn meta<'a>(
        &'a mut self,
        height: Option<Float>,
        vertex_degree: Option<Vec<usize>>,
        regular: bool,
        change_name: bool,
    ) -> &'a mut Self {
        self.kis(
            height,
            match vertex_degree {
                // By default meta works on verts.
                // of valence three.
                None => Some(vec![3]),
                _ => vertex_degree.clone(),
            },
            regular,
            false,
        );
        self.join(false);

        if change_name {
            let mut params = String::new();
            if let Some(height) = height {
                write!(&mut params, "{}", height).unwrap();
            }
            if let Some(vertex_degree) = vertex_degree {
                write!(&mut params, ",{}", format_vec(&vertex_degree)).unwrap();
            }
            if regular {
                params.push_str(",{t}");
            }
            self.name = format!("m{}{}", params, self.name);
        }

        self
    }

    pub fn needle<'a>(
        &'a mut self,
        height: Option<Float>,
        vertex_degree: Option<Vec<usize>>,
        regular: bool,
        change_name: bool,
    ) -> &'a mut Self {
        self.dual(false);
        self.truncate(height.clone(), vertex_degree.clone(), regular, false);

        if change_name {
            let mut params = String::new();
            if let Some(height) = height {
                write!(&mut params, "{}", height).unwrap();
            }
            if let Some(vertex_degree) = vertex_degree {
                write!(&mut params, ",{}", format_vec(&vertex_degree)).unwrap();
            }
            if regular {
                params.push_str(",{t}");
            }
            self.name = format!("n{}{}", params, self.name);
        }

        self
    }

    pub fn ortho<'a>(&'a mut self, change_name: bool) -> &'a mut Self {
        self.join(false);
        self.join(false);

        if change_name {
            self.name = format!("o{}", self.name);
        }

        self
    }

    pub fn propellor<'a>(
        &'a mut self,
        ratio: Option<Float>,
        change_name: bool,
    ) -> &'a mut Self {
        let ratio_ = match ratio {
            Some(r) => r,
            None => 1. / 3.,
        };

        let edges = self.edges();
        let reversed_edges: EdgeIndex =
            edges.iter().map(|edge| vec![edge[1], edge[0]]).collect();

        let new_points = edges
            .iter()
            .zip(reversed_edges.iter())
            .flat_map(|(edge, reversed_edge)| {
                let edge_points = as_points(edge, &self.points);
                vec![
                    (
                        edge,
                        *edge_points[0]
                            + ratio_ * (*edge_points[1] - *edge_points[0]),
                    ),
                    (
                        reversed_edge,
                        *edge_points[1]
                            + ratio_ * (*edge_points[0] - *edge_points[1]),
                    ),
                ]
            })
            .collect::<Vec<_>>();

        let new_ids =
            vertex_ids_ref_ref(&new_points, self.points_len() as Index);

        let mut face_index: FaceIndex = self
            .face_index
            .iter()
            .map(|face| {
                (0..face.len())
                    .map(|j| {
                        vertex(
                            &vec![face[j], face[(j + 1) % face.len()]],
                            &new_ids,
                        )
                        .unwrap()
                    })
                    .collect()
            })
            .collect();

        face_index.extend(
            self.face_index
                .iter()
                .flat_map(|face| {
                    (0..face.len())
                        .map(|j| {
                            let a = face[j];
                            let b = face[(j + 1) % face.len()];
                            let z = face[(j + face.len() - 1) % face.len()];
                            let eab = vertex(&vec![a, b], &new_ids).unwrap();
                            let eba = vertex(&vec![b, a], &new_ids).unwrap();
                            let eza = vertex(&vec![z, a], &new_ids).unwrap();
                            vec![a, eba, eab, eza]
                        })
                        .collect::<FaceIndex>()
                })
                .collect::<FaceIndex>(),
        );

        self.face_index = face_index;
        self.points.extend(vertex_values_as_ref(&new_points));

        if change_name {
            self.name = format!("p{}", self.name);
        }

        self
    }

    pub fn quinto<'a>(&'a mut self, change_name: bool) -> &'a mut Self {
        let edges = self.edges();
        let mut new_points: Vec<(Face, Point)> = edges
            .iter()
            .map(|edge| {
                let edge_points = as_points(edge, &self.points);
                (edge.clone(), 0.5 * (*edge_points[0] + *edge_points[1]))
            })
            .collect();

        new_points.extend(
            self.face_index
                .iter()
                .flat_map(|face| {
                    let edge_points = as_points(face, &self.points);
                    let centroid = centroid_ref(&edge_points);
                    (0..face.len())
                        .map(|i| {
                            (
                                extend![..face, i as u32],
                                (*edge_points[i]
                                    + *edge_points[(i + 1) % face.len()]
                                    + centroid)
                                    / 3.,
                            )
                        })
                        .collect::<Vec<(Face, Point)>>()
                })
                .collect::<Vec<(Face, Point)>>(),
        );

        let new_ids = vertex_ids_ref(&new_points, self.points_len() as u32);

        let mut face_index: FaceIndex = self
            .face_index
            .iter()
            .map(|face| {
                (0..face.len())
                    .map(|face_vertex| {
                        vertex(&extend![..face, face_vertex as u32], &new_ids)
                            .unwrap()
                    })
                    .collect()
            })
            .collect();

        face_index.extend(
            self.face_index
                .iter()
                .flat_map(|face| {
                    (0..face.len())
                        .map(|i| {
                            vec![
                                face[i],
                                vertex(
                                    &distinct_edge(&vec![
                                        face[(i + face.len() - 1) % face.len()],
                                        face[i],
                                    ]),
                                    &new_ids,
                                )
                                .unwrap(),
                                vertex(
                                    &distinct_edge(&vec![
                                        face[i],
                                        face[(i + 1) % face.len()],
                                    ]),
                                    &new_ids,
                                )
                                .unwrap(),
                                vertex(
                                    &extend![
                                        ..face,
                                        ((i + face.len() - 1) % face.len())
                                            as u32
                                    ],
                                    &new_ids,
                                )
                                .unwrap(),
                                vertex(&extend![..face, i as u32], &new_ids)
                                    .unwrap(),
                            ]
                        })
                        .collect::<FaceIndex>()
                })
                .collect::<FaceIndex>(),
        );

        self.face_index = face_index;
        self.points.extend(vertex_values_as_ref(&new_points));

        if change_name {
            self.name = format!("q{}", self.name);
        }

        self
    }

    pub fn reflect<'a>(&'a mut self, change_name: bool) -> &'a mut Self {
        self.points = self
            .points
            .par_iter()
            .map(|v| Point::new(v.x, -v.y, v.z))
            .collect();
        self.reverse();

        if change_name {
            self.name = format!("r{}", self.name);
        }

        self
    }

    pub fn snub<'a>(
        &'a mut self,
        ratio: Option<Float>,
        height: Option<Float>,
        change_name: bool,
    ) -> &'a mut Self {
        self.dual(false);
        self.gyro(ratio, height, false);
        self.dual(false);

        if change_name {
            let mut params = String::new();
            if let Some(ratio) = ratio {
                write!(&mut params, "{}", ratio).unwrap();
            }
            if let Some(height) = height {
                write!(&mut params, ",{}", height).unwrap();
            }
            self.name = format!("s{}{}", params, self.name);
        }

        self
    }

    pub fn truncate<'a>(
        &'a mut self,
        height: Option<Float>,
        vertex_degree: Option<Vec<usize>>,
        regular: bool,
        change_name: bool,
    ) -> &'a mut Self {
        self.dual(false);
        self.kis(height, vertex_degree.clone(), regular, false);
        self.dual(false);

        if change_name {
            let mut params = String::new();
            if let Some(height) = height {
                write!(&mut params, "{}", height).unwrap();
            }
            if let Some(vertex_degree) = vertex_degree {
                write!(&mut params, ",{}", format_vec(&vertex_degree)).unwrap();
            }
            if regular {
                params.push_str(",{t}");
            }
            self.name = format!("t{}{}", params, self.name);
        }

        self
    }

    pub fn whirl<'a>(
        &'a mut self,
        ratio: Option<Float>,
        height: Option<Float>,
        change_name: bool,
    ) -> &'a mut Self {
        let ratio_ = match ratio {
            Some(r) => r,
            None => 1. / 3.,
        };
        let height_ = match height {
            Some(h) => h,
            None => 0.,
        };

        let mut new_points: Vec<(Face, Point)> = self
            .face_index
            .iter()
            .flat_map(|face| {
                let face_points = as_points(face, &self.points);
                let center = centroid_ref(&face_points)
                    + face_normal(&face_points).unwrap() * height_;
                face.iter()
                    .enumerate()
                    .map(|v| {
                        let edge_points = [
                            face_points[v.0],
                            face_points[(v.0 + 1) % face.len()],
                        ];
                        let middle: ultraviolet::vec::Vec3 = *edge_points[0]
                            + ratio_ * (*edge_points[1] - *edge_points[0]);
                        (
                            extend![..face, *v.1],
                            middle + ratio_ * (center - middle),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let edges = self.edges();

        let new_points2: Vec<(Face, Point)> = edges
            .par_iter()
            .flat_map(|edge| {
                let edge_points = as_points(edge, &self.points);
                vec![
                    (
                        edge.clone(),
                        *edge_points[0]
                            + ratio_ * (*edge_points[1] - *edge_points[0]),
                    ),
                    (
                        vec![edge[1], edge[0]],
                        *edge_points[1]
                            + ratio_ * (*edge_points[0] - *edge_points[1]),
                    ),
                ]
            })
            .collect();

        new_points.extend(new_points2);

        let new_ids = vertex_ids_ref(&new_points, self.points_len() as Index);

        let mut face_index: FaceIndex = self
            .face_index
            .par_iter()
            .flat_map(|face| {
                (0..face.len())
                    .map(|j| {
                        let a = face[j];
                        let b = face[(j + 1) % face.len()];
                        let c = face[(j + 2) % face.len()];
                        let eab = vertex(&vec![a, b], &new_ids).unwrap();
                        let eba = vertex(&vec![b, a], &new_ids).unwrap();
                        let ebc = vertex(&vec![b, c], &new_ids).unwrap();
                        let mut mid = face.clone();
                        mid.push(a);
                        let mida = vertex(&mid, &new_ids).unwrap();
                        mid.pop();
                        mid.push(b);
                        let midb = vertex(&mid, &new_ids).unwrap();
                        vec![eab, eba, b, ebc, midb, mida]
                    })
                    .collect::<FaceIndex>()
            })
            .collect();

        face_index.extend(
            self.face_index
                .par_iter()
                .map(|face| {
                    let mut new_face = face.clone();
                    face.iter()
                        .map(|a| {
                            new_face.push(*a);
                            let result = vertex(&new_face, &new_ids).unwrap();
                            new_face.pop();
                            result
                        })
                        .collect()
                })
                .collect::<FaceIndex>(),
        );

        self.append_new_face_set(face_index.len() - self.face_index.len());

        self.points.extend(vertex_values(&new_points));
        self.face_index = face_index;

        if change_name {
            let mut params = String::new();
            if let Some(ratio) = ratio {
                write!(&mut params, "{}", ratio).unwrap();
            }
            if let Some(height) = height {
                write!(&mut params, ",{}", height).unwrap();
            }
            self.name = format!("w{}{}", params, self.name);
        }

        self
    }

    pub fn zip<'a>(
        &'a mut self,
        height: Option<Float>,
        vertex_degree: Option<Vec<usize>>,
        regular: bool,
        change_name: bool,
    ) -> &'a mut Self {
        self.dual(false);
        self.kis(height, vertex_degree.clone(), regular, false);

        if change_name {
            let mut params = String::new();
            if let Some(height) = height {
                write!(&mut params, "{}", height).unwrap();
            }
            if let Some(vertex_degree) = vertex_degree {
                write!(&mut params, ",{}", format_vec(&vertex_degree)).unwrap();
            }
            if regular {
                params.push_str(",{t}");
            }
            self.name = format!("z{}{}", params, self.name);
        }

        self
    }

    /// Reverses the winding order of faces.
    /// Clockwise(default) becomes counter-clockwise and vice versa.
    pub fn reverse(&mut self) {
        self.face_index
            .par_iter_mut()
            .for_each(|face| face.reverse());
    }

    /// Returns the name of this polyhedron.
    /// This can be used to reconstruct the polyhedron
    /// using Polyhedron::from<&str>().
    #[inline]
    pub fn name(&self) -> &String {
        &self.name
    }

    #[inline]
    pub fn points_len(&self) -> usize {
        self.points.len()
    }

    #[inline]
    pub fn points<'a>(&'a self) -> &'a Points {
        &self.points
    }

    pub fn face_index<'a>(&'a self) -> &'a FaceIndex {
        &self.face_index
    }

    #[inline]
    pub fn normalize(&mut self) {
        max_resize(&mut self.points, 1.);
    }

    #[inline]
    pub fn edges(&self) -> EdgeIndex {
        distinct_edges(&self.face_index)
    }

    pub fn normals(&self, normal_type: NormalType) -> Normals {
        match normal_type {
            NormalType::Smooth(_angle) => vec![],
            NormalType::Flat => self
                .face_index
                .par_iter()
                .flat_map(|f| {
                    f.iter()
                        // Cycle forever.
                        .cycle()
                        // Start at 3-tuple belonging to the
                        // face's last vertex.
                        .skip(f.len() - 1)
                        // Grab the next three vertex index
                        // entries.
                        .tuple_windows::<(_, _, _)>()
                        // Create a normal from that
                        .map(|t| {
                            -orthogonal(
                                &self.points[*t.0 as usize],
                                &self.points[*t.1 as usize],
                                &self.points[*t.2 as usize],
                            )
                            .normalized()
                        })
                        .take(f.len())
                        .collect::<Normals>()
                })
                .collect(),
            /*NormalType::Flat => self
            .face_index
            .par_iter()
            .for_each(|f| {
                normals.extend(
                    f.par_iter()
                        // Cycle forever.
                        .cycle()
                        // Start at 3-tuple belonging to the
                        // face's last vertex.
                        .skip(f.len() - 1)
                        // Grab the next three vertex index
                        // entries.
                        .tuple_windows::<(_, _, _)>()
                        // Create a normal from that
                        .for_each(|t| {
                            -orthogonal(
                                &self.points[*t.0 as usize],
                                &self.points[*t.1 as usize],
                                &self.points[*t.2 as usize],
                            )
                            .normalize()
                        })
                        .take(f.len())
                        .collect::<Normals>(),
                );
                face_index.extend(f.par_iter())
            })
            .flatten()
            .collect(),*/
        }
    }

    #[inline]
    pub fn triangulate(&mut self, shortest: bool) {
        self.face_index = self
            .face_index
            .iter()
            .flat_map(|face| match face.len() {
                // Bitriangulate quadrilateral faces
                // use shortest diagonal so triangles are
                // most nearly equilateral.
                4 => {
                    let p = as_points(face, &self.points);

                    if shortest
                        == ((*p[0] - *p[2]).mag_sq() < (*p[1] - *p[3]).mag_sq())
                    {
                        vec![
                            vec![face[0], face[1], face[2]],
                            vec![face[0], face[2], face[3]],
                        ]
                    } else {
                        vec![
                            vec![face[1], face[2], face[3]],
                            vec![face[1], face[3], face[0]],
                        ]
                    }
                }
                5 => vec![
                    vec![face[0], face[1], face[4]],
                    vec![face[1], face[2], face[4]],
                    vec![face[4], face[2], face[3]],
                ],
                _ => {
                    let a = face[0];
                    let mut bb = face[1];
                    face.iter()
                        .skip(2)
                        .map(|c| {
                            let b = bb;
                            bb = *c;
                            vec![a, b, *c]
                        })
                        .collect()
                } //_ => vec![face.clone()],
            })
            .collect();
    }

    pub fn to_nsi(&self, ctx: &nsi::Context) -> String {
        // Create a new mesh node and call it 'dodecahedron'.
        ctx.create(self.name.clone(), nsi::NodeType::Mesh, &[]);

        // Connect the 'dodecahedron' node to the scene's root.
        ctx.connect(self.name.clone(), "", nsi::NodeType::Root, "objects", &[]);

        /*
        let positions: FlatPoints = self
            .points
            .into_par_iter()
            .flat_map(|p3| once(p3.x).chain(once(p3.y)).chain(once(p3.z)))
            .collect();
        */

        let positions = unsafe {
            std::slice::from_raw_parts(
                self.points.as_ptr().cast::<Float>(),
                3 * self.points_len(),
            )
        };

        let face_arity = self
            .face_index
            .par_iter()
            .map(|face| face.len() as u32)
            .collect::<Vec<_>>();

        let edges = self.edges().into_iter().flatten().collect::<Vec<_>>();
        let face_index = self.face_index.concat();

        ctx.set_attribute(
            self.name.clone(),
            &[
                nsi::points!("P", positions),
                nsi::unsigneds!("P.indices", &face_index),
                // 5 points per each face.
                nsi::unsigneds!("nvertices", &face_arity),
                // Render this as a subdivison surface.
                nsi::string!("subdivision.scheme", "catmull-clark"),
                // Crease each of our 30 edges a bit.
                nsi::unsigneds!("subdivision.creasevertices", &edges),
                nsi::floats!(
                    "subdivision.creasesharpness",
                    &vec![10.; edges.len()]
                ),
                nsi::unsigned!("subdivision.smoothcreasecorners", 0),
            ],
        );

        self.name.clone()
    }
    /*
    function average_normal(fp) =
        let(fl=len(fp))

            let unit_normals = normale(face)
            let(unitns=
                [for(i=[0:fl-1])
                    let(n=orthogonal(fp[i],fp[(i+1)%fl],fp[(i+2)%fl]))
                    let(normn=norm(n))
                    normn==0 ? [] : n/normn
          ]
         )
    vsum(unitns)/len(unitns);*/

    pub fn export_as_obj(
        &self,
        destination: &Path,
        reverse_winding: bool,
    ) -> Result<PathBuf, Box<dyn Error>> {
        let path = destination.join(format!("polyhedron-{}.obj", self.name));
        let mut file = File::create(path.clone())?;

        writeln!(file, "o {}", self.name)?;

        for vertex in &self.points {
            writeln!(file, "v {} {} {}", vertex.x, vertex.y, vertex.z)?;
        }

        match reverse_winding {
            true => {
                for face in &self.face_index {
                    write!(file, "f")?;
                    for vertex_index in face.iter().rev() {
                        write!(file, " {}", vertex_index + 1)?;
                    }
                    writeln!(file, "")?;
                }
            }
            false => {
                for face in &self.face_index {
                    write!(file, "f")?;
                    for vertex_index in face {
                        write!(file, " {}", vertex_index + 1)?;
                    }
                    writeln!(file, "")?;
                }
            }
        };

        file.flush()?;

        Ok(path)
    }

    pub fn tetrahedron() -> Self {
        let c0 = 1.0;

        Self {
            points: vec![
                Point::new(c0, c0, c0),
                Point::new(c0, -c0, -c0),
                Point::new(-c0, c0, -c0),
                Point::new(-c0, -c0, c0),
            ],
            face_index: vec![
                vec![2, 1, 0],
                vec![3, 2, 0],
                vec![1, 3, 0],
                vec![2, 3, 1],
            ],
            face_set_index: vec![(0..4).collect()],
            name: String::from("T"),
        }
    }

    pub fn cube() -> Self {
        Self::hexahedron()
    }

    pub fn hexahedron() -> Self {
        let c0 = 1.0;

        Self {
            points: vec![
                Point::new(c0, c0, c0),
                Point::new(c0, c0, -c0),
                Point::new(c0, -c0, c0),
                Point::new(c0, -c0, -c0),
                Point::new(-c0, c0, c0),
                Point::new(-c0, c0, -c0),
                Point::new(-c0, -c0, c0),
                Point::new(-c0, -c0, -c0),
            ],
            face_index: vec![
                vec![4, 5, 1, 0],
                vec![2, 6, 4, 0],
                vec![1, 3, 2, 0],
                vec![6, 2, 3, 7],
                vec![5, 4, 6, 7],
                vec![3, 1, 5, 7],
            ],
            face_set_index: vec![(0..6).collect()],
            name: String::from("C"),
        }
    }

    pub fn octahedron() -> Self {
        let c0 = 0.7071067811865475244008443621048;

        Self {
            points: vec![
                Point::new(0.0, 0.0, c0),
                Point::new(0.0, 0.0, -c0),
                Point::new(c0, 0.0, 0.0),
                Point::new(-c0, 0.0, 0.0),
                Point::new(0.0, c0, 0.0),
                Point::new(0.0, -c0, 0.0),
            ],
            face_index: vec![
                vec![4, 2, 0],
                vec![3, 4, 0],
                vec![5, 3, 0],
                vec![2, 5, 0],
                vec![5, 2, 1],
                vec![3, 5, 1],
                vec![4, 3, 1],
                vec![2, 4, 1],
            ],
            face_set_index: vec![(0..8).collect()],
            name: String::from("O"),
        }
    }

    pub fn dodecahedron() -> Self {
        let c0 = 0.809016994374947424102293417183;
        let c1 = 1.30901699437494742410229341718;

        Self {
            points: vec![
                Point::new(0.0, 0.5, c1),
                Point::new(0.0, 0.5, -c1),
                Point::new(0.0, -0.5, c1),
                Point::new(0.0, -0.5, -c1),
                Point::new(c1, 0.0, 0.5),
                Point::new(c1, 0.0, -0.5),
                Point::new(-c1, 0.0, 0.5),
                Point::new(-c1, 0.0, -0.5),
                Point::new(0.5, c1, 0.0),
                Point::new(0.5, -c1, 0.0),
                Point::new(-0.5, c1, 0.0),
                Point::new(-0.5, -c1, 0.0),
                Point::new(c0, c0, c0),
                Point::new(c0, c0, -c0),
                Point::new(c0, -c0, c0),
                Point::new(c0, -c0, -c0),
                Point::new(-c0, c0, c0),
                Point::new(-c0, c0, -c0),
                Point::new(-c0, -c0, c0),
                Point::new(-c0, -c0, -c0),
            ],
            face_index: vec![
                vec![12, 4, 14, 2, 0],
                vec![16, 10, 8, 12, 0],
                vec![2, 18, 6, 16, 0],
                vec![17, 10, 16, 6, 7],
                vec![19, 3, 1, 17, 7],
                vec![6, 18, 11, 19, 7],
                vec![15, 3, 19, 11, 9],
                vec![14, 4, 5, 15, 9],
                vec![11, 18, 2, 14, 9],
                vec![8, 10, 17, 1, 13],
                vec![5, 4, 12, 8, 13],
                vec![1, 3, 15, 5, 13],
            ],
            face_set_index: vec![(0..12).collect()],
            name: String::from("D"),
        }
    }

    pub fn icosahedron() -> Self {
        let c0 = 0.809016994374947424102293417183;

        Self {
            points: vec![
                Point::new(0.5, 0.0, c0),
                Point::new(0.5, 0.0, -c0),
                Point::new(-0.5, 0.0, c0),
                Point::new(-0.5, 0.0, -c0),
                Point::new(c0, 0.5, 0.0),
                Point::new(c0, -0.5, 0.0),
                Point::new(-c0, 0.5, 0.0),
                Point::new(-c0, -0.5, 0.0),
                Point::new(0.0, c0, 0.5),
                Point::new(0.0, c0, -0.5),
                Point::new(0.0, -c0, 0.5),
                Point::new(0.0, -c0, -0.5),
            ],
            face_index: vec![
                vec![10, 2, 0],
                vec![5, 10, 0],
                vec![4, 5, 0],
                vec![8, 4, 0],
                vec![2, 8, 0],
                vec![6, 8, 2],
                vec![7, 6, 2],
                vec![10, 7, 2],
                vec![11, 7, 10],
                vec![5, 11, 10],
                vec![1, 11, 5],
                vec![4, 1, 5],
                vec![9, 1, 4],
                vec![8, 9, 4],
                vec![6, 9, 8],
                vec![3, 9, 6],
                vec![7, 3, 6],
                vec![11, 3, 7],
                vec![1, 3, 11],
                vec![9, 3, 1],
            ],
            face_set_index: vec![(0..20).collect()],
            name: String::from("I"),
        }
    }
}
