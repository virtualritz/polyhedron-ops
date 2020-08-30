use itertools::Itertools;
use nsi;
use ultraviolet::vec;

use std::{
    fs::File,
    io::Write,
    iter::Iterator,
    path::{Path, PathBuf},
};

use kiss3d::resource::Mesh;

use rayon::prelude::*;

type Float = f32;
type Index = u32;
type Face = Vec<Index>;
type FaceIndex = Vec<Face>;
// We treat an Edge as a Face with arity 2 to avoid copying in certain
// cases.
type Edge = Face;
type EdgeIndex = Vec<Edge>;

type FlatPoints = Vec<Float>;
type Point = ultraviolet::vec::Vec3; //  cgmath::Point3<Float>;
type Vector = ultraviolet::vec::Vec3;
type Normal = Vector;
type Points = Vec<Point>;
type PointsRef<'a> = Vec<&'a Point>;
type Normals = Vec<Normal>;

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

#[inline]
fn to_vadd(points: &Points, v: &Vector) -> Points {
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
fn to_centroid_points(points: &Points) -> Points {
    to_vadd(points, &-centroid(points))
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
fn average_magnitude(points: &Points) -> Float {
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
fn ordered_face_edges_(face: &Face) -> EdgeIndex {
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
fn edge_lengths(edges: &EdgeIndex, points: &Points) -> Vec<Float> {
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
fn circumscribed_resize(points: &mut Points, radius: Float) {
    center_on_centroid(points);
    let average = average_magnitude(points);

    points.par_iter_mut().for_each(|v| *v *= radius / average);
}

fn max_resize(points: &mut Points, radius: Float) {
    center_on_centroid(points);
    let max = max_magnitude(points);

    points.par_iter_mut().for_each(|v| *v *= radius / max);
}

#[inline]
fn project_on_sphere(points: &mut Points, radius: Float) {
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
// FIXME: rename to face_as_points
fn as_points<'a>(f: &[Index], points: &'a Points) -> PointsRef<'a> {
    f.par_iter().map(|index| &points[*index as usize]).collect()
}

#[inline]
fn orthogonal(v0: &Point, v1: &Point, v2: &Point) -> Vector {
    (*v1 - *v0).cross(*v2 - *v1)
}

/// Computes the normal of a face.
/// Assumes the face is planar.
#[inline]
fn face_normal(points: &PointsRef) -> Vector {
    // FIXME iterate over all points to make this work for
    // non-planar faces.
    -orthogonal(&points[0], &points[1], &points[2]).normalized()
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
fn vertex_ids(
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
    let edge_index: EdgeIndex = faces
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
        .collect();

    edge_index.into_iter().unique().collect()
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
    name: String,
}

impl Polyhedron {
    //[ for (f=faces) if(v!=[] && search(v,f)) f ];

    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            face_index: Vec::new(),
            name: String::new(),
        }
    }

    #[inline]
    pub fn name(&self) -> &String {
        &self.name
    }

    #[inline]
    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    #[inline]
    pub fn normalize(&mut self) {
        max_resize(&mut self.points, 1.);
    }

    #[inline]
    pub fn edges(&self) -> EdgeIndex {
        distinct_edges(&self.face_index)
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

    #[inline]
    pub fn ambo(&mut self, change_name: bool) {
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

        self.face_index = face_index;
        self.points = vertex_values(&points);

        if change_name {
            self.name = format!("a{}", self.name);
        }
    }

    pub fn bevel(&mut self, change_name: bool) {
        self.truncate(None, false);
        self.ambo(false);
        if change_name {
            self.name = format!("b{}", self.name);
        }
    }

    pub fn chamfer(&mut self, ratio: Float, change_name: bool) {
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
                        *face_points[j] + ratio * (centroid - *face_points[j]),
                    ))
                });
                result
            })
            .collect::<Vec<_>>();

        let new_ids = vertex_ids_ref(&new_points, self.num_points() as Index);

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

        self.face_index = face_index;
        self.points.par_iter_mut().for_each(|point| {
            *point = (1.5 * ratio) * *point;
        });
        self.points.extend(vertex_values(&new_points));

        if change_name {
            self.name = format!("g{}", self.name);
        }
    }

    pub fn dual(&mut self, change_name: bool) {
        let new_points = self
            .face_index
            .par_iter()
            .map(|face| centroid_ref(&as_points(face, &self.points)))
            .collect();
        self.face_index = Self::points_to_faces(self);
        self.points = new_points;
        if change_name {
            self.name = format!("d{}", self.name);
        }
    }

    pub fn expand(&mut self, change_name: bool) {
        self.ambo(false);
        self.ambo(false);
        if change_name {
            self.name = format!("e{}", self.name);
        }
    }

    pub fn gyro(
        &mut self,
        ratio: f32, /* 0.3333 */
        height: f32,
        change_name: bool,
    ) {
        // retain original points, add face centroids and directed
        // edge points each N-face becomes N pentagons
        let mut new_points: Vec<(&Face, Point)> = self
            .face_index
            .par_iter()
            .map(|face| {
                let fp = as_points(face, &self.points);
                (
                    face,
                    centroid_ref(&fp).normalized() + face_normal(&fp) * height,
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
                            + ratio * (*edge_points[1] - *edge_points[0]),
                    ),
                    (
                        &reversed_edges[edge.0],
                        *edge_points[1]
                            + ratio * (*edge_points[0] - *edge_points[1]),
                    ),
                ]
            })
            .collect::<Vec<_>>();

        new_points.extend(new_points2);
        //  2 points per edge

        let new_ids =
            vertex_ids_ref_ref(&new_points, self.num_points() as Index);

        //self.points =
        //self.points.iter_mut().map(|p| normalize(p)).collect();
        self.points.extend(vertex_values_as_ref(&new_points));

        self.face_index = self
            .face_index
            .par_iter()
            .flat_map(|face| {
                let mut new_faces = Vec::with_capacity(face.len());
                for j in 0..face.len() {
                    let a = face[j];
                    let b = face[(j + 1) % face.len()];
                    let z = face[(j + face.len() - 1) % face.len()];
                    let eab = vertex(&vec![a, b], &new_ids).unwrap();
                    let eza = vertex(&vec![z, a], &new_ids).unwrap();
                    let eaz = vertex(&vec![a, z], &new_ids).unwrap();
                    let centroid = vertex(face, &new_ids).unwrap();
                    new_faces.push(vec![a, eab, centroid, eza, eaz]);
                }
                new_faces
            })
            .collect();

        if change_name {
            self.name = format!("g{}", self.name);
        }
    }

    pub fn join(&mut self, change_name: bool) {
        self.dual(false);
        self.ambo(false);
        self.dual(false);
        if change_name {
            self.name = format!("j{}", self.name);
        }
    }

    /// kis – each face with a specified arity n is divided into n
    /// triangles which extend to the face centroid existimg points
    /// retained.
    pub fn kis(
        &mut self,
        height: Float,
        face_arity: Option<&Vec<usize>>,
        regular: bool,
        change_name: bool,
    ) {
        let new_points: Vec<(&Face, Point)> = self
            .face_index
            .par_iter()
            .filter(|face| {
                selected_face(face, face_arity) && !regular
                    || ((face_irregularity(face, &self.points) - 1.0).abs()
                        < 0.1)
            })
            .map(|face| {
                let fp = as_points(face, &self.points);
                (face, centroid_ref(&fp) + face_normal(&fp) * height)
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
                        vec![face[j], face[(j + 1) % face.len()], centroid as Index]
                    })
                    .collect(),
                None => vec![face.clone()],
            })
            .collect();

        if change_name {
            match face_arity {
                Some(face_arity) => {
                    self.name = format!("k{:?}{}", face_arity, self.name)
                }
                None => self.name = format!("k{}", self.name),
            }
        }
    } // end kis

    pub fn meta(&mut self, change_name: bool) {
        self.kis(0., Some(&vec![3]), false, false);
        self.join(false);
        if change_name {
            self.name = format!("m{}", self.name);
        }
    }

    pub fn needle(&mut self, change_name: bool) {
        self.dual(false);
        self.truncate(None, false);
        if change_name {
            self.name = format!("n{}", self.name);
        }
    }

    pub fn ortho(&mut self, change_name: bool) {
        self.join(false);
        self.join(false);
        if change_name {
            self.name = format!("o{}", self.name);
        }
    }

    pub fn propellor(&mut self, ratio: Float, change_name: bool) {
        let edges = self.edges();

        let reversed_edges: EdgeIndex = edges
            .par_iter()
            .map(|edge| vec![edge[1], edge[0]])
            .collect();

        let new_points = edges
            .par_iter()
            .enumerate()
            .flat_map(|edge| {
                let egdge_points = as_points(edge.1, &self.points);
                vec![
                    (
                        edge.1,
                        *egdge_points[0]
                            + ratio * (*egdge_points[1] - *egdge_points[0]),
                    ),
                    (
                        &reversed_edges[edge.0],
                        *egdge_points[1]
                            + ratio * (*egdge_points[0] - *egdge_points[1]),
                    ),
                ]
            })
            .collect::<Vec<_>>();

        let new_ids =
            vertex_ids_ref_ref(&new_points, self.num_points() as Index);

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
        println!("{}", self.num_points());
        self.points.extend(vertex_values_as_ref(&new_points));
        println!("{}", self.num_points());
        if change_name {
            self.name = format!("p{}", self.name);
        }
    }

    pub fn quinto(&mut self, change_name: bool) {
        let edges = self.edges();
        let mut new_points: Vec<(Face, Point)> = edges
            .par_iter()
            .map(|edge| {
                let edge_points = as_points(edge, &self.points);
                (edge.clone(), 0.5 * (*edge_points[0] + *edge_points[1]))
            })
            .collect();

        new_points.extend(
            self.face_index
                .par_iter()
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

        let new_ids = vertex_ids_ref(&new_points, self.num_points() as u32);

        let mut new_faces: FaceIndex = self
            .face_index
            .par_iter()
            .map(|face| {
                (0..face.len())
                    .map(|face_vertex| {
                        vertex(&extend![..face, face_vertex as u32], &new_ids)
                            .unwrap()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        new_faces.extend(
            self.face_index
                .par_iter()
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

        self.face_index = new_faces;
        self.points.extend(vertex_values_as_ref(&new_points));
        if change_name {
            self.name = format!("q{}", self.name);
        }
    }

    /*
         let (newids = vertex_ids(newv,len(pv)))
         let (newf =
              concat(
              [for (face=pf)    // reduced faces
                  [for (j=[0:len(face)-1])
                   let (nv=vertex([face,j],newids))
                   nv
                  ]
                  ]

                  ,

               [for (face=pf)
                   for (i = [0:len(face)-1])
                   let (v = face[i])
                   let (e0 = [face[(i-1+len(face)) % len(face)],face[i]])
                   let (e1 = [face[i] , face[(i + 1) % len(face)]])
                   let (e0p = vertex(distinct_edge(e0),newids))
                   let (e1p = vertex(distinct_edge(e1),newids))
                   let (iv0 = vertex([face,(i -1 + len(face)) % len(face)],newids))
                   let (iv1 = vertex([face,i],newids))
                   [v,e1p,iv1,iv0,e0p]

                  // [v,e0p,iv0,iv1,e1p]
                  ]

               ) )

         poly(name=str("q",p_name(obj)),
              vertices= concat(pv,vertex_values(newv)),
              faces=newf
             )
    ; // end quinta
    */
    pub fn reflect(&mut self, change_name: bool) {
        self.points = self
            .points
            .par_iter()
            .map(|v| Point::new(v.x, -v.y, v.z))
            .collect();
        self.reverse();
        if change_name {
            self.name = format!("r{}", self.name);
        }
    }

    pub fn snub(&mut self, change_name: bool) {
        self.dual(false);
        self.gyro(1. / 3., 0., false);
        self.dual(false);
        if change_name {
            self.name = format!("s{}", self.name);
        }
    }

    pub fn truncate(
        &mut self,
        vertex_valence: Option<&Vec<usize>>,
        change_name: bool,
    ) {
        self.dual(false);
        self.kis(0., vertex_valence, false, false);
        self.dual(false);

        if change_name {
            match vertex_valence {
                Some(vertex_valence) => {
                    self.name = format!("t{:?}{}", vertex_valence, self.name)
                }
                None => self.name = format!("t{}", self.name),
            }
        }
    }

    pub fn reverse(&mut self) {
        self.face_index = self
            .face_index
            .par_iter()
            .map(|f| {
                let mut new_face = f.clone();
                new_face.reverse();
                new_face
            })
            .collect();
    }

    pub fn normals(&self, normal_type: NormalType) -> Normals {
        match normal_type {
            NormalType::Smooth(angle) => vec![],
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
                3 * self.num_points(),
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
    ) -> std::io::Result<PathBuf> {
        let path = destination.join(format!("polyhedron-{}.obj", self.name));
        let mut file = File::create(path.clone())?;

        writeln!(file, "o {}", self.name)?;

        self.points.iter().for_each(|vertex| {
            writeln!(file, "v {} {} {}", vertex.x, vertex.y, vertex.z).unwrap()
        });

        match reverse_winding {
            true => self.face_index.iter().for_each(|face| {
                write!(file, "f");
                face.iter().rev().for_each(|vertex_index| {
                    write!(file, " {}", vertex_index + 1).unwrap()
                });
                writeln!(file, "");
            }),
            false => self.face_index.iter().for_each(|face| {
                write!(file, "f");
                face.iter().for_each(|vertex_index| {
                    write!(file, " {}", vertex_index + 1).unwrap()
                });
                writeln!(file, "");
            }),
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
            name: String::from("T"),
        }
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
            name: String::from("I"),
        }
    }
}

use nalgebra as na;

impl From<Polyhedron> for kiss3d::resource::Mesh {
    fn from(mut polyhedron: Polyhedron) -> kiss3d::resource::Mesh {
        polyhedron.reverse();

        /*
        let mut normals_polyhedron = Polyhedron {
            points: normals.clone(),
            face_index: {
                let mut index = 0u32;
                polyhedron
                    .face_index
                    .par_iter()
                    .map(|f| {
                        let face =
                            (index..index + f.len() as u32).collect();
                        index += f.len() as u32;
                        face
                    })
                    .collect()
            },
        };

        polyhedron.triangulate(false);
        normals_polyhedron.triangulate(false);

        // We now have two meshes with identical topology but different
        // index arrays. We unify the mapping.
        // FIXME: some points will be written to multiple
        let mut normals = vec![
            na::Vector3::new(0.0f32, 0., 0.);
            polyhedron.num_points()
        ];

        for f in 0..polyhedron.face_index.len() {
            for i in 0..polyhedron.face_index[f].len() {
                let v = normals_polyhedron.points
                    [normals_polyhedron.face_index[f][i] as usize];

                normals[polyhedron.face_index[f][i] as usize] =
                    na::Vector3::new(v.x, v.y, v.z);
            }
        }*/
        polyhedron.triangulate(true);

        let normals = polyhedron
            .normals(NormalType::Flat)
            .par_iter()
            .map(|n| na::Vector3::new(-n.x, -n.y, -n.z))
            .collect::<Vec<_>>();

        let face_index = (0..normals.len() as u16)
            .step_by(3)
            .map(|i| na::Point3::new(i, i + 1, i + 2))
            .collect::<Vec<_>>();

        Mesh::new(
            // Dupliacate points per face so we can
            // match the normals per face.
            polyhedron
                .face_index
                .par_iter()
                .flat_map(|f| {
                    as_points(f, &polyhedron.points)
                        .par_iter()
                        .map(|v| na::Point3::<f32>::new(v.x, v.y, v.z))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
            face_index,
            Some(normals),
            None,
            false,
        )

        /* smooth shaded mesh
        Mesh::new(
            mesh.points
                .par_iter()
                .map(|v| na::Point3::<f32>::new(v.x, v.y, v.z))
                .collect(),
            mesh.face_index
                .par_iter()
                .map(|f| na::Point3::new(f[0] as u16, f[1] as u16, f[2] as u16))
                .collect(),
            None,
            None,
            false,
        )*/
    }
}
