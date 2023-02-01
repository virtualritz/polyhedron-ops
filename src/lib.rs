#![allow(clippy::many_single_char_names)]
//! # Conway-Hart Polyhedron Operations
//!
//! This crate implements the [Conway Polyhedron
//! Operators](http://en.wikipedia.org/wiki/Conway_polyhedron_notation)
//! and their extensions by [George W. Hart](http://www.georgehart.com/) and others.
//!
//! The internal representation uses *n*-gon mesh buffers.  These need
//! preprocessing before they can be sent to a GPU but are almost fine to send
//! to an offline renderer, as-is.
//!
//! See the `playground` example for code on how to do either.
//!
//! ## Example
//! ```
//! use polyhedron_ops::Polyhedron;
//! use std::path::Path;
//!
//! // Conway notation: gapcD
//! let polyhedron = Polyhedron::dodecahedron()
//!     .chamfer(None, true)
//!     .propellor(None, true)
//!     .ambo(None, true)
//!     .gyro(None, None, true)
//!     .finalize();
//!
//! // Export as ./polyhedron-gapcD.obj
//! # #[cfg(feature = "obj")]
//! # {
//! polyhedron.write_to_obj(&Path::new("."), false);
//! # }
//! ```
//! The above code starts from a [dodecahedron](https://en.wikipedia.org/wiki/Dodecahedron)
//! and iteratively applies four operators.
//!
//! The resulting shape is shown below.
//!
//! ![](https://raw.githubusercontent.com/virtualritz/polyhedron-operators/HEAD/gapcD.jpg)
//!
//! ## Cargo Features
//!
//! ```toml
//! [dependencies]
//! polyhedron-ops = { version = "0.3", features = [ "bevy", "nsi", "obj" ] }
//! ```
//!
//! * `bevy` – A polyhedro can be converted into a [`bevy`](https://bevyengine.org/)
//!   [`Mesh`](https://docs.rs/bevy/latest/bevy/render/mesh/struct.Mesh.html).
//!   See the `bevy` example. ```ignore Mesh::from(polyhedron) ```
//!
//! * `nsi` – Add supports for sending data to renderers implementing the [ɴsɪ](https://crates.io/crates/nsi/)
//!   API. The function is called [`to_nsi()`](Polyhedron::to_nsi()).
//!
//! * `obj` – Add support for output to [Wavefront OBJ](https://en.wikipedia.org/wiki/Wavefront_.obj_file)
//!   via the [`write_to_obj()`](Polyhedron::write_to_obj()) function.
use itertools::Itertools;
use num_traits::FloatConst;
use rayon::prelude::*;
#[cfg(feature = "obj")]
use std::{
    error::Error,
    fs::File,
    io::Write as IoWrite,
    path::{Path, PathBuf},
};
use std::{
    fmt::{Display, Write},
    iter::Iterator,
};
use ultraviolet as uv;

mod helpers;
use helpers::*;

#[cfg(test)]
mod tests;

pub mod prelude {
    //! Re-exports commonly used types and traits.
    //!
    //! Importing the contents of this module is recommended.
    pub use crate::*;
}

static EPSILON: f32 = 0.00000001;

pub type Float = f32;
pub type VertexKey = u32;
pub type FaceKey = u32;
pub type Face = Vec<VertexKey>;
pub(crate) type FaceSlice = [VertexKey];
pub type Faces = Vec<Face>;
pub(crate) type FacesSlice = [Face];
pub type FaceSet = Vec<VertexKey>;
pub type Edge = [VertexKey; 2];
pub type Edges = Vec<Edge>;
pub type EdgesSlice = [Edge];
pub(crate) type _EdgeSlice = [Edge];
pub type Point = uv::vec::Vec3;
pub type Vector = uv::vec::Vec3;
pub type Normal = Vector;
#[allow(dead_code)]
pub type Normals = Vec<Normal>;
pub type Points = Vec<Point>;
pub(crate) type PointsSlice = [Point];
pub(crate) type PointsRefSlice<'a> = [&'a Point];

#[derive(Clone, Debug)]
pub struct Polyhedron {
    face_index: Faces,
    positions: Points,
    name: String,
    // This stores a FaceSet for each
    // set of faces belonging to the
    // same operations.
    face_set_index: Vec<FaceSet>,
}

impl Default for Polyhedron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "tilings")]
use tilings::RegularTiling;
#[cfg(feature = "tilings")]
impl From<RegularTiling> for Polyhedron {
    fn from(rt: RegularTiling) -> Polyhedron {
        Polyhedron {
            positions: rt
                .positions()
                .iter()
                .map(|p| Point::new(p.x, 0.0, p.y))
                .collect(),
            face_index: rt.faces().clone(),
            face_set_index: Vec::new(),
            name: rt.name().to_string(),
        }
    }
}

#[cfg(feature = "tilings")]
use tilings::SemiRegularTiling;
#[cfg(feature = "tilings")]
impl From<SemiRegularTiling> for Polyhedron {
    fn from(rt: SemiRegularTiling) -> Polyhedron {
        Polyhedron {
            positions: rt
                .positions()
                .iter()
                .map(|p| Point::new(p.x, 0.0, p.y))
                .collect(),
            face_index: rt.faces().clone(),
            face_set_index: Vec::new(),
            name: rt.name().to_string(),
        }
    }
}

impl Polyhedron {
    #[inline]
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            face_index: Vec::new(),
            face_set_index: Vec::new(),
            name: String::new(),
        }
    }

    #[inline]
    pub fn from(
        name: &str,
        positions: Points,
        face_index: Faces,
        face_set_index: Option<Vec<FaceSet>>,
    ) -> Self {
        Self {
            positions,
            face_index,
            face_set_index: face_set_index.unwrap_or_default(),
            name: name.to_string(),
        }
    }

    /// Returns the axis-aligned bounding box of the polyhedron in the format
    /// `[x_min, y_min, z_min, x_max, y_max, z_max]`.
    #[inline]
    pub fn bounding_box(&self) -> [f64; 6] {
        let mut bounds = [0.0f64; 6];
        self.positions.iter().for_each(|point| {
            if bounds[0] > point.x as _ {
                bounds[0] = point.x as _;
            } else if bounds[3] < point.x as _ {
                bounds[3] = point.x as _;
            }

            if bounds[1] > point.y as _ {
                bounds[1] = point.y as _;
            } else if bounds[4] < point.y as _ {
                bounds[4] = point.y as _;
            }

            if bounds[2] > point.z as _ {
                bounds[2] = point.z as _;
            } else if bounds[5] < point.z as _ {
                bounds[5] = point.z as _;
            }
        });
        bounds
    }

    /// Appends indices for newly added faces as a new FaceSet to the
    /// FaceSetIndex.
    #[inline]
    fn append_new_face_set(&mut self, size: usize) {
        self.face_set_index
            .append(&mut vec![((self.face_index.len() as VertexKey)
                ..((self.face_index.len() + size) as VertexKey))
                .collect()]);
    }

    /// Creates vertices with valence (aka degree) four.

    /// It is also called [rectification](https://en.wikipedia.org/wiki/Rectification_(geometry)),
    /// or the  [medial graph](https://en.wikipedia.org/wiki/Medial_graph) in graph theory.
    #[inline]
    pub fn ambo(
        &mut self,
        ratio: Option<Float>,
        change_name: bool,
    ) -> &mut Self {
        let ratio_ = match ratio {
            Some(r) => r.clamp(0.0, 1.0),
            None => 1. / 2.,
        };

        let edges = self.to_edges();

        let positions: Vec<(&Edge, Point)> = edges
            .par_iter()
            .map(|edge| {
                let edge_positions = index_as_positions(edge, &self.positions);
                (
                    edge,
                    ratio_ * *edge_positions[0]
                        + (1.0 - ratio_) * *edge_positions[1],
                )
            })
            .collect();

        let new_ids = vertex_ids_edge_ref_ref(&positions, 0);

        let face_index: Faces = self
            .face_index
            .par_iter()
            .map(|face| {
                let edges = distinct_face_edges(face);
                let result = edges
                    .iter()
                    .filter_map(|edge| vertex_edge(edge, &new_ids))
                    .collect::<Vec<_>>();
                result
            })
            .chain(
                self.positions
                    // Each old vertex creates a new face ...
                    .par_iter()
                    .enumerate()
                    .map(|(polygon_vertex, _)| {
                        let vertex_number = polygon_vertex as VertexKey;
                        ordered_vertex_edges(
                            vertex_number,
                            &vertex_faces(vertex_number, &self.face_index),
                        )
                        .iter()
                        .map(|ve| {
                            vertex_edge(&distinct_edge(ve), &new_ids).unwrap()
                        })
                        .collect::<Vec<_>>()
                    }),
            )
            .collect();

        self.append_new_face_set(face_index.len());

        self.face_index = face_index;
        self.positions = vertex_values(&positions);

        if change_name {
            let mut params = String::new();
            if let Some(ratio) = ratio {
                write!(&mut params, "{:.2}", ratio).unwrap();
            }
            self.name = format!("a{}{}", params, self.name);
        }

        self
    }

    pub fn bevel(
        &mut self,
        ratio: Option<Float>,
        height: Option<Float>,
        vertex_valence: Option<&[usize]>,
        regular_faces_only: Option<bool>,
        change_name: bool,
    ) -> &mut Self {
        self.truncate(height, vertex_valence, regular_faces_only, false);
        self.ambo(ratio, false);

        if change_name {
            let mut params = String::new();
            if let Some(ratio) = ratio {
                write!(&mut params, "{:.2}", ratio).unwrap();
            }
            if let Some(height) = height {
                write!(&mut params, ",{:.2}", height).unwrap();
            } else {
                write!(&mut params, ",").unwrap();
            }
            if let Some(vertex_valence) = vertex_valence {
                write!(&mut params, ",{}", format_slice(vertex_valence))
                    .unwrap();
            } else {
                write!(&mut params, ",").unwrap();
            }
            if let Some(regular_faces_only) = regular_faces_only {
                if regular_faces_only {
                    params.push_str(",{t}");
                }
            } else {
                write!(&mut params, ",").unwrap();
            }
            params = params.trim_end_matches(',').to_string();
            self.name = format!("b{}{}", params, self.name);
        }

        self
    }

    /// Apply proper canoicalization. Typical number of `iterarations` are
    /// `200`+.
    /// FIXME: this is b0rked atm.
    #[inline]
    pub fn _canonicalize(
        &mut self,
        iterations: Option<usize>,
        change_name: bool,
    ) {
        let mut dual = self.clone().dual(false).finalize();

        for _ in 0..iterations.unwrap_or(200) {
            // Reciprocate faces.
            dual.positions =
                reciprocate_faces(&self.face_index, &self.positions);
            self.positions =
                reciprocate_faces(&dual.face_index, &dual.positions);
        }

        if change_name {
            let mut params = String::new();
            if let Some(iterations) = iterations {
                write!(&mut params, "{}", iterations).unwrap();
            }
            self.name = format!("N{}{}", params, self.name);
        }
    }

    /// Performs Catmull-Clark subdivision.
    ///
    /// Each face is replaced with *n* quadralaterals based on edge midpositions
    /// vertices and centroid edge midpositions are average of edge endpositions
    /// and adjacent centroids original vertices replaced by weighted
    /// average of original vertex, face centroids and edge midpositions.
    pub fn catmull_clark_subdivide(&mut self, change_name: bool) -> &mut Self {
        let new_face_vertices = self
            .face_index
            .par_iter()
            .map(|face| {
                let face_positions = index_as_positions(face, &self.positions);
                (face.as_slice(), centroid_ref(&face_positions))
            })
            .collect::<Vec<_>>();

        let edges = self.to_edges();

        let new_edge_vertices = edges
            .par_iter()
            .map(|edge| {
                let ep = index_as_positions(edge, &self.positions);
                let af1 = face_with_edge(edge, &self.face_index);
                let af2 = face_with_edge(&[edge[1], edge[0]], &self.face_index);
                let fc1 =
                    vertex_point(&af1, new_face_vertices.as_slice()).unwrap();
                let fc2 =
                    vertex_point(&af2, new_face_vertices.as_slice()).unwrap();
                (edge, (*ep[0] + *ep[1] + *fc1 + *fc2) * 0.25)
            })
            .collect::<Vec<_>>();

        let new_face_vertex_ids = vertex_ids_ref_ref(
            new_face_vertices.as_slice(),
            self.positions.len() as _,
        );
        let new_edge_vertex_ids = vertex_ids_edge_ref_ref(
            new_edge_vertices.as_slice(),
            (self.positions.len() + new_face_vertices.len()) as _,
        );

        let new_face_index = self
            .face_index
            .par_iter()
            .flat_map(|face| {
                let centroid = vertex(face, &new_face_vertex_ids).unwrap();

                face.iter()
                    .circular_tuple_windows::<(_, _, _)>()
                    .map(|triplet| {
                        let mid1 = vertex_edge(
                            &distinct_edge(&[*triplet.0, *triplet.1]),
                            new_edge_vertex_ids.as_slice(),
                        )
                        .unwrap();
                        let mid2 = vertex_edge(
                            &distinct_edge(&[*triplet.1, *triplet.2]),
                            new_edge_vertex_ids.as_slice(),
                        )
                        .unwrap();
                        vec![centroid, mid1, *triplet.1, mid2]
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let new_positions = self
            .positions
            .par_iter()
            .enumerate()
            .map(|point| {
                let i = point.0 as u32;
                let v = point.1;
                let vertex_faces = vertex_faces(i, &self.face_index)
                    .iter()
                    .map(|face| {
                        vertex_point(face, new_face_vertices.as_slice())
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                let n = vertex_faces.len() as Float;
                let f = centroid_ref(&vertex_faces);
                let r = centroid_ref(
                    &vertex_edges(i, &edges)
                        .iter()
                        .map(|edge| {
                            vertex_edge_point(
                                edge,
                                new_edge_vertices.as_slice(),
                            )
                            .unwrap()
                        })
                        .collect::<Vec<_>>(),
                );
                (f + 2.0 * r + (n - 3.0) * *v) / n
            })
            .chain(vertex_values(new_face_vertices.as_slice()))
            .chain(vertex_values(new_edge_vertices.as_slice()))
            .collect::<Points>();

        self.positions = new_positions;
        self.face_index = new_face_index;

        if change_name {
            self.name = format!("v{}", self.name);
        }

        self
    }

    pub fn chamfer(
        &mut self,
        ratio: Option<Float>,
        change_name: bool,
    ) -> &mut Self {
        let ratio_ = match ratio {
            Some(r) => r.clamp(0.0, 1.0),
            None => 1. / 2.,
        };

        let new_positions: Vec<(Face, Point)> = self
            .face_index
            .par_iter()
            .flat_map(|face| {
                let face_positions = index_as_positions(face, &self.positions);
                let centroid = centroid_ref(&face_positions);
                // println!("{:?}", ep);
                let mut result = Vec::new();
                face.iter().enumerate().for_each(|face_point| {
                    let j = face_point.0;
                    let mut new_face = face.clone();
                    new_face.push(face[j]);
                    result.push((
                        new_face,
                        *face_positions[j]
                            + ratio_ * (centroid - *face_positions[j]),
                    ))
                });
                result
            })
            .collect();

        let new_ids =
            vertex_ids_ref(&new_positions, self.positions_len() as VertexKey);

        let face_index: Faces = self
            .face_index
            .par_iter()
            .map(|face| {
                // FIXME: use iterators with double collect
                let mut new_face = Vec::with_capacity(face.len());
                face.iter().for_each(|vertex_key| {
                    let mut face_key = face.clone();
                    face_key.push(*vertex_key);
                    new_face.push(vertex(&face_key, &new_ids).unwrap());
                });
                new_face
            })
            .chain(self.face_index.par_iter().flat_map(|face| {
                face.iter()
                    .circular_tuple_windows::<(_, _, _)>()
                    .filter_map(|v| {
                        if v.0 < v.1 {
                            let a: VertexKey = *v.0;
                            let b: VertexKey = *v.1;
                            let opposite_face = face_with_edge(&[b, a], &self.face_index);

                            /*once(a)
                            .chain(once(vertex(&extend![..opposite_face, a], &new_ids).unwrap()))
                            .chain(once(vertex(&extend![..opposite_face, b], &new_ids).unwrap()))
                            .chain(once(b))
                            .chain(once(vertex(&extend![..face, b], &new_ids).unwrap()))
                            .chain(once(vertex(&extend![..face, a], &new_ids).unwrap()))*/

                            Some(vec![
                                a,
                                vertex(&extend![..opposite_face, a], &new_ids).unwrap(),
                                vertex(&extend![..opposite_face, b], &new_ids).unwrap(),
                                b,
                                vertex(&extend![..face, b], &new_ids).unwrap(),
                                vertex(&extend![..face, a], &new_ids).unwrap(),
                            ])
                        } else {
                            None
                        }
                    })
                    .collect::<Faces>()
            }))
            .collect::<Faces>();

        self.append_new_face_set(face_index.len());

        self.face_index = face_index;
        self.positions.par_iter_mut().for_each(|point| {
            *point = (1.5 * ratio_) * *point;
        });
        self.positions.extend(vertex_values(&new_positions));

        if change_name {
            let mut params = String::new();
            if let Some(ratio) = ratio {
                write!(&mut params, "{:.2}", ratio).unwrap();
            }
            self.name = format!("c{}{}", params, self.name);
        }

        self
    }

    /// Replaces each face with a vertex, and each vertex with a face.
    pub fn dual(&mut self, change_name: bool) -> &mut Self {
        let new_positions = face_centers(&self.face_index, &self.positions);
        self.face_index = positions_to_faces(&self.positions, &self.face_index);
        self.positions = new_positions;
        // FIXME: FaceSetIndex

        if change_name {
            self.name = format!("d{}", self.name);
        }

        self
    }

    pub fn expand(
        &mut self,
        ratio: Option<Float>,
        change_name: bool,
    ) -> &mut Self {
        self.ambo(ratio, false);
        self.ambo(ratio, false);

        if change_name {
            let mut params = String::new();
            if let Some(ratio) = ratio {
                write!(&mut params, "{:.2}", ratio).unwrap();
            }
            self.name = format!("e{}{}", params, self.name);
        }

        self
    }

    /// Splits each edge and connects new edges at the split point to the face
    /// centroid. Existing positions are retained.
    /// ![Gyro](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Conway_gC.png/200px-Conway_gC.png)
    /// # Arguments
    /// * `ratio` – The ratio at which the adjacent edges get split.
    /// * `height` – An offset to add to the face centroid point along the face
    ///   normal.
    /// * `regular_faces_only` – Only faces whose edges are 90% the same length,
    ///   within the same face, are affected.
    pub fn gyro(
        &mut self,
        ratio: Option<f32>,
        height: Option<f32>,
        change_name: bool,
    ) -> &mut Self {
        let ratio_ = match ratio {
            Some(r) => r.clamp(0.0, 1.0),
            None => 1. / 3.,
        };
        let height_ = height.unwrap_or(0.);

        let edges = self.to_edges();
        let reversed_edges: Edges =
            edges.par_iter().map(|edge| [edge[1], edge[0]]).collect();

        // Retain original positions, add face centroids and directed
        // edge positions each N-face becomes N pentagons.
        let new_positions: Vec<(&FaceSlice, Point)> = self
            .face_index
            .par_iter()
            .map(|face| {
                let fp = index_as_positions(face, &self.positions);
                (
                    face.as_slice(),
                    centroid_ref(&fp).normalized()
                        + average_normal_ref(&fp).unwrap() * height_,
                )
            })
            .chain(edges.par_iter().enumerate().flat_map(|edge| {
                let edge_positions =
                    index_as_positions(edge.1, &self.positions);
                vec![
                    (
                        &edge.1[..],
                        *edge_positions[0]
                            + ratio_
                                * (*edge_positions[1] - *edge_positions[0]),
                    ),
                    (
                        &reversed_edges[edge.0][..],
                        *edge_positions[1]
                            + ratio_
                                * (*edge_positions[0] - *edge_positions[1]),
                    ),
                ]
            }))
            .collect();

        let new_ids = vertex_ids_ref_ref(
            &new_positions,
            self.positions_len() as VertexKey,
        );

        self.positions.extend(vertex_values_as_ref(&new_positions));

        self.face_index = self
            .face_index
            .par_iter()
            .flat_map(|face| {
                face.iter()
                    .cycle()
                    .skip(face.len() - 1)
                    .tuple_windows::<(_, _, _)>()
                    .take(face.len())
                    .map(|v| {
                        let a = *v.1;
                        let b = *v.2;
                        let z = *v.0;
                        let eab = vertex(&[a, b], &new_ids).unwrap();
                        let eza = vertex(&[z, a], &new_ids).unwrap();
                        let eaz = vertex(&[a, z], &new_ids).unwrap();
                        let centroid = vertex(face, &new_ids).unwrap();
                        vec![a, eab, centroid, eza, eaz]
                    })
                    .collect::<Faces>()
            })
            .collect();

        if change_name {
            let mut params = String::new();
            if let Some(ratio) = ratio {
                write!(&mut params, "{:.2}", ratio).unwrap();
            }
            if let Some(height) = height {
                write!(&mut params, ",{:.2}", height).unwrap();
            }
            self.name = format!("g{}{}", params, self.name);
        }

        self
    }

    /// Creates quadrilateral faces around each original edge. Original
    /// edges are discarded.
    /// # Arguments
    /// * `ratio` – The ratio at which the adjacent edges get split. Will be
    ///   clamped to `[0, 1]`. Default value is `0.5`.
    pub fn join(
        &mut self,
        ratio: Option<Float>,
        change_name: bool,
    ) -> &mut Self {
        self.dual(false);
        self.ambo(ratio, false);
        self.dual(false);

        if change_name {
            let mut params = String::new();
            if let Some(ratio) = ratio {
                write!(&mut params, "{:.2}", ratio).unwrap();
            }
            self.name = format!("j{}{}", params, self.name);
        }

        self
    }

    /// Splits each face into triangles, one for each edge, which
    /// extend to the face centroid. Existing positions are retained.
    /// # Arguments
    /// * `height` - An offset to add to the face centroid point along the face
    ///   normal.
    /// * `face_arity` - Only faces matching the given arities will be affected.
    /// * `regular_faces_only` - Only faces whose edges are 90% the same length,
    ///   within the same face, are affected.
    pub fn kis(
        &mut self,
        height: Option<Float>,
        face_arity_mask: Option<&[usize]>,
        face_index_mask: Option<&[FaceKey]>,
        regular_faces_only: Option<bool>,
        change_name: bool,
    ) -> &mut Self {
        let new_positions: Vec<(&FaceSlice, Point)> = self
            .face_index
            .par_iter()
            .enumerate()
            .filter_map(|(index, face)| {
                if is_face_selected(
                    face,
                    index,
                    &self.positions,
                    face_arity_mask,
                    face_index_mask,
                    regular_faces_only,
                ) {
                    let face_positions =
                        index_as_positions(face, &self.positions);
                    Some((
                        face.as_slice(),
                        centroid_ref(&face_positions)
                            + average_normal_ref(&face_positions).unwrap()
                                * height.unwrap_or(0.),
                    ))
                } else {
                    None
                }
            })
            .collect();

        let new_ids = vertex_ids_ref_ref(
            &new_positions,
            self.positions.len() as VertexKey,
        );

        self.positions.extend(vertex_values_as_ref(&new_positions));

        self.face_index = self
            .face_index
            .par_iter()
            .flat_map(|face: &Face| match vertex(face, &new_ids) {
                Some(centroid) => face
                    .iter()
                    .cycle()
                    .tuple_windows::<(&VertexKey, _)>()
                    .take(face.len())
                    .map(|v| vec![*v.0, *v.1, centroid as VertexKey])
                    .collect(),
                None => vec![face.clone()],
            })
            .collect();

        if change_name {
            let mut params = String::new();
            if let Some(height) = height {
                write!(&mut params, "{:.2}", height).unwrap();
            }
            if let Some(face_arity_mask) = face_arity_mask {
                write!(&mut params, ",{}", format_slice(face_arity_mask))
                    .unwrap();
            } else {
                write!(&mut params, ",").unwrap();
            }
            if let Some(face_index_mask) = face_index_mask {
                write!(&mut params, ",{}", format_slice(face_index_mask))
                    .unwrap();
            } else {
                write!(&mut params, ",").unwrap();
            }
            if let Some(regular_faces_only) = regular_faces_only {
                if regular_faces_only {
                    params.push_str(",{t}");
                }
            } else {
                write!(&mut params, ",").unwrap();
            }
            params = params.trim_end_matches(',').to_string();
            self.name = format!("k{}{}", params, self.name);
        }

        self
    }

    // Inset faces by `offset` from the original edges.
    pub fn inset(
        &mut self,
        offset: Option<Float>,
        face_arity: Option<&[usize]>,
        change_name: bool,
    ) -> &mut Self {
        if change_name {
            let mut params = String::new();
            if let Some(offset) = offset {
                write!(&mut params, "{:.2}", offset).unwrap();
            }
            if let Some(face_arity) = &face_arity {
                write!(&mut params, ",{}", format_slice(face_arity)).unwrap();
            }
            self.name = format!("i{}{}", params, self.name);
        }

        self.extrude(Some(0.0), Some(offset.unwrap_or(0.3)), face_arity, false);

        self
    }

    // Extrudes faces by `height` and shrinks the extruded faces by `distance`
    // from the original edges.
    pub fn extrude(
        &mut self,
        height: Option<Float>,
        offset: Option<Float>,
        face_arity: Option<&[usize]>,
        change_name: bool,
    ) -> &mut Self {
        let new_positions = self
            .face_index
            .par_iter()
            .filter(|face| face_arity_matches(face, face_arity))
            .flat_map(|face| {
                let face_positions = index_as_positions(face, &self.positions);
                let centroid = centroid_ref(&face_positions);
                face.iter()
                    .zip(&face_positions)
                    .map(|face_vertex_point| {
                        (
                            extend![..face, *face_vertex_point.0],
                            **face_vertex_point.1
                                + offset.unwrap_or(0.0)
                                    * (centroid - **face_vertex_point.1)
                                + average_normal_ref(&face_positions).unwrap()
                                    * height.unwrap_or(0.3),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let new_ids =
            vertex_ids_ref(&new_positions, self.positions_len() as VertexKey);

        self.face_index = self
            .face_index
            .par_iter()
            .flat_map(|face| {
                if face_arity_matches(face, face_arity) {
                    face.iter()
                        .enumerate()
                        .flat_map(|index_vertex| {
                            let a = *index_vertex.1;
                            let inset_a =
                                vertex(&extend![..face, a], &new_ids).unwrap();
                            let b = face[(index_vertex.0 + 1) % face.len()];
                            let inset_b =
                                vertex(&extend![..face, b], &new_ids).unwrap();
                            if height.unwrap_or(0.3).is_sign_positive() {
                                vec![vec![a, b, inset_b, inset_a]]
                            } else {
                                vec![vec![inset_a, inset_b, b, a]]
                            }
                        })
                        .chain(vec![face
                            .iter()
                            .map(|v| {
                                vertex(&extend![..face, *v], &new_ids).unwrap()
                            })
                            .collect::<Vec<_>>()])
                        .collect::<Vec<_>>()
                } else {
                    vec![face.clone()]
                }
            })
            .collect();

        self.positions.extend(vertex_values_as_ref(&new_positions));

        if change_name {
            let mut params = String::new();
            if let Some(height) = height {
                write!(&mut params, "{:.2}", height).unwrap();
            }
            if let Some(offset) = offset {
                write!(&mut params, ",{:.2}", offset).unwrap();
            } else {
                write!(&mut params, ",").unwrap();
            }
            if let Some(face_arity) = face_arity {
                write!(&mut params, ",{}", format_slice(face_arity)).unwrap();
            } else {
                write!(&mut params, ",").unwrap();
            }
            params = params.trim_end_matches(',').to_string();
            self.name = format!("x{}{}", params, self.name);
        }

        self
    }

    pub fn medial(
        &mut self,
        ratio: Option<Float>,
        height: Option<Float>,
        vertex_valence: Option<&[usize]>,
        regular_faces_only: Option<bool>,
        change_name: bool,
    ) -> &mut Self {
        self.dual(false);
        self.truncate(height, vertex_valence, regular_faces_only, false);
        self.ambo(ratio, false);

        if change_name {
            let mut params = String::new();
            if let Some(ratio) = ratio {
                write!(&mut params, "{:.2}", ratio).unwrap();
            }
            if let Some(height) = height {
                write!(&mut params, ",{:.2}", height).unwrap();
            } else {
                write!(&mut params, ",").unwrap();
            }
            if let Some(vertex_valence) = vertex_valence {
                write!(&mut params, ",{}", format_slice(vertex_valence))
                    .unwrap();
            } else {
                write!(&mut params, ",").unwrap();
            }
            if let Some(regular_faces_only) = regular_faces_only {
                if regular_faces_only {
                    params.push_str(",{t}");
                }
            } else {
                write!(&mut params, ",").unwrap();
            }
            params = params.trim_end_matches(',').to_string();
            self.name = format!("M{}{}", params, self.name);
        }

        self
    }

    pub fn meta(
        &mut self,
        ratio: Option<Float>,
        height: Option<Float>,
        vertex_valence: Option<&[usize]>,
        regular_faces_only: Option<bool>,
        change_name: bool,
    ) -> &mut Self {
        self.kis(
            height,
            match vertex_valence {
                // By default meta works on vertices of valence three.
                None => Some(&[3]),
                _ => vertex_valence,
            },
            None,
            regular_faces_only,
            false,
        );
        self.join(ratio, false);

        if change_name {
            let mut params = String::new();
            if let Some(ratio) = ratio {
                write!(&mut params, "{:.2}", ratio).unwrap();
            }
            if let Some(height) = height {
                write!(&mut params, ",{:.2}", height).unwrap();
            } else {
                write!(&mut params, ",").unwrap();
            }
            if let Some(vertex_valence) = vertex_valence {
                write!(&mut params, ",{}", format_slice(vertex_valence))
                    .unwrap();
            } else {
                write!(&mut params, ",").unwrap();
            }
            if let Some(regular_faces_only) = regular_faces_only {
                if regular_faces_only {
                    params.push_str(",{t}");
                }
            } else {
                write!(&mut params, ",").unwrap();
            }
            params = params.trim_end_matches(',').to_string();
            self.name = format!("m{}{}", params, self.name);
        }

        self
    }

    pub fn needle(
        &mut self,
        height: Option<Float>,
        vertex_valence: Option<&[usize]>,
        regular_faces_only: Option<bool>,
        change_name: bool,
    ) -> &mut Self {
        self.dual(false);
        self.truncate(height, vertex_valence, regular_faces_only, false);

        if change_name {
            let mut params = String::new();
            if let Some(height) = height {
                write!(&mut params, "{:.2}", height).unwrap();
            }
            if let Some(vertex_valence) = vertex_valence {
                write!(&mut params, ",{}", format_slice(vertex_valence))
                    .unwrap();
            } else {
                write!(&mut params, ",").unwrap();
            }
            if let Some(regular_faces_only) = regular_faces_only {
                if regular_faces_only {
                    params.push_str(",{t}");
                }
            } else {
                write!(&mut params, ",").unwrap();
            }
            params = params.trim_end_matches(',').to_string();
            self.name = format!("n{}{}", params, self.name);
        }

        self
    }

    pub fn ortho(
        &mut self,
        ratio: Option<Float>,
        change_name: bool,
    ) -> &mut Self {
        self.join(ratio, false);
        self.join(ratio, false);

        if change_name {
            let mut params = String::new();
            if let Some(ratio) = ratio {
                write!(&mut params, "{:.2}", ratio).unwrap();
            }
            self.name = format!("o{}{}", params, self.name);
        }

        self
    }

    /// Apply quick and dirty canonicalization. Typical number of `iterations
    /// are `100`+.
    #[inline]
    pub fn planarize(&mut self, iterations: Option<usize>, change_name: bool) {
        let mut dual = self.clone().dual(false).finalize();

        for _ in 0..iterations.unwrap_or(100) {
            // Reciprocate face centers.
            dual.positions =
                reciprocate_face_centers(&self.face_index, &self.positions);
            self.positions =
                reciprocate_face_centers(&dual.face_index, &dual.positions);
        }

        if change_name {
            let mut params = String::new();
            if let Some(iterations) = iterations {
                write!(&mut params, "{}", iterations).unwrap();
            }
            self.name = format!("K{}{}", params, self.name);
        }
    }

    pub fn propeller(
        &mut self,
        ratio: Option<Float>,
        change_name: bool,
    ) -> &mut Self {
        let ratio_ = match ratio {
            Some(r) => r.clamp(0.0, 1.0),
            None => 1. / 3.,
        };

        let edges = self.to_edges();
        let reversed_edges: Edges =
            edges.iter().map(|edge| [edge[1], edge[0]]).collect();

        let new_positions = edges
            .iter()
            .zip(reversed_edges.iter())
            .flat_map(|(edge, reversed_edge)| {
                let edge_positions = index_as_positions(edge, &self.positions);
                vec![
                    (
                        edge,
                        *edge_positions[0]
                            + ratio_
                                * (*edge_positions[1] - *edge_positions[0]),
                    ),
                    (
                        reversed_edge,
                        *edge_positions[1]
                            + ratio_
                                * (*edge_positions[0] - *edge_positions[1]),
                    ),
                ]
            })
            .collect::<Vec<_>>();

        let new_ids = vertex_ids_edge_ref_ref(
            &new_positions,
            self.positions_len() as VertexKey,
        );

        self.face_index = self
            .face_index
            .par_iter()
            .map(|face| {
                face.iter()
                    .circular_tuple_windows::<(_, _)>()
                    .map(|f| vertex_edge(&[*f.0, *f.1], &new_ids).unwrap())
                    .collect()

                /*(0..face.len())
                .map(|j| vertex_edge(&[face[j], face[(j + 1) % face.len()]], &new_ids).unwrap())
                .collect()*/
            })
            .chain(self.face_index.par_iter().flat_map(|face| {
                (0..face.len())
                    .map(|j| {
                        let a = face[j];
                        let b = face[(j + 1) % face.len()];
                        let z = face[(j + face.len() - 1) % face.len()];
                        let eab = vertex_edge(&[a, b], &new_ids).unwrap();
                        let eba = vertex_edge(&[b, a], &new_ids).unwrap();
                        let eza = vertex_edge(&[z, a], &new_ids).unwrap();
                        vec![eba, eab, eza, a]
                        //vec![eza, eab, eba, a]
                    })
                    .collect::<Faces>()
            }))
            .collect::<Faces>();

        self.positions.extend(vertex_values_as_ref(&new_positions));

        if change_name {
            let mut params = String::new();
            if let Some(ratio) = ratio {
                write!(&mut params, "{:.2}", ratio).unwrap();
            }
            self.name = format!("p{}{}", params, self.name);
        }

        self
    }

    pub fn quinto(
        &mut self,
        height: Option<Float>,
        change_name: bool,
    ) -> &mut Self {
        let height_ = match height {
            Some(h) => {
                if h < 0.0 {
                    0.0
                } else {
                    h
                }
            }
            None => 0.5,
        };

        let mut new_positions: Vec<(Face, Point)> = self
            .to_edges()
            .par_iter()
            .map(|edge| {
                let edge_positions = index_as_positions(edge, &self.positions);
                (
                    edge.to_vec(),
                    height_ * (*edge_positions[0] + *edge_positions[1]),
                )
            })
            .collect();

        new_positions.extend(
            self.face_index
                .par_iter()
                .flat_map(|face| {
                    let edge_positions =
                        index_as_positions(face, &self.positions);
                    let centroid = centroid_ref(&edge_positions);
                    (0..face.len())
                        .map(|i| {
                            (
                                extend![..face, i as VertexKey],
                                (*edge_positions[i]
                                    + *edge_positions[(i + 1) % face.len()]
                                    + centroid)
                                    / 3.,
                            )
                        })
                        .collect::<Vec<(Face, Point)>>()
                })
                .collect::<Vec<(Face, Point)>>(),
        );

        let new_ids =
            vertex_ids_ref(&new_positions, self.positions_len() as VertexKey);

        self.positions.extend(vertex_values_as_ref(&new_positions));

        self.face_index = self
            .face_index
            .par_iter()
            .map(|face| {
                (0..face.len())
                    .map(|face_vertex| {
                        vertex(
                            &extend![..face, face_vertex as VertexKey],
                            &new_ids,
                        )
                        .unwrap()
                    })
                    .collect()
            })
            .chain(self.face_index.par_iter().flat_map(|face| {
                (0..face.len())
                    .map(|i| {
                        let v = face[i];
                        let e0 =
                            [face[(i + face.len() - 1) % face.len()], face[i]];
                        let e1 = [face[i], face[(i + 1) % face.len()]];
                        let e0p =
                            vertex(&distinct_edge(&e0), &new_ids).unwrap();
                        let e1p =
                            vertex(&distinct_edge(&e1), &new_ids).unwrap();
                        let iv0 = vertex(
                            &extend![
                                ..face,
                                ((i + face.len() - 1) % face.len())
                                    as VertexKey
                            ],
                            &new_ids,
                        )
                        .unwrap();
                        let iv1 =
                            vertex(&extend![..face, i as VertexKey], &new_ids)
                                .unwrap();
                        vec![v, e1p, iv1, iv0, e0p]
                    })
                    .collect::<Faces>()
            }))
            .collect::<Faces>();

        if change_name {
            let mut params = String::new();
            if let Some(h) = height {
                write!(&mut params, "{:.2}", h).unwrap();
            }
            self.name = format!("q{}{}", params, self.name);
        }

        self
    }

    pub fn reflect(&mut self, change_name: bool) -> &mut Self {
        self.positions = self
            .positions
            .par_iter()
            .map(|v| Point::new(v.x, -v.y, v.z))
            .collect();
        self.reverse();

        if change_name {
            self.name = format!("r{}", self.name);
        }

        self
    }

    pub fn snub(
        &mut self,
        ratio: Option<Float>,
        height: Option<Float>,
        change_name: bool,
    ) -> &mut Self {
        self.dual(false);
        self.gyro(ratio, height, false);
        self.dual(false);

        if change_name {
            let mut params = String::new();
            if let Some(ratio) = ratio {
                write!(&mut params, "{:.2}", ratio).unwrap();
            }
            if let Some(height) = height {
                write!(&mut params, ",{:.2}", height).unwrap();
            }
            self.name = format!("s{}{}", params, self.name);
        }

        self
    }

    /// Projects all positions on the unit sphere (at `strength` `1.0`).
    ///
    /// If `strength` is zero this is a no-op and will neither change the
    /// geometry nor the name. Even if `change_name` is `true`.
    pub fn spherize(
        &mut self,
        strength: Option<Float>,
        change_name: bool,
    ) -> &mut Self {
        let strength_ = strength.unwrap_or(1.0);

        if 0.0 != strength_ {
            self.positions.par_iter_mut().for_each(|point| {
                *point =
                    (1.0 - strength_) * *point + strength_ * point.normalized();
            });

            if change_name {
                let mut params = String::new();
                if let Some(strength) = strength {
                    write!(&mut params, "{:.2}", strength).unwrap();
                }
                self.name = format!("S{}{}", params, self.name);
            }
        }

        self
    }

    pub fn truncate(
        &mut self,
        height: Option<Float>,
        vertex_valence: Option<&[usize]>,
        regular_faces_only: Option<bool>,
        change_name: bool,
    ) -> &mut Self {
        self.dual(false);
        self.kis(height, vertex_valence, None, regular_faces_only, false);
        self.dual(false);

        if change_name {
            let mut params = String::new();
            if let Some(height) = height {
                write!(&mut params, "{:.2}", height).unwrap();
            }
            if let Some(vertex_valence) = vertex_valence {
                write!(&mut params, ",{}", format_slice(vertex_valence))
                    .unwrap();
            } else {
                write!(&mut params, ",").unwrap();
            }
            if let Some(regular_faces_only) = regular_faces_only {
                if regular_faces_only {
                    params.push_str(",{t}");
                }
            } else {
                write!(&mut params, ",").unwrap();
            }
            params = params.trim_end_matches(',').to_string();
            self.name = format!("t{}{}", params, self.name);
        }

        self
    }

    pub fn whirl(
        &mut self,
        ratio: Option<Float>,
        height: Option<Float>,
        change_name: bool,
    ) -> &mut Self {
        let ratio_ = match ratio {
            Some(r) => r.clamp(0.0, 1.0),
            None => 1. / 3.,
        };
        let height_ = height.unwrap_or(0.);

        let new_positions: Vec<(Face, Point)> = self
            .face_index
            .par_iter()
            .flat_map(|face| {
                let face_positions = index_as_positions(face, &self.positions);
                let center = centroid_ref(&face_positions)
                    + average_normal_ref(&face_positions).unwrap() * height_;
                face.iter()
                    .enumerate()
                    .map(|v| {
                        let edge_positions = [
                            face_positions[v.0],
                            face_positions[(v.0 + 1) % face.len()],
                        ];
                        let middle: Point = *edge_positions[0]
                            + ratio_
                                * (*edge_positions[1] - *edge_positions[0]);
                        (
                            extend![..face, *v.1],
                            middle + ratio_ * (center - middle),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .chain(self.to_edges().par_iter().flat_map(|edge| {
                let edge_positions = index_as_positions(edge, &self.positions);
                vec![
                    (
                        edge.to_vec(),
                        *edge_positions[0]
                            + ratio_
                                * (*edge_positions[1] - *edge_positions[0]),
                    ),
                    (
                        vec![edge[1], edge[0]],
                        *edge_positions[1]
                            + ratio_
                                * (*edge_positions[0] - *edge_positions[1]),
                    ),
                ]
            }))
            .collect();

        let new_ids =
            vertex_ids_ref(&new_positions, self.positions_len() as VertexKey);

        self.positions.extend(vertex_values(&new_positions));

        let old_face_index_len = self.face_index.len();

        self.face_index = self
            .face_index
            .par_iter()
            .flat_map(|face| {
                face.iter()
                    .circular_tuple_windows::<(_, _, _)>()
                    .map(|v| {
                        let edeg_ab = vertex(&[*v.0, *v.1], &new_ids).unwrap();
                        let edeg_ba = vertex(&[*v.1, *v.0], &new_ids).unwrap();
                        let edeg_bc = vertex(&[*v.1, *v.2], &new_ids).unwrap();
                        let mut mid = face.clone();
                        mid.push(*v.0);
                        let mid_a = vertex(&mid, &new_ids).unwrap();
                        mid.pop();
                        mid.push(*v.1);
                        let mid_b = vertex(&mid, &new_ids).unwrap();
                        vec![edeg_ab, edeg_ba, *v.1, edeg_bc, mid_b, mid_a]
                    })
                    .collect::<Faces>()
            })
            .chain(self.face_index.par_iter().map(|face| {
                let mut new_face = face.clone();
                face.iter()
                    .map(|a| {
                        new_face.push(*a);
                        let result = vertex(&new_face, &new_ids).unwrap();
                        new_face.pop();
                        result
                    })
                    .collect()
            }))
            .collect::<Faces>();

        self.append_new_face_set(self.face_index.len() - old_face_index_len);

        if change_name {
            let mut params = String::new();
            if let Some(ratio) = ratio {
                write!(&mut params, "{:.2}", ratio).unwrap();
            }
            if let Some(height) = height {
                write!(&mut params, ",{:.2}", height).unwrap();
            }
            self.name = format!("w{}{}", params, self.name);
        }

        self
    }

    pub fn zip(
        &mut self,
        height: Option<Float>,
        vertex_valence: Option<&[usize]>,
        regular_faces_only: Option<bool>,
        change_name: bool,
    ) -> &mut Self {
        self.dual(false);
        self.kis(height, vertex_valence, None, regular_faces_only, false);

        if change_name {
            let mut params = String::new();
            if let Some(height) = height {
                write!(&mut params, "{:.2}", height).unwrap();
            }
            if let Some(vertex_valence) = vertex_valence {
                write!(&mut params, ",{}", format_slice(vertex_valence))
                    .unwrap();
            } else {
                write!(&mut params, ",").unwrap();
            }
            if let Some(regular_faces_only) = regular_faces_only {
                if regular_faces_only {
                    params.push_str(",{t}");
                }
            } else {
                write!(&mut params, ",").unwrap();
            }
            params = params.trim_end_matches(',').to_string();
            self.name = format!("z{}{}", params, self.name);
        }

        self
    }

    /// Reverses the winding order of faces.
    ///
    /// Clockwise(default) becomes counter-clockwise and vice versa.
    pub fn reverse(&mut self) -> &mut Self {
        self.face_index
            .par_iter_mut()
            .for_each(|face| face.reverse());

        self
    }

    /// Returns the name of this polyhedron. This can be used to reconstruct the
    /// polyhedron using `Polyhedron::from<&str>()`.
    #[inline]
    pub fn name(&self) -> &String {
        &self.name
    }

    #[inline]
    pub fn positions_len(&self) -> usize {
        self.positions.len()
    }

    #[inline]
    pub fn positions(&self) -> &Points {
        &self.positions
    }

    pub fn faces(&self) -> &Faces {
        &self.face_index
    }

    // Resizes the polyhedron to fit inside a unit sphere.
    #[inline]
    pub fn normalize(&mut self) -> &mut Self {
        max_resize(&mut self.positions, 1.);
        self
    }

    /// Compute the edges of the polyhedron.
    #[inline]
    pub fn to_edges(&self) -> Edges {
        let edges = self
            .face_index
            .par_iter()
            .map(|face| {
                face.iter()
                    // Grab two index entries.
                    .circular_tuple_windows::<(_, _)>()
                    .filter(|t| t.0 < t.1)
                    // Create an edge from them.
                    .map(|t| [*t.0, *t.1])
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect::<Vec<_>>();

        edges.into_iter().unique().collect()
    }

    /// Returns a flat [`u32`] triangle index buffer and two matching point and
    /// normal buffers.
    ///
    /// All the faces are disconnected. I.e. positions & normals are duplicated
    /// for each shared vertex.
    pub fn to_triangle_mesh_buffers(&self) -> (Vec<u32>, Points, Normals) {
        let (positions, normals): (Vec<_>, Vec<_>) = self
            .face_index
            .par_iter()
            .flat_map(|face| {
                face.iter()
                    // Cycle forever.
                    .cycle()
                    // Start at 3-tuple belonging to the
                    // face's last vertex.
                    .skip(face.len() - 1)
                    // Grab the next three vertex index
                    // entries.
                    .tuple_windows::<(_, _, _)>()
                    .take(face.len())
                    .map(|t| {
                        // The middle point of out tuple
                        let point = self.positions[*t.1 as usize];
                        // Create a normal from that
                        let normal = -orthogonal(
                            &self.positions[*t.0 as usize],
                            &point,
                            &self.positions[*t.2 as usize],
                        );
                        let mag_sq = normal.mag_sq();

                        (
                            point,
                            // Check for collinearity:
                            if mag_sq < EPSILON as _ {
                                average_normal_ref(&index_as_positions(
                                    face,
                                    self.positions(),
                                ))
                                .unwrap()
                            } else {
                                normal / mag_sq.sqrt()
                            },
                        )
                    })
                    // For each vertex of the face.
                    .collect::<Vec<_>>()
            })
            .unzip();

        // Build a new face index. Same topology as the old one, only with new
        // keys.
        let triangle_face_index = self
            .face_index
            .iter()
            // Build a new index where each face has the original arity and the
            // new numbering.
            .scan(0.., |counter, face| {
                Some(counter.take(face.len()).collect::<Vec<u32>>())
            })
            // Now split each of these faces into triangles.
            .flat_map(|face| match face.len() {
                // Filter out degenerate faces.
                1 | 2 => vec![],
                // Bitriangulate quadrilateral faces use shortest diagonal so
                // triangles are most nearly equilateral.
                4 => {
                    let p = index_as_positions(&face, &positions);

                    if (*p[0] - *p[2]).mag_sq() < (*p[1] - *p[3]).mag_sq() {
                        vec![
                            face[0], face[1], face[2], face[0], face[2],
                            face[3],
                        ]
                    } else {
                        vec![
                            face[1], face[2], face[3], face[1], face[3],
                            face[0],
                        ]
                    }
                }
                5 => vec![
                    face[0], face[1], face[4], face[1], face[2], face[4],
                    face[4], face[2], face[3],
                ],
                // FIXME: a nicer way to triangulate n-gons.
                _ => {
                    let a = face[0];
                    let mut bb = face[1];
                    face.iter()
                        .skip(2)
                        .flat_map(|c| {
                            let b = bb;
                            bb = *c;
                            vec![a, b, *c]
                        })
                        .collect()
                }
            })
            .collect();

        (triangle_face_index, positions, normals)
    }

    #[inline]
    pub fn triangulate(&mut self, shortest: Option<bool>) -> &mut Self {
        self.face_index = self
            .face_index
            .par_iter()
            .flat_map(|face| match face.len() {
                // Bitriangulate quadrilateral faces use shortest diagonal so
                // triangles are most nearly equilateral.
                4 => {
                    let p = index_as_positions(face, &self.positions);

                    if shortest.unwrap_or(true)
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
                // FIXME: a nicer way to triangulate n-gons.
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
                }
            })
            .collect();

        self
    }

    #[allow(clippy::too_many_arguments)]
    pub fn _open_face(
        &self,
        outer_inset_ratio: Option<Float>,
        outer_inset: Option<Float>,
        inner_inset_ratio: Option<Float>,
        inner_inset: Option<Float>,
        depth: Option<Float>,
        face_arity: Option<&[usize]>,
        min_edge_length: Option<Float>,
        _no_cut: Option<bool>,
    ) {
        // upper and lower inset can be specified by ratio or absolute distance
        //  let(inner_inset_ratio= inner_inset_ratio == undef ?
        // outer_inset_ratio : inner_inset_ratio,

        //pf=p_faces(obj),
        //pv=p_vertices(obj))

        // Corresponding positions on inner surface.
        let inverse_positions = self
            .positions
            .iter()
            .enumerate()
            .map(|point| {
                let vertex_faces = vertex_faces(point.0 as _, &self.face_index);
                // Calculate average normal at vertex.
                let average_normal_ref = vertex_faces
                    .iter()
                    .map(|face| {
                        average_normal_ref(&index_as_positions(
                            face,
                            &self.positions,
                        ))
                        .unwrap()
                    })
                    .fold(Normal::zero(), |accumulate, normal| {
                        accumulate + normal
                    })
                    / vertex_faces.len() as Float;

                *point.1 + depth.unwrap_or(0.2) * average_normal_ref
            })
            .collect::<Vec<_>>();

        let _new_vertices = self
            .face_index
            .iter()
            // Filter out faces that have an unwanted arity or are too small.
            .filter(|face| {
                face_arity_matches(face, face_arity)
                    && minimal_edge_length(face, &self.positions)
                        > min_edge_length.unwrap_or(0.01)
            })
            .flat_map(|face| {
                let face_positions = index_as_positions(face, &self.positions);
                let ofp = index_as_positions(face, &inverse_positions);
                let c = centroid_ref(&face_positions);
                let oc = centroid_ref(&ofp);

                face.iter()
                    .enumerate()
                    .flat_map(|f| {
                        let _v = *f.1;
                        let p = face_positions[f.0];
                        let p1 = face_positions[(f.0 + 1) % face.len()];
                        let p0 =
                            face_positions[(f.0 + face.len() - 1) % face.len()];

                        let sa = angle_between(&(*p0 - *p), &(*p1 - *p), None);
                        let bv = 0.5
                            * ((*p1 - *p).normalized()
                                + (*p0 - *p).normalized());
                        let op = ofp[f.0];

                        let _ip = match outer_inset {
                            None => {
                                *p + (c - *p) * outer_inset_ratio.unwrap_or(0.2)
                            }
                            Some(outer_inset) => {
                                *p + outer_inset / sa.sin() * bv
                            }
                        };
                        let _oip = match inner_inset {
                            None => {
                                *op + (oc - *op)
                                    * inner_inset_ratio.unwrap_or(0.2)
                            }
                            Some(inner_inset) => {
                                *op + inner_inset / sa.sin() * bv
                            }
                        };
                        //vec![[[face, v], ip], [[face, -v - 1], oip]]
                        vec![]
                    })
                    .collect::<Vec<_>>()
                //vec![]
            })
            .collect::<Vec<Point>>();
        /*
        // the inset positions on outer and inner surfaces
        // outer inset positions keyed by face, v, inner positions by face,-v-1
                flatten(
                  [ for (face = pf)
                    if(face_arity_matches(face,fn)
                       && min_edge_length(face,pv) > min_edge_length)
                        let(fp=as_positions(face,pv),
                            ofp=as_positions(face,inv),
                            c=centroid(fp),
                            oc=centroid(ofp))

                        flatten(
                           [for (i=[0:len(face)-1])
                            let(v=face[i],
                                p = fp[i],
                                p1= fp[(i+1)%len(face)],
                                p0=fp[(i-1 + len(face))%len(face)],
                                sa = angle_between(p0-p,p1-p),
                                bv = (unitv(p1-p)+unitv(p0-p))/2,
                                op= ofp[i],
                                ip = outer_inset ==  undef
                                    ? p + (c-p)*outer_inset_ratio
                                    : p + outer_inset/sin(sa) * bv ,
                                oip = inner_inset == undef
                                    ? op + (oc-op)*inner_inset_ratio
                                    : op + inner_inset/sin(sa) * bv)
                            [ [[face,v],ip],[[face,-v-1],oip]]
                           ])
                    ])
                  )
          let(newids=vertex_ids(newv,2*len(pv)))
          let(newf =
                flatten(
                 [ for (i = [0:len(pf)-1])
                   let(face = pf[i])
                   flatten(
                     face_arity_matches(face,fn)
                       && min_edge_length(face,pv) > min_edge_length
                       && i  >= nocut

                       ? [for (j=[0:len(face)-1])   //  replace N-face with 3*N quads
                         let (a=face[j],
                              inseta = vertex([face,a],newids),
                              oinseta= vertex([face,-a-1],newids),
                              b=face[(j+1)%len(face)],
                              insetb= vertex([face,b],newids),
                              oinsetb=vertex([face,-b-1],newids),
                              oa=len(pv) + a,
                              ob=len(pv) + b)

                            [
                              [a,b,insetb,inseta]  // outer face
                             ,[inseta,insetb,oinsetb,oinseta]  //wall
                             ,[oa,oinseta,oinsetb,ob]  // inner face
                            ]
                          ]
                       :  [[face],  //outer face
                           [reverse([  //inner face
                                  for (j=[0:len(face)-1])
                                  len(pv) +face[j]
                                ])
                           ]
                          ]
                      )
                ] ))

          poly(name=str("L",p_name(obj)),
              vertices=  concat(pv, inv, vertex_values(newv)) ,
              faces= newf,
              debug=newv
              )
           ; // end openface
           */
    }

    /// Turns the builder into a final object.
    pub fn finalize(&self) -> Self {
        self.clone()
    }

    /// Sends the polyhedron to the specified
    /// [ɴsɪ](https:://crates.io/crates/nsi) context.
    /// # Arguments
    /// * `handle` – Handle of the node being created. If omitted, the name of
    ///   the polyhedron will be used as a handle.
    ///
    /// * `crease_hardness` - The hardness of edges (default: 10).
    ///
    /// * `corner_hardness` - The hardness of vertices (default: 0).
    ///
    /// * `smooth_corners` - Whether to keep corners smooth, where more than two
    ///   edges meet. When set to `false` these automatically form a hard corner
    ///   with the same hardness as `crease_hardness`.
    #[cfg(feature = "nsi")]
    pub fn to_nsi(
        &self,
        ctx: &nsi::Context,
        handle: Option<&str>,
        crease_hardness: Option<f32>,
        corner_hardness: Option<f32>,
        smooth_corners: Option<bool>,
    ) -> String {
        let handle = handle.unwrap_or_else(|| self.name.as_str()).to_string();
        // Create a new mesh node.
        ctx.create(handle.clone(), nsi::NodeType::Mesh, &[]);

        // Flatten point vector.
        // Fast, unsafe version. May exploce on some platforms.
        // If so, use commented out code below instead.
        let positions = unsafe {
            std::slice::from_raw_parts(
                self.positions.as_ptr().cast::<Float>(),
                3 * self.positions_len(),
            )
        };

        /*
        let positions: Vec<f32> = self
            .positions
            .into_par_iter()
            .flat_map(|p3| once(p3.x as _).chain(once(p3.y as _)).chain(once(p3.z as _)))
            .collect();
        */

        ctx.set_attribute(
            handle.clone(),
            &[
                // Positions.
                nsi::points!("P", positions),
                // VertexKey into the position array.
                nsi::integers!(
                    "P.indices",
                    bytemuck::cast_slice(
                        &self
                            .face_index
                            .par_iter()
                            .flat_map(|face| face.clone())
                            .collect::<Vec<_>>()
                    )
                ),
                // Arity of each face.
                nsi::integers!(
                    "nvertices",
                    &self
                        .face_index
                        .par_iter()
                        .map(|face| face.len() as i32)
                        .collect::<Vec<_>>()
                ),
                // Render this as a C-C subdivison surface.
                nsi::string!("subdivision.scheme", "catmull-clark"),
                // This saves us from having to reverse the mesh ourselves.
                nsi::integer!("clockwisewinding", true as _),
            ],
        );

        // Default: semi sharp creases.
        let crease_hardness = crease_hardness.unwrap_or(10.);

        // Crease each of our edges a bit?
        if 0.0 != crease_hardness {
            let edges = self
                .to_edges()
                .into_iter()
                .flat_map(|edge| edge.to_vec())
                .collect::<Vec<_>>();
            ctx.set_attribute(
                handle.clone(),
                &[
                    nsi::integers!(
                        "subdivision.creasevertices",
                        bytemuck::cast_slice(&edges)
                    ),
                    nsi::floats!(
                        "subdivision.creasesharpness",
                        &vec![crease_hardness; edges.len() / 2]
                    ),
                ],
            );
        }

        match corner_hardness {
            Some(hardness) => {
                if 0.0 < hardness {
                    let corners = self
                        .positions
                        .par_iter()
                        .enumerate()
                        .map(|(i, _)| i as u32)
                        .collect::<Vec<_>>();
                    ctx.set_attribute(
                        handle.clone(),
                        &[
                            nsi::integers!(
                                "subdivision.cornervertices",
                                bytemuck::cast_slice(&corners)
                            ),
                            nsi::floats!(
                                "subdivision.cornersharpness",
                                &vec![hardness; corners.len()]
                            ),
                        ],
                    );
                }
            }

            // Have the renderer semi create sharp corners automagically.
            None => ctx.set_attribute(
                handle.clone(),
                &[
                    // Disabling below flag activates the specific
                    // deRose extensions for the C-C creasing
                    // algorithm that causes any vertex with where
                    // more then three creased edges meet to forma a
                    // corner.
                    // See fig. 8c/d in this paper:
                    // http://graphics.pixar.com/people/derose/publications/Geri/paper.pdf
                    nsi::integer!(
                        "subdivision.smoothcreasecorners",
                        smooth_corners.unwrap_or(false) as _
                    ),
                ],
            ),
        };

        handle
    }

    #[cfg(feature = "obj")]
    pub fn read_from_obj(&self, source: &Path, reverse_winding: bool) {
        //Result<Self, Box<dyn Error>> {

        let obj = tobj::load_obj(source, &tobj::LoadOptions::default());
    }

    /// Write the polyhedron to a
    /// [Wavefront OBJ](https://en.wikipedia.org/wiki/Wavefront_.obj_file)
    /// file.
    ///
    /// The [`name`](Polyhedron::name()) of the polyhedron is appended
    /// to the given `destination` and postfixed with the extension
    /// `.obj`.
    ///
    /// Depending on the target coordinate system (left- or right
    /// handed) the mesh’s winding order can be reversed with the
    /// `reverse_face_winding` flag.
    ///
    /// The return value, on success, is the final, complete path of
    /// the OBJ file.
    #[cfg(feature = "obj")]
    pub fn write_to_obj(
        &self,
        destination: &Path,
        reverse_winding: bool,
    ) -> Result<PathBuf, Box<dyn Error>> {
        let path = destination.join(format!("polyhedron-{}.obj", self.name));
        let mut file = File::create(path.clone())?;

        writeln!(file, "o {}", self.name)?;

        for vertex in &self.positions {
            writeln!(file, "v {} {} {}", vertex.x, vertex.y, vertex.z)?;
        }

        match reverse_winding {
            true => {
                for face in &self.face_index {
                    write!(file, "f")?;
                    for vertex_index in face.iter().rev() {
                        write!(file, " {}", vertex_index + 1)?;
                    }
                    writeln!(file)?;
                }
            }
            false => {
                for face in &self.face_index {
                    write!(file, "f")?;
                    for vertex_index in face {
                        write!(file, " {}", vertex_index + 1)?;
                    }
                    writeln!(file)?;
                }
            }
        };

        file.flush()?;

        Ok(path)
    }

    pub fn tetrahedron() -> Self {
        let c0 = 1.0;

        Self {
            positions: vec![
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

    #[inline]
    pub fn cube() -> Self {
        Self::hexahedron()
    }

    pub fn hexahedron() -> Self {
        let c0 = 1.0;

        Self {
            positions: vec![
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
        let c0 = 0.707_106_77;

        Self {
            positions: vec![
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
        let c0 = 0.809_017;
        let c1 = 1.309_017;

        Self {
            positions: vec![
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
        let c0 = 0.809_017;

        Self {
            positions: vec![
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

    /// common code for prism and antiprism
    fn protoprism(n: usize, anti: bool) -> Self {
        let n = if n < 3 { 3 } else { n };

        // Angles.
        let theta = f32::TAU() / n as f32;
        let twist = if anti { theta / 2.0 } else { 0.0 };
        // Half-edge.
        let h = (theta * 0.5).sin();

        let mut face_index = vec![
            (0..n).map(|i| i as VertexKey).collect::<Vec<_>>(),
            (n..2 * n).rev().map(|i| i as VertexKey).collect::<Vec<_>>(),
        ];

        // Sides.
        if anti {
            face_index.extend(
                (0..n)
                    .map(|i| {
                        vec![
                            i as VertexKey,
                            (i + n) as VertexKey,
                            ((i + 1) % n) as VertexKey,
                        ]
                    })
                    .chain((0..n).map(|i| {
                        vec![
                            (i + n) as VertexKey,
                            ((i + 1) % n + n) as VertexKey,
                            ((i + 1) % n) as VertexKey,
                        ]
                    })),
            );
        } else {
            face_index.extend((0..n).map(|i| {
                vec![
                    i as VertexKey,
                    (i + n) as VertexKey,
                    ((i + 1) % n + n) as VertexKey,
                    ((i + 1) % n) as VertexKey,
                ]
            }));
        };

        Self {
            name: format!("{}{}", if anti { "AP" } else { "P" }, n),
            positions: (0..n)
                .map(move |i| {
                    Point::new(
                        (i as f32 * theta).cos() as _,
                        h,
                        (i as f32 * theta).sin() as _,
                    )
                })
                .chain((0..n).map(move |i| {
                    Point::new(
                        (twist + i as f32 * theta).cos() as _,
                        -h,
                        (twist + i as f32 * theta).sin() as _,
                    )
                }))
                .collect(),

            face_index,
            face_set_index: Vec::new(),
        }
    }

    pub fn prism(n: usize) -> Self {
        Self::protoprism(n, false)
    }

    pub fn antiprism(n: usize) -> Self {
        Self::protoprism(n, true)
    }
}

#[cfg(feature = "bevy")]
use bevy::render::mesh::{
    Indices, Mesh, PrimitiveTopology, VertexAttributeValues,
};

#[cfg(feature = "bevy")]
impl From<Polyhedron> for Mesh {
    fn from(mut polyhedron: Polyhedron) -> Self {
        polyhedron.reverse();

        let (index, positions, normals) = polyhedron.to_triangle_mesh_buffers();

        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
        mesh.set_indices(Some(Indices::U32(index)));

        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            VertexAttributeValues::Float32x3(
                positions
                    .par_iter()
                    .map(|p| [p.x, p.y, p.z])
                    .collect::<Vec<_>>(),
            ),
        );

        mesh.insert_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            VertexAttributeValues::Float32x3(
                normals
                    .par_iter()
                    .map(|n| [-n.x, -n.y, -n.z])
                    .collect::<Vec<_>>(),
            ),
        );

        mesh
    }
}
