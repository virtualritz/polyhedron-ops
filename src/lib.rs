//! # Conway-Hart Polyhedron Operations
//!
//! This crate implements the [Conway Polyhedron
//! Operators](http://en.wikipedia.org/wiki/Conway_polyhedron_notation)
//! and their extensions by
//! [George W. Hart](http://www.georgehart.com/) and others.
//!
//! The internal representation uses mesh buffers. These need
//! furter preprocessing before they can be sent to a GPU but
//! are almost fine to send to an offline renderer, as-is.
//!
//! See the `playground` example for code on how to do either.
//! ## Example
//! ```
//! use polyhedron_ops::Polyhedron;
//! use std::path::Path;
//!
//! // Conway notation: gapcD
//! let polyhedron =
//!     Polyhedron::dodecahedron()
//!         .chamfer(None, true)
//!         .propellor(None, true)
//!         .ambo(None, true)
//!         .gyro(None, None, true)
//!         .finalize();
//!
//! // Export as ./polyhedron-gapcD.obj
//! # #[cfg(feature = "obj")]
//! # {
//! polyhedron.write_to_obj(&Path::new("."), false);
//! # }
//!```
//! The above code starts from a
//! [dodecahedron](https://en.wikipedia.org/wiki/Dodecahedron) and
//! iteratively applies four operators.
//!
//! The resulting shape is shown below.
//!
//! ![](https://raw.githubusercontent.com/virtualritz/polyhedron-operators/HEAD/gapcD.jpg)
//!
//! ## Cargo Features
//! The crate supports sending data to renderers implementing the
//! [ɴsɪ](https://crates.io/crates/nsi/) API. The function is called
//! [`to_nsi()`](Polyhedron::to_nsi()) and is enabled through the
//! `"nsi"` feature.
//!
//! Output to
//! [Wavefront OBJ](https://en.wikipedia.org/wiki/Wavefront_.obj_file)
//! is supported via the `"obj"` feature which adds the
//! [`write_to_obj()`](Polyhedron::write_to_obj()) function.
//! ```toml
//! [dependencies]
//! polyhedron-ops = { version = "0.1.4", features = [ "nsi", "obj" ] }
//! ```
use clamped::Clamp;
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

pub type Float = f32;
pub type VertexKey = u32;
pub type Face = Vec<VertexKey>;
pub(crate) type FaceSlice = [VertexKey];
pub type Faces = Vec<Face>;
pub(crate) type FacesSlice = [Face];
pub type FaceSet = Vec<VertexKey>;
pub type Edge = [VertexKey; 2];
pub type Edges = Vec<Edge>;
pub(crate) type _EdgeSlice = [Edge];
pub type Point = ultraviolet::vec::Vec3;
pub type Vector = ultraviolet::vec::Vec3;
pub type Normal = Vector;
#[allow(dead_code)]
pub type Normals = Vec<Normal>;
pub type Points = Vec<Point>;
pub(crate) type PointsSlice = [Point];
pub(crate) type PointRefSlice<'a> = [&'a Point];

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

pub enum NormalType {
    Smooth(Float),
    Flat,
}

#[derive(Clone, Debug)]
pub struct Polyhedron {
    face_index: Faces,
    points: Points,
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
            points: rt
                .points()
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
            points: rt
                .points()
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
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            face_index: Vec::new(),
            face_set_index: Vec::new(),
            name: String::new(),
        }
    }

    pub fn from(
        name: &str,
        points: Points,
        face_index: Faces,
        face_set_index: Option<Vec<FaceSet>>,
    ) -> Self {
        Self {
            points,
            face_index,
            face_set_index: face_set_index.unwrap_or_default(),
            name: name.to_string(),
        }
    }

    /// Returns the axis-aligned bounding box of the polyhedron in the
    /// format `[x_min, y_min, z_min, x_max, y_max, z_max]`.
    pub fn bounding_box(&self) -> [f64; 6] {
        let mut bounds = [0.0f64; 6];
        self.points.iter().for_each(|point| {
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

    /// Appends indices for newly added faces
    /// as a new FaceSet to the FaceSetIndex.
    fn append_new_face_set(&mut self, size: usize) {
        self.face_set_index
            .append(&mut vec![((self.face_index.len() as VertexKey)
                ..((self.face_index.len() + size) as VertexKey))
                .collect()]);
    }

    pub fn planarize(&mut self, iterations: usize) {
        let mut dual = self.clone().dual(false).finalize();
        for _ in 0..iterations {
            // Reciprocate face centers.
            dual.points =
                reciprocate_face_centers(&self.face_index, &self.points);
            self.points =
                reciprocate_face_centers(&dual.face_index, &dual.points);
        }
    }

    /// Creates vertices with valence (aka degree) four. It is also
    /// called
    /// [rectification](https://en.wikipedia.org/wiki/Rectification_(geometry)),
    /// or the
    /// [medial graph](https://en.wikipedia.org/wiki/Medial_graph)
    /// in graph theory.
    #[inline]
    pub fn ambo(
        &mut self,
        ratio: Option<Float>,
        change_name: bool,
    ) -> &mut Self {
        let ratio_ = match ratio {
            Some(r) => r.clamped(0.0, 1.0),
            None => 1. / 2.,
        };

        let edges = distinct_edges(&self.face_index);

        let points: Vec<(&Edge, Point)> = edges
            .par_iter()
            .map(|edge| {
                let edge_points = index_as_points(edge, &self.points);
                (
                    edge,
                    ratio_ * *edge_points[0] + (1.0 - ratio_) * *edge_points[1],
                )
            })
            .collect();

        let new_ids = vertex_ids_edge_ref_ref(&points, 0);

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
                self.points
                    // Each old vertex creates a new face ...
                    .par_iter()
                    .enumerate()
                    .map(|polygon_vertex| {
                        let vertex_number = polygon_vertex.0 as VertexKey;
                        ordered_vertex_edges(
                            vertex_number,
                            &vertex_faces(vertex_number, &self.face_index),
                        )
                        .iter()
                        .map(|ve| {
                            vertex_edge(&distinct_edge(ve), &new_ids).unwrap()
                                as VertexKey
                        })
                        .collect::<Vec<_>>()
                    }),
            )
            .collect();

        self.append_new_face_set(face_index.len());

        self.face_index = face_index;
        self.points = vertex_values(&points);

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
        vertex_valence: Option<Vec<usize>>,
        regular_faces_only: Option<bool>,
        change_name: bool,
    ) -> &mut Self {
        self.truncate(
            height,
            vertex_valence.clone(),
            regular_faces_only,
            false,
        );
        self.ambo(ratio, false);

        if change_name {
            let mut params = String::new();
            if let Some(ratio) = ratio {
                write!(&mut params, "{:.2}", ratio).unwrap();
            }
            if let Some(height) = height {
                write!(&mut params, "{:.2}", height).unwrap();
            }
            if let Some(vertex_valence) = vertex_valence {
                write!(&mut params, ",{}", format_vec(&vertex_valence))
                    .unwrap();
            }
            if let Some(regular_faces_only) = regular_faces_only {
                if regular_faces_only {
                    params.push_str(",{t}");
                }
            }
            self.name = format!("b{}{}", params, self.name);
        }

        self
    }

    pub fn chamfer(
        &mut self,
        ratio: Option<Float>,
        change_name: bool,
    ) -> &mut Self {
        let ratio_ = match ratio {
            Some(r) => r.clamped(0.0, 1.0),
            None => 1. / 2.,
        };

        let new_points: Vec<(Face, Point)> = self
            .face_index
            .par_iter()
            .flat_map(|face| {
                let face_points = index_as_points(face, &self.points);
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
            .collect();

        let new_ids =
            vertex_ids_ref(&new_points, self.points_len() as VertexKey);

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
                (0..face.len())
                    .filter_map(|j| {
                        if face[j] < face[(j + 1) % face.len()] {
                            let a: VertexKey = face[j];
                            let b: VertexKey = face[(j + 1) % face.len()];
                            let opposite_face =
                                face_with_edge(&[b, a], &self.face_index);

                            Some(vec![
                                a,
                                vertex(&extend![..opposite_face, a], &new_ids)
                                    .unwrap(),
                                vertex(&extend![..opposite_face, b], &new_ids)
                                    .unwrap(),
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
        self.points.par_iter_mut().for_each(|point| {
            *point = (1.5 * ratio_) * *point;
        });
        self.points.extend(vertex_values(&new_points));

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
        let new_points = face_centers(&self.face_index, &self.points);
        self.face_index = points_to_faces(&self.points, &self.face_index);
        self.points = new_points;
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

    /// Splits each edge and connects new edges at the split point
    /// to the face centroid. Existing points are retained.
    /// ![Gyro](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Conway_gC.png/200px-Conway_gC.png)
    /// # Arguments
    /// * `ratio` – – The ratio at which the adjacent edges get split.
    /// * `height` – An offset to add to the face centroid point along
    ///     the face normal.
    /// * `regular_faces_only` – Only faces whose edges are 90% the
    ///     same length, within the same face, are affected.
    pub fn gyro(
        &mut self,
        ratio: Option<f32>,
        height: Option<f32>,
        change_name: bool,
    ) -> &mut Self {
        let ratio_ = match ratio {
            Some(r) => r.clamped(0.0, 1.0),
            None => 1. / 3.,
        };
        let height_ = height.unwrap_or(0.);

        let edges = self.to_edges();
        let reversed_edges: Edges =
            edges.par_iter().map(|edge| [edge[1], edge[0]]).collect();

        // Retain original points, add face centroids and directed
        // edge points each N-face becomes N pentagons.
        let new_points: Vec<(&FaceSlice, Point)> = self
            .face_index
            .par_iter()
            .map(|face| {
                let fp = index_as_points(face, &self.points);
                (
                    face.as_slice(),
                    centroid_ref(&fp).normalized()
                        + face_normal(&fp).unwrap() * height_,
                )
            })
            .chain(edges.par_iter().enumerate().flat_map(|edge| {
                let edge_points = index_as_points(edge.1, &self.points);
                vec![
                    (
                        &edge.1[..],
                        *edge_points[0]
                            + ratio_ * (*edge_points[1] - *edge_points[0]),
                    ),
                    (
                        &reversed_edges[edge.0][..],
                        *edge_points[1]
                            + ratio_ * (*edge_points[0] - *edge_points[1]),
                    ),
                ]
            }))
            .collect();

        let new_ids =
            vertex_ids_ref_ref(&new_points, self.points_len() as VertexKey);

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
    /// * `ratio` – The ratio at which the adjacent edges get split.
    ///     Will be clamped to `[0, 1]`. Default value is `0.5`.
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
    /// extend to the face centroid. Existing points are retained.
    /// # Arguments
    /// * `height` - An offset to add to the face centroid point along
    ///     the face normal.
    /// * `face_arity` - Only faces matching the given arities will be
    ///     affected.
    /// * `regular_faces_only` - Only faces whose edges are 90% the
    ///     same length, within the same face, are affected.
    pub fn kis(
        &mut self,
        height: Option<Float>,
        face_arity: Option<Vec<usize>>,
        regular_faces_only: Option<bool>,
        change_name: bool,
    ) -> &mut Self {
        let new_points: Vec<(&FaceSlice, Point)> = self
            .face_index
            .par_iter()
            .filter_map(|face| {
                if selected_face(face, face_arity.as_ref())
                    && !regular_faces_only.unwrap_or(false)
                    || ((face_irregularity(face, &self.points) - 1.0).abs()
                        < 0.1)
                {
                    let face_points = index_as_points(face, &self.points);
                    Some((
                        face.as_slice(),
                        centroid_ref(&face_points)
                            + face_normal(&face_points).unwrap()
                                * height.unwrap_or(0.),
                    ))
                } else {
                    None
                }
            })
            .collect();

        let new_ids =
            vertex_ids_ref_ref(&new_points, self.points.len() as VertexKey);

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
                            centroid as VertexKey,
                        ]
                    })
                    .collect(),
                None => vec![face.clone()],
            })
            .collect();

        if change_name {
            let mut params = String::new();
            if let Some(height) = height {
                write!(&mut params, "{:.2}", height).unwrap();
            }
            if let Some(face_arity) = face_arity {
                write!(&mut params, ",{:.2}", format_vec(&face_arity)).unwrap();
            }
            if let Some(regular_faces_only) = regular_faces_only {
                if regular_faces_only {
                    params.push_str(",{t}");
                }
            }
            self.name = format!("k{}{}", params, self.name);
        }

        self
    }

    pub fn inset(
        &mut self,
        distance: Option<Float>,
        height: Option<Float>,
        face_arity: Option<Vec<usize>>,
        change_name: bool,
    ) -> &mut Self {
        let distance = distance.unwrap_or(0.3);

        let new_points = self
            .face_index
            .iter()
            .filter(|face| selected_face(face, face_arity.as_ref()))
            .flat_map(|face| {
                let face_points = index_as_points(face, &self.points);
                let centroid = centroid_ref(&face_points);
                face.iter()
                    .zip(&face_points)
                    .map(|face_vertex_point| {
                        (
                            extend![..face, *face_vertex_point.0],
                            **face_vertex_point.1
                                + distance * (centroid - **face_vertex_point.1)
                                + face_normal(&face_points).unwrap()
                                    * height.unwrap_or(0.),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let new_ids =
            vertex_ids_ref(&new_points, self.points_len() as VertexKey);

        self.face_index = self
            .face_index
            .iter()
            .flat_map(|face| {
                if selected_face(face, face_arity.as_ref()) {
                    face.iter()
                        .enumerate()
                        .flat_map(|index_vertex| {
                            let a = *index_vertex.1;
                            let inset_a =
                                vertex(&extend![..face, a], &new_ids).unwrap();
                            let b = face[(index_vertex.0 + 1) % face.len()];
                            let inset_b =
                                vertex(&extend![..face, b], &new_ids).unwrap();
                            vec![vec![a, b, inset_b, inset_a]]
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

        self.points.extend(vertex_values_as_ref(&new_points));

        if change_name {
            let mut params = String::new();
            write!(&mut params, "{:.2}", distance).unwrap();
            if let Some(height) = height {
                write!(&mut params, "{:.2}", height).unwrap();
            }
            if let Some(face_arity) = face_arity {
                write!(&mut params, ",{}", format_vec(&face_arity)).unwrap();
            }
            self.name = format!("i{}{}", params, self.name);
        }

        self
    }

    fn _loft(&self) {
        // FIXME
    }

    pub fn medial(
        &mut self,
        ratio: Option<Float>,
        height: Option<Float>,
        vertex_valence: Option<Vec<usize>>,
        regular_faces_only: Option<bool>,
        change_name: bool,
    ) -> &mut Self {
        self.dual(false);
        self.truncate(
            height,
            vertex_valence.clone(),
            regular_faces_only,
            false,
        );
        self.ambo(ratio, false);

        if change_name {
            let mut params = String::new();
            if let Some(ratio) = ratio {
                write!(&mut params, "{:.2}", ratio).unwrap();
            }
            if let Some(height) = height {
                write!(&mut params, "{:.2}", height).unwrap();
            }
            if let Some(vertex_valence) = vertex_valence {
                write!(&mut params, ",{}", format_vec(&vertex_valence))
                    .unwrap();
            }
            if let Some(regular_faces_only) = regular_faces_only {
                if regular_faces_only {
                    params.push_str(",{t}");
                }
            }
            self.name = format!("M{}{}", params, self.name);
        }

        self
    }

    pub fn meta(
        &mut self,
        ratio: Option<Float>,
        height: Option<Float>,
        vertex_valence: Option<Vec<usize>>,
        regular_faces_only: Option<bool>,
        change_name: bool,
    ) -> &mut Self {
        self.kis(
            height,
            match vertex_valence {
                // By default meta works on verts.
                // of valence three.
                None => Some(vec![3]),
                _ => vertex_valence.clone(),
            },
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
                write!(&mut params, "{:.2}", height).unwrap();
            }
            if let Some(vertex_valence) = vertex_valence {
                write!(&mut params, ",{}", format_vec(&vertex_valence))
                    .unwrap();
            }
            if let Some(regular_faces_only) = regular_faces_only {
                if regular_faces_only {
                    params.push_str(",{t}");
                }
            }
            self.name = format!("m{}{}", params, self.name);
        }

        self
    }

    pub fn needle(
        &mut self,
        height: Option<Float>,
        vertex_valence: Option<Vec<usize>>,
        regular_faces_only: Option<bool>,
        change_name: bool,
    ) -> &mut Self {
        self.dual(false);
        self.truncate(
            height,
            vertex_valence.clone(),
            regular_faces_only,
            false,
        );

        if change_name {
            let mut params = String::new();
            if let Some(height) = height {
                write!(&mut params, "{:.2}", height).unwrap();
            }
            if let Some(vertex_valence) = vertex_valence {
                write!(&mut params, ",{}", format_vec(&vertex_valence))
                    .unwrap();
            }
            if let Some(regular_faces_only) = regular_faces_only {
                if regular_faces_only {
                    params.push_str(",{t}");
                }
            }
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

    pub fn propellor(
        &mut self,
        ratio: Option<Float>,
        change_name: bool,
    ) -> &mut Self {
        let ratio_ = match ratio {
            Some(r) => r.clamped(0.0, 1.0),
            None => 1. / 3.,
        };

        let edges = self.to_edges();
        let reversed_edges: Edges =
            edges.iter().map(|edge| [edge[1], edge[0]]).collect();

        let new_points = edges
            .iter()
            .zip(reversed_edges.iter())
            .flat_map(|(edge, reversed_edge)| {
                let edge_points = index_as_points(edge, &self.points);
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

        let new_ids = vertex_ids_edge_ref_ref(
            &new_points,
            self.points_len() as VertexKey,
        );

        self.face_index = self
            .face_index
            .par_iter()
            .map(|face| {
                (0..face.len())
                    .map(|j| {
                        vertex_edge(
                            &[face[j], face[(j + 1) % face.len()]],
                            &new_ids,
                        )
                        .unwrap()
                    })
                    .collect()
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
                        vec![a, eba, eab, eza]
                    })
                    .collect::<Faces>()
            }))
            .collect::<Faces>();

        self.points.extend(vertex_values_as_ref(&new_points));

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

        let mut new_points: Vec<(Face, Point)> = self
            .to_edges()
            .par_iter()
            .map(|edge| {
                let edge_points = index_as_points(edge, &self.points);
                (edge.to_vec(), height_ * (*edge_points[0] + *edge_points[1]))
            })
            .collect();

        new_points.extend(
            self.face_index
                .par_iter()
                .flat_map(|face| {
                    let edge_points = index_as_points(face, &self.points);
                    let centroid = centroid_ref(&edge_points);
                    (0..face.len())
                        .map(|i| {
                            (
                                extend![..face, i as VertexKey],
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

        let new_ids =
            vertex_ids_ref(&new_points, self.points_len() as VertexKey);

        self.points.extend(vertex_values_as_ref(&new_points));

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

    /// Projects all points on the unit sphere (at `strength` `1.0`).
    ///
    /// If `strength` is zero this is a no-op and will neither change
    /// the geometry nor the name. Even if `change_name` is `true`.
    pub fn spherize(
        &mut self,
        strength: Option<Float>,
        change_name: bool,
    ) -> &mut Self {
        let strength_ = strength.unwrap_or(1.0);

        if 0.0 != strength_ {
            self.points.par_iter_mut().for_each(|point| {
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
        vertex_valence: Option<Vec<usize>>,
        regular_faces_only: Option<bool>,
        change_name: bool,
    ) -> &mut Self {
        self.dual(false);
        self.kis(height, vertex_valence.clone(), regular_faces_only, false);
        self.dual(false);

        if change_name {
            let mut params = String::new();
            if let Some(height) = height {
                write!(&mut params, "{:.2}", height).unwrap();
            }
            if let Some(vertex_valence) = vertex_valence {
                write!(&mut params, ",{}", format_vec(&vertex_valence))
                    .unwrap();
            }
            if let Some(regular_faces_only) = regular_faces_only {
                if regular_faces_only {
                    params.push_str(",{t}");
                }
            }
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
            Some(r) => r.clamped(0.0, 1.0),
            None => 1. / 3.,
        };
        let height_ = height.unwrap_or(0.);

        let new_points: Vec<(Face, Point)> = self
            .face_index
            .par_iter()
            .flat_map(|face| {
                let face_points = index_as_points(face, &self.points);
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
            .chain(self.to_edges().par_iter().flat_map(|edge| {
                let edge_points = index_as_points(edge, &self.points);
                vec![
                    (
                        edge.to_vec(),
                        *edge_points[0]
                            + ratio_ * (*edge_points[1] - *edge_points[0]),
                    ),
                    (
                        vec![edge[1], edge[0]],
                        *edge_points[1]
                            + ratio_ * (*edge_points[0] - *edge_points[1]),
                    ),
                ]
            }))
            .collect();

        let new_ids =
            vertex_ids_ref(&new_points, self.points_len() as VertexKey);

        self.points.extend(vertex_values(&new_points));

        let old_face_index_len = self.face_index.len();

        self.face_index = self
            .face_index
            .par_iter()
            .flat_map(|face| {
                (0..face.len())
                    .map(|j| {
                        let a = face[j];
                        let b = face[(j + 1) % face.len()];
                        let c = face[(j + 2) % face.len()];
                        let eab = vertex(&[a, b], &new_ids).unwrap();
                        let eba = vertex(&[b, a], &new_ids).unwrap();
                        let ebc = vertex(&[b, c], &new_ids).unwrap();
                        let mut mid = face.clone();
                        mid.push(a);
                        let mida = vertex(&mid, &new_ids).unwrap();
                        mid.pop();
                        mid.push(b);
                        let midb = vertex(&mid, &new_ids).unwrap();
                        vec![eab, eba, b, ebc, midb, mida]
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
        vertex_valence: Option<Vec<usize>>,
        regular_faces_only: Option<bool>,
        change_name: bool,
    ) -> &mut Self {
        self.dual(false);
        self.kis(height, vertex_valence.clone(), regular_faces_only, false);

        if change_name {
            let mut params = String::new();
            if let Some(height) = height {
                write!(&mut params, "{:.2}", height).unwrap();
            }
            if let Some(vertex_valence) = vertex_valence {
                write!(&mut params, ",{}", format_vec(&vertex_valence))
                    .unwrap();
            }
            if let Some(regular_faces_only) = regular_faces_only {
                if regular_faces_only {
                    params.push_str(",{t}");
                }
            }
            self.name = format!("z{}{}", params, self.name);
        }

        self
    }

    /// Reverses the winding order of faces.
    /// Clockwise(default) becomes counter-clockwise and vice versa.
    pub fn reverse(&mut self) -> &mut Self {
        self.face_index
            .par_iter_mut()
            .for_each(|face| face.reverse());

        self
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
    pub fn points(&self) -> &Points {
        &self.points
    }

    pub fn faces(&self) -> &Faces {
        &self.face_index
    }

    // Resizes the polyhedron to fit inside a unit sphere.
    #[inline]
    pub fn normalize(&mut self) -> &mut Self {
        max_resize(&mut self.points, 1.);
        self
    }

    /// Compute the edges of the polyhedron.
    #[inline]
    pub fn to_edges(&self) -> Edges {
        distinct_edges(&self.face_index)
    }

    /// Returns a flat u32 trinagle index buffer and two matching point
    /// and normal buffers.
    ///
    /// All the faces are disconnected. I.e. points & normals are
    /// duplicated for each shared vertex.
    pub fn to_triangle_mesh_buffers(&self) -> (Vec<u32>, Points, Normals) {
        let (points, normals): (Vec<_>, Vec<_>) = self
            .face_index
            .par_iter()
            .flat_map(|f| {
                let average_normal =
                    face_normal(&index_as_points(f, self.points())).unwrap();

                f.iter()
                    // Cycle forever.
                    .cycle()
                    // Start at 3-tuple belonging to the
                    // face's last vertex.
                    .skip(f.len() - 1)
                    // Grab the next three vertex index
                    // entries.
                    .tuple_windows::<(_, _, _)>()
                    .map(|t| {
                        // The middle point of out tuple
                        let point = self.points[*t.1 as usize];
                        // Create a normal from that
                        let normal = -orthogonal(
                            &self.points[*t.0 as usize],
                            &point,
                            &self.points[*t.2 as usize],
                        );
                        let mag_sq = normal.mag_sq();

                        (
                            point,
                            // Check for collinearity:
                            if mag_sq < EPSILON as _ {
                                average_normal
                            } else {
                                normal / mag_sq.sqrt()
                            },
                        )
                    })
                    // For each vertex of the face.
                    .take(f.len())
                    .collect::<Vec<_>>()
            })
            .unzip();

        // Build a new face index. Same topology as the old one, only
        // with new keys.
        let triangle_face_index = self
            .face_index
            .iter()
            // Build a new index where each face has the original arity
            // and the new numbering.
            .scan(0.., |counter, face| {
                Some(counter.take(face.len()).collect::<Vec<u32>>())
            })
            // Now split each of these faces into triangles.
            .flat_map(|face| match face.len() {
                // Bitriangulate quadrilateral faces
                // use shortest diagonal so triangles are
                // most nearly equilateral.
                4 => {
                    let p = index_as_points(&face, &points);

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

        (triangle_face_index, points, normals)
    }

    #[inline]
    pub fn triangulate(&mut self, shortest: Option<bool>) -> &mut Self {
        self.face_index = self
            .face_index
            .par_iter()
            .flat_map(|face| match face.len() {
                // Bitriangulate quadrilateral faces
                // use shortest diagonal so triangles are
                // most nearly equilateral.
                4 => {
                    let p = index_as_points(face, &self.points);

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

    /// Turns the builder into a final object.
    pub fn finalize(&self) -> Self {
        self.clone()
    }

    /// Sends the polyhedron to the specified
    /// [ɴsɪ](https:://crates.io/crates/nsi) context.
    /// # Arguments
    /// * `handle` – Handle of the node being created. If omitted, the
    ///     name of the polyhedron will be used as a handle.
    ///
    /// * `crease_hardness` - The hardness of edges (default: 10).
    ///
    /// * `corner_hardness` - The hardness of vertices (default: 0).
    ///
    /// * `smooth_corners` - Whether to keep corners smooth, where more
    ///     than two edges meet. When set to `false` these
    ///     automatically form a hard corner with the same hardness
    ///     as `crease_hardness`..
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
                self.points.as_ptr().cast::<Float>(),
                3 * self.points_len(),
            )
        };

        /*
        let positions: Vec<f32> = self
            .points
            .into_par_iter()
            .flat_map(|p3| once(p3.x as _).chain(once(p3.y as _)).chain(once(p3.z as _)))
            .collect();
        */

        let face_arity = self
            .face_index
            .par_iter()
            .map(|face| face.len() as u32)
            .collect::<Vec<_>>();

        let face_index = self.face_index.concat();

        ctx.set_attribute(
            handle.clone(),
            &[
                // Positions.
                nsi::points!("P", positions),
                // VertexKey into the position array.
                nsi::integers!("P.indices", bytemuck::cast_slice(&face_index)),
                // Arity of each face.
                nsi::integers!("nvertices", bytemuck::cast_slice(&face_arity)),
                // Render this as a C-C subdivison surface.
                nsi::string!("subdivision.scheme", "catmull-clark"),
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
                        &vec![crease_hardness; edges.len()]
                    ),
                ],
            );
        }

        match corner_hardness {
            Some(hardness) => {
                if 0.0 < hardness {
                    let corners = self
                        .points
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
                    // Disabling below flag activates the specific deRose
                    // extensions for the C-C creasing algorithm
                    // that causes any vertex with where more then three
                    // creased edges meet to forma a corner.
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

    /// Write the polyhedron to a
    /// [Wavefront OBJ](https://en.wikipedia.org/wiki/Wavefront_.obj_file)
    /// file.
    ///
    /// The [`name`](Polyhedron::name()) of the polyhedron is appended to the given
    /// `destination` and postfixed with the extension `.obj`.
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

    #[inline]
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
        let c0 = 0.707_106_77;

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
        let c0 = 0.809_017;
        let c1 = 1.309_017;

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
        let c0 = 0.809_017;

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

    pub fn prism(n: usize) -> Self {
        let n = if n < 3 { 3 } else { n };

        // Angle.
        let theta = f32::TAU() / n as f32;
        // Half-edge.
        let h = (theta * 0.5).sin();

        let mut face_index = Vec::new();

        // Top- & bottom faces
        face_index.push((0..n).map(|i| i as VertexKey).collect::<Vec<_>>());
        face_index
            .push((n..2 * n).rev().map(|i| i as VertexKey).collect::<Vec<_>>());
        // Sides.
        face_index.extend((0..n).map(|i| {
            vec![
                i as VertexKey,
                (i + n) as VertexKey,
                ((i + 1) % n + n) as VertexKey,
                ((i + 1) % n) as VertexKey,
            ]
        }));

        Self {
            name: format!("P{}", n),
            points: (0..n)
                .map(move |i| {
                    Point::new(
                        (i as f32 * theta).cos() as _,
                        h,
                        (i as f32 * theta).sin() as _,
                    )
                })
                .chain((0..n).map(move |i| {
                    Point::new(
                        (i as f32 * theta).cos() as _,
                        -h,
                        (i as f32 * theta).sin() as _,
                    )
                }))
                .collect(),

            face_index,
            face_set_index: Vec::new(),
        }
    }
}

#[cfg(feature = "bevy")]
use bevy::render::{
    mesh::{Indices, Mesh},
    pipeline::PrimitiveTopology,
};

#[cfg(feature = "bevy")]
impl From<Polyhedron> for Mesh {
    fn from(mut polyhedron: Polyhedron) -> Self {
        polyhedron.reverse();

        let (index, points, normals) = polyhedron.to_triangle_mesh_buffers();

        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
        mesh.set_indices(Some(Indices::U32(index)));
        mesh.set_attribute(
            Mesh::ATTRIBUTE_POSITION,
            points
                .par_iter()
                .map(|p| [p.x, p.y, p.z])
                .collect::<Vec<_>>(),
        );
        mesh.set_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            normals
                .par_iter()
                .map(|n| [-n.x, -n.y, -n.z])
                .collect::<Vec<_>>(),
        );
        // Bevy forces UVs. So we create some fake UVs by just
        // projecting through, onto the XY plane.
        mesh.set_attribute(
            Mesh::ATTRIBUTE_UV_0,
            points.par_iter().map(|p| [p.x, p.y]).collect::<Vec<_>>(),
        );

        mesh
    }
}
