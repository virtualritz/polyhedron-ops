#![allow(clippy::many_single_char_names)]
#![feature(iter_array_chunks)]
//! # Conway-Hart Polyhedron Operations
//!
//! This crate implements the [Conway Polyhedron
//! Operators](https://en.wikipedia.org/wiki/Conway_polyhedron_notation)
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
//! polyhedron.write_obj(&Path::new("."), false);
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
//! * `bevy` – Adds support for converting a polyhedron into a [`bevy`](https://bevyengine.org/)
//!   [`Mesh`](https://docs.rs/bevy/latest/bevy/render/mesh/struct.Mesh.html).
//!   See the `bevy` example.
//!
//! * `nsi` – Add supports for sending data to renderers implementing the [ɴsɪ](https://crates.io/crates/nsi/)
//!   API. The function is called [`to_nsi()`](Polyhedron::to_nsi()).
//!
//! * `obj` – Add support for output to [Wavefront OBJ](https://en.wikipedia.org/wiki/Wavefront_.obj_file)
//!   via the [`write_obj()`](Polyhedron::write_obj()) function.
//!
//! * `parser` – Add support for parsing strings in [Conway Polyhedron Notation](https://en.wikipedia.org/wiki/Conway_polyhedron_notation).
//!   This feature implements
//!   [`Polyhedron::TryFrom<&str>`](Polyhedron::try_from<&str>).
use crate::helpers::*;
use itertools::Itertools;
use rayon::prelude::*;
use ultraviolet as uv;
#[cfg(feature = "parser")]
#[macro_use]
extern crate pest_derive;

mod base_polyhedra;
mod helpers;
mod io;
mod mesh_buffers;
mod operators;
#[cfg(feature = "parser")]
mod parser;
mod selection;
mod text_helpers;

#[cfg(test)]
mod tests;

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
    pub(crate) fn positions_len(&self) -> usize {
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

    /// Turns the builder into a final object.
    pub fn finalize(&self) -> Self {
        self.clone()
    }
}

impl Polyhedron {
    /// Appends indices for newly added faces as a new [`FaceSet`] to the
    /// [`FaceSetIndex`].
    #[inline]
    fn append_new_face_set(&mut self, size: usize) {
        self.face_set_index
            .append(&mut vec![((self.face_index.len() as VertexKey)
                ..((self.face_index.len() + size) as VertexKey))
                .collect()]);
    }
}
