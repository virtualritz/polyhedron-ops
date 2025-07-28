use crate::*;
use nsi_core as nsi;

/// Conversion to [ɴsɪ](https:://crates.io/crates/nsi).
impl<'a> Polyhedron {
    /// Sends the polyhedron to the specified
    /// [ɴsɪ](https:://crates.io/crates/nsi) context.
    ///
    /// # Arguments
    ///
    /// * `handle` – Handle of the node being created. If omitted, the name of
    ///   the polyhedron will be used as a handle.
    ///
    /// * `crease_hardness` - The hardness of edges (default: 10).
    ///
    /// * `corner_hardness` - The hardness of vertices (default: 0).
    ///
    /// * `auto_corners` - Whether to create corners where more than two edges
    ///   meet. When set to `true` these automatically form a hard corner with
    ///   the same hardness as `crease_hardness`. This is ignored if
    ///   `corner_hardness` is `Some`.
    ///
    ///   This activates the specific *deRose* extensions for the Catmull-Clark
    ///   subdivision creasing algorithm. See fig. 8c/d in
    ///   [this paper](http://graphics.pixar.com/people/derose/publications/Geri/paper.pdf).
    #[cfg(feature = "nsi")]
    pub fn to_nsi(
        &self,
        ctx: &nsi::Context,
        handle: Option<&str>,
        crease_hardness: Option<f32>,
        corner_hardness: Option<f32>,
        auto_corners: Option<bool>,
    ) -> String {
        let handle = handle.unwrap_or(self.name.as_str()).to_string();
        // Create a new mesh node.
        ctx.create(&handle, nsi::node::MESH, None);

        // Flatten point vector.
        let position = unsafe {
            std::slice::from_raw_parts(
                self.positions.as_ptr().cast::<f32>(),
                self.positions_len() * 3,
            )
        };

        ctx.set_attribute(
            &handle,
            &[
                // Positions.
                nsi::points!("P", position),
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
                .flat_map(|edge| edge)
                .collect::<Vec<_>>();
            ctx.set_attribute(
                &handle,
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
                        .map(|(i, _)| i as i32)
                        .collect::<Vec<_>>();
                    ctx.set_attribute(
                        &handle,
                        &[
                            nsi::integers!(
                                "subdivision.cornervertices",
                                &corners
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
                &handle,
                &[
                    // Disabling below flag activates the specific
                    // deRose extensions for the C-C creasing algorithm
                    // that causes any vertex where more then three
                    // creased edges meet to form a corner.
                    // See fig. 8c/d in this paper:
                    // http://graphics.pixar.com/people/derose/publications/Geri/paper.pdf
                    nsi::integer!(
                        "subdivision.smoothcreasecorners",
                        !auto_corners.unwrap_or(true) as _
                    ),
                ],
            ),
        };

        handle
    }

    /// Creates the buffers to send a polyhedron to an
    /// [ɴsɪ](https:://crates.io/crates/nsi) context.
    ///
    /// # Arguments
    ///
    /// * `crease_hardness` - The hardness of edges (default: 10).
    ///
    /// * `corner_hardness` - The hardness of vertices (default: 0).
    ///
    /// # Examples
    ///
    /// ```
    /// # use nsi::*;
    /// # use polyhedron_ops::Polyhedron;
    /// # let ctx = Context::new(None).unwrap();
    /// # let polyhedron = Polyhedron::dodecahedron().chamfer(None, true).propellor(None, true).finalize();
    /// let (
    ///     position,
    ///     position_index,
    ///     face_arity,
    ///     crease_index,
    ///     crease_hardness,
    ///     corner_index,
    ///     corner_hardness,
    /// ) = polyhedron.to_nsi(Some(10.0), Some(5.0));
    ///
    /// ctx.create("polyhedron", nsi::node::MESH, None);
    ///
    /// ctx.set_attribute(
    ///     "polyhedron",
    ///     &[
    ///         nsi::points!("P", position),
    ///         nsi::integers!("P.indices", &position_index),
    ///         nsi::integers!("nvertices", &face_arity),
    ///         nsi::integers!("subdivision.creasevertices", &crease_index.unwrap()),
    ///         nsi::floats!("subdivision.creasehardness", &crease_hardness.unwrap()),
    ///         nsi::integers!("subdivision.cornervertices", &corner_index.unwrap()),
    ///         nsi::floats!("subdivision.cornerhardness", &corner_hardness.unwrap()),
    ///         nsi::integer!("clockwisewinding", 1)
    ///     ]
    /// );
    /// ```
    #[cfg(feature = "nsi")]
    pub fn to_nsi_buffers(
        &'a self,
        crease_hardness: Option<f32>,
        corner_hardness: Option<f32>,
    ) -> (
        &'a [[f32; 3]],
        Vec<i32>,
        Vec<i32>,
        Option<Vec<i32>>,
        Option<Vec<f32>>,
        Option<Vec<i32>>,
        Option<Vec<f32>>,
    ) {
        // Flatten point vector.
        let position = unsafe {
            std::slice::from_raw_parts(
                self.positions.as_ptr().cast::<[f32; 3]>(),
                self.positions_len(),
            )
        };

        let face_index = self
            .face_index
            .par_iter()
            .flat_map(|face| bytemuck::cast_slice(face).to_vec())
            .collect::<Vec<_>>();

        let face_arity = self
            .face_index
            .par_iter()
            .map(|face| face.len() as i32)
            .collect::<Vec<_>>();

        let (crease_index, crease_hardness) =
            if let Some(crease_hardness) = crease_hardness {
                let edge = self
                    .to_edges()
                    .into_iter()
                    .flat_map(|edge| edge)
                    .collect::<Vec<_>>();

                let edge_len = edge.len();
                (
                    Some(bytemuck::cast_vec(edge)),
                    Some(vec![crease_hardness; edge_len / 2]),
                )
            } else {
                (None, None)
            };

        let (corner_index, corner_hardness) =
            if let Some(corner_hardness) = corner_hardness {
                let corner = self
                    .positions
                    .par_iter()
                    .enumerate()
                    .map(|(i, _)| i as i32)
                    .collect::<Vec<_>>();

                let corner_len = corner.len();
                (Some(corner), Some(vec![corner_hardness; corner_len]))
            } else {
                (None, None)
            };

        (
            position,
            face_index,
            face_arity,
            crease_index,
            crease_hardness,
            corner_index,
            corner_hardness,
        )
    }
}
