use crate::*;

/// Conversion to [ɴsɪ](https:://crates.io/crates/nsi).
impl Polyhedron {
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
    /// * `smooth_corners` - Whether to keep corners smooth where more than two
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
        let handle = handle.unwrap_or(self.name.as_str()).to_string();
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
}
