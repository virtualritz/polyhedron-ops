use crate::{helpers::*, text_helpers::*, *};
use std::fmt::Write;

/// # Operators
impl Polyhedron {
    /// Creates vertices with valence (aka degree) four.
    ///
    /// It is also called [rectification](https://en.wikipedia.org/wiki/Rectification_(geometry)),
    /// or the [medial graph](https://en.wikipedia.org/wiki/Medial_graph) in graph theory.
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
                write!(&mut params, "{}", format_float(ratio)).unwrap();
            }
            self.name = format!("a{}{}", params, self.name);
        }

        self
    }

    /// Adds faces at the center, original vertices, and along the edges.
    ///
    /// # Arguments
    ///
    /// * `ratio` - The ratio of the new vertices to the original vertices.
    /// * `height` - The height (depth) of the bevel.
    /// * `face_arity_mask` - Only faces matching the given arities will be
    ///   affected.
    /// * `regular_faces_only` - Only regular faces will be affected.
    pub fn bevel(
        &mut self,
        ratio: Option<Float>,
        height: Option<Float>,
        face_arity_mask: Option<&[usize]>,
        regular_faces_only: Option<bool>,
        change_name: bool,
    ) -> &mut Self {
        self.truncate(height, face_arity_mask, regular_faces_only, false);
        self.ambo(ratio, false);

        if change_name {
            let mut params = String::new();
            if let Some(ratio) = ratio {
                write!(&mut params, "{}", format_float(ratio)).unwrap();
            }
            if let Some(height) = height {
                write!(&mut params, ",{}", format_float(height)).unwrap();
            } else {
                write!(&mut params, ",").unwrap();
            }
            if let Some(face_arity_mask) = face_arity_mask {
                write!(
                    &mut params,
                    ",{}",
                    format_integer_slice(face_arity_mask)
                )
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

    /// Apply proper canonicalization. t yypical number of `iterarations` is
    /// `200`+.
    /// FIXME: this is b0rked atm.
    #[inline]
    fn _canonicalize(&mut self, iterations: Option<usize>, change_name: bool) {
        let mut dual = self.clone().dual(false).finalize();

        for _ in 0..iterations.unwrap_or(200) {
            // Reciprocate faces.
            dual.positions =
                _reciprocate_faces(&self.face_index, &self.positions);
            self.positions =
                _reciprocate_faces(&dual.face_index, &dual.positions);
        }

        if change_name {
            let mut params = String::new();
            if let Some(iterations) = iterations {
                write!(&mut params, "{}", iterations).unwrap();
            }
            self.name = format!("N{}{}", params, self.name);
        }
    }

    /// Performs [Catmull-Clark subdivision](https://en.wikipedia.org/wiki/Catmull%E2%80%93Clark_subdivision_surface).
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
                let i = point.0 as _;
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

    /// [Chamfers](https://en.wikipedia.org/wiki/Chamfer_(geometry)) edges.
    /// I.e. adds a new hexagonal face in place of each original edge.
    ///
    /// # Arguments
    ///
    /// * `ratio` - The ratio of the new faces to the old faces.
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
        self.positions.par_iter_mut().for_each(|point| {
            *point = (1.5 * ratio_) * *point;
        });
        self.positions.extend(vertex_values(&new_positions));

        if change_name {
            let mut params = String::new();
            if let Some(ratio) = ratio {
                write!(&mut params, "{}", format_float(ratio)).unwrap();
            }
            self.name = format!("c{}{}", params, self.name);
        }

        self
    }

    /// Creates the [dual](https://en.wikipedia.org/wiki/Dual_polyhedron).
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

    /// [Cantellates](https://en.wikipedia.org/wiki/Cantellation_(geometry)).
    /// I.e. creates a new facet in place of each edge and of each vertex.
    ///
    /// # Arguments
    ///
    /// * `ratio` - The ratio of the new faces to the old faces.
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
                write!(&mut params, "{}", format_float(ratio)).unwrap();
            }
            self.name = format!("e{}{}", params, self.name);
        }

        self
    }

    /// Extrudes faces by `height` and shrinks the extruded faces by `distance`
    /// from the original edges.
    ///
    /// # Arguments
    ///
    /// * `height` – The distance to extrude the faces. Default value is `0.3`.
    /// * `offset` – The distance to inset the extruded faces. Default value is
    ///   `0.0`.
    /// * `face_arity_mask` – Only faces matching the given arities will be
    ///   affected.
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
                write!(&mut params, "{}", format_float(height)).unwrap();
            }
            if let Some(offset) = offset {
                write!(&mut params, ",{}", format_float(offset)).unwrap();
            } else {
                write!(&mut params, ",").unwrap();
            }
            if let Some(face_arity) = face_arity {
                write!(&mut params, ",{}", format_integer_slice(face_arity))
                    .unwrap();
            } else {
                write!(&mut params, ",").unwrap();
            }
            params = params.trim_end_matches(',').to_string();
            self.name = format!("x{}{}", params, self.name);
        }

        self
    }

    /// Splits each edge and connects new edges at the split point to the face
    /// centroid. Existing positions are retained.
    /// ![Gyro](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Conway_gC.png/200px-Conway_gC.png)
    ///
    /// # Arguments
    ///
    /// * `ratio` – The ratio at which the adjacent edges get split.
    /// * `height` – An offset to add to the face centroid point along the face
    ///   normal.
    /// * `change_name` – Whether to change the name of the mesh.
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
                write!(&mut params, "{}", format_float(ratio)).unwrap();
            }
            if let Some(height) = height {
                write!(&mut params, ",{}", format_float(height)).unwrap();
            }
            self.name = format!("g{}{}", params, self.name);
        }

        self
    }

    /// Inset faces by `offset` from the original edges.
    ///
    /// # Arguments
    ///
    /// * `offset` – The distance to inset the faces. Default value is `0.3`.
    /// * `face_arity_mask` – Only faces matching the given arities will be
    ///   affected.
    pub fn inset(
        &mut self,
        offset: Option<Float>,
        face_arity_mask: Option<&[usize]>,
        change_name: bool,
    ) -> &mut Self {
        if change_name {
            let mut params = String::new();
            if let Some(offset) = offset {
                write!(&mut params, "{}", format_float(offset)).unwrap();
            }
            if let Some(face_arity_mask) = &face_arity_mask {
                write!(
                    &mut params,
                    ",{}",
                    format_integer_slice(face_arity_mask)
                )
                .unwrap();
            }
            self.name = format!("i{}{}", params, self.name);
        }

        self.extrude(
            Some(0.0),
            Some(offset.unwrap_or(0.3)),
            face_arity_mask,
            false,
        );

        self
    }

    /// Creates quadrilateral faces around each original edge. Original
    /// edges are discarded.
    ///
    /// # Arguments
    ///
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
                write!(&mut params, "{}", format_float(ratio)).unwrap();
            }
            self.name = format!("j{}{}", params, self.name);
        }

        self
    }

    /// Creates a [kleetrope](https://en.wikipedia.org/wiki/Kleetope) from the
    /// input. Splits each face into triangles, one for each edge, which
    /// extend to the face centroid. Existing positions are retained.
    ///
    /// # Arguments
    ///
    /// * `height` - An offset to add to the face centroid point along the face
    ///   normal.
    /// * `face_arity_mask` - Only faces matching the given arities will be
    ///   affected.
    /// * `face_index_mask` - Only faces matching the given indices will be
    ///   affected.
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
                write!(&mut params, "{}", format_float(height)).unwrap();
            }
            if let Some(face_arity_mask) = face_arity_mask {
                write!(
                    &mut params,
                    ",{}",
                    format_integer_slice(face_arity_mask)
                )
                .unwrap();
            } else {
                write!(&mut params, ",").unwrap();
            }
            if let Some(face_index_mask) = face_index_mask {
                write!(
                    &mut params,
                    ",{}",
                    format_integer_slice(face_index_mask)
                )
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

    /// Adds edges from the center to each original vertex.
    ///
    /// # Arguments
    ///
    /// * `ratio` – The ratio of the new vertices to the original vertices.
    /// * `height` – The height of the new vertices.
    /// * `vertex_valence_mask` – Only vertices matching the given valences will
    ///   be affected.
    /// * `regular_faces_only` – Only regular faces will be affected.
    pub fn medial(
        &mut self,
        ratio: Option<Float>,
        height: Option<Float>,
        vertex_valence_mask: Option<&[usize]>,
        regular_faces_only: Option<bool>,
        change_name: bool,
    ) -> &mut Self {
        self.dual(false);
        self.truncate(height, vertex_valence_mask, regular_faces_only, false);
        self.ambo(ratio, false);

        if change_name {
            let mut params = String::new();
            if let Some(ratio) = ratio {
                write!(&mut params, "{}", format_float(ratio)).unwrap();
            }
            if let Some(height) = height {
                write!(&mut params, ",{}", format_float(height)).unwrap();
            } else {
                write!(&mut params, ",").unwrap();
            }
            if let Some(vertex_valence_mask) = vertex_valence_mask {
                write!(
                    &mut params,
                    ",{}",
                    format_integer_slice(vertex_valence_mask)
                )
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

    /// Adds vertices at the center and along the edges.
    ///
    /// # Arguments
    ///
    /// * `ratio` – The ratio of the new vertices to the original vertices.
    /// * `height` – The height of the new vertices.
    /// * `vertex_valence_mask` – Only vertices matching the given valences
    ///  will be affected.
    /// * `regular_faces_only` – Only regular faces will be affected.
    pub fn meta(
        &mut self,
        ratio: Option<Float>,
        height: Option<Float>,
        vertex_valence_mask: Option<&[usize]>,
        regular_faces_only: Option<bool>,
        change_name: bool,
    ) -> &mut Self {
        self.kis(
            height,
            match vertex_valence_mask {
                // By default meta works on vertices of valence three.
                None => Some(&[3]),
                _ => vertex_valence_mask,
            },
            None,
            regular_faces_only,
            false,
        );
        self.join(ratio, false);

        if change_name {
            let mut params = String::new();
            if let Some(ratio) = ratio {
                write!(&mut params, "{}", format_float(ratio)).unwrap();
            }
            if let Some(height) = height {
                write!(&mut params, ",{}", format_float(height)).unwrap();
            } else {
                write!(&mut params, ",").unwrap();
            }
            if let Some(vertex_valence_mask) = vertex_valence_mask {
                write!(
                    &mut params,
                    ",{}",
                    format_integer_slice(vertex_valence_mask)
                )
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

    /// Like [`kis`](Polyhedron::kis) but also splits each edge in the middle.
    ///
    /// # Arguments
    ///
    /// * `height` – The offset of the new face centers.
    /// * `vertex_valence_mask` – Only vertices matching the given valences will
    ///   be affected.
    /// * `regular_faces_only` – Only regular faces will be affected.
    pub fn needle(
        &mut self,
        height: Option<Float>,
        vertex_valence_mask: Option<&[usize]>,
        regular_faces_only: Option<bool>,
        change_name: bool,
    ) -> &mut Self {
        self.dual(false);
        self.truncate(height, vertex_valence_mask, regular_faces_only, false);

        if change_name {
            let mut params = String::new();
            if let Some(height) = height {
                write!(&mut params, "{}", format_float(height)).unwrap();
            }
            if let Some(vertex_valence_mask) = vertex_valence_mask {
                write!(
                    &mut params,
                    ",{}",
                    format_integer_slice(vertex_valence_mask)
                )
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

    /// Connects the center of each face to the center of each edge.
    ///
    /// # Arguments
    ///
    /// * `ratio` – The ratio of the new two parts each original edge is split
    ///   into.
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
                write!(&mut params, "{}", format_float(ratio)).unwrap();
            }
            self.name = format!("o{}{}", params, self.name);
        }

        self
    }

    /// Applies quick and dirty canonicalization.
    ///
    /// # Arguments
    ///
    /// * `iterations` – The number of iterations to perform. Typical number of
    ///   `iterations are `100`+. The default is `100`.
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

    /// Splits each edge into three parts and creates edges on each face
    /// connecting the new vertices.
    ///
    /// # Arguments
    ///
    /// * `ratio` – The ratio of the edge splits.
    pub fn propellor(
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
                write!(&mut params, "{}", format_float(ratio)).unwrap();
            }
            self.name = format!("p{}{}", params, self.name);
        }

        self
    }

    /// Splits each edge in the middle and creates new faces in the middle of
    /// each face then connects those.
    ///
    /// # Arguments
    ///
    /// * `height` – The offset of the new faces from the original face.
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
            if let Some(height) = height {
                write!(&mut params, "{}", format_float(height)).unwrap();
            }
            self.name = format!("q{}{}", params, self.name);
        }

        self
    }

    /// [Reflects](https://en.wikipedia.org/wiki/Reflection_(mathematics)) the shape.
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

    /// Applies a [snub](https://en.wikipedia.org/wiki/Snub_(geometry)) to the shape.
    ///
    /// # Arguments
    ///
    /// * `ratio` – The ratio at which the adjacent edges get split.
    /// * `height` – The height of the newly created centers.
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
                write!(&mut params, "{}", format_float(ratio)).unwrap();
            }
            if let Some(height) = height {
                write!(&mut params, ",{}", format_float(height)).unwrap();
            }
            self.name = format!("s{}{}", params, self.name);
        }

        self
    }

    /// Projects all positions on the unit sphere (at `strength` `1.0`).
    ///
    /// # Arguments
    ///
    /// * `strength` – The strength of the spherization. If `strength` is zero
    ///   this is a no-op and will neither change the geometry nor the name.
    ///   Even if `change_name` is `true`.
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
                    write!(&mut params, "{}", format_float(strength)).unwrap();
                }
                self.name = format!("S{}{}", params, self.name);
            }
        }

        self
    }

    /// Cuts off the shape at its vertices but leaves a portion of the original
    /// edges.
    ///
    /// # Arguments
    ///
    /// * `height` – The height of the newly created centers.
    /// * `face_arity_mask` - Only faces matching the given arities will be
    ///   affected.
    /// * `regular_faces_only` – Only regular faces will be affected.
    pub fn truncate(
        &mut self,
        height: Option<Float>,
        face_arity_mask: Option<&[usize]>,
        regular_faces_only: Option<bool>,
        change_name: bool,
    ) -> &mut Self {
        self.dual(false);
        self.kis(height, face_arity_mask, None, regular_faces_only, false);
        self.dual(false);

        if change_name {
            let mut params = String::new();
            if let Some(height) = height {
                write!(&mut params, "{}", format_float(height)).unwrap();
            }
            if let Some(face_arity_mask) = face_arity_mask {
                write!(
                    &mut params,
                    ",{}",
                    format_integer_slice(face_arity_mask)
                )
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

    /// Splits each edge into three parts and connexts the new vertices. But
    /// also splits the newly formed connections and connects those.
    ///
    /// # Arguments
    ///
    /// * `ratio` – The ratio at which the adjacent edges get split.
    /// * `height` – The height offset of the newly created vertices.
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
                write!(&mut params, "{}", format_float(ratio)).unwrap();
            }
            if let Some(height) = height {
                write!(&mut params, ",{}", format_float(height)).unwrap();
            }
            self.name = format!("w{}{}", params, self.name);
        }

        self
    }

    /// [Bitruncates](https://en.wikipedia.org/wiki/Bitruncation) the shape.
    ///
    /// # Arguments
    ///
    /// * `height` – The height offset of the newly created vertices.
    /// * `face_arity_mask` – Only faces with the given arity will be affected.
    /// * `regular_faces_only` – Only regular faces will be affected.
    pub fn zip(
        &mut self,
        height: Option<Float>,
        face_arity_mask: Option<&[usize]>,
        regular_faces_only: Option<bool>,
        change_name: bool,
    ) -> &mut Self {
        self.dual(false);
        self.kis(height, face_arity_mask, None, regular_faces_only, false);

        if change_name {
            let mut params = String::new();
            if let Some(height) = height {
                write!(&mut params, "{}", format_float(height)).unwrap();
            }
            if let Some(face_arity_mask) = face_arity_mask {
                write!(
                    &mut params,
                    ",{}",
                    format_integer_slice(face_arity_mask)
                )
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

    #[allow(clippy::too_many_arguments)]
    fn _open_face(
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
                    && _minimal_edge_length(face, &self.positions)
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

                        let sa = _angle_between(&(*p0 - *p), &(*p1 - *p), None);
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
}

/// # Triangulation
impl Polyhedron {
    #[inline]
    /// Bitriangulates quadrilateral faces.
    ///
    /// N-gon trinagulation is naive and may yield inferor results.
    ///
    /// # Arguments
    ///
    /// * `shortest` - If `true`, use shortest diagonal so triangles are most
    ///   nearly equilateral. On by default.
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
}
