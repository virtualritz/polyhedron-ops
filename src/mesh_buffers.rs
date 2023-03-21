use crate::*;

impl Polyhedron {
    /// Returns a flat [`u32`] triangle index buffer and two matching point and
    /// normal buffers.
    ///
    /// This is mostly useful for realtime rendering, e.g. sending data to a
    /// GPU.
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
                    let p = index_as_positions(face.as_slice(), &positions);

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
}
