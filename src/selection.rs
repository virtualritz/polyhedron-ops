use crate::*;

/// Selection methods.
impl Polyhedron {
    /// Selects all faces within a half space defined by a plane through
    /// `origin` and the the give plane `normal`.
    pub fn select_faces_above_plane(
        &self,
        origin: Point,
        normal: Vector,
    ) -> Vec<FaceKey> {
        #[inline]
        fn is_inside_half_space(
            point: Point,
            plane_origin: Point,
            plane_normal: Vector,
        ) -> bool {
            let point_to_plane = point - plane_origin;
            let distance = point_to_plane.dot(plane_normal);
            distance > 0.0
        }

        self.face_index
            .iter()
            .enumerate()
            .filter_map(|(face_number, face)| {
                if face.iter().all(|&vertex_key| {
                    is_inside_half_space(
                        self.positions[vertex_key as usize],
                        origin,
                        normal,
                    )
                }) {
                    Some(face_number as _)
                } else {
                    None
                }
            })
            .collect()
    }
}
