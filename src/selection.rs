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
        self.face_index
            .iter()
            .enumerate()
            .filter_map(|(face_number, face)| {
                if face.iter().all(|&vertex_key| {
                    is_point_inside_half_space(
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

pub(crate) fn is_face_selected(
    face: &Face,
    index: usize,
    positions: &Points,
    face_arity_mask: Option<&[usize]>,
    face_index_mask: Option<&[FaceKey]>,
    regular_faces_only: Option<bool>,
) -> bool {
    face_arity_mask.map_or_else(
        || true,
        |face_arity_mask| face_arity_mask.contains(&face.len()),
    ) && face_index_mask.map_or_else(
        || true,
        |face_index_mask| face_index_mask.contains(&(index as _)),
    ) && (!regular_faces_only.unwrap_or(false)
        || ((face_irregularity(face, positions) - 1.0).abs() < 0.1))
}

#[inline]
fn is_point_inside_half_space(
    point: Point,
    plane_origin: Point,
    plane_normal: Vector,
) -> bool {
    let point_to_plane = point - plane_origin;
    let distance = point_to_plane.dot(plane_normal);
    distance > 0.0
}
