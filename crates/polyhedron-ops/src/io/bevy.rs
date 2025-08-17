use crate::*;
use bevy::render::{
    mesh::{Indices, Mesh, PrimitiveTopology, VertexAttributeValues},
    render_asset::RenderAssetUsages,
};

/// Conversion to a bevy [`Mesh`].
impl From<Polyhedron> for Mesh {
    fn from(mut polyhedron: Polyhedron) -> Self {
        // Reverse winding order for Bevy's counter-clockwise convention
        polyhedron.reverse();
        let (index, positions, normals) = polyhedron.to_triangle_mesh_buffers();

        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        );

        mesh.insert_indices(Indices::U32(index));

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
                    .map(|n| [-n.x, -n.y, -n.z]) // Flip normals
                    .collect::<Vec<_>>(),
            ),
        );

        // Add UV coordinates - required for StandardMaterial in Bevy 0.16
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_UV_0,
            VertexAttributeValues::Float32x2(vec![[0.0, 0.0]; positions.len()]),
        );

        mesh
    }
}
