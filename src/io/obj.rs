use crate::*;
use std::{
    error::Error,
    fs::File,
    io::Write as IoWrite,
    path::{Path, PathBuf},
};

impl Polyhedron {
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
    /// `reverse_winding` flag.
    ///
    /// The return value, on success, is the final, complete path of
    /// the OBJ file.
    #[cfg(feature = "obj")]
    pub fn write_obj(
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

    pub fn read_obj(
        source: &Path,
        reverse_winding: bool,
    ) -> Result<Self, tobj::LoadError> {
        let (geometry, _) =
            tobj::load_obj(source, &tobj::OFFLINE_RENDERING_LOAD_OPTIONS)?;

        Ok(Polyhedron {
            face_index: {
                let mut index = 0;
                geometry[0]
                    .mesh
                    .face_arities
                    .iter()
                    .map(|&face_arity| {
                        assert!(0 != face_arity);
                        let face_arity = face_arity as usize;
                        let mut face_indices = geometry[0].mesh.indices
                            [index..index + face_arity]
                            .to_vec();
                        if reverse_winding {
                            face_indices.reverse();
                        }
                        index += face_arity;

                        face_indices
                    })
                    .collect()
            },
            positions: geometry[0]
                .mesh
                .positions
                .iter()
                .array_chunks::<3>()
                .map(|p| Point::new(*p[0], *p[1], *p[2]))
                .collect(),
            name: geometry[0].name.clone(),
            ..Default::default()
        })
    }
}
