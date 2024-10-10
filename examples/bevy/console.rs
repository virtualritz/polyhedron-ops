use crate::RootPolyhedron;
use bevy::prelude::{error, info, Assets, Handle, Mesh, Query, ResMut, With};
use bevy_console::ConsoleCommand;
use clap::Parser;
use polyhedron_ops::Polyhedron;
use std::{error::Error, mem::replace};

pub mod prelude {
    pub use crate::console::{render_command, RenderCommand};
    pub use bevy_console::{AddConsoleCommand, ConsolePlugin};
}

fn render(conway: String) -> Result<Polyhedron, Box<dyn Error>> {
    let polyhedron = Polyhedron::try_from(conway.as_str())?
        .normalize()
        .finalize();
    Ok(polyhedron)
}

#[derive(Parser, ConsoleCommand)]
#[command(name = "render")]
pub struct RenderCommand {
    conway: String,
}

pub fn render_command(
    mesh_query: Query<&Handle<Mesh>, With<RootPolyhedron>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut log: ConsoleCommand<RenderCommand>,
) {
    if let Some(Ok(RenderCommand { conway })) = log.take() {
        let update = || -> Result<String, Box<dyn Error>> {
            let polyhedron = render(conway)?;
            let mesh_handle = mesh_query.get_single()?;
            let mesh = meshes
                .get_mut(mesh_handle)
                .ok_or("Root polyhedron mesh not found")?;
            let name = polyhedron.name().clone();
            let _ = replace::<Mesh>(mesh, Mesh::from(polyhedron));
            Ok(name)
        };

        match update() {
            Ok(name) => {
                info!("Rendered polyhedron: {name}")
            }
            Err(e) => error!("Unable to render polyhedron: {e:?}"),
        }
    }
}
