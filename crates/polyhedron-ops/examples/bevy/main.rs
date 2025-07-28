use bevy::prelude::*;
// use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use polyhedron_ops::Polyhedron;

#[cfg(feature = "console")]
mod console;
#[cfg(feature = "console")]
use console::prelude::*;

#[derive(Component)]
pub struct RootPolyhedron;

fn main() {
    let mut app = App::new();

    app.add_plugins(DefaultPlugins)
        // .add_plugins(PanOrbitCameraPlugin)
        .add_systems(Startup, setup);

    #[cfg(feature = "console")]
    app.add_plugins(ConsolePlugin)
        .add_console_command::<RenderCommand, _>(render_command);

    app.run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // chamfered_tetrahedron
    let polyhedron = Polyhedron::dodecahedron() // D
        .bevel(None, None, None, None, true) // b
        .normalize()
        .finalize();

    commands.spawn((
        Mesh3d(meshes.add(Mesh::from(polyhedron))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.4, 0.35, 0.3),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.5, 0.0),
        RootPolyhedron,
    ));

    // Light.
    commands.spawn((
        PointLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));

    // Camera.
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}
