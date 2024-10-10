use bevy::{
    app::{App, Startup},
    asset::Assets,
    color::Color,
    core_pipeline::core_3d::Camera3dBundle,
    ecs::system::{Commands, ResMut},
    math::Vec3,
    pbr::{DirectionalLightBundle, PbrBundle, StandardMaterial},
    render::{mesh::Mesh, view::Msaa},
    transform::components::Transform,
    utils::default,
    DefaultPlugins,
};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use polyhedron_ops as p_ops;

fn main() {
    App::new()
        .insert_resource(Msaa::Sample4)
        .add_plugins(DefaultPlugins)
        .add_plugins(PanOrbitCameraPlugin)
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // chamfered_tetrahedron
    let polyhedron = p_ops::Polyhedron::dodecahedron() // D
        .bevel(None, None, None, None, true) // b
        .normalize()
        .finalize();

    commands.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(polyhedron)),
        material: materials.add(Color::srgb(0.4, 0.35, 0.3)),
        ..Default::default()
    });

    // Light.
    commands.spawn(DirectionalLightBundle {
        transform: Transform::from_translation(Vec3::new(4.0, 8.0, 4.0)),
        ..Default::default()
    });

    // Camera.
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(-3.0, 3.0, 5.0)),
            ..default()
        },
        PanOrbitCamera::default(),
    ));
}
