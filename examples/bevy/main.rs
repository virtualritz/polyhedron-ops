use bevy::prelude::*;
use polyhedron_ops as p_ops;
use smooth_bevy_cameras::{
    controllers::orbit::{
        OrbitCameraBundle, OrbitCameraController, OrbitCameraPlugin,
    },
    LookTransformPlugin,
};

fn main() {
    App::new()
        .insert_resource(Msaa::Sample4)
        .add_plugins(DefaultPlugins)
        .add_plugin(LookTransformPlugin)
        .add_plugin(OrbitCameraPlugin::default())
        .add_startup_system(setup)
        .add_system(bevy::window::close_on_esc)
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
        material: materials.add(Color::rgb(0.4, 0.35, 0.3).into()),
        ..Default::default()
    });

    commands
        // light
        .spawn(DirectionalLightBundle {
            transform: Transform::from_translation(Vec3::new(4.0, 8.0, 4.0)),
            ..Default::default()
        });

    commands
        // camera
        .spawn(Camera3dBundle::default())
        .insert(OrbitCameraBundle::new(
            OrbitCameraController::default(),
            Vec3::new(-3.0, 3.0, 5.0),
            Vec3::new(0., 0., 0.),
            Vec3::new(0., -1., 0.),
        ));
}
