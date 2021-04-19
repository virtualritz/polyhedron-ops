use bevy::prelude::*;
use bevy_orbit_controls::*;
use polyhedron_ops as p_ops;

fn main() {
    App::build()
        .insert_resource(Msaa { samples: 4 })
        .add_plugins(DefaultPlugins)
        .add_plugin(OrbitCameraPlugin)
        .add_startup_system(setup.system())
        .add_system(bevy::input::system::exit_on_esc_system.system())
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // chamfered_tetrahedron
    let polyhedron = p_ops::Polyhedron::tetrahedron() // D
        .kis(None, None, None, true)
        .normalize()
        .bevel(None, None, None, None, true)
        .normalize()
        .needle(None, None, None, true)
        .normalize()
        .gyro(None, None, true)
        .normalize()
        .finalize();

    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(polyhedron)),
        material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
        ..Default::default()
    });

    commands
        // light
        .spawn_bundle(LightBundle {
            transform: Transform::from_translation(Vec3::new(4.0, 8.0, 4.0)),
            ..Default::default()
        });

    commands
        // camera
        .spawn_bundle(PerspectiveCameraBundle {
            transform: Transform::from_translation(Vec3::new(-3.0, 3.0, 5.0))
                .looking_at(Vec3::default(), Vec3::Y),
            ..Default::default()
        })
        .insert(OrbitCamera::default());
}
