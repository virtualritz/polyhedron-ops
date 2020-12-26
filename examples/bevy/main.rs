use bevy::prelude::*;
use bevy_orbit_controls::*;
use polyhedron_ops as p_ops;

#[bevy_main]
fn main() {
    App::build()
        .add_plugins(DefaultPlugins)
        .add_plugin(OrbitCameraPlugin)
        .add_startup_system(startup.system())
        .run();
}

fn startup(
    commands: &mut Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
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

    commands
        .spawn(PbrBundle {
            mesh: meshes.add(Mesh::from(polyhedron)),
            material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
            ..Default::default()
        })
        // light
        .spawn(LightBundle {
            transform: Transform::from_translation(Vec3::new(4.0, 8.0, 4.0)),
            ..Default::default()
        })
        // camera
        .spawn(Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(-3.0, 3.0, 5.0))
                .looking_at(Vec3::default(), Vec3::unit_y()),
            ..Default::default()
        })
        .with(OrbitCamera::default());
}
