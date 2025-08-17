use bevy::{
    prelude::*,
    render::view::RenderLayers,
    window::{Window, WindowPlugin},
};
use bevy_egui::{
    EguiGlobalSettings, EguiPlugin, EguiPrimaryContextPass, PrimaryEguiContext,
};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

use playground::{
    state::{NsiRenderCommand, PlaygroundState, PolyhedronMesh},
    ui::{ui_system, update_nsi_texture, viewport_overlay},
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Polyhedron Playground".to_string(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin::default())
        .add_plugins(PanOrbitCameraPlugin)
        .init_resource::<PlaygroundState>()
        .add_systems(Startup, setup)
        .add_systems(Update, rebuild_polyhedron)
        .add_systems(Update, update_nsi_camera)
        .add_systems(Update, toggle_mesh_visibility)
        .add_systems(EguiPrimaryContextPass, ui_system)
        .add_systems(EguiPrimaryContextPass, handle_zoom)
        .add_systems(EguiPrimaryContextPass, update_nsi_texture)
        .add_systems(
            EguiPrimaryContextPass,
            viewport_overlay.after(update_nsi_texture),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    mut egui_global_settings: ResMut<EguiGlobalSettings>,
    _meshes: ResMut<Assets<Mesh>>,
    _materials: ResMut<Assets<StandardMaterial>>,
) {
    // Disable automatic creation of primary context to set it up manually
    egui_global_settings.auto_create_primary_context = false;
    // Camera - positioned to look at origin
    commands
        .spawn((
            Camera3d::default(),
            Transform::from_xyz(0.0, 5.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
            PanOrbitCamera {
                focus: Vec3::ZERO,
                ..default()
            },
        ))
        .with_children(|parent| {
            // Key light - main light from upper right relative to camera view
            parent.spawn((
                PointLight {
                    intensity: 1000000.0,
                    shadows_enabled: true,
                    ..default()
                },
                Transform::from_xyz(5.0, 3.0, -8.0),
            ));

            // Fill light - softer light from the left relative to camera view
            parent.spawn((
                PointLight {
                    intensity: 400000.0,
                    shadows_enabled: false,
                    ..default()
                },
                Transform::from_xyz(-4.0, 1.0, -6.0),
            ));

            // Back light - rim lighting from behind the subject
            parent.spawn((
                PointLight {
                    intensity: 600000.0,
                    shadows_enabled: false,
                    ..default()
                },
                Transform::from_xyz(0.0, 5.0, -15.0),
            ));
        });

    // Add subtle ambient light for overall illumination
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 0.05,
        affects_lightmapped_meshes: false,
    });

    // Egui camera - renders UI to full window
    commands.spawn((
        PrimaryEguiContext,
        Camera3d::default(),
        RenderLayers::none(), // Don't render any 3D objects
        Camera {
            order: 1, // Render after the world camera
            ..default()
        },
    ));
}

fn rebuild_polyhedron(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut state: ResMut<PlaygroundState>,
    query: Query<Entity, With<PolyhedronMesh>>,
) {
    if !state.needs_rebuild {
        return;
    }

    // Remove existing mesh
    for entity in query.iter() {
        commands.entity(entity).despawn();
    }

    // Create base polyhedron
    let mut polyhedron = state.base_shape.create();

    // Apply operators
    for operator in &state.operators {
        operator.apply(&mut polyhedron);
    }

    // Finalize and convert to mesh
    polyhedron.finalize();
    let mesh: Mesh = polyhedron.into();

    // Spawn new mesh
    commands.spawn(PolyhedronMesh).insert((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.5, 0.5, 0.8),
            perceptual_roughness: 0.3,
            metallic: 0.0,
            ..default()
        })),
        Transform::default(),
        Visibility::default(),
    ));

    state.needs_rebuild = false;
}

fn handle_zoom(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<PlaygroundState>,
) {
    if keyboard_input.pressed(KeyCode::ControlLeft)
        || keyboard_input.pressed(KeyCode::ControlRight)
    {
        if keyboard_input.just_pressed(KeyCode::KeyZ) {
            if keyboard_input.pressed(KeyCode::ShiftLeft)
                || keyboard_input.pressed(KeyCode::ShiftRight)
            {
                // Redo with Ctrl+Shift+Z
                state.redo();
            } else {
                // Undo with Ctrl+Z
                state.undo();
            }
        } else if keyboard_input.just_pressed(KeyCode::KeyY) {
            // Redo with Ctrl+Y
            state.redo();
        }
    }
}

fn update_nsi_camera(
    mut state: ResMut<PlaygroundState>,
    camera_query: Query<
        &Transform,
        (With<Camera3d>, With<PanOrbitCamera>, Changed<Transform>),
    >,
) {
    // Check if NSI is rendering and camera has moved
    if state.nsi_state.is_rendering {
        if let Ok(camera_transform) = camera_query.single() {
            // Check if camera has moved significantly
            if let Some(last_transform) = state.nsi_state.last_camera_transform
            {
                let position_delta = (camera_transform.translation
                    - last_transform.translation)
                    .length();
                let rotation_delta = camera_transform
                    .rotation
                    .angle_between(last_transform.rotation);

                if position_delta > 0.01 || rotation_delta > 0.001 {
                    println!(
                        "Camera delta - position: {:.6}, rotation: {:.6} radians",
                        position_delta, rotation_delta
                    );
                    println!("Camera moved, updating NSI camera");

                    // Send camera update command
                    if let Some(tx) = &state.nsi_state.render_thread_tx {
                        let _ = tx.send(NsiRenderCommand::UpdateCamera(
                            *camera_transform,
                        ));
                    }

                    // Update last transform
                    state.nsi_state.last_camera_transform =
                        Some(*camera_transform);
                }
            }
        }
    }
}

fn toggle_mesh_visibility(
    state: Res<PlaygroundState>,
    mut visibility_query: Query<&mut Visibility, With<PolyhedronMesh>>,
) {
    // Hide realtime mesh when NSI is active
    for mut visibility in visibility_query.iter_mut() {
        *visibility = if state.nsi_render {
            Visibility::Hidden
        } else {
            Visibility::Visible
        };
    }
}
