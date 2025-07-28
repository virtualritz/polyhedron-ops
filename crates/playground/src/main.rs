use bevy::prelude::*;
use bevy::render::camera::Viewport;
use bevy::window::{Window, WindowPlugin};
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use polyhedron_ops::Polyhedron;

#[derive(Debug, Clone, Copy, PartialEq)]
enum OperatorType {
    Ambo,
    Bevel,
    Chamfer,
    Dual,
    Expand,
    Gyro,
    Inset,
    Join,
    Kis,
    Meta,
    Needle,
    Ortho,
    Propellor,
    Quinto,
    Reflect,
    Snub,
    Spherize,
    Truncate,
    Whirl,
    Zip,
}

impl OperatorType {
    fn all() -> Vec<Self> {
        vec![
            Self::Ambo,
            Self::Bevel,
            Self::Chamfer,
            Self::Dual,
            Self::Expand,
            Self::Gyro,
            Self::Inset,
            Self::Join,
            Self::Kis,
            Self::Meta,
            Self::Needle,
            Self::Ortho,
            Self::Propellor,
            Self::Quinto,
            Self::Reflect,
            Self::Snub,
            Self::Spherize,
            Self::Truncate,
            Self::Whirl,
            Self::Zip,
        ]
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Ambo => "Ambo",
            Self::Bevel => "Bevel",
            Self::Chamfer => "Chamfer",
            Self::Dual => "Dual",
            Self::Expand => "Expand",
            Self::Gyro => "Gyro",
            Self::Inset => "Inset",
            Self::Join => "Join",
            Self::Kis => "Kis",
            Self::Meta => "Meta",
            Self::Needle => "Needle",
            Self::Ortho => "Ortho",
            Self::Propellor => "Propellor",
            Self::Quinto => "Quinto",
            Self::Reflect => "Reflect",
            Self::Snub => "Snub",
            Self::Spherize => "Spherize",
            Self::Truncate => "Truncate",
            Self::Whirl => "Whirl",
            Self::Zip => "Zip",
        }
    }

    fn letter(&self) -> &'static str {
        match self {
            Self::Ambo => "a",
            Self::Bevel => "b",
            Self::Chamfer => "c",
            Self::Dual => "d",
            Self::Expand => "e",
            Self::Gyro => "g",
            Self::Inset => "I",
            Self::Join => "j",
            Self::Kis => "k",
            Self::Meta => "m",
            Self::Needle => "n",
            Self::Ortho => "o",
            Self::Propellor => "p",
            Self::Quinto => "q",
            Self::Reflect => "r",
            Self::Snub => "s",
            Self::Spherize => "S",
            Self::Truncate => "t",
            Self::Whirl => "w",
            Self::Zip => "z",
        }
    }

    fn has_ratio(&self) -> bool {
        matches!(
            self,
            Self::Ambo
                | Self::Bevel
                | Self::Chamfer
                | Self::Gyro
                | Self::Inset
                | Self::Join
                | Self::Meta
                | Self::Ortho
                | Self::Propellor
                | Self::Snub
                | Self::Spherize
                | Self::Truncate
                | Self::Whirl
                | Self::Zip
        )
    }

    fn has_height(&self) -> bool {
        matches!(
            self,
            Self::Bevel
                | Self::Kis
                | Self::Meta
                | Self::Quinto
                | Self::Snub
                | Self::Whirl
        )
    }

    fn has_regular_faces_param(&self) -> bool {
        matches!(
            self,
            Self::Kis
                | Self::Truncate
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Operator {
    op_type: OperatorType,
    ratio: Option<f32>,
    height: Option<f32>,
    nsides: Option<u32>,
    regular_faces: bool,
    enabled: bool,
}

impl Operator {
    fn new(op_type: OperatorType) -> Self {
        Self {
            op_type,
            ratio: if op_type.has_ratio() { Some(0.5) } else { None },
            height: if op_type.has_height() {
                Some(0.3)
            } else {
                None
            },
            nsides: None, // Will be set based on operator type if needed
            regular_faces: false,
            enabled: true,
        }
    }

    fn apply(&self, polyhedron: &mut Polyhedron) {
        if !self.enabled {
            return;
        }

        match self.op_type {
            OperatorType::Ambo => {
                polyhedron.ambo(self.ratio, false);
            }
            OperatorType::Bevel => {
                polyhedron.bevel(self.ratio, self.height, None, None, false);
            }
            OperatorType::Chamfer => {
                polyhedron.chamfer(self.ratio, false);
            }
            OperatorType::Dual => {
                polyhedron.dual(false);
            }
            OperatorType::Expand => {
                polyhedron.expand(None, false);
            }
            OperatorType::Gyro => {
                polyhedron.gyro(self.ratio, None, false);
            }
            OperatorType::Inset => {
                polyhedron.inset(self.ratio, None, false);
            }
            OperatorType::Join => {
                polyhedron.join(self.ratio, false);
            }
            OperatorType::Kis => {
                polyhedron.kis(self.height, None, None, None, false);
            }
            OperatorType::Meta => {
                polyhedron.meta(self.ratio, self.height, None, None, false);
            }
            OperatorType::Needle => {
                polyhedron.needle(None, None, None, false);
            }
            OperatorType::Ortho => {
                polyhedron.ortho(self.ratio, false);
            }
            OperatorType::Propellor => {
                polyhedron.propellor(self.ratio, false);
            }
            OperatorType::Quinto => {
                polyhedron.quinto(self.height, false);
            }
            OperatorType::Reflect => {
                polyhedron.reflect(false);
            }
            OperatorType::Snub => {
                polyhedron.snub(self.ratio, self.height, false);
            }
            OperatorType::Spherize => {
                polyhedron.spherize(self.ratio, false);
            }
            OperatorType::Truncate => {
                polyhedron.truncate(self.ratio, None, None, false);
            }
            OperatorType::Whirl => {
                polyhedron.whirl(self.ratio, self.height, false);
            }
            OperatorType::Zip => {
                polyhedron.zip(self.ratio, None, None, false);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum BaseShape {
    Tetrahedron,
    Cube,
    Octahedron,
    Dodecahedron,
    Icosahedron,
    Prism(u8),
    Antiprism(u8),
}

impl BaseShape {
    fn name(&self) -> String {
        match self {
            Self::Tetrahedron => "Tetrahedron".to_string(),
            Self::Cube => "Cube".to_string(),
            Self::Octahedron => "Octahedron".to_string(),
            Self::Dodecahedron => "Dodecahedron".to_string(),
            Self::Icosahedron => "Icosahedron".to_string(),
            Self::Prism(n) => format!("{}-Prism", n),
            Self::Antiprism(n) => format!("{}-Antiprism", n),
        }
    }

    fn create(&self) -> Polyhedron {
        match self {
            Self::Tetrahedron => Polyhedron::tetrahedron(),
            Self::Cube => Polyhedron::hexahedron(),
            Self::Octahedron => Polyhedron::octahedron(),
            Self::Dodecahedron => Polyhedron::dodecahedron(),
            Self::Icosahedron => Polyhedron::icosahedron(),
            Self::Prism(n) => Polyhedron::prism(Some(*n as usize)),
            Self::Antiprism(n) => Polyhedron::antiprism(Some(*n as usize)),
        }
    }
}

#[derive(Clone, PartialEq)]
struct EditableState {
    base_shape: BaseShape,
    operators: Vec<Operator>,
}

#[derive(Resource)]
struct PlaygroundState {
    base_shape: BaseShape,
    operators: Vec<Operator>,
    selected_operator: Option<usize>,
    needs_rebuild: bool,
    prism_sides: u8,
    antiprism_sides: u8,
    panel_width: f32,
    nsi_render: bool,
    undoer: egui::util::undoer::Undoer<EditableState>,
}

impl PlaygroundState {
    fn save_state(&mut self) {
        let state = EditableState {
            base_shape: self.base_shape.clone(),
            operators: self.operators.clone(),
        };
        self.undoer.add_undo(&state);
    }
    
    fn current_state(&self) -> EditableState {
        EditableState {
            base_shape: self.base_shape.clone(),
            operators: self.operators.clone(),
        }
    }
    
    fn undo(&mut self) -> bool {
        let current = self.current_state();
        if let Some(state) = self.undoer.undo(&current) {
            self.base_shape = state.base_shape.clone();
            self.operators = state.operators.clone();
            self.selected_operator = None;
            self.needs_rebuild = true;
            true
        } else {
            false
        }
    }
    
    fn redo(&mut self) -> bool {
        let current = self.current_state();
        if let Some(state) = self.undoer.redo(&current) {
            self.base_shape = state.base_shape.clone();
            self.operators = state.operators.clone();
            self.selected_operator = None;
            self.needs_rebuild = true;
            true
        } else {
            false
        }
    }
}

impl Default for PlaygroundState {
    fn default() -> Self {
        let base_shape = BaseShape::Dodecahedron;
        let operators = Vec::new();
        
        Self {
            base_shape,
            operators,
            selected_operator: None,
            needs_rebuild: true,
            prism_sides: 6,
            antiprism_sides: 6,
            panel_width: 500.0,
            nsi_render: false,
            undoer: egui::util::undoer::Undoer::default(),
        }
    }
}

#[derive(Component)]
struct PolyhedronMesh;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Polyhedron Playground".to_string(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin {
            enable_multipass_for_primary_context: false,
        })
        .add_plugins(PanOrbitCameraPlugin)
        .init_resource::<PlaygroundState>()
        .add_systems(Startup, setup)
        .add_systems(Update, ui_system)
        .add_systems(Update, rebuild_polyhedron)
        .add_systems(Update, handle_zoom)
        .add_systems(Update, update_camera_viewport.after(ui_system))
        .add_systems(Update, viewport_overlay.after(ui_system))
        .run();
}

fn setup(
    mut commands: Commands,
    _meshes: ResMut<Assets<Mesh>>,
    _materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera - positioned to look at origin  
    commands.spawn((
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
}

fn ui_system(
    mut contexts: EguiContexts,
    mut state: ResMut<PlaygroundState>,
    mut operator_types: Local<Vec<OperatorType>>,
) {
    if operator_types.is_empty() {
        *operator_types = OperatorType::all();
    }
    let operator_types_list = operator_types.clone();

    // Extract values to avoid borrow checker issues
    let prism_sides = state.prism_sides;
    let antiprism_sides = state.antiprism_sides;
    
    let panel_response = egui::SidePanel::left("operators_panel")
        .default_width(500.0)
        .resizable(true)
        .show(contexts.ctx_mut(), |ui| {
            // Base shape as first "operator" - always pinned at top
            ui.group(|ui| {
                ui.set_width(ui.available_width());
                
                ui.horizontal(|ui| {
                    // No drag handle for base shape
                    ui.label("📦"); // Icon for base shape
                    
                    // Always enabled
                    ui.add_enabled(false, egui::Checkbox::new(&mut true, ""));
                    
                    // "Polyhedron" label
                    ui.label(egui::RichText::new("Polyhedron").strong());
                });
                
                // Parameters section
                ui.separator();
                egui::Grid::new("polyhedron_params")
                    .num_columns(2)
                    .spacing([40.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Shape:");
                        egui::ComboBox::from_label("")
                            .selected_text(state.base_shape.name())
                            .show_ui(ui, |ui| {
                                for shape in [
                                    BaseShape::Tetrahedron,
                                    BaseShape::Cube,
                                    BaseShape::Octahedron,
                                    BaseShape::Dodecahedron,
                                    BaseShape::Icosahedron,
                                ] {
                                    if ui.selectable_value(&mut state.base_shape, shape, shape.name()).clicked() {
                                        state.save_state();
                                        state.needs_rebuild = true;
                                    }
                                }
                                
                                if ui.selectable_value(
                                    &mut state.base_shape,
                                    BaseShape::Prism(prism_sides),
                                    format!("{}-Prism", prism_sides)
                                ).clicked() {
                                    state.save_state();
                                    state.needs_rebuild = true;
                                }
                                
                                if ui.selectable_value(
                                    &mut state.base_shape,
                                    BaseShape::Antiprism(antiprism_sides),
                                    format!("{}-Antiprism", antiprism_sides)
                                ).clicked() {
                                    state.save_state();
                                    state.needs_rebuild = true;
                                }
                            });
                        ui.end_row();
                        
                        // Prism sides control
                        if matches!(state.base_shape, BaseShape::Prism(_)) {
                            ui.label("Sides:");
                            if ui.add(egui::Slider::new(&mut state.prism_sides, 3..=20)).changed() {
                                state.base_shape = BaseShape::Prism(state.prism_sides);
                                state.needs_rebuild = true;
                            }
                            ui.end_row();
                        }
                        
                        // Antiprism sides control
                        if matches!(state.base_shape, BaseShape::Antiprism(_)) {
                            ui.label("Sides:");
                            if ui.add(egui::Slider::new(&mut state.antiprism_sides, 3..=20)).changed() {
                                state.base_shape = BaseShape::Antiprism(state.antiprism_sides);
                                state.needs_rebuild = true;
                            }
                            ui.end_row();
                        }
                    });
            });
            
            ui.separator();
            ui.heading("Operators");
            
            // Grid of operator buttons
            egui::Grid::new("operator_buttons")
                .num_columns(8)
                .spacing([4.0, 4.0])
                .show(ui, |ui| {
                    for (i, op_type) in operator_types_list.iter().enumerate() {
                        let response = ui.add_sized(
                            [32.0, 32.0],
                            egui::Button::new(
                                egui::RichText::new(op_type.letter())
                                    .size(16.0)
                                    .strong()
                            )
                        );
                        
                        if response.clicked() {
                            state.save_state();
                            state.operators.push(Operator::new(op_type.clone()));
                            // Select the newly added operator to show its parameters
                            state.selected_operator = Some(state.operators.len() - 1);
                            state.needs_rebuild = true;
                        }
                        
                        response.on_hover_text(op_type.name());
                        
                        // New row every 8 buttons
                        if (i + 1) % 8 == 0 {
                            ui.end_row();
                        }
                    }
                });
            
            ui.separator();
            
            // Operators list with drag and drop
            let mut to_remove = None;
            let mut changed = false;
            
            // Track drag and drop
            let mut source_row = None;
            let mut dest_row = None;
            
            let selected_operator = state.selected_operator;
            let mut new_selected = selected_operator;
            
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    // Drop zone for the entire list
                    let _response = ui.scope(|ui| {
                        for (i, operator) in state.operators.iter_mut().enumerate() {
                            let item_id = egui::Id::new(("operator", i));
                            let is_selected = selected_operator == Some(i);
                            
                            ui.group(|ui| {
                                ui.set_width(ui.available_width());
                                
                                // Operator header
                                ui.horizontal(|ui| {
                                    // Drag handle (only this is draggable)
                                    let drag_response = ui.dnd_drag_source(item_id, i, |ui| {
                                        ui.label("☰");
                                    }).response;
                                    
                                    // Handle drop on this item
                                    if let Some(pointer_pos) = ui.ctx().pointer_interact_pos() {
                                        if let Some(_source) = drag_response.dnd_hover_payload::<usize>() {
                                            // Visual feedback for drop location
                                            let rect = ui.min_rect();
                                            let stroke = egui::Stroke::new(2.0, egui::Color32::WHITE);
                                            
                                            if pointer_pos.y < rect.center().y {
                                                ui.painter().hline(rect.x_range(), rect.top(), stroke);
                                            } else {
                                                ui.painter().hline(rect.x_range(), rect.bottom(), stroke);
                                            }
                                            
                                            if let Some(released) = drag_response.dnd_release_payload::<usize>() {
                                                source_row = Some(*released);
                                                dest_row = Some(if pointer_pos.y < rect.center().y { i } else { i + 1 });
                                            }
                                        }
                                    }
                                    
                                    // Enable checkbox
                                    if ui.checkbox(&mut operator.enabled, "").changed() {
                                        changed = true;
                                    }
                                    
                                    // Operator name button
                                    if ui.selectable_label(is_selected, operator.op_type.name()).clicked() {
                                        new_selected = if is_selected { None } else { Some(i) };
                                    }
                                    
                                    // Fill remaining space
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        // Remove button with trash icon
                                        if ui.small_button("🗑").on_hover_text("Remove operator").clicked() {
                                            to_remove = Some(i);
                                        }
                                    });
                                });
                                
                                // Parameters section (when selected)
                                if is_selected {
                                    ui.separator();
                                    egui::Grid::new(("params_grid", i))
                                        .num_columns(2)
                                        .spacing([40.0, 4.0])
                                        .show(ui, |ui| {
                                            // Ratio parameter
                                            if let Some(ratio) = &mut operator.ratio {
                                                ui.label("Ratio:");
                                                if ui.add(egui::Slider::new(ratio, 0.0..=1.0).step_by(0.01)).changed() {
                                                    changed = true;
                                                }
                                                ui.end_row();
                                            }
                                            
                                            // Height parameter
                                            if let Some(height) = &mut operator.height {
                                                ui.label("Height:");
                                                if ui.add(egui::Slider::new(height, 0.0..=2.0).step_by(0.01)).changed() {
                                                    changed = true;
                                                }
                                                ui.end_row();
                                            }
                                            
                                            // Nsides parameter
                                            if let Some(nsides) = &mut operator.nsides {
                                                ui.label("Sides:");
                                                if ui.add(egui::Slider::new(nsides, 3..=20)).changed() {
                                                    changed = true;
                                                }
                                                ui.end_row();
                                            }
                                            
                                            // Regular faces parameter
                                            if operator.op_type.has_regular_faces_param() {
                                                ui.label("Regular faces:");
                                                if ui.checkbox(&mut operator.regular_faces, "").changed() {
                                                    changed = true;
                                                }
                                                ui.end_row();
                                            }
                                        });
                                }
                            });
                        }
                    });
                });
            
            // Update selected operator
            state.selected_operator = new_selected;
            
            // Apply drag and drop reordering
            if let (Some(source), Some(dest)) = (source_row, dest_row) {
                if source != dest && source != dest.saturating_sub(1) {
                    let item = state.operators.remove(source);
                    let insert_idx = if source < dest { dest - 1 } else { dest };
                    state.operators.insert(insert_idx, item);
                    
                    // Update selected index
                    if let Some(selected) = state.selected_operator {
                        if selected == source {
                            state.selected_operator = Some(insert_idx);
                        } else if source < selected && selected <= insert_idx {
                            state.selected_operator = Some(selected - 1);
                        } else if insert_idx <= selected && selected < source {
                            state.selected_operator = Some(selected + 1);
                        }
                    }
                    
                    state.needs_rebuild = true;
                }
            }
            
            // Remove operator if requested
            if let Some(idx) = to_remove {
                state.operators.remove(idx);
                if let Some(selected) = state.selected_operator {
                    if selected == idx {
                        state.selected_operator = None;
                    } else if selected > idx {
                        state.selected_operator = Some(selected - 1);
                    }
                }
                state.needs_rebuild = true;
            }
            
            if changed {
                state.needs_rebuild = true;
            }
            
            // Add some space before bottom buttons
            ui.add_space(ui.available_height() - 100.0);
            
            ui.separator();
            
            // Export button
            if ui.add_sized(
                [ui.available_width(), 40.0],
                egui::Button::new("Export...")
            ).clicked() {
                // TODO: Implement export dialog
            }
            
            ui.separator();
            
            // Clear All button at the bottom
            if ui.add_sized(
                [ui.available_width(), 40.0],
                egui::Button::new("Clear All Operators")
            ).clicked() {
                state.save_state();
                state.operators.clear();
                state.selected_operator = None;
                state.needs_rebuild = true;
            }
            
            ui.separator();
        });
    
    // Update panel width
    state.panel_width = panel_response.response.rect.width();
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

    // Create new polyhedron
    let mut polyhedron = state.base_shape.create();

    // Apply operators
    for operator in &state.operators {
        operator.apply(&mut polyhedron);
    }

    // Normalize and finalize
    polyhedron.normalize().finalize();
    

    // Create mesh
    commands.spawn((
        Mesh3d(meshes.add(Mesh::from(polyhedron))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb_u8(124, 144, 255),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0).with_scale(Vec3::splat(2.0)),
        PolyhedronMesh,
    ));

    state.needs_rebuild = false;
}

fn handle_zoom(
    mut contexts: EguiContexts,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<PlaygroundState>,
) {
    let ctx = contexts.ctx_mut();
    
    let ctrl_pressed = keyboard_input.pressed(KeyCode::ControlLeft) || keyboard_input.pressed(KeyCode::ControlRight);
    let shift_pressed = keyboard_input.pressed(KeyCode::ShiftLeft) || keyboard_input.pressed(KeyCode::ShiftRight);
    
    if ctrl_pressed {
        let current_scale = ctx.pixels_per_point();
        
        if keyboard_input.just_pressed(KeyCode::Equal) || keyboard_input.just_pressed(KeyCode::NumpadAdd) {
            // Zoom in
            let new_scale = (current_scale * 1.2).min(3.0);
            ctx.set_pixels_per_point(new_scale);
        } else if keyboard_input.just_pressed(KeyCode::Minus) || keyboard_input.just_pressed(KeyCode::NumpadSubtract) {
            // Zoom out  
            let new_scale = (current_scale / 1.2).max(0.5);
            ctx.set_pixels_per_point(new_scale);
        } else if keyboard_input.just_pressed(KeyCode::KeyZ) {
            if shift_pressed {
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

fn update_camera_viewport(
    windows: Query<&Window>,
    state: Res<PlaygroundState>,
    mut contexts: EguiContexts,
    mut camera_query: Query<&mut Camera, With<Camera3d>>,
) {
    if let Ok(window) = windows.single() {
        let ctx = contexts.ctx_mut();
        let pixels_per_point = ctx.pixels_per_point();
        
        for mut camera in camera_query.iter_mut() {
            // Set viewport to exclude the egui panel
            let panel_width_pixels = (state.panel_width * pixels_per_point) as u32;
            camera.viewport = Some(Viewport {
                physical_position: UVec2::new(panel_width_pixels, 0),
                physical_size: UVec2::new(
                    ((window.width() * pixels_per_point) as u32).saturating_sub(panel_width_pixels).max(1),
                    (window.height() * pixels_per_point) as u32,
                ),
                depth: 0.0..1.0,
            });
        }
    }
}

fn viewport_overlay(
    mut contexts: EguiContexts,
    mut state: ResMut<PlaygroundState>,
    windows: Query<&Window>,
) {
    if let Ok(window) = windows.single() {
        let ctx = contexts.ctx_mut();
        
        // Position the overlay in the top-right of the viewport
        let panel_width = state.panel_width;
        let window_width = window.width();
        let viewport_width = window_width - panel_width;
        
        egui::Area::new(egui::Id::new("viewport_overlay"))
            .fixed_pos(egui::pos2(panel_width + viewport_width - 100.0, 10.0))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let response = ui.toggle_value(&mut state.nsi_render, "NSI");
                    if response.changed() {
                        // TODO: Toggle NSI rendering
                        if state.nsi_render {
                            println!("NSI rendering enabled");
                        } else {
                            println!("NSI rendering disabled");
                        }
                    }
                });
            });
    }
}
