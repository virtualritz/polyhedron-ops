use crate::{
    nsi::start_nsi_render,
    operators::{BaseShape, Operator, OperatorType},
    state::{NsiRenderCommand, PlaygroundState},
};
use bevy::{prelude::*, render::camera::Viewport};
use bevy_egui::{EguiContexts, egui};
use bevy_panorbit_camera::PanOrbitCamera;

pub fn ui_system(
    mut contexts: EguiContexts,
    mut state: ResMut<PlaygroundState>,
    mut operator_types: Local<Vec<OperatorType>>,
    windows: Query<&Window>,
    mut camera_query: Query<
        (&mut Camera, &Transform, &Projection),
        (With<Camera3d>, With<PanOrbitCamera>),
    >,
) {
    if operator_types.is_empty() {
        *operator_types = OperatorType::all();
    }
    let operator_types_list = operator_types.clone();

    // Extract values to avoid borrow checker issues
    let prism_sides = state.prism_sides;
    let antiprism_sides = state.antiprism_sides;

    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };

    let panel_response = egui::SidePanel::left("operators_panel")
        .default_width(500.0)
        .resizable(true)
        .show(ctx, |ui| {
            // Base shape section
            draw_base_shape_section(
                ui,
                &mut state,
                prism_sides,
                antiprism_sides,
            );

            ui.separator();
            ui.heading("Operators");

            // Operator buttons
            draw_operator_buttons(ui, &operator_types_list, &mut state);

            ui.separator();

            // Operators list
            let operators_changes = draw_operators_list(ui, &mut state);

            // Apply operators changes
            if let Some(changes) = operators_changes {
                if changes.needs_rebuild {
                    state.needs_rebuild = true;
                }
                if let Some(new_selected) = changes.new_selected {
                    state.selected_operator = Some(new_selected);
                }
                if let Some(to_remove) = changes.to_remove {
                    state.save_state();
                    state.operators.remove(to_remove);
                    // Update selection
                    if let Some(sel) = state.selected_operator {
                        if sel == to_remove {
                            state.selected_operator = None;
                        } else if sel > to_remove {
                            state.selected_operator = Some(sel - 1);
                        }
                    }
                    state.needs_rebuild = true;
                }
                if let Some((src, dst)) = changes.drag_drop {
                    state.save_state();
                    let op = state.operators.remove(src);
                    let insert_idx = if src < dst { dst - 1 } else { dst };
                    state.operators.insert(insert_idx, op);

                    // Update selected index if needed
                    if let Some(sel) = state.selected_operator {
                        if sel == src {
                            state.selected_operator = Some(insert_idx);
                        } else if src < sel && sel <= dst {
                            state.selected_operator = Some(sel - 1);
                        } else if dst <= sel && sel < src {
                            state.selected_operator = Some(sel + 1);
                        }
                    }
                    state.needs_rebuild = true;
                }
            }

            ui.separator();

            // Clear button
            if ui
                .button("Clear All Operators")
                .on_hover_text("Remove all operators")
                .clicked()
            {
                state.save_state();
                state.operators.clear();
                state.selected_operator = None;
                state.needs_rebuild = true;
            }

            ui.separator();

            // Conway notation display
            draw_conway_notation(ui, &state);

            ui.separator();

            // Undo/Redo buttons
            ui.horizontal(|ui| {
                if ui
                    .button("↶ Undo")
                    .on_hover_text("Undo last change (Ctrl+Z)")
                    .clicked()
                {
                    state.undo();
                }
                if ui
                    .button("↷ Redo")
                    .on_hover_text("Redo last change (Ctrl+Y)")
                    .clicked()
                {
                    state.redo();
                }
            });

            // NSI and Export sections
            draw_nsi_section(ui, &mut state, &windows, &camera_query);

            ui.separator();

            draw_export_section(ui, &state, &camera_query);
        });

    // Update panel width
    let panel_width = panel_response.response.rect.width();

    // Check for keyboard shortcuts
    if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Z)) {
        state.undo();
    }
    if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Y)) {
        state.redo();
    }

    // Update camera viewport
    if let Ok((mut camera, _, _)) = camera_query.single_mut() {
        if let Ok(window) = windows.single() {
            let scale_factor = window.scale_factor();
            let panel_width_physical = (panel_width * scale_factor) as u32;
            let window_width_physical = window.physical_width();
            let window_height_physical = window.physical_height();

            let viewport_width =
                window_width_physical.saturating_sub(panel_width_physical);

            camera.viewport = Some(Viewport {
                physical_position: UVec2::new(panel_width_physical, 0),
                physical_size: UVec2::new(
                    viewport_width,
                    window_height_physical,
                ),
                ..default()
            });
        }
    }

    state.panel_width = panel_width;
}

fn draw_base_shape_section(
    ui: &mut egui::Ui,
    state: &mut PlaygroundState,
    prism_sides: u8,
    antiprism_sides: u8,
) {
    ui.group(|ui| {
        ui.set_width(ui.available_width());

        ui.horizontal(|ui| {
            ui.label("📦");
            ui.add_enabled(false, egui::Checkbox::new(&mut true, ""));
            ui.label(egui::RichText::new("Polyhedron").strong());
        });

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
                            if ui
                                .selectable_value(
                                    &mut state.base_shape,
                                    shape,
                                    shape.name(),
                                )
                                .clicked()
                            {
                                state.save_state();
                                state.needs_rebuild = true;
                            }
                        }

                        if ui
                            .selectable_value(
                                &mut state.base_shape,
                                BaseShape::Prism(prism_sides),
                                format!("{}-Prism", prism_sides),
                            )
                            .clicked()
                        {
                            state.save_state();
                            state.needs_rebuild = true;
                        }

                        if ui
                            .selectable_value(
                                &mut state.base_shape,
                                BaseShape::Antiprism(antiprism_sides),
                                format!("{}-Antiprism", antiprism_sides),
                            )
                            .clicked()
                        {
                            state.save_state();
                            state.needs_rebuild = true;
                        }
                    });
                ui.end_row();

                // Prism sides control
                if matches!(state.base_shape, BaseShape::Prism(_)) {
                    ui.label("Sides:");
                    if ui
                        .add(egui::Slider::new(&mut state.prism_sides, 3..=20))
                        .changed()
                    {
                        state.base_shape = BaseShape::Prism(state.prism_sides);
                        state.needs_rebuild = true;
                    }
                    ui.end_row();
                }

                // Antiprism sides control
                if matches!(state.base_shape, BaseShape::Antiprism(_)) {
                    ui.label("Sides:");
                    if ui
                        .add(egui::Slider::new(
                            &mut state.antiprism_sides,
                            3..=20,
                        ))
                        .changed()
                    {
                        state.base_shape =
                            BaseShape::Antiprism(state.antiprism_sides);
                        state.needs_rebuild = true;
                    }
                    ui.end_row();
                }
            });
    });
}

fn draw_operator_buttons(
    ui: &mut egui::Ui,
    operator_types_list: &[OperatorType],
    state: &mut PlaygroundState,
) {
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
                            .strong(),
                    ),
                );

                if response.clicked() {
                    state.save_state();
                    state.operators.push(Operator::new(*op_type));
                    state.selected_operator = Some(state.operators.len() - 1);
                    state.needs_rebuild = true;
                }

                response.on_hover_text(op_type.name());

                if (i + 1) % 8 == 0 {
                    ui.end_row();
                }
            }
        });
}

struct OperatorsListChanges {
    needs_rebuild: bool,
    new_selected: Option<usize>,
    to_remove: Option<usize>,
    drag_drop: Option<(usize, usize)>,
}

fn draw_operators_list(
    ui: &mut egui::Ui,
    state: &mut PlaygroundState,
) -> Option<OperatorsListChanges> {
    let mut changes = OperatorsListChanges {
        needs_rebuild: false,
        new_selected: state.selected_operator,
        to_remove: None,
        drag_drop: None,
    };

    let mut source_row = None;
    let mut dest_row = None;

    let selected_operator = state.selected_operator;
    let operators_len = state.operators.len();

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let _response = ui.scope(|ui| {
                for i in 0..operators_len {
                    let item_id = egui::Id::new(("operator", i));
                    let is_selected = selected_operator == Some(i);

                    ui.group(|ui| {
                        ui.set_width(ui.available_width());

                        // We need to access the operator data
                        let operator = &mut state.operators[i];
                        let op_type = operator.op_type;
                        let op_name = op_type.name();
                        let mut enabled = operator.enabled;

                        // Operator header
                        ui.horizontal(|ui| {
                            // Drag handle
                            let drag_response = ui.add(
                                egui::Label::new("≡")
                                    .sense(egui::Sense::click_and_drag()),
                            );

                            if drag_response.drag_started() {
                                ui.ctx().set_dragged_id(item_id);
                                source_row = Some(i);
                            }

                            let dropped_id = ui.ctx().drag_stopped_id();
                            if let Some(dropped_id) = dropped_id {
                                if drag_response.hovered()
                                    && dropped_id != item_id
                                {
                                    dest_row = Some(i);
                                }
                            }

                            // Enabled checkbox
                            if ui.checkbox(&mut enabled, "").changed() {
                                operator.enabled = enabled;
                                changes.needs_rebuild = true;
                            }

                            // Operator name
                            let name_response = ui.selectable_label(
                                is_selected,
                                egui::RichText::new(op_name).strong(),
                            );

                            if name_response.clicked() {
                                changes.new_selected = Some(i);
                            }

                            // Delete button
                            ui.with_layout(
                                egui::Layout::right_to_left(
                                    egui::Align::Center,
                                ),
                                |ui| {
                                    if ui
                                        .small_button("❌")
                                        .on_hover_text("Remove operator")
                                        .clicked()
                                    {
                                        changes.to_remove = Some(i);
                                    }
                                },
                            );
                        });

                        // Parameters section
                        if is_selected && operator.enabled {
                            ui.separator();

                            let mut param_changed = false;

                            egui::Grid::new(("operator_params", i))
                                .num_columns(2)
                                .spacing([40.0, 4.0])
                                .show(ui, |ui| {
                                    // Ratio parameter
                                    if op_type.has_ratio() {
                                        ui.label("Ratio:");
                                        if let Some(ratio) = &mut operator.ratio
                                        {
                                            if ui
                                                .add(
                                                    egui::Slider::new(
                                                        ratio,
                                                        0.0..=1.0,
                                                    )
                                                    .step_by(0.01),
                                                )
                                                .changed()
                                            {
                                                param_changed = true;
                                            }
                                        }
                                        ui.end_row();
                                    }

                                    // Height parameter
                                    if op_type.has_height() {
                                        ui.label("Height:");
                                        if let Some(height) =
                                            &mut operator.height
                                        {
                                            if ui
                                                .add(
                                                    egui::Slider::new(
                                                        height,
                                                        0.0..=2.0,
                                                    )
                                                    .step_by(0.01),
                                                )
                                                .changed()
                                            {
                                                param_changed = true;
                                            }
                                        }
                                        ui.end_row();
                                    }

                                    // Regular faces parameter
                                    if op_type.has_regular_faces_param() {
                                        ui.label("Regular faces:");
                                        if ui
                                            .checkbox(
                                                &mut operator.regular_faces,
                                                "",
                                            )
                                            .changed()
                                        {
                                            param_changed = true;
                                        }
                                        ui.end_row();
                                    }
                                });

                            if param_changed {
                                changes.needs_rebuild = true;
                            }
                        }
                    });

                    // Visual separator
                    if i < operators_len - 1 {
                        ui.add_space(4.0);
                    }
                }
            });
        });

    // Handle drag and drop
    let dragged_id = ui.ctx().drag_stopped_id();
    if dragged_id.is_some() {
        if let (Some(src), Some(dst)) = (source_row, dest_row) {
            if src != dst
                && dragged_id == Some(egui::Id::new(("operator", src)))
            {
                changes.drag_drop = Some((src, dst));
            }
        }
    }

    Some(changes)
}

fn draw_conway_notation(ui: &mut egui::Ui, state: &PlaygroundState) {
    ui.horizontal(|ui| {
        ui.label("Conway notation:");
        let notation = state
            .operators
            .iter()
            .filter(|op| op.enabled)
            .map(|op| op.op_type.letter())
            .collect::<Vec<_>>()
            .join("");

        let polyhedron_letter = match state.base_shape {
            BaseShape::Tetrahedron => "T",
            BaseShape::Cube => "C",
            BaseShape::Octahedron => "O",
            BaseShape::Dodecahedron => "D",
            BaseShape::Icosahedron => "I",
            BaseShape::Prism(n) => {
                ui.label(format!("P{}", n));
                return;
            }
            BaseShape::Antiprism(n) => {
                ui.label(format!("A{}", n));
                return;
            }
        };

        ui.label(
            egui::RichText::new(format!("{}{}", notation, polyhedron_letter))
                .monospace()
                .strong(),
        );
    });
}

fn draw_nsi_section(
    ui: &mut egui::Ui,
    state: &mut PlaygroundState,
    windows: &Query<&Window>,
    camera_query: &Query<
        (&mut Camera, &Transform, &Projection),
        (With<Camera3d>, With<PanOrbitCamera>),
    >,
) {
    ui.separator();
    ui.heading("Rendering");

    // Create polyhedron for NSI if needed
    let polyhedron = if state.needs_rebuild {
        None
    } else {
        let mut poly = state.base_shape.create();
        for operator in &state.operators {
            operator.apply(&mut poly);
        }
        poly.finalize();
        Some(poly)
    };

    ui.horizontal(|ui| {
        let prev_nsi = state.nsi_render;
        if ui.toggle_value(&mut state.nsi_render, "NSI").clicked() {
            println!("NSI toggle changed to: {}", state.nsi_render);
            if state.nsi_render && !prev_nsi {
                // Start NSI render
                println!(
                    "NSI enabled, is_rendering: {}",
                    state.nsi_state.is_rendering
                );
                if let Some(polyhedron) = polyhedron {
                    if let Ok(window) = windows.single() {
                        if let Ok((_, camera_transform, camera_projection)) =
                            camera_query.single()
                        {
                            println!("Starting NSI render");
                            let panel_width_physical = (state.panel_width
                                * window.scale_factor())
                                as u32;
                            let viewport_width_physical = window
                                .physical_width()
                                .saturating_sub(panel_width_physical);
                            let viewport_height_physical =
                                window.physical_height();

                            start_nsi_render(
                                state,
                                polyhedron,
                                viewport_width_physical,
                                viewport_height_physical,
                                *camera_transform,
                                camera_projection,
                            );
                        }
                    }
                }
            } else if !state.nsi_render && prev_nsi {
                // Stop NSI render
                if let Some(tx) = &state.nsi_state.render_thread_tx {
                    let _ = tx.send(NsiRenderCommand::Stop);
                }
                state.nsi_state.is_rendering = false;
                // Clear the texture handle so it won't be displayed
                state.nsi_state.texture_handle = None;
            }
        }

        if state.nsi_state.is_rendering {
            ui.label(format!("Progress: {:.0}%", state.nsi_state.progress));
        }
    });
}

fn draw_export_section(
    ui: &mut egui::Ui,
    state: &PlaygroundState,
    camera_query: &Query<
        (&mut Camera, &Transform, &Projection),
        (With<Camera3d>, With<PanOrbitCamera>),
    >,
) {
    ui.separator();
    ui.heading("Export");

    // Create polyhedron for export
    let polyhedron = if state.needs_rebuild {
        None
    } else {
        let mut poly = state.base_shape.create();
        for operator in &state.operators {
            operator.apply(&mut poly);
        }
        poly.finalize();
        Some(poly)
    };

    ui.horizontal(|ui| {
        if ui.button("Export as .obj").clicked() {
            if let Some(polyhedron) = polyhedron.as_ref() {
                match rfd::FileDialog::new()
                    .add_filter("Wavefront OBJ", &["obj"])
                    .set_file_name(&format!("{}.obj", polyhedron.name()))
                    .save_file()
                {
                    Some(path) => {
                        match polyhedron
                            .write_obj(path.parent().unwrap(), false)
                        {
                            Ok(_) => {
                                println!(
                                    "Exported polyhedron to: {}",
                                    path.display()
                                );
                            }
                            Err(e) => {
                                eprintln!("Failed to export: {}", e);
                            }
                        }
                    }
                    None => {
                        println!("Export cancelled");
                    }
                }
            }
        }

        if ui.button("Export as .nsi").clicked() {
            if let Some(polyhedron) = polyhedron {
                export_as_nsi(polyhedron, camera_query);
            }
        }
    });
}

fn export_as_nsi(
    polyhedron: polyhedron_ops::Polyhedron,
    camera_query: &Query<
        (&mut Camera, &Transform, &Projection),
        (With<Camera3d>, With<PanOrbitCamera>),
    >,
) {
    match rfd::FileDialog::new()
        .add_filter("NSI Scene", &["nsi"])
        .set_file_name(&format!("{}.nsi", polyhedron.name()))
        .save_file()
    {
        Some(path) => {
            // Create NSI context for export
            let ctx = nsi::Context::new(None).unwrap();

            // Set up stream to file
            ctx.set_attribute(
                nsi::ROOT,
                &[
                    nsi::string!("streamfilename", path.to_str().unwrap()),
                    nsi::integer!("streamcompression", 1),
                    nsi::integer!("streamindentation", 1),
                ],
            );

            // Create screen
            ctx.create("screen", nsi::SCREEN, None);
            ctx.set_attribute(
                "screen",
                &[
                    nsi::integers!("resolution", &[1920, 1080]).array_len(2),
                    nsi::integer!("oversampling", 32),
                ],
            );

            ctx.create("beauty", nsi::OUTPUT_LAYER, None);
            ctx.set_attribute(
                "beauty",
                &[
                    nsi::string!("variablename", "Ci"),
                    nsi::integer!("withalpha", 1),
                    nsi::string!("scalarformat", "float"),
                ],
            );
            ctx.connect("beauty", None, "screen", "outputlayers", None);

            // Setup output driver
            ctx.create("driver", nsi::OUTPUT_DRIVER, None);
            ctx.set_attribute(
                "driver",
                &[
                    nsi::string!("drivername", "exr"),
                    nsi::string!("imagefilename", "output.exr"),
                ],
            );
            ctx.connect("driver", None, "beauty", "outputdrivers", None);

            // Add the polyhedron
            let poly_handle =
                polyhedron.to_nsi(&ctx, None, Some(10.0), None, None);

            // Camera setup
            if let Ok((_, camera_transform, _)) = camera_query.single() {
                setup_nsi_camera(&ctx, camera_transform);
            }

            // Create basic shading
            setup_nsi_shading(&ctx, &poly_handle);

            // Basic environment light
            setup_nsi_environment(&ctx);

            // Add render control statements
            ctx.render_control(nsi::Action::Start, None);
            ctx.render_control(nsi::Action::Wait, None);

            println!("Exported NSI scene to: {}", path.display());
        }
        None => {
            println!("Export cancelled");
        }
    }
}

fn setup_nsi_camera(ctx: &nsi::Context, camera_transform: &Transform) {
    ctx.create("camera_xform", nsi::TRANSFORM, None);
    ctx.connect("camera_xform", None, nsi::ROOT, "objects", None);

    let nsi_position = camera_transform.translation;
    let look_dir = camera_transform.forward();
    let distance = nsi_position.length();
    let nsi_target = nsi_position + look_dir.as_vec3() * distance;
    let nsi_up = camera_transform.up();

    let forward = (nsi_target - nsi_position).normalize();
    let right = forward.cross(nsi_up.into()).normalize();
    let up = right.cross(forward);

    let transform_matrix = [
        right.x as f64,
        right.y as f64,
        right.z as f64,
        0.0,
        up.x as f64,
        up.y as f64,
        up.z as f64,
        0.0,
        -forward.x as f64,
        -forward.y as f64,
        -forward.z as f64,
        0.0,
        nsi_position.x as f64,
        nsi_position.y as f64,
        nsi_position.z as f64,
        1.0,
    ];

    ctx.set_attribute(
        "camera_xform",
        &[nsi::doubles!("transformationmatrix", &transform_matrix)
            .array_len(16)],
    );

    ctx.create("camera", nsi::PERSPECTIVE_CAMERA, None);
    ctx.connect("camera", None, "camera_xform", "objects", None);
    ctx.set_attribute("camera", &[nsi::float!("fov", 60.0)]);
    ctx.connect("screen", None, "camera", "screens", None);
}

fn setup_nsi_shading(ctx: &nsi::Context, poly_handle: &str) {
    ctx.create("plastic_shader", nsi::SHADER, None);
    ctx.set_attribute(
        "plastic_shader",
        &[
            nsi::string!("shaderfilename", "${DELIGHT}/osl/dlPrincipled"),
            nsi::color!("baseColor", &[0.18, 0.18, 0.8]),
            nsi::float!("roughness", 0.2),
            nsi::float!("metallic", 0.0),
            nsi::float!("specular", 0.5),
        ],
    );

    ctx.create("poly_attrib", nsi::ATTRIBUTES, None);
    ctx.connect("poly_attrib", None, poly_handle, "geometryattributes", None);
    ctx.connect("plastic_shader", None, "poly_attrib", "surfaceshader", None);
}

fn setup_nsi_environment(ctx: &nsi::Context) {
    ctx.create("env_xform", nsi::TRANSFORM, None);
    ctx.connect("env_xform", None, nsi::ROOT, "objects", None);

    ctx.create("environment", nsi::ENVIRONMENT, None);
    ctx.connect("environment", None, "env_xform", "objects", None);

    ctx.create("env_shader", nsi::SHADER, None);
    ctx.set_attribute(
        "env_shader",
        &[
            nsi::string!("shaderfilename", "${DELIGHT}/osl/environmentLight"),
            nsi::float!("intensity", 1.0),
        ],
    );

    ctx.create("env_attrib", nsi::ATTRIBUTES, None);
    ctx.connect(
        "env_attrib",
        None,
        "environment",
        "geometryattributes",
        None,
    );
    ctx.connect("env_shader", None, "env_attrib", "surfaceshader", None);
}
