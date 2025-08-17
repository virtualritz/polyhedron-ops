use crate::{
    nsi::start_nsi_render,
    state::{NsiRenderCommand, PlaygroundState},
};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};
use bevy_panorbit_camera::PanOrbitCamera;

pub fn viewport_overlay(
    mut contexts: EguiContexts,
    mut state: ResMut<PlaygroundState>,
    windows: Query<&Window>,
    camera_query: Query<
        (&Transform, &Projection),
        (With<Camera3d>, With<PanOrbitCamera>),
    >,
) {
    if let Ok(window) = windows.single() {
        let Ok(ctx) = contexts.ctx_mut() else {
            return;
        };

        // Position the overlay in the top-right of the viewport
        let panel_width = state.panel_width;
        let window_width = window.width();
        let viewport_width = window_width - panel_width;

        // AIDEV-NOTE: Ensure NSI toggle is on top layer and fully interactive.
        egui::Area::new(egui::Id::new("viewport_overlay"))
            .fixed_pos(egui::pos2(panel_width + viewport_width - 100.0, 10.0))
            .order(egui::Order::Foreground) // Ensure it's on top
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let response =
                        ui.toggle_value(&mut state.nsi_render, "NSI")
                            .on_hover_text("Toggle NSI rendering");

                    if response.changed() {
                        println!("NSI toggle changed from overlay to: {}", state.nsi_render);

                        if state.nsi_render {
                            // Start NSI render
                            if !state.nsi_state.is_rendering && !state.needs_rebuild {
                                // Clear the old texture to avoid showing stale render
                                state.nsi_state.texture_handle = None;

                                // Create polyhedron
                                let mut poly = state.base_shape.create();
                                for operator in &state.operators {
                                    operator.apply(&mut poly);
                                }
                                poly.finalize();

                                if let Ok((camera_transform, camera_projection)) = camera_query.single() {
                                    println!("Starting NSI render from viewport overlay");
                                    let panel_width_physical = (state.panel_width * window.scale_factor()) as u32;
                                    let viewport_width_physical = window.physical_width().saturating_sub(panel_width_physical);
                                    let viewport_height_physical = window.physical_height();

                                    start_nsi_render(
                                        &mut state,
                                        poly,
                                        viewport_width_physical,
                                        viewport_height_physical,
                                        *camera_transform,
                                        camera_projection,
                                    );
                                }
                            }
                        } else {
                            // Stop render
                            if let Some(tx) = &state.nsi_state.render_thread_tx {
                                let _ = tx.send(NsiRenderCommand::Stop);
                            }
                            state.nsi_state.is_rendering = false;
                            // Keep the texture handle so the last render is still visible
                            // state.nsi_state.texture_handle = None;
                        }
                    }
                });
            });
    }

    // Display NSI render overlay if texture is available
    // AIDEV-NOTE: Fixed race condition - overlay now shows when NSI is enabled
    // and texture exists. Previously required both nsi_render AND
    // (is_rendering OR has_texture), causing timing issues.
    if state.nsi_render && state.nsi_state.texture_handle.is_some() {
        if let Some(texture_handle) = &state.nsi_state.texture_handle {
            let Ok(ctx) = contexts.ctx_mut() else {
                return;
            };

            // Create a fullscreen overlay for the NSI render
            let panel_width = state.panel_width;

            if let Ok(window) = windows.single() {
                let window_width = window.width();
                let window_height = window.height();
                let viewport_width = window_width - panel_width;

                egui::Area::new(egui::Id::new("nsi_render_overlay"))
                    .fixed_pos(egui::pos2(panel_width, 0.0))
                    .order(egui::Order::Middle) // Render between background and foreground UI
                    .interactable(false) // Make sure overlay doesn't block interactions
                    .show(ctx, |ui| {
                        // Add a dark background for debugging
                        let rect = egui::Rect::from_min_size(
                            ui.cursor().min,
                            egui::vec2(viewport_width, window_height),
                        );
                        ui.painter().rect_filled(
                            rect,
                            0.0,
                            egui::Color32::from_rgba_unmultiplied(
                                64, 0, 64, 128,
                            ), // Dark purple background
                        );

                        // Display the NSI rendered image
                        ui.image((
                            texture_handle.id(),
                            egui::vec2(viewport_width, window_height),
                        ));
                    });
            }
        }
    }
}

pub fn update_nsi_texture(
    mut contexts: EguiContexts,
    mut state: ResMut<PlaygroundState>,
) {
    // Update texture if tiles have been updated
    if let Some(tile_updated) = &state.nsi_state.tile_updated {
        let was_updated = *tile_updated.lock();
        if was_updated {
            *tile_updated.lock() = false;

            // Update the egui texture from the image buffer
            let texture_update = if let Some(image) = &state.nsi_state.image {
                let Ok(ctx) = contexts.ctx_mut() else {
                    return;
                };

                // Update texture from image buffer
                let img = image.lock();
                let size = [img.width() as usize, img.height() as usize];
                let pixels: Vec<egui::Color32> = img
                    .pixels()
                    .map(|p| {
                        // Use premultiplied alpha for proper blending
                        egui::Color32::from_rgba_premultiplied(
                            p[0], p[1], p[2], p[3],
                        )
                    })
                    .collect();

                // Check if we have any visible pixels
                let visible_count = pixels
                    .iter()
                    .filter(|p| {
                        let [r, g, b, a] = p.to_array();
                        a > 0 && (r > 0 || g > 0 || b > 0)
                    })
                    .count();

                let texture = ctx.load_texture(
                    "nsi_render",
                    egui::ColorImage { size, pixels },
                    egui::TextureOptions::LINEAR,
                );

                // Request egui to repaint to show the updated texture
                ctx.request_repaint();

                Some(texture)
            } else {
                None
            };

            if let Some(texture) = texture_update {
                state.nsi_state.texture_handle = Some(texture);
            }
        }
    }
}
