use crate::operators::{BaseShape, Operator};
use bevy::prelude::*;
use bevy_egui::egui;
use crossbeam::channel::Sender;
use image::RgbaImage;
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Debug)]
pub enum NsiRenderCommand {
    Start,
    Stop,
    UpdateCamera(Transform),
}

#[derive(Clone, PartialEq)]
pub struct EditableState {
    pub base_shape: BaseShape,
    pub operators: Vec<Operator>,
}

#[derive(Clone)]
pub struct NsiRenderState {
    pub is_rendering: bool,
    pub progress: f32,
    pub image: Option<Arc<Mutex<RgbaImage>>>,
    pub texture_handle: Option<egui::TextureHandle>,
    pub context: Option<Arc<nsi::Context<'static>>>,
    pub tile_updated: Option<Arc<Mutex<bool>>>,
    pub render_finished: Option<Arc<Mutex<bool>>>,
    pub last_camera_transform: Option<Transform>,
    pub render_thread_tx: Option<Sender<NsiRenderCommand>>,
}

#[derive(Resource)]
pub struct PlaygroundState {
    pub base_shape: BaseShape,
    pub operators: Vec<Operator>,
    pub selected_operator: Option<usize>,
    pub needs_rebuild: bool,
    pub prism_sides: u8,
    pub antiprism_sides: u8,
    pub panel_width: f32,
    pub nsi_render: bool,
    pub nsi_state: NsiRenderState,
    pub undoer: egui::util::undoer::Undoer<EditableState>,
}

impl PlaygroundState {
    pub fn save_state(&mut self) {
        let state = EditableState {
            base_shape: self.base_shape,
            operators: self.operators.clone(),
        };
        self.undoer.add_undo(&state);
    }

    pub fn current_state(&self) -> EditableState {
        EditableState {
            base_shape: self.base_shape,
            operators: self.operators.clone(),
        }
    }

    pub fn undo(&mut self) -> bool {
        let current = self.current_state();
        if let Some(state) = self.undoer.undo(&current) {
            self.base_shape = state.base_shape;
            self.operators = state.operators.clone();
            self.selected_operator = None;
            self.needs_rebuild = true;
            true
        } else {
            false
        }
    }

    pub fn redo(&mut self) -> bool {
        let current = self.current_state();
        if let Some(state) = self.undoer.redo(&current) {
            self.base_shape = state.base_shape;
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
            nsi_state: NsiRenderState {
                is_rendering: false,
                progress: 0.0,
                image: None,
                texture_handle: None,
                context: None,
                tile_updated: None,
                render_finished: None,
                last_camera_transform: None,
                render_thread_tx: None,
            },
            undoer: egui::util::undoer::Undoer::default(),
        }
    }
}

#[derive(Component)]
pub struct PolyhedronMesh;
