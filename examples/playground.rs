//use cgmath::prelude::*;
//use itertools::Itertools;

#[macro_use]
extern crate slice_as_array;

pub enum RenderType {
    Normal,
    Cloud,
    Dump,
}

mod nsi_render;

extern crate nalgebra;
use nalgebra as na;

extern crate kiss3d;
use kiss3d::{
    camera::{ArcBall, Camera, FirstPerson},
    event::{Action, Key, Modifiers, WindowEvent},
    light::Light,
    resource::Mesh,
    window::Window,
};

use na::{Point3, UnitQuaternion, Vector3};
use std::{cell::RefCell, io, io::Write, rc::Rc, path::Path};

extern crate polyhedron_ops;
use polyhedron_ops::prelude::*;

/// Struct storing indices corresponding to the vertex
/// Some points may not have texcoords or normals, 0 is used to
/// indicate this as OBJ indices begin at 1
/*#[derive(Hash, Eq, PartialEq, PartialOrd, Ord, Debug, Copy, Clone)]
struct VertexIndex {
    pub position: Index,
    pub texture: Index,
    pub normal: Index,
}*/

fn main() {
    let distance = 2.0f32;
    let eye = Point3::new(distance, distance, distance);
    let at = Point3::origin();
    let mut first_person = FirstPerson::new(eye, at);
    let mut arc_ball = ArcBall::new(eye, at);
    let mut use_arc_ball = true;

    let mut window = Window::new("Polyhedron Operations");
    window.set_light(Light::StickToCamera);

    let mut poly = Polyhedron::tetrahedron();
    poly.normalize();

    let mesh = Rc::new(RefCell::new(Mesh::from(poly.clone())));
    let mut c = window.add_mesh(mesh, Vector3::new(1.0, 1.0, 1.0));

    c.set_color(0.9, 0.8, 0.7);
    c.enable_backface_culling(false);
    c.set_points_size(10.);

    window.set_light(Light::StickToCamera);
    window.set_framerate_limit(Some(60));

    let mut last_op = 'n';
    let mut last_op_value = 0.;
    let mut alter_last_op = false;
    let mut last_poly = poly.clone();

    let path = dirs::home_dir().unwrap();

    println!(
        "Press one of\n\
        [A]mbo\n\
        [B]evel\n\
        [C]chamfer\n\
        [D]ual\n\
        [E]xpand\n\
        [G]yro↑↓\n\
        [J]oin\n\
        [K]iss↑↓\n\
        [M]eta\n\
        [N]eedle\n\
        [O]rtho\n\
        [P]propellor\n\
        [Q]uinto\n\
        [R]eflect\n\
        [S]nub\n\
        [T]runcate\n\
        (Shift)+⬆/⬇︎ – modify the last ↑↓ operation\n\
        (Shift)+[Enter] – Render (in the cloud w. [Shift])\n\
        (Shift)+[Space] – save as OBJ (dump to NSI w. [Shift])"
    );

    while !window.should_close() {
        // rotate the arc-ball camera.
        let curr_yaw = arc_ball.yaw();
        arc_ball.set_yaw(curr_yaw + 0.01);

        // update the current camera.
        for event in window.events().iter() {
            match event.value {
                WindowEvent::Key(key, Action::Release, modifiers) => {
                    match key {
                        Key::Numpad1 => use_arc_ball = true,
                        Key::Numpad2 => use_arc_ball = false,
                        Key::A => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            poly.ambo(true);
                            poly.normalize();
                            last_op = '_';
                        }
                        Key::C => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            poly.chamfer(1. / 2., true);
                            poly.normalize();
                            last_op = '_';
                        }
                        Key::B => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            poly.bevel(true);
                            poly.normalize();
                            last_op = '_';
                        }
                        Key::D => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            poly.dual(true);
                            poly.normalize();
                            last_op = '_';
                        }
                        Key::E => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            poly.expand(true);
                            poly.normalize();
                            last_op = '_';
                        }
                        Key::G => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            last_op_value = 0.;
                            poly.gyro(1. / 3., last_op_value, true);
                            poly.normalize();
                            last_op = 'g';
                        }
                        Key::J => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            poly.join(true);
                            poly.normalize();
                            last_op = '_';
                        }
                        Key::K => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            last_op_value = 0.;
                            poly.kis(last_op_value, None, false, true);
                            poly.normalize();
                            last_op = 'k';
                        }
                        Key::M => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            poly.meta(true);
                            poly.normalize();
                            last_op = '_';
                        }
                        Key::N => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            poly.needle(true);
                            poly.normalize();
                            last_op = '_';
                        }
                        Key::O => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            poly.ortho(true);
                            poly.normalize();
                            last_op = '_';
                        }
                        Key::P => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            poly.propellor(1. / 3., true);
                            poly.normalize();
                            last_op = '_';
                        }
                        Key::Q => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            poly.quinto(true);
                            poly.normalize();
                            last_op = '_';
                        }
                        Key::R => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            poly.reflect(true);
                            poly.normalize();
                            last_op = '_';
                        }
                        /*Key::T => {
                            if Super == modifiers {
                                return;
                            }
                        }*/
                        Key::S => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            poly.snub(true);
                            poly.normalize();
                            last_op = '_';
                        }

                        Key::T => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            poly.truncate(None, true);
                            poly.normalize();
                            last_op = '_';
                        }
                        Key::Space => {
                            if modifiers.intersects(Modifiers::Shift) {
                                let xform = arc_ball
                                    .inverse_transformation()
                                    .iter()
                                    .map(|e| *e as f64)
                                    .collect::<Vec<_>>();

                                println!(
                                    "Dumped to {}",
                                    nsi_render::nsi_render(
                                        &path,
                                        &poly,
                                        slice_as_array!(
                                            xform.as_slice(),
                                            [f64; 16]
                                        )
                                        .unwrap(),
                                        RenderType::Dump,
                                    )
                                );
                            } else {
                                println!(
                                    "Exported to {}",
                                    poly.export_as_obj(&path, true)
                                        .unwrap()
                                        .display()
                                );
                            }
                        }
                        Key::Up => {
                            alter_last_op = true;
                            if modifiers.intersects(Modifiers::Shift) {
                                last_op_value += 0.1;
                            } else {
                                last_op_value += 0.01;
                            }
                        }
                        Key::Down => {
                            alter_last_op = true;
                            if modifiers.intersects(Modifiers::Shift) {
                                last_op_value -= 0.1;
                            } else {
                                last_op_value -= 0.01;
                            }
                        }
                        Key::Delete => {
                            poly = last_poly.clone();
                        }
                        Key::Return => {
                            let xform = arc_ball
                                .inverse_transformation()
                                .iter()
                                .map(|e| *e as f64)
                                .collect::<Vec<_>>();

                            nsi_render::nsi_render(
                                Path::new(""),
                                &poly,
                                slice_as_array!(xform.as_slice(), [f64; 16])
                                    .unwrap(),
                                if modifiers.intersects(Modifiers::Shift) {
                                    RenderType::Cloud
                                } else {
                                    RenderType::Normal
                                },
                            );
                        }
                        _ => {
                            break;
                        }
                    };
                    if alter_last_op {
                        alter_last_op = false;
                        match last_op {
                            'g' => {
                                poly = last_poly.clone();
                                poly.gyro(1. / 3., last_op_value, true);
                                poly.normalize();
                            }
                            'k' => {
                                poly = last_poly.clone();
                                poly.kis(last_op_value, None, false, true);
                                poly.normalize();
                            }
                            _ => (),
                        }
                    }
                    c.unlink();
                    let mesh = Rc::new(RefCell::new(Mesh::from(poly.clone())));
                    c = window.add_mesh(mesh, Vector3::new(1.0, 1.0, 1.0));
                    c.set_color(0.9, 0.8, 0.7);
                    c.enable_backface_culling(false);
                    c.set_points_size(10.);

                    print!("{}\r", poly.name());
                    io::stdout().flush().unwrap();
                }
                _ => {}
            }
        }

        /*
        window.draw_line(
            &Point3::origin(),
            &Point3::new(10.0, 0.0, 0.0),
            &Point3::new(10.0, 0.0, 0.0),
        );
        window.draw_line(
            &Point3::origin(),
            &Point3::new(0.0, 10.0, 0.0),
            &Point3::new(0.0, 10.0, 0.0),
        );
        window.draw_line(
            &Point3::origin(),
            &Point3::new(0.0, 0.0, 10.0),
            &Point3::new(0.0, 0.0, 10.0),
        );*/

        if use_arc_ball {
            window.render_with_camera(&mut arc_ball);
        } else {
            window.render_with_camera(&mut first_person);
        }
    }
}
