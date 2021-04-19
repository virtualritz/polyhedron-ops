use kiss3d::{
    camera::{ArcBall, FirstPerson},
    event::{Action, Key, Modifiers, WindowEvent},
    light::Light,
    nalgebra as na,
    resource::Mesh,
    window::Window,
};
use na::{Point3, Vector3};
use rayon::prelude::*;
use std::{cell::RefCell, io, io::Write, rc::Rc};
//extern crate polyhedron_ops;
use polyhedron_ops::prelude::*;

#[cfg(feature = "nsi")]
mod nsi_render;

#[cfg(feature = "nsi")]
use slice_as_array::*;

#[cfg(feature = "nsi")]
use std::path::Path;

#[cfg(feature = "nsi")]
use kiss3d::camera::Camera;

use itertools::Itertools;

#[derive(PartialEq)]
pub enum RenderType {
    Normal,
    Cloud,
    Dump,
}

fn into_mesh(polyhedron: Polyhedron) -> Mesh {
    let (face_index, points, normals) = polyhedron.to_triangle_mesh_buffers();

    Mesh::new(
        // Duplicate points per face so we can
        // match the normals per face.
        points
            .par_iter()
            .map(|p| na::Point3::<f32>::new(p.x, p.y, p.z))
            .collect::<Vec<_>>(),
        face_index
            .iter()
            .tuples::<(_, _, _)>()
            .map(|i| na::Point3::<u16>::new(*i.0 as _, *i.1 as _, *i.2 as _))
            .collect::<Vec<_>>(),
        Some(
            normals
                .par_iter()
                .map(|n| na::Vector3::new(n.x, n.y, n.z))
                .collect::<Vec<_>>(),
        ),
        None,
        false,
    )
}

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

    let mesh = Rc::new(RefCell::new(into_mesh(poly.clone())));
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
    let mut render_quality = 0;

    let mut turntable = false;

    println!(
        "Press one of:\n\
        ____________________________________________ Start Shapes (Reset) ______\n\
        [T]etrahedron              [P]rism ↑↓\n\
        [C]ube (hexahedron)\n\
        [O]ctahedron\n\
        [D]dodecahedron\n\
        [I]cosehedron\n\
        ______________________________________________________ Operations ______\n\
        [a]mbo ↑↓\n\
        [b]evel ↑↓\n\
        [c]chamfer ↑↓\n\
        [d]ual\n\
        [e]xpand ↑↓\n\
        [g]yro ↑↓\n\
        [i]nset ↑↓\n\
        [j]oin ↑↓\n\
        [k]iss ↑↓\n\
        [M]edial ↑↓\n\
        [m]eta ↑↓\n\
        [n]eedle ↑↓\n\
        [o]rtho ↑↓\n\
        [p]propeller ↑↓\n\
        [q]uinto ↑↓\n\
        [r]eflect\n\
        [s]nub ↑↓\n\
        [S]pherize ↑↓\n\
        [t]runcate ↑↓\n\
        Catmull-Clark subdi[v]ide\n\
        [w]hirl ↑↓\n\
        e[x]trude ↑↓\n\
        [z]ip ↑↓\n\
        _______________________________________________________ Modifiers ______\n\
        (Shift)+⬆⬇︎  – modify the last operation marked with ↑↓ (10× w. [Shift])\n\
        [Delete]    – Undo last operation\n\
        _______________________________________________________ Exporting ______"
    );
    #[cfg(feature = "nsi")]
    print!("([Shift])+");
    print!("[Space] – save as OBJ");
    #[cfg(feature = "nsi")]
    print!(
        " (dump to NSI w. [Shift])\n\
        _______________________________________________________ Rendering ______\n\
        (Shift)+[Enter]   – Render (in the cloud w. [Shift])\n\
        [Shift]+[0]..[9]  – Set render quality: [preview]..[super high quality]\n\
        [Ctrl]+[T]        – Toggle turntable rendering (72 frames)"
    );
    print!(
        "\n________________________________________________________________________\n\
        ❯ {} – render quality {} {:<80}\r",
        poly.name(),
        render_quality,
        if turntable { "(turntable)" } else { "" },
    );
    io::stdout().flush().unwrap();

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
                        Key::Key0 => render_quality = 0,
                        Key::Key1 => {
                            if modifiers.intersects(Modifiers::Shift) {
                                render_quality = 1;
                            }
                        }
                        Key::Key2 => {
                            if modifiers.intersects(Modifiers::Shift) {
                                render_quality = 2;
                            }
                        }
                        Key::Key3 => {
                            if modifiers.intersects(Modifiers::Shift) {
                                render_quality = 3;
                            }
                        }
                        Key::Key4 => {
                            if modifiers.intersects(Modifiers::Shift) {
                                render_quality = 4;
                            }
                        }
                        Key::Key5 => {
                            if modifiers.intersects(Modifiers::Shift) {
                                render_quality = 5;
                            }
                        }
                        Key::Key6 => {
                            if modifiers.intersects(Modifiers::Shift) {
                                render_quality = 6;
                            }
                        }
                        Key::Key7 => {
                            if modifiers.intersects(Modifiers::Shift) {
                                render_quality = 7;
                            }
                        }
                        Key::Key8 => {
                            if modifiers.intersects(Modifiers::Shift) {
                                render_quality = 8;
                            }
                        }
                        Key::Key9 => {
                            if modifiers.intersects(Modifiers::Shift) {
                                render_quality = 9;
                            }
                        }
                        Key::A => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            last_op_value = 0.5;
                            poly.ambo(None, true);
                            poly.normalize();
                            last_op = 'a';
                        }
                        Key::B => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            last_op_value = 0.;
                            poly.bevel(None, None, None, None, true);
                            poly.normalize();
                            last_op = 'b';
                        }
                        Key::C => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            if modifiers.intersects(Modifiers::Shift) {
                                poly = Polyhedron::hexahedron();
                                poly.normalize();
                            } else {
                                last_op_value = 0.5;
                                poly.chamfer(None, true);
                                poly.normalize();
                                last_op = 'c';
                            }
                        }
                        Key::D => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            if modifiers.intersects(Modifiers::Shift) {
                                poly = Polyhedron::dodecahedron();
                                poly.normalize();
                            } else {
                                last_op_value = 0.;
                                poly.dual(true);
                                poly.normalize();
                            }
                            last_op = '_';
                        }
                        Key::E => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            last_op_value = 0.5;
                            poly.expand(None, true);
                            poly.normalize();
                            last_op = 'e';
                        }
                        Key::G => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            last_op_value = 0.;
                            poly.gyro(None, None, true);
                            poly.normalize();
                            last_op = 'g';
                        }
                        Key::I => {
                            if modifiers.intersects(Modifiers::Shift) {
                                alter_last_op = false;
                                last_poly = poly.clone();
                                poly = Polyhedron::icosahedron();
                                poly.normalize();
                            } else {
                                alter_last_op = false;
                                last_poly = poly.clone();
                                last_op_value = 0.3;
                                poly.inset(None, None, true);
                                poly.normalize();
                                last_op = 'i';
                            }
                        }
                        Key::J => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            last_op_value = 0.5;
                            poly.join(None, true);
                            poly.normalize();
                            last_op = 'j';
                        }
                        Key::K => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            last_op_value = 0.;
                            poly.kis(None, None, None, true);
                            poly.normalize();
                            last_op = 'k';
                        }
                        Key::I => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            last_op_value = 0.3;
                            poly.inset(None, None, true);
                            poly.normalize();
                            last_op = 'l';
                        }
                        Key::M => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            last_op_value = 0.;
                            if modifiers.intersects(Modifiers::Shift) {
                                poly.medial(None, None, None, None, true);
                                last_op = 'M';
                            } else {
                                poly.meta(None, None, None, None, true);
                                last_op = 'm';
                            }
                            poly.normalize();
                        }
                        Key::N => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            last_op_value = 0.;
                            poly.needle(None, None, None, true);
                            poly.normalize();
                            last_op = 'n';
                        }
                        Key::O => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            if modifiers.intersects(Modifiers::Shift) {
                                poly = Polyhedron::octahedron();
                                poly.normalize();
                            } else {
                                last_op_value = 0.5;
                                poly.ortho(None, true);
                                poly.normalize();
                                last_op = 'o';
                            }
                        }
                        Key::P => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            if modifiers.intersects(Modifiers::Shift) {
                                last_op_value = 0.03;
                                poly = Polyhedron::prism(3);
                                poly.normalize();
                                last_op = 'P';
                            } else {
                                last_op_value = 1. / 3.;
                                poly.propeller(None, true);
                                poly.normalize();
                                last_op = 'p';
                            }
                        }
                        Key::Q => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            last_op_value = 0.5;
                            poly.quinto(None, true);
                            poly.normalize();
                            last_op = 'q';
                        }
                        Key::R => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            poly.reflect(true);
                            poly.normalize();
                            last_op = '_';
                        }
                        Key::S => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            if modifiers.intersects(Modifiers::Shift) {
                                last_op_value = 1.0;
                                poly.spherize(None, true);
                                last_op = 'S';
                            } else {
                                last_op_value = 0.;
                                poly.snub(None, None, true);
                                poly.normalize();
                                last_op = 's';
                            }
                        }
                        Key::T => {
                            if modifiers.intersects(Modifiers::Shift) {
                                alter_last_op = false;
                                last_poly = poly.clone();
                                poly = Polyhedron::tetrahedron();
                                poly.normalize();
                            } else if modifiers.intersects(Modifiers::Control) {
                                turntable = !turntable;
                            } else {
                                alter_last_op = false;
                                last_poly = poly.clone();
                                last_op_value = 0.;
                                poly.truncate(None, None, None, true);
                                poly.normalize();
                                last_op = 't';
                            }
                        }
                        Key::V => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            last_op_value = 0.;
                            poly.catmull_clark_subdivide(true);
                            poly.normalize();
                            last_op = 'v';
                        }
                        Key::W => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            last_op_value = 0.;
                            poly.whirl(None, None, true);
                            poly.normalize();
                            last_op = 'w';
                        }
                        Key::X => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            last_op_value = 0.3;
                            poly.extrude(None, None, None, true);
                            poly.normalize();
                            last_op = 'j';
                        }
                        Key::Z => {
                            alter_last_op = false;
                            last_poly = poly.clone();
                            last_op_value = 0.;
                            poly.zip(None, None, None, true);
                            poly.normalize();
                            last_op = 'z';
                        }
                        Key::Space => {
                            if modifiers.intersects(Modifiers::Shift) {
                                #[cfg(feature = "nsi")]
                                {
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
                                            slice_as_array!(xform.as_slice(), [f64; 16]).unwrap(),
                                            render_quality,
                                            RenderType::Dump,
                                            turntable,
                                        )
                                    );
                                }
                            } else {
                                println!(
                                    "Exported to {}",
                                    poly.write_to_obj(&path, true).unwrap().display()
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
                        #[cfg(feature = "nsi")]
                        Key::Return => {
                            let xform = arc_ball
                                .inverse_transformation()
                                .iter()
                                .map(|e| *e as f64)
                                .collect::<Vec<_>>();

                            nsi_render::nsi_render(
                                Path::new(""),
                                &poly,
                                slice_as_array!(xform.as_slice(), [f64; 16]).unwrap(),
                                render_quality,
                                if modifiers.intersects(Modifiers::Shift) {
                                    RenderType::Cloud
                                } else {
                                    RenderType::Normal
                                },
                                turntable,
                            );
                        }
                        _ => {
                            break;
                        }
                    };
                    if alter_last_op {
                        alter_last_op = false;
                        if '_' != last_op {
                            poly = last_poly.clone();
                        }
                        match last_op {
                            'a' => {
                                poly.ambo(Some(last_op_value), true);
                            }
                            'b' => {
                                poly.bevel(
                                    Some(last_op_value),
                                    Some(last_op_value),
                                    None,
                                    None,
                                    true,
                                );
                            }
                            'c' => {
                                poly.chamfer(Some(last_op_value), true);
                            }
                            'e' => {
                                poly.expand(Some(last_op_value), true);
                            }
                            'g' => {
                                poly.gyro(None, Some(last_op_value), true);
                            }
                            'i' => {
                                poly.inset(Some(last_op_value), None, true);
                            }
                            'j' => {
                                poly.join(Some(last_op_value), true);
                            }
                            'k' => {
                                poly.kis(Some(last_op_value), None, None, true);
                            }
                            'm' => {
                                poly.meta(
                                    Some(last_op_value),
                                    Some(last_op_value),
                                    None,
                                    None,
                                    true,
                                );
                            }
                            'o' => {
                                poly.ortho(Some(last_op_value), true);
                            }
                            'p' => {
                                poly.propeller(Some(last_op_value), true);
                            }
                            'P' => {
                                poly = Polyhedron::prism((last_op_value * 100.) as _);
                                poly.normalize();
                            }
                            'q' => {
                                poly.quinto(Some(last_op_value), true);
                            }
                            'M' => {
                                poly.medial(
                                    Some(last_op_value),
                                    Some(last_op_value),
                                    None,
                                    None,
                                    true,
                                );
                            }
                            'n' => {
                                poly.needle(Some(last_op_value), None, None, true);
                            }
                            's' => {
                                poly.snub(None, Some(last_op_value), true);
                            }
                            'S' => {
                                poly.spherize(Some(last_op_value), true);
                            }
                            't' => {
                                poly.truncate(Some(last_op_value), None, None, true);
                            }
                            'w' => {
                                poly.whirl(None, Some(last_op_value), true);
                            }
                            'x' => {
                                poly.extrude(Some(last_op_value), None, None, true);
                            }
                            'z' => {
                                poly.zip(Some(last_op_value), None, None, true);
                            }

                            _ => (),
                        }
                        if '_' != last_op {
                            poly.normalize();
                        }
                    }
                    c.unlink();
                    let mesh = Rc::new(RefCell::new(into_mesh(poly.clone())));
                    c = window.add_mesh(mesh, Vector3::new(1.0, 1.0, 1.0));
                    c.set_color(0.9, 0.8, 0.7);
                    c.enable_backface_culling(false);
                    c.set_points_size(10.);

                    print!(
                        "❯ {} – render quality {} {:<80}\r",
                        poly.name(),
                        render_quality,
                        if turntable { "(turntable)" } else { "" },
                    );
                    io::stdout().flush().unwrap();
                }
                _ => {}
            }
        }

        window.draw_line(
            &Point3::origin(),
            &Point3::new(1.0, 0.0, 0.0),
            &Point3::new(1.0, 0.0, 0.0),
        );
        window.draw_line(
            &Point3::origin(),
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(0.0, 1.0, 0.0),
        );
        window.draw_line(
            &Point3::origin(),
            &Point3::new(0.0, 0.0, 1.0),
            &Point3::new(0.0, 0.0, 1.0),
        );

        if use_arc_ball {
            window.render_with_camera(&mut arc_ball);
        } else {
            window.render_with_camera(&mut first_person);
        }
    }
}
