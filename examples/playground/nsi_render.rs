pub use crate::*;
use nsi::*;
use nsi_core as nsi;
use std::f64::consts::TAU;
use ultraviolet as uv;

const FPS: u32 = 60;
const TURNTABLE_SECONDS: u32 = 1;
const FRAME_STEP: f64 =
    360.0 / TURNTABLE_SECONDS as f64 / FPS as f64 * TAU / 90.0;

/// Returns the name of the `screen` node that was created.
fn nsi_globals_and_camera(
    c: &Context,
    name: &str,
    _camera_xform: &[f64; 16],
    render_quality: u32,
    turntable: bool,
) {
    // Setup a camera transform.
    c.create("camera_xform", TRANSFORM, None);
    c.connect("camera_xform", None, ROOT, "objects", None);

    c.set_attribute(
        "camera_xform",
        &[double_matrix!(
            "transformationmatrix",
            //camera_xform
            &[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 5., 1.,]
        )],
    );

    // Setup a camera.
    c.create("camera", PERSPECTIVE_CAMERA, None);

    c.set_attribute(
        "camera",
        &[
            float!("fov", 35.),
            doubles!("shutterrange", &[0.0, 1.0]), //.array_len(2),
            doubles!("shutteropening", &[0.5, 0.5]), //.array_len(2)
        ],
    );
    c.connect("camera", None, "camera_xform", "objects", None);

    // Setup a screen.
    c.create("screen", SCREEN, None);
    c.connect("screen", None, "camera", "screens", None);

    c.set_attribute(
        "screen",
        &[
            integers!("resolution", &[512, 512]).array_len(2),
            integer!(
                "oversampling",
                if turntable {
                    1 << (3 + render_quality)
                } else {
                    32
                }
            ),
        ],
    );

    c.set_attribute(
        ".global",
        &[
            integer!("renderatlowpriority", 1),
            string!("bucketorder", "circle"),
            integer!("quality.shadingsamples", 1 << (3 + render_quality)),
            integer!("maximumraydepth.reflection", 6),
        ],
    );

    c.create("albedo", OUTPUT_LAYER, None);
    c.set_attribute(
        "albedo",
        &[
            string!("variablename", "albedo"),
            string!("variablesource", "shader"),
            string!("layertype", "color"),
            string!("scalarformat", "float"),
            string!("filter", "box"),
            double!("filterwidth", 1.),
        ],
    );
    c.connect("albedo", None, "screen", "outputlayers", None);

    // Normal layer.
    c.create("normal", OUTPUT_LAYER, None);
    c.set_attribute(
        "normal",
        &[
            string!("variablename", "N.world"),
            string!("variablesource", "builtin"),
            string!("layertype", "vector"),
            string!("scalarformat", "float"),
            string!("filter", "box"),
            double!("filterwidth", 1.),
        ],
    );
    c.connect("normal", None, "screen", "outputlayers", None);

    // Setup an output layer.
    c.create(name, OUTPUT_LAYER, None);
    c.set_attribute(
        name,
        &[
            string!("variablename", "Ci"),
            integer!("withalpha", 1),
            string!("scalarformat", "float"),
            double!("filterwidth", 1.),
        ],
    );
    c.connect(name, None, "screen", "outputlayers", None);

    // Setup an output driver.
    c.create("driver", OUTPUT_DRIVER, None);
    c.connect("driver", None, name, "outputdrivers", None);
    c.set_attribute(
        "driver",
        &[
            string!("drivername", "idisplay"),
            string!("imagefilename", name.to_string() + ".exr"),
            //string!("filename", name.to_string() + ".exr"),
        ],
    );

    /*
    c.create("driver2", OUTPUT_DRIVER, None);
    c.connect("driver2", None, name, "outputdrivers", None);
    c.connect("driver2", None, "albedo", "outputdrivers", None);
    c.connect("driver2", None, "normal", "outputdrivers", None);
    c.set_attribute(
        "driver2",
        &[
            string!("drivername", "r-display"),
            string!("imagefilename", name),
            float!("denoise", 1.),
        ],
    );*/
}

fn nsi_environment(c: &Context) {
    // Set up an environment light.
    c.create("env_xform", TRANSFORM, None);
    c.connect("env_xform", None, ROOT, "objects", None);

    c.create("environment", ENVIRONMENT, None);
    c.connect("environment", None, "env_xform", "objects", None);

    c.create("env_attrib", ATTRIBUTES, None);
    c.connect(
        "env_attrib",
        None,
        "environment",
        "geometryattributes",
        None,
    );

    c.set_attribute("env_attrib", &[integer!("visibility.camera", 0)]);

    c.create("env_shader", SHADER, None);
    c.connect("env_shader", None, "env_attrib", "surfaceshader", None);

    // Environment light attributes.
    c.set_attribute(
        "env_shader",
        &[
            string!("shaderfilename", "${DELIGHT}/osl/environmentLight"),
            float!("intensity", 1.),
        ],
    );

    c.set_attribute(
        "env_shader",
        &[string!("image", "assets/wooden_lounge_1k.tdl")],
    );
}

fn nsi_material(c: &Context, name: &str) {
    // Particle attributes.
    let attribute_name = format!("{}_attrib", name);
    c.create(&attribute_name, ATTRIBUTES, None);
    c.connect(&attribute_name, None, name, "geometryattributes", None);

    // Particle shader.
    let shader_name = format!("{}_shader", name);
    c.create(&shader_name, SHADER, None);
    c.connect(&shader_name, None, &attribute_name, "surfaceshader", None);

    /*
    c.set_attribute(
        &shader_name,
        &[
            string!(
                "shaderfilename",
                PathBuf::from(path)
                    .join("osl")
                    .join("dlPrincipled")
                    .to_string_lossy()
                    .into_owned()
            ),
            color!("i_color", &[1.0f32, 0.6, 0.3]),
            //arg!("coating_thickness", &0.1f32),
            float!("roughness", 0.3f32),
            float!("specular_level", 0.5f32),
            float!("metallic", 1.0f32),
            float!("anisotropy", 0.9f32),
            float!("thin_film_thickness", 0.6),
            float!("thin_film_ior", 3.0),
            //color!("incandescence", &[0.0f32, 0.0, 0.0]),
        ],
    );


    c.set_attribute(
        &shader_name,
        &[
            string!(
                "shaderfilename",
                PathBuf::from(path)
                    .join("osl")
                    .join("dlGlass")
                    .to_string_lossy()
                    .into_owned()
            ),
            float!("refract_roughness", 0.666f32),
        ],
    );*/

    c.set_attribute(
        &shader_name,
        &[
            string!("shaderfilename", "${DELIGHT}/osl/dlMetal"),
            color!("i_color", &[1.0f32, 0.6, 0.3]),
            float!("roughness", 0.3),
            //float!("anisotropy", 0.9f32),
            float!("thin_film_thickness", 0.6),
            float!("thin_film_ior", 3.0),
        ],
    );
}

pub fn nsi_render(
    path: &Path,
    polyhedron: &crate::Polyhedron,
    camera_xform: &[f64; 16],
    render_quality: u32,
    render_type: crate::RenderType,
    turntable: bool,
) -> std::string::String {
    let destination =
        path.join(format!("polyhedron-{}.nsi", polyhedron.name()));

    let ctx = {
        match render_type {
            RenderType::Normal => Context::new(None),
            RenderType::Cloud => Context::new(Some(&[integer!("cloud", 1)])),
            RenderType::Dump => Context::new(Some(&[
                string!("type", "apistream"),
                string!("streamfilename", destination.to_str().unwrap()),
            ])),
        }
    }
    .unwrap();

    nsi_globals_and_camera(
        &ctx,
        polyhedron.name(),
        camera_xform,
        render_quality,
        turntable,
    );

    nsi_environment(&ctx);

    let name = polyhedron.to_nsi(
        &ctx,
        Some(&(polyhedron.name().to_string() + "-mesh")),
        None,
        None,
        None,
    );

    nsi_material(&ctx, &name);

    /*
    ctx.append(
        ROOT,
        None,
        ctx.append(
            &ctx.rotation(Some("mesh-rotation"), (frame * 5) as f64, &[0., 1., 0.]),
            None,
            &name,
        )
        .0,
    );*/

    if turntable {
        ctx.create("rotation", TRANSFORM, None);
        ctx.connect("rotation", None, ROOT, "objects", None);
        ctx.connect(&name, None, "rotation", "objects", None);

        for frame in 0..TURNTABLE_SECONDS * FPS {
            ctx.set_attribute(
                "driver",
                &[string!("filename", format!("{}_{:02}.exr", name, frame))],
            );

            ctx.set_attribute_at_time(
                "rotation",
                0.0,
                &[double_matrix!(
                    "transformationmatrix",
                    uv::DMat4::from_angle_plane(
                        (frame as f64 * FRAME_STEP) as _,
                        uv::DBivec3::from_normalized_axis(uv::DVec3::new(
                            0., 1., 0.
                        ))
                    )
                    .transposed()
                    .as_array()
                )],
            );

            ctx.set_attribute_at_time(
                "rotation",
                1.0,
                &[double_matrix!(
                    "transformationmatrix",
                    uv::DMat4::from_angle_plane(
                        ((frame + 1) as f64 * FRAME_STEP) as _,
                        uv::DBivec3::from_normalized_axis(uv::DVec3::new(
                            0., 1., 0.
                        ))
                    )
                    .transposed()
                    .as_array()
                )],
            );

            ctx.render_control(nsi::Action::Synchronize, None);
            ctx.render_control(nsi::Action::Start, None);
            ctx.render_control(nsi::Action::Wait, None);
        }
    } else {
        ctx.connect(&name, None, ROOT, "objects", None);

        //if RenderType::Dump != render_type {
        ctx.render_control(nsi::Action::Start, None);
        //}
    }

    //if RenderType::Dump != render_type {
    ctx.render_control(nsi::Action::Wait, None);
    //}

    destination.to_string_lossy().to_string()
}
