pub use crate::*;
use std::{
    env,
    path::{Path, PathBuf},
};
use ultraviolet as uv;

fn nsi_globals_and_camera(
    c: &nsi::Context,
    name: &str,
    _camera_xform: &[f64; 16],
    render_quality: u32,
) {
    // Setup a camera transform.
    c.create("camera_xform", nsi::NodeType::Transform, None);
    c.connect("camera_xform", "", ".root", "objects", None);

    c.set_attribute(
        "camera_xform",
        &[nsi::double_matrix!(
            "transformationmatrix",
            //camera_xform
            &[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 5., 1.,]
        )],
    );

    // Setup a camera.
    c.create("camera", nsi::NodeType::PerspectiveCamera, None);

    c.set_attribute("camera", &[nsi::float!("fov", 35.)]);
    c.connect("camera", "", "camera_xform", "objects", None);

    // Setup a screen.
    c.create("screen", nsi::NodeType::Screen, None);
    c.connect("screen", "", "camera", "screens", None);
    c.set_attribute(
        "screen",
        &[
            nsi::integers!("resolution", &[2048, 2048]).array_len(2),
            nsi::integer!("oversampling", 32),
        ],
    );

    c.set_attribute(
        ".global",
        &[
            nsi::integer!("renderatlowpriority", 1),
            nsi::string!("bucketorder", "circle"),
            nsi::integer!("quality.shadingsamples", 1 << (3 + render_quality)),
            nsi::integer!("maximumraydepth.reflection", 6),
        ],
    );

    c.create("albedo", nsi::NodeType::OutputLayer, None);
    c.set_attribute(
        "albedo",
        &[
            nsi::string!("variablename", "albedo"),
            nsi::string!("variablesource", "shader"),
            nsi::string!("layertype", "color"),
            nsi::string!("scalarformat", "float"),
            nsi::string!("filter", "box"),
            nsi::double!("filterwidth", 1.),
        ],
    );
    c.connect("albedo", "", "screen", "outputlayers", None);

    // Normal layer.
    c.create("normal", nsi::NodeType::OutputLayer, None);
    c.set_attribute(
        "normal",
        &[
            nsi::string!("variablename", "N.world"),
            nsi::string!("variablesource", "builtin"),
            nsi::string!("layertype", "vector"),
            nsi::string!("scalarformat", "float"),
            nsi::string!("filter", "box"),
            nsi::double!("filterwidth", 1.),
        ],
    );
    c.connect("normal", "", "screen", "outputlayers", None);

    // Setup an output layer.
    c.create(name, nsi::NodeType::OutputLayer, None);
    c.set_attribute(
        name,
        &[
            nsi::string!("variablename", "Ci"),
            nsi::integer!("withalpha", 1),
            nsi::string!("scalarformat", "float"),
            nsi::double!("filterwidth", 1.),
        ],
    );
    c.connect(name, "", "screen", "outputlayers", None);

    // Setup an output driver.
    c.create("driver", nsi::NodeType::OutputDriver, None);
    c.connect("driver", "", name, "outputdrivers", None);
    c.set_attribute(
        "driver",
        &[
            nsi::string!("drivername", "idisplay"),
            nsi::string!("filename", name.to_string() + ".exr"),
        ],
    );

    /*
    c.create("driver2", nsi::NodeType::OutputDriver, None);
    c.connect("driver2", "", name, "outputdrivers", None);
    c.connect("driver2", "", "albedo", "outputdrivers", None);
    c.connect("driver2", "", "normal", "outputdrivers", None);
    c.set_attribute(
        "driver2",
        &[
            nsi::string!("drivername", "r-display"),
            nsi::string!("imagefilename", name),
            nsi::float!("denoise", 1.),
        ],
    );*/
}

fn nsi_environment(c: &nsi::Context) {
    if let Ok(path) = &env::var("DELIGHT") {
        // Set up an environment light.
        c.create("env_xform", nsi::NodeType::Transform, None);
        c.connect("env_xform", "", ".root", "objects", None);

        c.create("environment", nsi::NodeType::Environment, None);
        c.connect("environment", "", "env_xform", "objects", None);

        c.create("env_attrib", nsi::NodeType::Attributes, None);
        c.connect("env_attrib", "", "environment", "geometryattributes", None);

        c.set_attribute("env_attrib", &[nsi::integer!("visibility.camera", 0)]);

        c.create("env_shader", nsi::NodeType::Shader, None);
        c.connect("env_shader", "", "env_attrib", "surfaceshader", None);

        // Environment light attributes.
        c.set_attribute(
            "env_shader",
            &[
                nsi::string!(
                    "shaderfilename",
                    PathBuf::from(path)
                        .join("osl")
                        .join("environmentLight")
                        .to_string_lossy()
                        .into_owned()
                ),
                nsi::float!("intensity", 1.),
            ],
        );

        c.set_attribute(
            "env_shader",
            &[nsi::string!("image", "assets/wooden_lounge_1k.tdl")],
        );
    }
}

fn nsi_material(c: &nsi::Context, name: &str) {
    if let Ok(path) = &env::var("DELIGHT") {
        // Particle attributes.
        let attribute_name = format!("{}_attrib", name);
        c.create(attribute_name.clone(), nsi::NodeType::Attributes, None);
        c.connect(attribute_name.clone(), "", name, "geometryattributes", None);

        // Particle shader.
        let shader_name = format!("{}_shader", name);
        c.create(shader_name.clone(), nsi::NodeType::Shader, None);
        c.connect(
            shader_name.clone(),
            "",
            attribute_name,
            "surfaceshader",
            None,
        );

        /*
        c.set_attribute(
            shader_name,
            &[
                nsi::string!(
                    "shaderfilename",
                    PathBuf::from(path)
                        .join("osl")
                        .join("dlPrincipled")
                        .to_string_lossy()
                        .into_owned()
                ),
                nsi::color!("i_color", &[1.0f32, 0.6, 0.3]),
                //nsi::arg!("coating_thickness", &0.1f32),
                nsi::float!("roughness", 0.3f32),
                nsi::float!("specular_level", 0.5f32),
                nsi::float!("metallic", 1.0f32),
                nsi::float!("anisotropy", 0.9f32),
                nsi::float!("thin_film_thickness", 0.6),
                nsi::float!("thin_film_ior", 3.0),
                //nsi::color!("incandescence", &[0.0f32, 0.0, 0.0]),
            ],
        );


        c.set_attribute(
            shader_name,
            &[
                nsi::string!(
                    "shaderfilename",
                    PathBuf::from(path)
                        .join("osl")
                        .join("dlGlass")
                        .to_string_lossy()
                        .into_owned()
                ),
                nsi::float!("refract_roughness", 0.666f32),
            ],
        );*/

        c.set_attribute(
            shader_name,
            &[
                nsi::string!(
                    "shaderfilename",
                    PathBuf::from(path)
                        .join("osl")
                        .join("dlMetal")
                        .to_string_lossy()
                        .into_owned()
                ),
                nsi::color!("i_color", &[1.0f32, 0.6, 0.3]),
                nsi::float!("roughness", 0.3),
                //nsi::float!("anisotropy", 0.9f32),
                nsi::float!("thin_film_thickness", 0.6),
                nsi::float!("thin_film_ior", 3.0),
            ],
        );
    }
}

pub fn nsi_render(
    path: &Path,
    polyhedron: &crate::Polyhedron,
    camera_xform: &[f64; 16],
    render_quality: u32,
    render_type: crate::RenderType,
    turntable: bool,
) -> String {
    let destination =
        path.join(format!("polyhedron-{}.nsi", polyhedron.name()));

    let ctx = {
        match render_type {
            RenderType::Normal => nsi::Context::new(None),
            RenderType::Cloud => {
                nsi::Context::new(Some(&[nsi::integer!("cloud", 1)]))
            }
            RenderType::Dump => nsi::Context::new(Some(&[
                nsi::string!("type", "apistream"),
                nsi::string!("streamfilename", destination.to_str().unwrap()),
            ])),
        }
    }
    .unwrap();

    nsi_globals_and_camera(
        &ctx,
        polyhedron.name(),
        camera_xform,
        render_quality,
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
        ".root",
        None,
        ctx.append(
            &ctx.rotation(Some("mesh-rotation"), (frame * 5) as f64, &[0., 1., 0.]),
            None,
            &name,
        )
        .0,
    );

    ctx.append(
        &polyhedron.name(),
        Some("outputdrivers"),
        &ctx.node(
            Some("exr-driver"),
            nsi::NodeType::OutputDriver,
            &[
                nsi::string!("drivername", "r-display"),
                nsi::string!("imagefilename", format!("{}-{:02}.exr", name, frame)),
            ],
        ),
    );*/

    if turntable {
        ctx.create("rotation", nsi::NodeType::Transform, None);
        ctx.connect("rotation", "", ".root", "objects", None);
        ctx.connect(name, "", "rotation", "objects", None);

        for frame in 0..72 {
            ctx.set_attribute(
                "rotation",
                &[nsi::double_matrix!(
                    "transformationmatrix",
                    uv::DMat4::from_angle_plane(
                        (frame as f64 * 5.0 * core::f64::consts::TAU / 90.0)
                            as _,
                        uv::DBivec3::from_normalized_axis(uv::DVec3::new(
                            0., 1., 0.
                        ))
                    )
                    .transposed()
                    .as_array()
                )],
            );

            ctx.render_control(&[nsi::string!("action", "synchronize")]);
            ctx.render_control(&[nsi::string!("action", "start")]);
            ctx.render_control(&[nsi::string!("action", "wait")]);
        }
    } else {
        ctx.connect(name, "", ".root", "objects", None);

        //if RenderType::Dump != render_type {
        ctx.render_control(&[nsi::string!("action", "start")]);
        //}
    }

    //if RenderType::Dump != render_type {
    ctx.render_control(&[nsi::string!("action", "wait")]);
    //}

    destination.to_string_lossy().to_string()
}
