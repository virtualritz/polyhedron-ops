extern crate nsi;

use std::{env, path::PathBuf};

fn nsi_camera(c: &nsi::Context, name: &str, camera_xform: &[f64; 16]) {
    // Setup a camera transform.
    c.create("camera_xform", nsi::NodeType::Transform, &[]);
    c.connect("camera_xform", "", ".root", "objects", &[]);

    c.set_attribute(
        "camera_xform",
        &[nsi::double_matrix!(
            "transformationmatrix",
            //camera_xform
            &[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 5., 1.,]
        )],
    );

    // Setup a camera.
    c.create("camera", nsi::NodeType::PerspectiveCamera, &[]);

    c.set_attribute("camera", &[nsi::float!("fov", 35.)]);
    c.connect("camera", "", "camera_xform", "objects", &[]);

    // Setup a screen.
    c.create("screen", nsi::NodeType::Screen, &[]);
    c.connect("screen", "", "camera", "screens", &[]);
    c.set_attribute(
        "screen",
        &[
            nsi::integers!("resolution", &[2048, 2048]).array_len(2),
            nsi::integer!("oversampling", 16),
        ],
    );

    c.set_attribute(
        ".global",
        &[
            nsi::integer!("renderatlowpriority", 1),
            nsi::string!("bucketorder", "circle"),
            nsi::unsigned!("quality.shadingsamples", 128),
            nsi::integer!("maximumraydepth.reflection", 6),
        ],
    );

    c.create("albedo", nsi::NodeType::OutputLayer, &[]);
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
    c.connect("albedo", "", "screen", "outputlayers", &[]);

    // Normal layer.
    c.create("normal", nsi::NodeType::OutputLayer, &[]);
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
    c.connect("normal", "", "screen", "outputlayers", &[]);

    // Setup an output layer.
    c.create(name, nsi::NodeType::OutputLayer, &[]);
    c.set_attribute(
        name,
        &[
            nsi::string!("variablename", "Ci"),
            nsi::integer!("withalpha", 1),
            nsi::string!("scalarformat", "float"),
            nsi::double!("filterwidth", 1.),
        ],
    );
    c.connect(name, "", "screen", "outputlayers", &[]);

    // Setup an output driver.
    c.create("driver1", nsi::NodeType::OutputDriver, &[]);
    c.connect("driver1", "", name, "outputdrivers", &[]);
    c.set_attribute("driver1", &[nsi::string!("drivername", "idisplay")]);

    c.create("driver2", nsi::NodeType::OutputDriver, &[]);
    c.connect("driver2", "", name, "outputdrivers", &[]);
    c.connect("driver2", "", "albedo", "outputdrivers", &[]);
    c.connect("driver2", "", "normal", "outputdrivers", &[]);
    c.set_attribute(
        "driver2",
        &[
            nsi::string!("drivername", "r-display"),
            nsi::string!("imagefilename", name),
            nsi::float!("denoise", 1.),
        ],
    );
}

fn nsi_environment(c: &nsi::Context) {
    if let Ok(path) = &env::var("DELIGHT") {
        // Set up an environment light.
        c.create("env_xform", nsi::NodeType::Transform, &[]);
        c.connect("env_xform", "", ".root", "objects", &[]);

        c.create("environment", nsi::NodeType::Environment, &[]);
        c.connect("environment", "", "env_xform", "objects", &[]);

        c.create("env_attrib", nsi::NodeType::Attributes, &[]);
        c.connect("env_attrib", "", "environment", "geometryattributes", &[]);

        c.set_attribute("env_attrib", &[nsi::integer!("visibility.camera", 0)]);

        c.create("env_shader", nsi::NodeType::Shader, &[]);
        c.connect("env_shader", "", "env_attrib", "surfaceshader", &[]);

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
            &[nsi::string!("image", "assets/wooden_lounge_2k.tdl")],
        );
    }
}

fn nsi_material(c: &nsi::Context, name: &str) {
    if let Ok(path) = &env::var("DELIGHT") {
        // Particle attributes.
        let attribute_name = format!("{}_attrib", name);
        c.create(attribute_name.clone(), nsi::NodeType::Attributes, &[]);
        c.connect(attribute_name.clone(), "", name, "geometryattributes", &[]);

        // Particle shader.
        let shader_name = format!("{}_shader", name);
        c.create(shader_name.clone(), nsi::NodeType::Shader, &[]);
        c.connect(
            shader_name.clone(),
            "",
            attribute_name,
            "surfaceshader",
            &[],
        );

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
                nsi::float!("anisotropy", 0.0f32),
                nsi::float!("sss_weight", 0.0f32),
                nsi::color!("sss_color", &[0.5f32, 0.5, 0.5]),
                nsi::float!("sss_scale", 0.0f32),
                nsi::color!("incandescence", &[0.0f32, 0.0, 0.0]),
                nsi::float!("incandescence_intensity", 0.0f32),
                //nsi::color!("incandescence_multiplier", &[1.0f32, 1.0, 1.0]),
            ],
        );
    }
}

pub fn nsi_render(
    polyhedron: &crate::Polyhedron,
    camera_xform: &[f64; 16],
    name: &str,
    cloud_render: bool,
) {
    let ctx = {
        if cloud_render {
            nsi::Context::new(&[
                nsi::integer!("cloud", 1),
                nsi::string!("software", "HOUDINI"),
            ])
        } else {
            nsi::Context::new(&[])
        }
    }
    .expect("Could not create NSI rendering context.");

    nsi_camera(&ctx, name, camera_xform);

    nsi_environment(&ctx);

    let name = polyhedron.to_nsi(&ctx);

    nsi_material(&ctx, &name);

    // And now, render it!
    ctx.render_control(&[nsi::string!("action", "start")]);

    // And now, render it!
    ctx.render_control(&[nsi::string!("action", "wait")]);
}
