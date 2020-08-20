extern crate nsi;

use std::{env, path::PathBuf};

fn nsi_camera(c: &nsi::Context, name: &str, camera_xform: &[f64; 16]) {
    // Setup a camera transform.
    c.create("cam1_trs", nsi::NodeType::Transform, &[]);
    c.connect("cam1_trs", "", ".root", "objects", &[]);

    c.set_attribute(
        "cam1_trs",
        &[nsi::double_matrix!(
            "transformationmatrix",
            //camera_xform
            &[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 5., 1.,]
        )],
    );

    // Setup a camera.
    c.create("cam1", nsi::NodeType::PerspectiveCamera, &[]);

    c.set_attribute("cam1", &[nsi::float!("fov", 35.)]);
    c.connect("cam1", "", "cam1_trs", "objects", &[]);

    // Setup a screen.
    c.create("s1", nsi::NodeType::Screen, &[]);
    c.connect("s1", "", "cam1", "screens", &[]);
    c.set_attribute(
        "s1",
        &[
            nsi::integers!("resolution", &[1536, 1536]).array_len(2),
            nsi::integer!("oversampling", 16),
        ],
    );

    c.set_attribute(
        ".global",
        &[
            nsi::integer!("renderatlowpriority", 1),
            nsi::string!("bucketorder", "circle"),
            nsi::unsigned!("quality.shadingsamples", 512),
            nsi::integer!("maximumraydepth.reflection", 6),
        ],
    );

    // Setup an output layer.
    c.create(name, nsi::NodeType::OutputLayer, &[]);
    c.set_attribute(
        name,
        &[
            nsi::string!("variablename", "Ci"),
            nsi::integer!("withalpha", 1),
            nsi::string!("scalarformat", "half"),
        ],
    );
    c.connect(name, "", "s1", "outputlayers", &[]);

    // Setup an output driver.
    c.create("driver1", nsi::NodeType::OutputDriver, &[]);
    c.connect("driver1", "", name, "outputdrivers", &[]);
    c.set_attribute("driver1", &[nsi::string!("drivername", "idisplay")]);
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
