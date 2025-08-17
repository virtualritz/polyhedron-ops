use crate::state::{NsiRenderCommand, PlaygroundState};
use bevy::prelude::*;
use crossbeam::channel::Receiver;
use image::{Rgba, RgbaImage};
use parking_lot::Mutex;
use polyhedron_ops::Polyhedron;
use std::sync::Arc;

pub fn start_nsi_render(
    state: &mut PlaygroundState,
    polyhedron: Polyhedron,
    width: u32,
    height: u32,
    camera_transform: Transform,
    camera_projection: &Projection,
) {
    // Reset render state
    state.nsi_state.is_rendering = true;
    state.nsi_state.progress = 0.0;
    state.nsi_state.last_camera_transform = Some(camera_transform);

    // Create image buffer with transparent pixels
    let mut img = RgbaImage::new(width, height);
    // Initialize with transparent black
    for pixel in img.pixels_mut() {
        *pixel = Rgba([0, 0, 0, 0]);
    }
    let image = Arc::new(Mutex::new(img));
    state.nsi_state.image = Some(image.clone());

    // Create tile update flag for progressive rendering
    let tile_updated = Arc::new(Mutex::new(false));
    state.nsi_state.tile_updated = Some(tile_updated.clone());

    // Create render finished flag
    let render_finished = Arc::new(Mutex::new(false));
    state.nsi_state.render_finished = Some(render_finished.clone());

    // Create NSI context
    let ctx = Arc::new(nsi::Context::new(None).unwrap());
    state.nsi_state.context = Some(ctx.clone());

    // Create communication channel if needed
    if state.nsi_state.render_thread_tx.is_none() {
        let (tx, rx) = crossbeam::channel::unbounded();
        state.nsi_state.render_thread_tx = Some(tx);

        // Start render thread
        let ctx_clone = ctx.clone();
        let render_finished_clone = render_finished.clone();
        std::thread::spawn(move || {
            nsi_render_thread(rx, ctx_clone, render_finished_clone);
        });
    }

    // Setup scene
    setup_nsi_scene(
        &ctx,
        polyhedron,
        width,
        height,
        camera_transform,
        camera_projection,
        image,
        tile_updated,
    );

    // Send start command
    if let Some(tx) = &state.nsi_state.render_thread_tx {
        let _ = tx.send(NsiRenderCommand::Start);
    }
}

fn setup_nsi_scene(
    ctx: &nsi::Context,
    polyhedron: Polyhedron,
    width: u32,
    height: u32,
    camera_transform: Transform,
    camera_projection: &Projection,
    image: Arc<Mutex<RgbaImage>>,
    tile_updated: Arc<Mutex<bool>>,
) {
    println!("Camera position: {:?}", camera_transform.translation);
    println!("Camera forward (Bevy): {:?}", camera_transform.forward());

    // Calculate camera distance from origin
    let distance = camera_transform.translation.length();
    let look_dir = camera_transform.forward();
    println!(
        "Camera distance from origin: {:.2}, looking direction: {:?}",
        distance, look_dir
    );

    // Setup camera transform
    ctx.create("camera_xform", nsi::TRANSFORM, None);
    ctx.connect("camera_xform", None, nsi::ROOT, "objects", None);

    // NSI expects +Z to be forward, but Bevy uses -Z
    // Also, NSI's up is +Y like Bevy
    let nsi_position = camera_transform.translation;
    let nsi_target = nsi_position + look_dir.as_vec3() * distance;
    let nsi_up = camera_transform.up();

    println!(
        "Setting NSI camera - pos: {:?}, target: {:?}, up: {:?}",
        nsi_position, nsi_target, nsi_up
    );

    // Build look-at matrix for NSI
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

    // Setup camera
    ctx.create("camera", nsi::PERSPECTIVE_CAMERA, None);
    ctx.connect("camera", None, "camera_xform", "objects", None);

    // Convert FOV based on projection
    let fov_degrees = match camera_projection {
        Projection::Perspective(persp) => {
            // Bevy's FOV is vertical, NSI expects the larger FOV (diagonal or
            // horizontal)
            let vertical_fov = persp.fov.to_degrees();
            let aspect_ratio = width as f32 / height as f32;

            // Convert vertical FOV to horizontal FOV
            let horizontal_fov = 2.0
                * ((vertical_fov.to_radians() / 2.0).tan() * aspect_ratio)
                    .atan()
                    .to_degrees();

            // Use diagonal FOV for best results
            let diagonal_fov = 2.0
                * ((vertical_fov.to_radians() / 2.0).tan().powi(2)
                    + (horizontal_fov.to_radians() / 2.0).tan().powi(2))
                .sqrt()
                .atan()
                .to_degrees();

            println!(
                "FOV conversion: vertical={:.1}°, horizontal={:.1}°, diagonal={:.1}°",
                vertical_fov, horizontal_fov, diagonal_fov
            );

            diagonal_fov
        }
        _ => 60.0, // Default FOV
    };

    println!("Camera FOV: {} degrees (diagonal)", fov_degrees);

    ctx.set_attribute("camera", &[nsi::float!("fov", fov_degrees)]);

    // Setup screen
    ctx.create("screen", nsi::SCREEN, None);
    ctx.connect("screen", None, "camera", "screens", None);
    ctx.set_attribute(
        "screen",
        &[
            nsi::integers!("resolution", &[width as i32, height as i32])
                .array_len(2),
            nsi::integer!("oversampling", 8), // Lower oversampling for faster interactive feedback
        ],
    );

    // Setup output layer
    ctx.create("beauty", nsi::OUTPUT_LAYER, None);
    ctx.set_attribute(
        "beauty",
        &[
            nsi::string!("variablename", "Ci"),
            nsi::integer!("withalpha", 1),
            nsi::string!("scalarformat", "float"),
            nsi::double!("filterwidth", 1.),
        ],
    );
    ctx.connect("beauty", None, "screen", "outputlayers", None);

    // Setup output driver with display callback
    ctx.create("driver", nsi::OUTPUT_DRIVER, None);
    ctx.connect("driver", None, "beauty", "outputdrivers", None);

    println!("Setting up output driver with callbacks...");

    let open_callback = nsi::output::OpenCallback::new(
        |name: &str,
         width: usize,
         height: usize,
         format: &nsi::output::PixelFormat| {
            println!(
                "NSI Open callback called: name={}, size={}x{}, channels={}",
                name,
                width,
                height,
                format.channels()
            );
            nsi::output::Error::None
        },
    );

    // Get the tile updated flag from state
    let tile_updated_clone = tile_updated.clone();

    let write_callback = nsi::output::WriteCallback::new({
        let image = image.clone();
        let mut tile_count = 0;
        move |_name: &str,
              width: usize,
              _height: usize,
              x_min: usize,
              x_max_plus_one: usize,
              y_min: usize,
              y_max_plus_one: usize,
              pixel_format: &nsi::output::PixelFormat,
              pixel_data: &[f32]| {
            tile_count += 1;

            // Debug: Verify buffer size and channels
            let expected_size = width * _height * pixel_format.channels();
            let actual_size = pixel_data.len();
            let channels = pixel_format.channels();

            println!(
                "Write callback called - tile {}, bounds: ({},{}) to ({},{})",
                tile_count, x_min, y_min, x_max_plus_one, y_max_plus_one
            );

            if tile_count == 1 {
                println!(
                    "Buffer info: expected_size={}, actual_size={}, channels={}, width={}, height={}",
                    expected_size, actual_size, channels, width, _height
                );

                if channels < 4 {
                    println!(
                        "WARNING: Channel count is less than 4 ({}), expecting RGBA!",
                        channels
                    );
                }

                // Check first few pixels to see if we have any data
                println!("First 16 values in pixel_data:");
                for i in 0..16.min(pixel_data.len()) {
                    print!("{:.3} ", pixel_data[i]);
                    if (i + 1) % 4 == 0 {
                        println!();
                    }
                }
                println!();

                // Check if values are in 0-1 range or 0-255 range
                let mut max_val = 0.0f32;
                for i in 0..pixel_data.len().min(1000) {
                    max_val = max_val.max(pixel_data[i]);
                }
                println!("Max value in first 1000 pixels: {}", max_val);
            }

            // Check if this tile has any actual content
            let mut has_any_pixels = false;
            let mut max_value = 0.0f32;
            for y in y_min..y_max_plus_one {
                for x in x_min..x_max_plus_one {
                    let idx = (y * width + x) * pixel_format.channels();
                    if idx + pixel_format.channels() <= pixel_data.len()
                        && pixel_format.channels() >= 4
                    {
                        let r = pixel_data[idx];
                        let g = pixel_data[idx + 1];
                        let b = pixel_data[idx + 2];
                        let a = pixel_data[idx + 3];

                        max_value = max_value.max(r).max(g).max(b).max(a);
                        if r > 0.0 || g > 0.0 || b > 0.0 || a > 0.0 {
                            has_any_pixels = true;
                        }
                    }
                }
            }

            if tile_count <= 5 || tile_count % 10 == 0 || has_any_pixels {
                println!(
                    "Tile {}: region x[{}-{}), y[{}-{}), has_pixels: {}, max_value: {}",
                    tile_count,
                    x_min,
                    x_max_plus_one,
                    y_min,
                    y_max_plus_one,
                    has_any_pixels,
                    max_value
                );
            }

            let mut img = image.lock();

            // Update pixels in the specified tile
            for y in y_min..y_max_plus_one {
                for x in x_min..x_max_plus_one {
                    let idx = (y * width + x) * pixel_format.channels();

                    // Debug first pixel of first tile
                    if tile_count == 1 && x == x_min && y == y_min {
                        println!(
                            "First pixel debug: x={}, y={}, width={}, channels={}, idx={}, pixel_data.len()={}",
                            x,
                            y,
                            width,
                            pixel_format.channels(),
                            idx,
                            pixel_data.len()
                        );
                    }

                    if idx + pixel_format.channels() <= pixel_data.len()
                        && pixel_format.channels() >= 4
                    {
                        let r =
                            (pixel_data[idx] * 255.0).clamp(0.0, 255.0) as u8;
                        let g = (pixel_data[idx + 1] * 255.0).clamp(0.0, 255.0)
                            as u8;
                        let b = (pixel_data[idx + 2] * 255.0).clamp(0.0, 255.0)
                            as u8;
                        let a = 255u8; // Force alpha to 1.0 for debugging
                        // let a = (pixel_data[idx + 3] * 255.0).clamp(0.0,
                        // 255.0) as u8;

                        if x < img.width() as usize && y < img.height() as usize
                        {
                            // Always update the pixel, even if black
                            img.put_pixel(
                                x as u32,
                                y as u32,
                                Rgba([r, g, b, a]),
                            );

                            // Debug first few pixels with content
                            if tile_count <= 3
                                && (r > 0 || g > 0 || b > 0 || a > 0)
                            {
                                println!(
                                    "  Pixel ({},{}) = [{}, {}, {}, {}]",
                                    x, y, r, g, b, a
                                );
                            }
                        }
                    }
                }
            }

            // Signal that a tile was updated
            *tile_updated_clone.lock() = true;

            // Force immediate texture update for every Nth tile
            if tile_count % 5 == 0 {
                println!("Tile {} - forcing texture update", tile_count);
            }

            nsi::output::Error::None
        }
    });

    let finish_callback = nsi::output::FinishCallback::new(
        |name: String,
         width: usize,
         height: usize,
         _pixel_format: nsi::output::PixelFormat,
         _pixel_data: Vec<f32>| {
            println!(
                "NSI Finish callback called: name={}, size={}x{}",
                name, width, height
            );
            nsi::output::Error::None
        },
    );

    ctx.set_attribute(
        "driver",
        &[
            nsi::string!("drivername", nsi::output::FERRIS),
            nsi::string!("imagefilename", "playground"),
            nsi::callback!("callback.open", open_callback),
            nsi::callback!("callback.write", write_callback),
            nsi::callback!("callback.finish", finish_callback),
        ],
    );

    // Setup environment
    ctx.create("env_xform", nsi::TRANSFORM, None);
    ctx.connect("env_xform", None, nsi::ROOT, "objects", None);

    ctx.create("environment", nsi::ENVIRONMENT, None);
    ctx.connect("environment", None, "env_xform", "objects", None);

    ctx.create("env_shader", nsi::SHADER, None);
    ctx.set_attribute(
        "env_shader",
        &[
            nsi::string!("shaderfilename", "${DELIGHT}/osl/environmentLight"),
            nsi::float!("intensity", 5.0), /* Increase intensity to ensure
                                            * visibility */
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

    // Add the polyhedron
    let poly_handle = polyhedron.to_nsi(&ctx, None, Some(10.0), None, None);

    // Create shader for the polyhedron
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

    // Create attributes node
    ctx.create("poly_attrib", nsi::ATTRIBUTES, None);
    ctx.connect(
        "poly_attrib",
        None,
        &poly_handle,
        "geometryattributes",
        None,
    );
    ctx.connect("plastic_shader", None, "poly_attrib", "surfaceshader", None);

    // Debug: Print some info about the polyhedron
    // Debug: Print some info about the polyhedron
    let positions = polyhedron.positions();
    println!("Polyhedron vertex count: {}", positions.len());
    println!("Polyhedron face count: {}", polyhedron.faces().len());

    // Get bounds for debugging
    let positions = polyhedron.positions();
    if !positions.is_empty() {
        let min = positions.iter().fold(Vec3::splat(f32::MAX), |a, b| {
            a.min(Vec3::new(b.x, b.y, b.z))
        });
        let max = positions.iter().fold(Vec3::splat(f32::MIN), |a, b| {
            a.max(Vec3::new(b.x, b.y, b.z))
        });
        println!("Polyhedron bounds: min {:?}, max {:?}", min, max);
    }

    // This ensures the scene is fully set up before rendering begins
}

pub fn nsi_render_thread(
    rx: Receiver<NsiRenderCommand>,
    ctx: Arc<nsi::Context<'static>>,
    render_finished: Arc<Mutex<bool>>,
) {
    let mut is_rendering = false;

    while let Ok(command) = rx.recv() {
        match command {
            NsiRenderCommand::Start => {
                if !is_rendering {
                    println!("NSI render thread: Starting render");
                    ctx.render_control(nsi::Action::Start, None);

                    // Synchronize to ensure scene is ready
                    ctx.render_control(nsi::Action::Synchronize, None);
                    println!("NSI render thread: Scene synchronized");

                    is_rendering = true;
                    *render_finished.lock() = false;

                    // Start progress monitoring
                    // AIDEV-NOTE: Progress monitoring removed temporarily as
                    // get_attribute API changed

                    println!("NSI render thread: Render started");
                }
            }
            NsiRenderCommand::Stop => {
                if is_rendering {
                    println!("NSI render thread: Stopping render");
                    ctx.render_control(nsi::Action::Stop, None);
                    ctx.render_control(nsi::Action::Wait, None);
                    is_rendering = false;
                    *render_finished.lock() = true;
                    println!("NSI render thread: Render stopped");
                }
            }
            NsiRenderCommand::UpdateCamera(transform) => {
                println!(
                    "NSI render thread: Stopping render for camera update"
                );

                // Stop current render
                ctx.render_control(nsi::Action::Stop, None);
                ctx.render_control(nsi::Action::Wait, None);

                // Update camera transform
                let nsi_position = transform.translation;
                let look_dir = transform.forward();
                let distance = nsi_position.length();
                let nsi_target = nsi_position + look_dir.as_vec3() * distance;
                let nsi_up = transform.up();

                // Build look-at matrix
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

                // Restart render
                ctx.render_control(nsi::Action::Start, None);
                ctx.render_control(nsi::Action::Synchronize, None);

                println!("NSI render thread: Render restarted with new camera");
            }
        }
    }
}
