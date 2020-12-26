use crate::*;
#[test]
fn tetrahedron_to_terahedron() {
    // Tetrahedron

    let mut tetrahedron = Polyhedron::tetrahedron();

    //tetrahedron.dual();
    tetrahedron.kis(Some(0.3), None, None, false);

    //let ctx = nsi::Context::new(&[nsi::string!("streamfilename",
    // "stdout")]).unwrap(); tetrahedron.to_nsi(ctx,
    // "terahedron");
    if cfg!(feature = "obj") {
        tetrahedron
            .write_to_obj(&std::path::PathBuf::from("."), false)
            .unwrap();
    }
}

#[test]
fn cube_to_octahedron() {
    let mut cube = Polyhedron::hexahedron();

    cube.dual(false);
    if cfg!(feature = "obj") {
        cube.write_to_obj(&std::path::PathBuf::from("."), false)
            .unwrap();
    }
}

#[test]
fn triangulate_cube() {
    let mut cube = Polyhedron::hexahedron();

    cube.triangulate(true);
    if cfg!(feature = "obj") {
        cube.write_to_obj(&std::path::PathBuf::from("."), false)
            .unwrap();
    }
}

#[test]
fn make_pentagonal_prism() {
    let pentagonal_prism = Polyhedron::prism(5);

    if cfg!(feature = "obj") {
        pentagonal_prism
            .write_to_obj(&std::path::PathBuf::from("."), false)
            .unwrap();
    }
}
