use crate::*;
#[test]
fn tetrahedron_to_terahedron() {
    // Tetrahedron

    let mut tetrahedron = Polyhedron::tetrahedron();

    //tetrahedron.dual();
    tetrahedron.kis(Some(0.3), None, None, None, false);

    //let ctx = nsi::Context::new(&[nsi::string!("streamfilename",
    // "stdout")]).unwrap(); tetrahedron.to_nsi(ctx,
    // "terahedron");
    #[cfg(feature = "obj")]
    tetrahedron
        .write_to_obj(&std::path::PathBuf::from("."), false)
        .unwrap();
}

#[test]
fn cube_to_octahedron() {
    let mut cube = Polyhedron::hexahedron();

    cube.dual(false);
    #[cfg(feature = "obj")]
    cube.write_to_obj(&std::path::PathBuf::from("."), false)
        .unwrap();
}

#[test]
fn triangulate_cube() {
    let mut cube = Polyhedron::hexahedron();

    cube.triangulate(Some(true));
    #[cfg(feature = "obj")]
    cube.write_to_obj(&std::path::PathBuf::from("."), false)
        .unwrap();
}

#[test]
fn make_pentagonal_prism() {
    let pentagonal_prism = Polyhedron::prism(5);

    #[cfg(feature = "obj")]
    pentagonal_prism
        .write_to_obj(&std::path::PathBuf::from("."), false)
        .unwrap();
}
