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
fn make_prisms() {
    for i in 3..9 {
        let prism = Polyhedron::prism(i);

        #[cfg(feature = "obj")]
        prism
            .write_to_obj(&std::path::PathBuf::from("."), false)
            .unwrap();

        let f = prism.faces().len();
        let v = prism.positions_len();
        let e = prism.to_edges().len();
        assert!(f == i + 2);
        assert!(v == i * 2);
        assert!(e == 2 * i + i);
        assert!(f + v - e == 2); // Euler's Formula
    }
}

#[test]
fn make_antiprisms() {
    for i in 3..9 {
        let antiprism = Polyhedron::antiprism(i);

        #[cfg(feature = "obj")]
        antiprism
            .write_to_obj(&std::path::PathBuf::from("."), false)
            .unwrap();

        let f = antiprism.faces().len();
        let v = antiprism.positions_len();
        let e = antiprism.to_edges().len();
        assert!(f == i * 2 + 2);
        assert!(v == i * 2);
        assert!(e == 2 * i + 2 * i);
        assert!(f + v - e == 2); // Euler's Formula
    }
}
