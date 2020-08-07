#[cfg(test)]
mod tests {

    use crate::*;
    #[test]
    fn tetrahedron_to_terahedron() {
        // Tetrahedron

        let mut tetrahedron = Polyhedron::tetrahedron();

        //tetrahedron.dual();
        tetrahedron.kis(0.3, None, false);

        //let ctx = nsi::Context::new(&[nsi::string!("streamfilename",
        // "stdout")]).unwrap(); tetrahedron.to_nsi(ctx,
        // "terahedron");
        tetrahedron.export_as_obj(
            &std::path::PathBuf::from("/Users/moritz/tetrahedron.obj"),
            true,
        );
    }

    #[test]
    fn cube_to_octahedron() {
        let mut cube = Polyhedron::hexahedron();

        cube.dual();
        cube.export_as_obj(
            &std::path::PathBuf::from("/Users/moritz/octahedron.obj"),
            true,
        );
    }

    #[test]
    fn trinangulate_cube() {
        let mut cube = Polyhedron::hexahedron();

        cube.triangulate(true);
        cube.export_as_obj(
            &std::path::PathBuf::from(
                "/Users/moritz/triangulated_cube.obj",
            ),
            true,
        );
    }
}
