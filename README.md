# Polyhedron Operators

This crate implements the [Conway Polyhedral
Operators](http://en.wikipedia.org/wiki/Conway_polyhedron_notation)
and their extensions by [George W. Hart](http://www.georgehart.com/)
and others.

This is an experiment to imporve my understanding of iterators
in Rust. It is based on Hart’s OpenSCAD code which, being
functional, leads itself well to being translated to functional Rust.

## Supported Operators

- [x] kN - kis on N-sided faces (if no N, then general kis)
- [ ] a - ambo
- [ ] g - gyro
- [x] d - dual
- [ ] r - reflect
- [ ] e - explode (a.k.a. expand, equiv. to aa)
- [ ] b - bevel (equiv. to ta)
- [ ] o - ortho (equiv. to jj)
- [ ] m - meta (equiv. to k3j)
- [ ] tN - truncate vertices of degree N (equiv. to dkNd; if no N, then truncate all vertices)
- [ ] j - join (equiv. to dad)
- [ ] s - snub (equiv. to dgd)
- [ ] p - propellor
- [ ] c - chamfer
- [ ] w - whirl
- [ ] q - quinto
