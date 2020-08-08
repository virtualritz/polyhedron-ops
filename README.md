# Polyhedron Operators

This crate implements the [Conway Polyhedral
Operators](http://en.wikipedia.org/wiki/Conway_polyhedron_notation)
and their extensions by [George W. Hart](http://www.georgehart.com/)
and others.

This is an experiment to improve my understanding of iterators
in Rust. It is based on Hart’s OpenSCAD code which, being
functional, leads itself well to being translated to functional Rust.

## Supported Operators

- [x] kN - kis on N-sided faces (if no N, then general kis)
- [ ] a - ambo
- [x] g - gyro
- [x] d - dual
- [x] r - reflect
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

## Playing

While I work on this crate there it's a binary
that allows me to test things.

```
cargo run --release
```

Use keys to matching the operator to apply.
Use `Up` and `Down` to adjust the parameter of the
last operator. Combine with `Shift` for
10× the change. Use `S` to save as
`$HOME/polyhedron.obj`.

I use `kiss3d` for realtime preview which is
close to the metal enough to limit meshes to
65k vertices. This means the preview will be 
broken if your mesh hits this limit.

Export will always yield a correct OBJ though.
Which you can view in Blender or another DCC
app.
