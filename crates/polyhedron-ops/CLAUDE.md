# `CLAUDE.md` -- polyhedron-ops -- Conway/Hart Polyhedron Operations

## The Golden Rule

When unsure about implementation details, ALWAYS ask the developer.

## Project Context

This is a Rust crate implementing Conway Polyhedron Operators and their extensions by George W. Hart and others. The crate provides high-performance geometric operations on polyhedra using n-gon mesh buffers.

## Critical Architecture Decisions

### Method Chaining API

Operations are designed to be chained together. Each operator method returns `&mut Self` to allow fluent API usage:
```rust
polyhedron.chamfer(None, true).propellor(None, true).ambo(None, true)
```

### Use External Crates Where Possible

- A lot of geometry processing infrastructure exists in the Rust ecosystem. Always search crates.io for the most appropriate crates before implementing functionality on your own. When in doubt, ALWAYS ask the developer.

- Examples are ultraviolet (for linear algebra), rayon (for parallelism), kiss3d (for visualization), etc.

### Core Data Structure

- **Flat vertex arrays**: Positions stored as `Vec<Vec3>`
- **Face indices**: Faces represented as `Vec<Vec<u32>>` (n-gons)
- **Face sets**: Grouping of faces by operation for tracking
- **Generic float type**: Currently `f32` but designed to be changeable

#### Important Types

- **[`Polyhedron`]** -- The central data structure holding vertex positions, face indices, and metadata.
- **[`Point`]** -- 3D point type (`ultraviolet::Vec3`).
- **[`Face`]** -- Variable-length vertex index list (`Vec<VertexKey>`).
- **[`Edge`]** -- Pair of vertex indices (`[VertexKey; 2]`).
- **[`VertexKey`]** & **[`FaceKey`]** -- Index types (`u32`).

### Key Features

- Conway notation operators (ambo, bevel, chamfer, dual, etc.).
- Support for n-gon faces (not limited to triangles).
- Parallel processing with [`rayon`] for performance.
- Parser for Conway notation strings (with `parser` feature).
- Export to various formats (OBJ, Bevy mesh, NSI).

### Performance Requirements

- Use `rayon` for parallelism in bulk operations
- Prefer `par_iter()`/`par_iter_mut()` for vertex/face operations
- Avoid unnecessary allocations, conversions, copies
- Use efficient data structures for topology queries

### Design Patterns

- Use Rust's type system as much as possible to make code more idiomatic/generic
- Prefer functional style over imperative
- Use generics and traits for type flexibility
- Employ macros to reduce code duplication
- Keep operators pure (modify self, return self)
- Use macros to avoid repetitive boilerplate code, i.e. keep code DRY

## File Format Support

- Target formats: Wavefront OBJ (with `obj` feature).
- Bevy Mesh (with `bevy` feature).
- NSI scene description (with `nsi` feature).

## External Crate Integration

- Linear algebra: `ultraviolet` crate for vectors and matrices.
- Parallelism: `rayon` for parallel iterators.
- Visualization: `kiss3d` for examples, `bevy` for game engine integration.
- Parsing: `pest` for Conway notation parser.

## Traits

- All public-facing types must implement `Debug`, `Clone`, `Hash`, `PartialEq`, and `Eq`. Also `Copy`, if it can be trivially derived.

## Code Style and Patterns

### Anchor comments

Add specially formatted comments throughout the codebase, where appropriate, for yourself as inline knowledge that can be easily `rg`ped (grepped) for.

### Guidelines:

- ALWAYS run `cargo clippy --fix` before committing. If clippy brings up any issues, fix them, then repeat untile there are no more issues brought up by clippy. Finally run `cargo fmt`, then commit.

- Use `AIDEV-NOTE:`, `AIDEV-TODO:`, or `AIDEV-QUESTION:` (all-caps prefix) for comments aimed at AI and developers.
- **Important:** Before scanning files, always first try to **grep for existing anchors** `AIDEV-*` in relevant subdirectories.
- **Update relevant anchors** when modifying associated code.
- **Do not remove `AIDEV-NOTE`s** without explicit human instruction.
- Make sure to add relevant anchor comments, whenever a file or piece of code is:

  - too complex, or
  - very important, or
  - confusing, or
  - could have a bug

- DO NOT change any public-facing API without presenting a change proposal to the user first, inclusing a rationale and getting permission to do so after.

- Write idiomatic and canonical Rust code. I.e. avoid patterns common in imperative languages like C/C++/JS/TS that can be expressed more elegantly, concise and with more leeway for the compiler to optimize, in Rust.

- PREFER functional style over imperative style. I.e. use for_each or map instead of for loops, use collect instead of pre-allocating a Vec and using push.

- USE rayon to parallelize whenever larger amounts of data are being processed.

- AVOID unnecessary allocations, conversions, copies

- AVOID using `unsafe` code unless absolutely necessary.

- AVOID return statements; structure functions with if ... if else ... else blocks instead.

- Prefer using the stack, use SmallVec whenever it makes sense.

- NAMING follows the rules laid out in this document: https://raw.githubusercontent.com/rust-lang/api-guidelines/refs/heads/master/src/naming.md

## Domain Glossary (Claude, learn these!)

- **Polyhedron**: A 3D solid bounded by flat polygonal faces, straight edges and sharp vertices.
- **Conway Operators**: A set of operations (ambo, bevel, chamfer, dual, etc.) that transform polyhedra.
- **Face**: A polygon (n-gon) defined by an ordered list of vertex indices.
- **Edge**: A line segment between two vertices.
- **Vertex**: A point in 3D space where edges meet.
- **n-gon**: A polygon with n sides (triangle=3-gon, quad=4-gon, etc.).
- **Dual**: Operation that replaces faces with vertices and vice versa.
- **Face arity**: The number of vertices/edges in a face.
- **Vertex valence/degree**: The number of edges meeting at a vertex.

## What AI Must NEVER Do

1. **Never modify test files** - Tests encode human intent
2. **Never change API contracts** - Breaks real applications
3. **Never alter migration files** - Data loss risk
4. **Never commit secrets** - Use environment variables
5. **Never assume business logic** - Always ask
6. **Never remove AIDEV- comments** - They're there for a reason

Remember: We optimize for maintainability over cleverness.
When in doubt, choose the boring solution.

## Project Overview

This crate provides a comprehensive set of Conway polyhedron operators that can be applied to 3D polyhedra. Starting from base shapes (Platonic solids, prisms, antiprisms), users can apply a sequence of operations to create complex geometric forms. The operations preserve topological validity while transforming the geometry.

### Base Polyhedra

The crate provides constructors for:
- Platonic solids (tetrahedron, cube, octahedron, dodecahedron, icosahedron)
- Prisms (n-sided)
- Antiprisms (n-sided)
- Pyramids (planned)

### Operator Categories

1. **Topology-changing**: ambo, bevel, chamfer, dual, expand, gyro, join, kis, meta, needle, ortho, propellor, quinto, snub, truncate, whirl, zip
2. **Geometry-preserving**: reflect, spherize, canonicalize
3. **Subdivision**: subdivide (Catmull-Clark)
4. **Extrusion**: extrude, inset/loft

## Build and Development Commands

```bash
# Build the project
cargo build

# Run tests
cargo test

# Run a specific test
cargo test test_name

# Build with optimizations
cargo build --release

# Format code
cargo fmt

# Run clippy linter
cargo clippy --fix --allow-dirty

# Check code without building
cargo check

# Search -- grep replacement
rg

# Find files -- find replacelemt
fd

# Xargs
xargs

# Awk
awk

# Find & replacele -- sed replacement
sd

# View - cat replacement
bat
```

## Writing Instructions For User Interaction And Documentation

These instructions apply to any communcation (e.g. feedback you print to the user) as well as any documentation you write.

- Be concise.

- AVOID weasel words.

- Use simple sentences. But feel free to use technical jargon.

- Do NOT overexplain basic concepts. Assume the user is technically proficient.

- AVOID flattering, corporate-ish or marketing language. Maintain a neutral viewpoint.

- AVOID vague and/or generic claims which may seem correct but are not substantiated by the the context.

## Documentation

- All code comments MUST end with a period.

- All doc comments should also end with a period unless they're headlines. This includes list items.

- ENSURE an en-dash is expressed as two dashes like so: --. En-dashes are not used for connecting words, e.g. "compile-time".

- All references to types, keywords, symbols etc. MUST be enclosed in backticks: `struct` `Foo`.

- For each part of the docs, every first reference to a type, keyword, symbol etc. that is NOT the item itself that is being described MUST be linked to the relevant section in the docs like so: [`Foo`].

- NEVER use fully qualified paths in doc links. Use [`Foo`](foo::bar::Foo) instead of [`foo::bar::Foo`].