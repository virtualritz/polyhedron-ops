use cgmath::prelude::*;
use itertools::Itertools;
//use nsi;
use std::{
    fs::File,
    io::Write,
    iter::{once, Iterator},
    path::{Path, PathBuf},
};

use dirs;

use rayon::prelude::*;

mod tests;

type Float = f32;
type Index = u32;
type Face = Vec<Index>;
type FaceIndex = Vec<Face>;
// We treat an Edge as a Face with arity 2 to avoid copying in certain
// cases.
type Edge = Face;
type EdgeIndex = Vec<Edge>;

type FlatVertices = Vec<Float>;
type Point = cgmath::Point3<Float>;
type Vector = cgmath::Vector3<Float>;
type Normal = Vector;
type Vertices = Vec<Point>;
type VerticesRef<'a> = Vec<&'a Point>;
type Normals = Vec<Normal>;

enum NormalType {
    Smooth(Float),
    Flat,
}

//impl Vertices {
#[inline]
fn to_vadd(vertices: &Vertices, v: &Vector) -> Vertices {
    vertices.par_iter().map(|p| p.clone() + v).collect()
}

#[inline]
fn vadd(vertices: &mut Vertices, v: &Vector) {
    vertices.par_iter_mut().for_each(|p| *p += *v);
}

#[inline]
fn centroid(vertices: &Vertices) -> Point {
    let identity = Point::new(0., 0., 0.);
    let total_displacement = vertices
        .into_par_iter()
        .cloned()
        .reduce(|| Point::new(0., 0., 0.), |acc, p| acc + p.to_vec());

    total_displacement / vertices.len() as Float
}

#[inline]
fn ordered_vertex_edges_recurse(v: u32, vfaces: &FaceIndex, face: &Face, k: usize) -> EdgeIndex {
    if k < vfaces.len() {
        let i = index_of(&v, face).unwrap();
        let j = (i + face.len() - 1) % face.len();
        let edge = vec![v, face[j]];
        let nface = face_with_edge(&edge, vfaces);
        let mut result = vec![edge];
        result.extend(ordered_vertex_edges_recurse(v, vfaces, &nface, k + 1));
        result
    } else {
        vec![]
    }
}

#[inline]
fn ordered_vertex_edges(v: u32, vfaces: &FaceIndex) -> EdgeIndex {
    if vfaces.is_empty() {
        vec![]
    } else {
        let face = &vfaces[0];
        let i = index_of(&v, face).unwrap();
        let j = (i + face.len() - 1) % face.len();
        let edge = vec![v, face[j]];
        let nface = face_with_edge(&edge, vfaces);
        let mut result = vec![edge];
        result.extend(ordered_vertex_edges_recurse(v, vfaces, &nface, 1));
        result
    }
}

#[inline]
fn distinct_edge(edge: &Edge) -> Edge {
    if edge[0] < edge[1] {
        edge.clone()
    } else {
        let mut e = edge.clone();
        e.reverse();
        e
    }
}

#[inline]
fn distinct_face_edges(face: &Face) -> EdgeIndex {
    face.iter()
        .cycle()
        .tuple_windows::<(_, _)>()
        .map(|t| {
            if t.0 < t.1 {
                vec![*t.0, *t.1]
            } else {
                vec![*t.1, *t.0]
            }
        })
        .take(face.len())
        .collect()
}

#[inline]
fn to_centroid_points(vertices: &Vertices) -> Vertices {
    let c = centroid(vertices);
    to_vadd(vertices, &cgmath::Vector3::new(-c.x, -c.y, -c.z))
}

#[inline]
fn center_on_centroid(vertices: &mut Vertices) {
    let c = centroid(vertices);
    vadd(vertices, &cgmath::Vector3::new(-c.x, -c.y, -c.z));
}

#[inline]
fn centroid_ref<'a>(vertices: &'a VerticesRef) -> Point {
    let total_displacement = vertices
        .into_iter()
        //.cloned()
        .fold(Point::new(0., 0., 0.), |acc, p| acc + (*p).to_vec());

    total_displacement / vertices.len() as Float
}

#[inline]
fn vnorm(points: &Vertices) -> Vec<Float> {
    points
        .par_iter()
        .map(|v| Normal::new(v.x, v.y, v.z).magnitude())
        .collect()
}
// FIXME rename to average_magnitude
#[inline]
fn average_norm(points: &Vertices) -> Float {
    vnorm(points).par_iter().sum::<Float>() / points.len() as Float
}

/// Returns a [`FaceIndex`] of faces
/// containing `vertex_number`.
#[inline]
fn vertex_faces(vertex_number: Index, face_index: &FaceIndex) -> FaceIndex {
    face_index
        .par_iter()
        .filter(|face| face.contains(&vertex_number))
        .cloned()
        .collect()
}

/// Returns a [`Vec`] of anticlockwise
/// ordered edges.
fn ordered_face_edges_(face: &Face) -> EdgeIndex {
    face.iter()
        .cycle()
        .tuple_windows::<(_, _)>()
        .map(|edge| vec![*edge.0, *edge.1])
        .take(face.len())
        .collect()
}

/// Returns a [`Vec`] of anticlockwise
/// ordered edges.
#[inline]
fn ordered_face_edges(face: &Face) -> EdgeIndex {
    // FIXME: make functional
    let mut result = EdgeIndex::with_capacity(face.len());
    for i in 0..face.len() {
        result.push(vec![face[i], face[(i + 1) % face.len()]]);
    }
    result
}

#[inline]
fn face_with_edge(edge: &Edge, faces: &FaceIndex) -> Face {
    let result = faces
        .par_iter()
        .filter(|face| ordered_face_edges(face).contains(edge))
        .flatten()
        .cloned()
        .collect();
    result
}

#[inline]
fn index_of<T: PartialEq>(element: &T, list: &Vec<T>) -> Option<usize> {
    list.iter().position(|e| *e == *element)
}

/// Used internally by [`ordered_vertex_faces()`].
#[inline]
fn ordered_vertex_faces_recurse(
    v: Index,
    face_index: &FaceIndex,
    cface: &Face,
    k: Index,
) -> FaceIndex {
    if (k as usize) < face_index.len() {
        let i = index_of(&v, &cface).unwrap() as i32;
        let j = ((i - 1 + cface.len() as i32) % cface.len() as i32) as usize;
        let edge = vec![v, cface[j]];
        let mut nfaces = vec![face_with_edge(&edge, face_index)];
        nfaces.extend(ordered_vertex_faces_recurse(
            v,
            face_index,
            &nfaces[0],
            k + 1,
        ));
        nfaces
    } else {
        FaceIndex::new()
    }
}

#[inline]
fn ordered_vertex_faces(vertex_number: Index, face_index: &FaceIndex) -> FaceIndex {
    let mut result = vec![face_index[0].clone()];
    result.extend(ordered_vertex_faces_recurse(
        vertex_number,
        face_index,
        &face_index[0],
        1,
    ));

    result
}

#[inline]
fn edge_length(edge: &Edge, vertices: &Vertices) -> Float {
    let edge = vec![edge[0], edge[1]];
    let vertices = as_points(&edge, vertices);
    (vertices[0] - vertices[1]).magnitude()
}

#[inline]
fn edge_lengths(edges: &EdgeIndex, points: &Vertices) -> Vec<Float> {
    edges
        .par_iter()
        .map(|edge| edge_length(edge, points))
        .collect()
}

#[inline]
fn face_edges(face: &Face, vertices: &Vertices) -> Vec<Float> {
    ordered_face_edges(face)
        .par_iter()
        .map(|edge| edge_length(edge, vertices))
        .collect()
}

#[inline]
fn circumscribed_resize(vertices: &mut Vertices, radius: Float) {
    center_on_centroid(vertices);
    let average = average_norm(vertices);

    vertices.par_iter_mut().for_each(|v| *v *= radius / average);
}

#[inline]
fn face_irregularity(face: &Face, points: &Vertices) -> Float {
    let lengths = face_edges(face, points);
    // The largest value in lengths or NaN (0./0.) otherwise.
    lengths.par_iter().cloned().reduce(|| 0. / 0., Float::max)
        // divide by the smallest value in lengths or NaN (0./0.) otherwise.
        / lengths.par_iter().cloned().reduce(|| 0. / 0., Float::min)
}

#[inline]
// FIXME: rename to face_as_vertices
fn as_points<'a>(f: &[Index], vertices: &'a Vertices) -> VerticesRef<'a> {
    f.par_iter()
        .map(|index| &vertices[*index as usize])
        .collect()
}

#[inline]
fn orthogonal(v0: &Point, v1: &Point, v2: &Point) -> Vector {
    (v1 - v0).cross(v2 - v1)
}

/// Computes the normal of a face.
/// Assumes the face is planar.
#[inline]
fn face_normal(vertices: &VerticesRef) -> Vector {
    // FIXME iterate over all points to make this work for
    // non-planar faces.
    -orthogonal(&vertices[0], &vertices[1], &vertices[2]).normalize()
}

#[inline]
fn vertex_ids_ref<'a>(entries: &Vec<(&'a Face, Point)>, offset: Index) -> Vec<(&'a Face, Index)> {
    entries
        .par_iter()
        .enumerate()
        // FIXME swap with next line once rustfmt is fixed.
        //.map(|i| (i.1.0, i.0 + offset))
        .map(|i| (entries[i.0].0, i.0 as Index + offset))
        .collect()
}

#[inline]
fn vertex_ids(entries: &Vec<(Face, Point)>, offset: usize) -> Vec<(Face, usize)> {
    entries
        .par_iter()
        .enumerate()
        // FIXME swap with next line once rustfmt is fixed.
        //.map(|i| (i.1.0, i.0 + offset))
        .map(|i| (entries[i.0].0.clone(), i.0 + offset))
        .collect()
}

#[inline]
fn vertex(key: &Face, entries: &Vec<(&Face, Index)>) -> Option<Index> {
    match entries.par_iter().find_first(|f| key == f.0) {
        Some(entry) => Some(entry.1),
        None => None,
    }
}

#[inline]
fn vertex_values_as_ref<'a>(entries: &'a Vec<(&Face, Point)>) -> VerticesRef<'a> {
    entries.par_iter().map(|e| &e.1).collect()
}

fn vertex_values(entries: &Vec<(&Face, Point)>) -> Vertices {
    entries.par_iter().map(|e| e.1).collect()
}

#[inline]
fn selected_face(face: &Face, face_arity: Option<&Vec<usize>>) -> bool {
    match face_arity {
        None => true,
        Some(arity) => arity.contains(&face.len()),
    }
}

#[inline]
fn distinct_edges(faces: &FaceIndex) -> EdgeIndex {
    let mut edge_index: EdgeIndex = faces
        .par_iter()
        .map(|face| {
            face.iter()
                .cycle()
                // Grab two index entries.
                .tuple_windows::<(_, _)>()
                .filter(|t| t.0 < t.1)
                // Create an edge from them.
                .map(|t| vec![*t.0, *t.1])
                .take(face.len())
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect();

    edge_index.into_iter().unique().collect()
}

#[derive(Clone, Debug)]
struct Polyhedron {
    vertices: Vertices,
    //face_arity: Vec<index>,
    face_index: FaceIndex,
}

impl Polyhedron {
    //[ for (f=faces) if(v!=[] && search(v,f)) f ];

    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            face_index: Vec::new(),
        }
    }

    #[inline]
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    pub fn normalize(&mut self) {
        circumscribed_resize(&mut self.vertices, 1.);
    }

    pub fn vertices_to_faces(mesh: &Self) -> FaceIndex {
        let mut fi = FaceIndex::with_capacity(mesh.num_vertices());
        for vertex_number in 0..mesh.num_vertices() as u32 {
            // each old vertex creates a new face, with
            let vertex_faces = vertex_faces(vertex_number, &mesh.face_index);
            // vertex faces in left-hand order
            let mut new_face = Face::new();
            for of in ordered_vertex_faces(vertex_number as u32, &vertex_faces) {
                new_face.push(index_of(&of, &mesh.face_index).unwrap() as Index)
            }
            fi.push(new_face)
        }
        fi
    }

    fn triangulate(&mut self, shortest: bool) {
        self.face_index = self
            .face_index
            .iter()
            .map(|face| match face.len() {
                // Bitriangulate quadrilateral faces
                // use shortest diagonal so triangles are
                // most nearly equilateral.
                4 => {
                    let p = as_points(face, &self.vertices);

                    if shortest == ((p[0] - p[2]).magnitude2() < (p[1] - p[3]).magnitude2()) {
                        vec![
                            vec![face[0], face[1], face[2]],
                            vec![face[0], face[2], face[3]],
                        ]
                    } else {
                        vec![
                            vec![face[1], face[2], face[3]],
                            vec![face[1], face[3], face[0]],
                        ]
                    }
                }
                5 => vec![
                    vec![face[0], face[1], face[4]],
                    vec![face[1], face[2], face[4]],
                    vec![face[4], face[2], face[3]],
                ],
                _ => {
                    let a = face[0];
                    let mut bb = face[1];
                    face.iter()
                        .skip(2)
                        .map(|c| {
                            let b = bb;
                            bb = *c;
                            vec![a, b, *c]
                        })
                        .collect()
                }
                _ => vec![face.clone()],
            })
            .flatten()
            .collect();
    }

    fn ambo(&mut self) {
        let edges = distinct_edges(&self.face_index);

        let vertices = edges
            .par_iter()
            .map(|edge| {
                let edge_points = as_points(edge, &self.vertices);
                (edge, 0.5 * (edge_points[0] + edge_points[1].to_vec()))
            })
            .collect::<Vec<_>>();

        let newids = vertex_ids_ref(&vertices, 0);

        let mut face_index: FaceIndex = self
            .face_index
            .par_iter()
            .map(|face| {
                let edges = distinct_face_edges(face);

                let result = edges
                    .iter()
                    .map(|edge| match vertex(edge, &newids) {
                        Some(index) => vec![index as Index],
                        _ => vec![],
                    })
                    .flatten()
                    .collect::<Vec<_>>();
                result
            })
            .collect::<Vec<_>>();

        let mut new_face_index: FaceIndex = self
            .vertices
            // Each old vertex creates a new face ...
            .par_iter()
            .enumerate()
            .map(|vi| {
                let vi = vi.0 as Index;
                let vf = vertex_faces(vi, &self.face_index);
                ordered_vertex_edges(vi, &vf)
                    .iter()
                    .map(|ve| vertex(&distinct_edge(ve), &newids).unwrap() as Index)
                    .collect::<Vec<_>>()
            })
            .collect();

        face_index.append(&mut new_face_index);

        self.face_index = face_index;
        self.vertices = vertex_values(&vertices);
    }

    pub fn ortho(&mut self) {
        self.join();
    }

    pub fn meta(&mut self) {
        self.kis(0., Some(&vec![3]), false);
        self.join();
    }

    pub fn join(&mut self) {
        self.dual();
        self.ambo();
        self.dual();
    }

    pub fn snub(&mut self) {
        self.dual();
        self.gyro(1. / 3., 0.);
        self.dual();
    }

    pub fn explode(&mut self) {
        self.ambo();
        self.ambo();
    }

    pub fn reflect(&mut self) {
        self.vertices = self
            .vertices
            .par_iter()
            .map(|v| Point::new(v.x, -v.y, v.z))
            .collect();
        self.reverse();
    }

    pub fn reverse(&mut self) {
        self.face_index = self
            .face_index
            .par_iter()
            .map(|f| {
                let mut new_face = f.clone();
                new_face.reverse();
                new_face
            })
            .collect();
    }

    fn dual(&mut self) {
        let new_vertices = self
            .face_index
            .par_iter()
            .map(|face| centroid_ref(&as_points(face, &self.vertices)))
            .collect();
        self.face_index = Self::vertices_to_faces(self);
        self.vertices = new_vertices;
    }

    /// kis – each face with a specified arity n is divided into n
    /// triangles which extend to the face centroid existimg vertices
    /// retained.
    fn kis(&mut self, height: Float, face_arity: Option<&Vec<usize>>, regular: bool) {
        let new_vertices: Vec<(&Face, Point)> = self
            .face_index
            .par_iter()
            .filter(|face| {
                selected_face(face, face_arity) && !regular
                    || ((face_irregularity(face, &self.vertices) - 1.0).abs() < 0.1)
            })
            .map(|face| {
                let fp = as_points(face, &self.vertices);
                (face, centroid_ref(&fp) + face_normal(&fp) * height)
            })
            .collect();

        let newids = vertex_ids_ref(&new_vertices, self.vertices.len() as Index);

        self.vertices.extend(vertex_values_as_ref(&new_vertices));

        self.face_index = self
            .face_index
            .par_iter()
            .map(|f: &Face| match vertex(f, &newids) {
                Some(centroid) => {
                    let mut result = Vec::with_capacity(f.len());
                    for j in 0..f.len() {
                        result.push(vec![f[j], f[(j + 1) % f.len()], centroid as Index]);
                    }
                    result
                }
                None => vec![f.clone()],
            })
            .flatten()
            .collect();
    } // end kis

    fn edges(&self) -> EdgeIndex {
        distinct_edges(&self.face_index)
    }

    fn gyro(&mut self, r: f32 /* 0.3333 */, h: f32) {
        // retain original vertices, add face centroids and directed
        // edge points each N-face becomes N pentagons

        let mut new_vertices: Vec<(&Face, Point)> = self
            .face_index
            .par_iter()
            .map(|face| {
                let fp = as_points(face, &self.vertices);
                (face, centroid_ref(&fp) + face_normal(&fp) * h)
            })
            .collect::<Vec<_>>();

        let edges = self.edges();

        let mut rev_edges: EdgeIndex = edges
            .par_iter()
            .map(|edge| vec![edge[1], edge[0]])
            .collect();

        let new_vertices2 = edges
            .par_iter()
            .enumerate()
            .map(|edge| {
                let ep = as_points(edge.1, &self.vertices);
                // println!("{:?}", ep);
                vec![
                    (edge.1, ep[0] + r * (ep[1] - ep[0])),
                    (&rev_edges[edge.0], ep[1] + r * (ep[0] - ep[1])),
                ]
            })
            .flatten()
            .collect::<Vec<_>>();

        new_vertices.extend(new_vertices2);
        //  2 points per edge

        let newids = vertex_ids_ref(&new_vertices, self.num_vertices() as Index);

        self.vertices.extend(vertex_values_as_ref(&new_vertices));

        let new_face_index = self
            .face_index
            .par_iter()
            .map(|face| {
                let mut new_faces = Vec::new();
                for j in 0..face.len() {
                    let a = face[j];
                    let b = face[(j + 1) % face.len()];
                    let z = face[(j + face.len() - 1) % face.len()];
                    let eab = vertex(&vec![a, b], &newids).unwrap();
                    let eza = vertex(&vec![z, a], &newids).unwrap();
                    let eaz = vertex(&vec![a, z], &newids).unwrap();
                    let centroid = vertex(face, &newids).unwrap();
                    new_faces.push(vec![a, eab, centroid, eza, eaz]);
                }
                new_faces
            })
            .flatten()
            .collect();

        //n!("{:?}", new_face_index);

        self.face_index = new_face_index;
    }

    pub fn normals(&self, normal_type: NormalType) -> Normals {
        match normal_type {
            NormalType::Smooth(angle) => vec![],
            NormalType::Flat => self
                .face_index
                .par_iter()
                .map(|f| {
                    f.iter()
                        // Cycle forever.
                        .cycle()
                        // Start at 3-tuple belonging to the
                        // face's last vertex.
                        .skip(f.len() - 1)
                        // Grab the next three vertex index
                        // entries.
                        .tuple_windows::<(_, _, _)>()
                        // Create a normal from that
                        .map(|t| {
                            -orthogonal(
                                &self.vertices[*t.0 as usize],
                                &self.vertices[*t.1 as usize],
                                &self.vertices[*t.2 as usize],
                            )
                            .normalize()
                        })
                        .take(f.len())
                        .collect::<Normals>()
                })
                .flatten()
                .collect(),
            /*NormalType::Flat => self
            .face_index
            .par_iter()
            .for_each(|f| {
                normals.extend(
                    f.par_iter()
                        // Cycle forever.
                        .cycle()
                        // Start at 3-tuple belonging to the
                        // face's last vertex.
                        .skip(f.len() - 1)
                        // Grab the next three vertex index
                        // entries.
                        .tuple_windows::<(_, _, _)>()
                        // Create a normal from that
                        .for_each(|t| {
                            -orthogonal(
                                &self.vertices[*t.0 as usize],
                                &self.vertices[*t.1 as usize],
                                &self.vertices[*t.2 as usize],
                            )
                            .normalize()
                        })
                        .take(f.len())
                        .collect::<Normals>(),
                );
                face_index.extend(f.par_iter())
            })
            .flatten()
            .collect(),*/
        }
    }

    /*
    pub fn render_with_nsi(&self, ctx: nsi::Context, name: &str) {
        // Create a new mesh node and call it 'dodecahedron'.
        ctx.create(name, nsi::NodeType::Mesh, &[]);

        // Connect the 'dodecahedron' node to the scene's root.
        ctx.connect(name, "", nsi::NodeType::Root, "objects", &[]);

        /*
        let positions: FlatVertices = self
            .vertices
            .into_par_iter()
            .flat_map(|p3| once(p3.x).chain(once(p3.y)).chain(once(p3.z)))
            .collect();
        */

        let positions = unsafe {
            std::slice::from_raw_parts(
                self.vertices.as_ptr().cast::<Float>(),
                3 * self.vertices.len(),
            )
        };

        let face_arity = self
            .face_index
            .par_iter()
            .map(|face| face.len() as u32)
            .collect::<Vec<u32>>();

        let face_index = self.face_index.concat();

        ctx.set_attribute(
            name,
            &[
                nsi::points!("P", positions),
                nsi::unsigneds!("P.indices", &face_index),
                // 5 vertices per each face.
                nsi::unsigneds!("nvertices", &face_arity),
                // Render this as a subdivison surface.
                // nsi::string!("subdivision.scheme", "catmull-clark"),
                // Crease each of our 30 edges a bit.
                //nsi::unsigneds!("subdivision.creasevertices", &face_index),
                //nsi::floats!("subdivision.creasesharpness", &[10.; 30]),
            ],
        );
    }*/

    /*
    function average_normal(fp) =
        let(fl=len(fp))

            let unit_normals = normale(face)
            let(unitns=
                [for(i=[0:fl-1])
                    let(n=orthogonal(fp[i],fp[(i+1)%fl],fp[(i+2)%fl]))
                    let(normn=norm(n))
                    normn==0 ? [] : n/normn
          ]
         )
    vsum(unitns)/len(unitns);*/

    pub fn export_as_obj(&self, destination: &Path, reverse_winding: bool) -> std::io::Result<()> {
        let mut file = File::create(destination)?;

        self.vertices
            .iter()
            .for_each(|vertex| write!(file, "v {} {} {}\n", vertex.x, vertex.y, vertex.z).unwrap());

        match reverse_winding {
            true => self.face_index.iter().for_each(|face| {
                write!(file, "f");
                face.iter()
                    .rev()
                    .for_each(|vertex_index| write!(file, " {}", vertex_index + 1).unwrap());
                write!(file, "\n");
            }),
            false => self.face_index.iter().for_each(|face| {
                write!(file, "f");
                face.iter()
                    .for_each(|vertex_index| write!(file, " {}", vertex_index + 1).unwrap());
                write!(file, "\n");
            }),
        };

        file.flush()?;

        Ok(())
    }

    pub fn tetrahedron() -> Self {
        let c0 = 1.0;

        Self {
            vertices: vec![
                Point::new(c0, c0, c0),
                Point::new(c0, -c0, -c0),
                Point::new(-c0, c0, -c0),
                Point::new(-c0, -c0, c0),
            ],
            face_index: vec![vec![2, 1, 0], vec![3, 2, 0], vec![1, 3, 0], vec![2, 3, 1]],
        }
    }

    pub fn hexahedron() -> Self {
        let c0 = 1.0;

        Self {
            vertices: vec![
                Point::new(c0, c0, c0),
                Point::new(c0, c0, -c0),
                Point::new(c0, -c0, c0),
                Point::new(c0, -c0, -c0),
                Point::new(-c0, c0, c0),
                Point::new(-c0, c0, -c0),
                Point::new(-c0, -c0, c0),
                Point::new(-c0, -c0, -c0),
            ],
            face_index: vec![
                vec![4, 5, 1, 0],
                vec![2, 6, 4, 0],
                vec![1, 3, 2, 0],
                vec![6, 2, 3, 7],
                vec![5, 4, 6, 7],
                vec![3, 1, 5, 7],
            ],
        }
    }

    pub fn octahedron() -> Self {
        let c0 = 0.7071067811865475244008443621048;

        Self {
            vertices: vec![
                Point::new(0.0, 0.0, c0),
                Point::new(0.0, 0.0, -c0),
                Point::new(c0, 0.0, 0.0),
                Point::new(-c0, 0.0, 0.0),
                Point::new(0.0, c0, 0.0),
                Point::new(0.0, -c0, 0.0),
            ],
            face_index: vec![
                vec![4, 2, 0],
                vec![3, 4, 0],
                vec![5, 3, 0],
                vec![2, 5, 0],
                vec![5, 2, 1],
                vec![3, 5, 1],
                vec![4, 3, 1],
                vec![2, 4, 1],
            ],
        }
    }

    pub fn dodecahedron() -> Self {
        let c0 = 0.809016994374947424102293417183;
        let c1 = 1.30901699437494742410229341718;

        Self {
            vertices: vec![
                Point::new(0.0, 0.5, c1),
                Point::new(0.0, 0.5, -c1),
                Point::new(0.0, -0.5, c1),
                Point::new(0.0, -0.5, -c1),
                Point::new(c1, 0.0, 0.5),
                Point::new(c1, 0.0, -0.5),
                Point::new(-c1, 0.0, 0.5),
                Point::new(-c1, 0.0, -0.5),
                Point::new(0.5, c1, 0.0),
                Point::new(0.5, -c1, 0.0),
                Point::new(-0.5, c1, 0.0),
                Point::new(-0.5, -c1, 0.0),
                Point::new(c0, c0, c0),
                Point::new(c0, c0, -c0),
                Point::new(c0, -c0, c0),
                Point::new(c0, -c0, -c0),
                Point::new(-c0, c0, c0),
                Point::new(-c0, c0, -c0),
                Point::new(-c0, -c0, c0),
                Point::new(-c0, -c0, -c0),
            ],
            face_index: vec![
                vec![12, 4, 14, 2, 0],
                vec![16, 10, 8, 12, 0],
                vec![2, 18, 6, 16, 0],
                vec![17, 10, 16, 6, 7],
                vec![19, 3, 1, 17, 7],
                vec![6, 18, 11, 19, 7],
                vec![15, 3, 19, 11, 9],
                vec![14, 4, 5, 15, 9],
                vec![11, 18, 2, 14, 9],
                vec![8, 10, 17, 1, 13],
                vec![5, 4, 12, 8, 13],
                vec![1, 3, 15, 5, 13],
            ],
        }
    }

    pub fn icosahedron() -> Self {
        let c0 = 0.809016994374947424102293417183;

        Self {
            vertices: vec![
                Point::new(0.5, 0.0, c0),
                Point::new(0.5, 0.0, -c0),
                Point::new(-0.5, 0.0, c0),
                Point::new(-0.5, 0.0, -c0),
                Point::new(c0, 0.5, 0.0),
                Point::new(c0, -0.5, 0.0),
                Point::new(-c0, 0.5, 0.0),
                Point::new(-c0, -0.5, 0.0),
                Point::new(0.0, c0, 0.5),
                Point::new(0.0, c0, -0.5),
                Point::new(0.0, -c0, 0.5),
                Point::new(0.0, -c0, -0.5),
            ],
            face_index: vec![
                vec![10, 2, 0],
                vec![5, 10, 0],
                vec![4, 5, 0],
                vec![8, 4, 0],
                vec![2, 8, 0],
                vec![6, 8, 2],
                vec![7, 6, 2],
                vec![10, 7, 2],
                vec![11, 7, 10],
                vec![5, 11, 10],
                vec![1, 11, 5],
                vec![4, 1, 5],
                vec![9, 1, 4],
                vec![8, 9, 4],
                vec![6, 9, 8],
                vec![3, 9, 6],
                vec![7, 3, 6],
                vec![11, 3, 7],
                vec![1, 3, 11],
                vec![9, 3, 1],
            ],
        }
    }
}

/// Struct storing indices corresponding to the vertex
/// Some vertices may not have texcoords or normals, 0 is used to
/// indicate this as OBJ indices begin at 1
#[derive(Hash, Eq, PartialEq, PartialOrd, Ord, Debug, Copy, Clone)]
struct VertexIndex {
    pub position: Index,
    pub texture: Index,
    pub normal: Index,
}

impl From<Polyhedron> for kiss3d::resource::Mesh {
    fn from(mut polyhedron: Polyhedron) -> kiss3d::resource::Mesh {
        polyhedron.reverse();

        /*
        let mut normals_polyhedron = Polyhedron {
            vertices: normals.clone(),
            face_index: {
                let mut index = 0u32;
                polyhedron
                    .face_index
                    .par_iter()
                    .map(|f| {
                        let face =
                            (index..index + f.len() as u32).collect();
                        index += f.len() as u32;
                        face
                    })
                    .collect()
            },
        };

        polyhedron.triangulate(false);
        normals_polyhedron.triangulate(false);

        // We now have two meshes with identical topology but different
        // index arrays. We unify the mapping.
        // FIXME: some vertices will be written to multiple
        let mut normals = vec![
            na::Vector3::new(0.0f32, 0., 0.);
            polyhedron.num_vertices()
        ];

        for f in 0..polyhedron.face_index.len() {
            for i in 0..polyhedron.face_index[f].len() {
                let v = normals_polyhedron.vertices
                    [normals_polyhedron.face_index[f][i] as usize];

                normals[polyhedron.face_index[f][i] as usize] =
                    na::Vector3::new(v.x, v.y, v.z);
            }
        }*/
        polyhedron.triangulate(true);

        let normals = polyhedron
            .normals(NormalType::Flat)
            .par_iter()
            .map(|n| Vector3::new(-n.x, -n.y, -n.z))
            .collect::<Vec<_>>();

        let face_index = (0..normals.len() as u16)
            .step_by(3)
            .map(|i| na::Point3::new(i, i + 1, i + 2))
            .collect::<Vec<_>>();

        Mesh::new(
            // Dupliacate vertices per face so we can
            // match the normals per face.
            polyhedron
                .face_index
                .par_iter()
                .map(|f| {
                    as_points(f, &polyhedron.vertices)
                        .par_iter()
                        .map(|v| na::Point3::<f32>::new(v.x, v.y, v.z))
                        .collect::<Vec<_>>()
                })
                .flatten()
                .collect::<Vec<_>>(),
            face_index,
            Some(normals),
            None,
            false,
        )

        /* smooth shaded mesh
        Mesh::new(
            mesh.vertices
                .par_iter()
                .map(|v| na::Point3::<f32>::new(v.x, v.y, v.z))
                .collect(),
            mesh.face_index
                .par_iter()
                .map(|f| na::Point3::new(f[0] as u16, f[1] as u16, f[2] as u16))
                .collect(),
            None,
            None,
            false,
        )*/
    }
}

use nalgebra as na;

use kiss3d::{
    camera::{ArcBall, FirstPerson},
    event::{Action, Key, Modifiers, WindowEvent},
    light::Light,
    resource::Mesh,
    window::Window,
};

use na::{Point3, UnitQuaternion, Vector3};
use std::{cell::RefCell, rc::Rc};

fn main() {
    let distance = 2.0f32;
    let eye = Point3::new(distance, distance, distance);
    let at = Point3::origin();
    let mut first_person = FirstPerson::new(eye, at);
    let mut arc_ball = ArcBall::new(eye, at);
    let mut use_arc_ball = true;

    let mut window = Window::new("Polyhedron Operations");
    window.set_light(Light::StickToCamera);

    let mut poly = Polyhedron::hexahedron();
    poly.normalize();

    let mesh = Rc::new(RefCell::new(Mesh::from(poly.clone())));
    let mut c = window.add_mesh(mesh, Vector3::new(1.0, 1.0, 1.0));

    c.set_color(0.9, 0.8, 0.7);
    c.enable_backface_culling(false);
    c.set_points_size(10.);

    window.set_light(Light::StickToCamera);
    window.set_framerate_limit(Some(60));

    let mut last_op = 'n';
    let mut last_op_value = 0.;
    let mut alter_last_op = false;
    let mut last_poly = poly.clone();

    let path = dirs::home_dir().unwrap().join("polyhedron.obj");

    println!(
        "Press one of\nA(mbo)\nD(ual)\nE(xplode)\nG(yro)\nJ(oin)\nK(iss)\nM(eta)\nO(rtho)\n(Shft) Up/Down – modify the last operation\nSpace – save"
    );

    while !window.should_close() {
        // rotate the arc-ball camera.
        let curr_yaw = arc_ball.yaw();
        arc_ball.set_yaw(curr_yaw + 0.01);

        // update the current camera.
        for event in window.events().iter() {
            match event.value {
                WindowEvent::Key(key, Action::Release, modifiers) => {
                    match key {
                        Key::Numpad1 => use_arc_ball = true,
                        Key::Numpad2 => use_arc_ball = false,
                        Key::A => {
                            last_poly = poly.clone();
                            poly.ambo();
                            poly.normalize();
                        }
                        Key::D => {
                            last_poly = poly.clone();
                            poly.dual();
                            poly.normalize();
                        }
                        Key::E => {
                            last_poly = poly.clone();
                            poly.explode();
                            poly.normalize();
                        }
                        Key::G => {
                            last_poly = poly.clone();
                            last_op_value = 0.;
                            poly.gyro(1. / 3., last_op_value);
                            last_op = 'g';
                        }
                        Key::J => {
                            last_poly = poly.clone();
                            poly.join();
                            poly.normalize();
                        }
                        Key::K => {
                            last_poly = poly.clone();
                            last_op_value = 0.;
                            poly.kis(last_op_value, None, false);
                            last_op = 'k';
                        }
                        Key::M => {
                            last_poly = poly.clone();
                            poly.meta();
                            poly.normalize();
                        }
                        Key::O => {
                            last_poly = poly.clone();
                            poly.ortho();
                            poly.normalize();
                        }
                        /*Key::T => {
                            if Super == modifiers {
                                return;
                            }
                        }*/
                        Key::S => {
                            last_poly = poly.clone();
                            poly.snub();
                            poly.normalize();
                        }
                        Key::Space => {
                            poly.export_as_obj(&path, true);
                            println!("Exported to {}", path.display());
                        }
                        Key::Up => {
                            alter_last_op = true;
                            if modifiers.intersects(Modifiers::Shift) {
                                last_op_value += 0.1;
                            } else {
                                last_op_value += 0.01;
                            }
                        }
                        Key::Down => {
                            alter_last_op = true;
                            if modifiers.intersects(Modifiers::Shift) {
                                last_op_value -= 0.1;
                            } else {
                                last_op_value -= 0.01;
                            }
                        }
                        Key::Delete => {
                            poly = last_poly.clone();
                        }
                        _ => {
                            break;
                        }
                    };
                    if alter_last_op {
                        alter_last_op = false;
                        match last_op {
                            'g' => {
                                poly = last_poly.clone();
                                poly.gyro(1. / 3., last_op_value);
                            }
                            'k' => {
                                poly = last_poly.clone();
                                poly.kis(last_op_value, None, false);
                            }
                            _ => (),
                        }
                    }
                    c.unlink();
                    let mesh = Rc::new(RefCell::new(Mesh::from(poly.clone())));
                    c = window.add_mesh(mesh, Vector3::new(1.0, 1.0, 1.0));
                    c.set_color(0.9, 0.8, 0.7);
                    c.enable_backface_culling(false);
                    c.set_points_size(10.);
                }
                _ => {}
            }
        }

        /*
        window.draw_line(
            &Point3::origin(),
            &Point3::new(10.0, 0.0, 0.0),
            &Point3::new(10.0, 0.0, 0.0),
        );
        window.draw_line(
            &Point3::origin(),
            &Point3::new(0.0, 10.0, 0.0),
            &Point3::new(0.0, 10.0, 0.0),
        );
        window.draw_line(
            &Point3::origin(),
            &Point3::new(0.0, 0.0, 10.0),
            &Point3::new(0.0, 0.0, 10.0),
        );*/

        if use_arc_ball {
            window.render_with_camera(&mut arc_ball);
        } else {
            window.render_with_camera(&mut first_person);
        }
    }
}
