use crate::*;
use pest::{iterators::Pairs, Parser};
use std::{fmt::Debug, str::FromStr};

#[derive(Parser)]
#[grammar = "grammar.pest"]
struct ConwayPolyhedronNotationParser;

impl TryFrom<&str> for Polyhedron {
    type Error = pest::error::Error<Rule>;

    /// Tries to create a polyhedron from a [Conway Polyedron Notation](https://en.wikipedia.org/wiki/Conway_polyhedron_notation)
    /// string.
    ///
    /// E.g. the string `aD` creates a
    /// [dodecahedron](Polyhedron::dodecahedron())
    /// with an [ambo](Polyhedron::ambo()) operation applied to it. Also known
    /// as an [icosidodecahedron](https://en.wikipedia.org/wiki/Icosidodecahedron).
    /// One of the [Archimedian solids](https://en.wikipedia.org/wiki/Archimedean_solid).
    ///
    /// # Overview
    ///
    /// * All parameters are optional (i.e. have defaults, if absent).
    ///
    /// * Any (number of) parameter(s) can be skipped by using commata (`,`).
    ///
    /// * Whitespace is ignored. This includes tabs, newlines and carriage
    /// returns.
    ///
    /// # Tokens
    ///
    /// **Integers** are written as decimal numbers.
    ///
    /// **Floats** are written as decimal numbers with an optional decimal
    /// point and an optional exponent.
    ///
    /// **Booleans** are written as `{t}` (true) or `{f}` (false).
    ///
    /// Parameter names in the operator list below are prefixed with the
    /// expected type:
    ///
    /// * `b_` – boolean
    /// * `f_` – float
    /// * `uf_` – unsigned float
    /// * `i_` – integer
    /// * `ui_` – unsigned integer
    /// * `[ui_, …]` – array of unsigned integers or single unsigned integer
    ///
    /// ## Platonic Solids
    ///
    /// * `T` – tetrahedron
    /// * `C` – hexahedron (cube)
    /// * `O` – octahedron
    /// * `D` – dodecahedron
    /// * `I` – icosahedron
    ///
    /// ## Prisms & Antiprisms
    ///
    /// * `P` *`ui_number`* – prism with resp. number of sides
    /// * `A` *`ui_number`* – antiprism with resp. number of sides
    ///
    /// ## Operators
    ///
    /// * `a` *`uf_ratio`* – ambo
    /// * `b` *`f_ratio`*, *`f_height`*, *`[ui_vertex_valence_mask, …]`*,
    ///   *`b_regular_faces_only`* – bevel (equiv. to `ta`)
    /// * `c` *`uf_ratio`* – chamfer
    /// * `d` – dual
    /// * `e` *`uf_ratio`* – expand (a.k.a. explode, equiv. to `aa`)
    /// * `g` *`uf_ratio`*, *`f_height`* – gyro
    /// * `i` *`f_offset`* – inset/loft (equiv. to `x,N`)
    /// * `j` *`uf_ratio`* – join (equiv. to `dad`)
    /// * `K` *`ui_iterations`* – planarize (quick & dirty canonicalization)
    /// * `k` *`f_height`*, *`[ui_face_arity_mask, …]`*, *`[ui_face_index_mask,
    ///   ]`*, *`b_regular_faces_only`* – kis
    /// * `M` *`uf_ratio`*, *`f_height`*, *`[ui_vertex_valence_mask, …]`*,
    ///   *`b_regular_faces_only`* – medial (equiv. to `dta`)
    /// * `m` *`uf_ratio`*, *`f_height`*, *`[ui_vertex_valence_mask, …]`*,
    ///   *`b_regular_faces_only`* – meta (equiv. to `k,,3j`)
    /// * `n` *`f_height`*, *`[ui_vertex_valence_mask, …]`*,
    ///   *`b_regular_faces_only`* – needle (equiv. to `dt`)
    /// * `o` *`uf_ratio`* – ortho (equiv. to `jj`)
    /// * `p` *`uf_ratio`* – propellor
    /// * `q` *`f_height`* – quinto
    /// * `r` – reflect
    /// * `S` *`uf_strength`* – spherize
    /// * `s` *`uf_ratio`*, *`f_height`* – snub (equiv. to `dgd`)
    /// * `t` *`f_height`*, *`[ui_vertex_valence_mask, …]`*,
    ///   *`b_regular_faces_only`* – truncate (equiv. to `dkd`)
    /// * `v` – subdivide (Catmull-Clark)
    /// * `w` *`uf_ratio`*, *`f_height`* – whirl
    /// * `x` *`f_height`*, *`f_offset`*, *`[ui_face_arity_mask, …]`* – extrude
    /// * `z` *`f_height`*, *`[ui_vertex_valence_mask, …]`*,
    ///   *`b_regular_faces_only`* – zip (equiv. to `dk`)
    ///
    /// # Examples
    ///
    /// ```
    /// # use polyhedron_ops::Polyhedron;
    /// let polyhedron_from_str =
    ///     Polyhedron::try_from("g0.2k0.1,[3,4],,{t}b,2T").unwrap();
    ///
    /// let polyhedron_from_builder = Polyhedron::tetrahedron()
    ///     .bevel(None, Some(2.0), None, None, true)
    ///     .kis(Some(0.1), Some(&[3, 4]), None, Some(true), true)
    ///     .gyro(Some(0.2), None, true)
    ///     .finalize();
    ///
    /// assert_eq!(polyhedron_from_str.name(), polyhedron_from_builder.name());
    /// ```
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        let mut poly = Polyhedron::new();

        let conway_notation_token_tree = ConwayPolyhedronNotationParser::parse(
            Rule::conway_notation_string,
            s,
        )?;

        // Reverse notation and skip end-of-input token now at the beginning
        // (EOI)
        conway_notation_token_tree.rev().skip(1).for_each(|pair| {
            let token = pair.clone().into_inner();
            match pair.as_rule() {
                Rule::tetrahedron => {
                    poly = Polyhedron::tetrahedron();
                }
                Rule::hexahedron => {
                    poly = Polyhedron::hexahedron();
                }
                Rule::octahedron => {
                    poly = Polyhedron::octahedron();
                }
                Rule::dodecahedron => {
                    poly = Polyhedron::dodecahedron();
                }
                Rule::icosahedron => {
                    poly = Polyhedron::icosahedron();
                }
                Rule::prism => {
                    poly = Polyhedron::prism(to_number(token).0);
                }
                Rule::antiprism => {
                    poly = Polyhedron::antiprism(to_number(token).0);
                }
                Rule::ambo => {
                    poly.ambo(to_number(token).0, true);
                }
                Rule::bevel => {
                    let (ratio, token) = to_number(token);
                    let (height, token) = to_number(token);
                    let (vertex_valence, token) = to_vec(token);
                    let (regular, _) = to_bool(token);
                    poly.bevel(
                        ratio,
                        height,
                        if vertex_valence.is_empty() {
                            None
                        } else {
                            Some(vertex_valence.as_slice())
                        },
                        regular,
                        true,
                    );
                }
                Rule::catmull_clark_subdivide => {
                    poly.catmull_clark_subdivide(true);
                }
                Rule::chamfer => {
                    poly.chamfer(to_number(token).0, true);
                }
                Rule::dual => {
                    poly.dual(true);
                }
                Rule::expand => {
                    poly.expand(to_number(token).0, true);
                }
                Rule::extrude => {
                    let (height, token) = to_number(token);
                    let (offset, token) = to_number(token);
                    let (face_arity_mask, _) = to_vec(token);
                    poly.extrude(
                        height,
                        offset,
                        if face_arity_mask.is_empty() {
                            None
                        } else {
                            Some(face_arity_mask.as_slice())
                        },
                        true,
                    );
                }
                Rule::gyro => {
                    let (ratio, token) = to_number(token);
                    let (height, _) = to_number(token);
                    poly.gyro(ratio, height, true);
                }
                Rule::inset => {
                    let (offset, token) = to_number(token);
                    let (face_arity_mask, _) = to_vec(token);
                    poly.inset(
                        offset,
                        if face_arity_mask.is_empty() {
                            None
                        } else {
                            Some(face_arity_mask.as_slice())
                        },
                        true,
                    );
                }
                Rule::join => {
                    poly.join(to_number(token).0, true);
                }
                Rule::kis => {
                    let (height, token) = to_number(token);
                    let (face_arity_mask, token) = to_vec(token);
                    let (face_index_mask, token) = to_vec(token);
                    let (regular, _) = to_bool(token);
                    poly.kis(
                        height,
                        if face_arity_mask.is_empty() {
                            None
                        } else {
                            Some(face_arity_mask.as_slice())
                        },
                        if face_index_mask.is_empty() {
                            None
                        } else {
                            Some(face_index_mask.as_slice())
                        },
                        regular,
                        true,
                    );
                }
                Rule::medial => {
                    let (ratio, token) = to_number(token);
                    let (height, token) = to_number(token);
                    let (vertex_valence, token) = to_vec(token);
                    let (regular, _) = to_bool(token);
                    poly.medial(
                        ratio,
                        height,
                        if vertex_valence.is_empty() {
                            None
                        } else {
                            Some(vertex_valence.as_slice())
                        },
                        regular,
                        true,
                    );
                }
                Rule::meta => {
                    let (ratio, token) = to_number(token);
                    let (height, token) = to_number(token);
                    let (vertex_valence, token) = to_vec(token);
                    let (regular, _) = to_bool(token);
                    poly.meta(
                        ratio,
                        height,
                        if vertex_valence.is_empty() {
                            None
                        } else {
                            Some(vertex_valence.as_slice())
                        },
                        regular,
                        true,
                    );
                }
                Rule::needle => {
                    let (height, token) = to_number(token);
                    let (vertex_valence, token) = to_vec(token);
                    let (regular, _) = to_bool(token);
                    poly.needle(
                        height,
                        if vertex_valence.is_empty() {
                            None
                        } else {
                            Some(vertex_valence.as_slice())
                        },
                        regular,
                        true,
                    );
                }
                Rule::ortho => {
                    poly.ortho(to_number(token).0, true);
                }
                Rule::planarize => {
                    poly.planarize(to_number(token).0, true);
                }
                Rule::propellor => {
                    poly.propellor(to_number(token).0, true);
                }
                Rule::quinto => {
                    poly.quinto(to_number(token).0, true);
                }
                Rule::reflect => {
                    poly.reflect(true);
                }
                Rule::snub => {
                    let (ratio, token) = to_number(token);
                    let (height, _) = to_number(token);
                    poly.snub(ratio, height, true);
                }
                Rule::spherize => {
                    poly.spherize(to_number(token).0, true);
                }
                Rule::truncate => {
                    let (height, token) = to_number(token);
                    let (vertex_valence_mask, token) = to_vec(token);
                    let (regular, _) = to_bool(token);
                    poly.truncate(
                        height,
                        if vertex_valence_mask.is_empty() {
                            None
                        } else {
                            Some(vertex_valence_mask.as_slice())
                        },
                        regular,
                        true,
                    );
                }
                Rule::whirl => {
                    let (ratio, token) = to_number(token);
                    let (height, _) = to_number(token);
                    poly.whirl(ratio, height, true);
                }
                Rule::zip => {
                    let (height, token) = to_number(token);
                    let (vertex_valence_mask, token) = to_vec(token);
                    let (regular, _) = to_bool(token);
                    poly.zip(
                        height,
                        if vertex_valence_mask.is_empty() {
                            None
                        } else {
                            Some(vertex_valence_mask.as_slice())
                        },
                        regular,
                        true,
                    );
                }
                _ => (),
            }
            poly.normalize();
        });

        Ok(poly)
    }
}

fn is_empty_or_comma(mut tokens: Pairs<'_, Rule>) -> (bool, Pairs<'_, Rule>) {
    // No more tokens? Return None.
    match tokens.clone().next() {
        Some(token) => {
            if Rule::separator == token.as_rule() {
                tokens.next();
                (true, tokens)
            } else {
                (false, tokens)
            }
        }
        None => (true, tokens),
    }
}

fn to_bool(tokens: Pairs<'_, Rule>) -> (Option<bool>, Pairs<'_, Rule>) {
    let (exit, mut tokens) = is_empty_or_comma(tokens);

    if exit {
        return (None, tokens);
    }

    let result = match tokens.next().unwrap().as_str() {
        "{t}" => Some(true),
        "{f}" => Some(false),
        _ => None,
    };

    (result, tokens)
}

fn to_number<T>(tokens: Pairs<'_, Rule>) -> (Option<T>, Pairs<'_, Rule>)
where
    T: FromStr,
    <T as FromStr>::Err: Debug,
{
    let (exit, mut tokens) = is_empty_or_comma(tokens);

    if exit {
        return (None, tokens);
    }

    // Parse the next token as a number.
    let value = tokens.next().unwrap().as_str().parse::<T>().unwrap();

    // Skip possible trailing seprarator.
    tokens.next();

    (Some(value), tokens)
}

fn to_vec<T>(tokens: Pairs<'_, Rule>) -> (Vec<T>, Pairs<'_, Rule>)
where
    T: FromStr,
    <T as FromStr>::Err: Debug,
{
    let (exit, mut tokens) = is_empty_or_comma(tokens);

    if exit {
        return (Vec::new(), tokens);
    }

    let vertex_valence = tokens
        .clone()
        .take_while(|token| Rule::separator != token.as_rule())
        .map(|token| token.as_str().parse::<T>().unwrap())
        .collect::<Vec<_>>();

    if !vertex_valence.is_empty() {
        tokens.next();
        tokens.next();
    }

    // Skip trailing separator.
    tokens.next();

    (vertex_valence, tokens)
}
