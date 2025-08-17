use polyhedron_ops::Polyhedron;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OperatorType {
    Ambo,
    Bevel,
    Chamfer,
    Dual,
    Expand,
    Gyro,
    Inset,
    Join,
    Kis,
    Meta,
    Needle,
    Ortho,
    Propellor,
    Quinto,
    Reflect,
    Snub,
    Spherize,
    Truncate,
    Whirl,
    Zip,
}

impl OperatorType {
    pub fn all() -> Vec<Self> {
        vec![
            Self::Ambo,
            Self::Bevel,
            Self::Chamfer,
            Self::Dual,
            Self::Expand,
            Self::Gyro,
            Self::Inset,
            Self::Join,
            Self::Kis,
            Self::Meta,
            Self::Needle,
            Self::Ortho,
            Self::Propellor,
            Self::Quinto,
            Self::Reflect,
            Self::Snub,
            Self::Spherize,
            Self::Truncate,
            Self::Whirl,
            Self::Zip,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Ambo => "Ambo",
            Self::Bevel => "Bevel",
            Self::Chamfer => "Chamfer",
            Self::Dual => "Dual",
            Self::Expand => "Expand",
            Self::Gyro => "Gyro",
            Self::Inset => "Inset",
            Self::Join => "Join",
            Self::Kis => "Kis",
            Self::Meta => "Meta",
            Self::Needle => "Needle",
            Self::Ortho => "Ortho",
            Self::Propellor => "Propellor",
            Self::Quinto => "Quinto",
            Self::Reflect => "Reflect",
            Self::Snub => "Snub",
            Self::Spherize => "Spherize",
            Self::Truncate => "Truncate",
            Self::Whirl => "Whirl",
            Self::Zip => "Zip",
        }
    }

    pub fn letter(&self) -> &'static str {
        match self {
            Self::Ambo => "a",
            Self::Bevel => "b",
            Self::Chamfer => "c",
            Self::Dual => "d",
            Self::Expand => "e",
            Self::Gyro => "g",
            Self::Inset => "I",
            Self::Join => "j",
            Self::Kis => "k",
            Self::Meta => "m",
            Self::Needle => "n",
            Self::Ortho => "o",
            Self::Propellor => "p",
            Self::Quinto => "q",
            Self::Reflect => "r",
            Self::Snub => "s",
            Self::Spherize => "S",
            Self::Truncate => "t",
            Self::Whirl => "w",
            Self::Zip => "z",
        }
    }

    pub fn has_ratio(&self) -> bool {
        matches!(
            self,
            Self::Ambo
                | Self::Bevel
                | Self::Chamfer
                | Self::Gyro
                | Self::Inset
                | Self::Join
                | Self::Meta
                | Self::Ortho
                | Self::Propellor
                | Self::Snub
                | Self::Spherize
                | Self::Truncate
                | Self::Whirl
                | Self::Zip
        )
    }

    pub fn has_height(&self) -> bool {
        matches!(
            self,
            Self::Bevel
                | Self::Kis
                | Self::Meta
                | Self::Quinto
                | Self::Snub
                | Self::Whirl
        )
    }

    pub fn has_regular_faces_param(&self) -> bool {
        matches!(self, Self::Kis | Self::Truncate)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Operator {
    pub op_type: OperatorType,
    pub ratio: Option<f32>,
    pub height: Option<f32>,
    pub nsides: Option<u32>,
    pub regular_faces: bool,
    pub enabled: bool,
}

impl Operator {
    pub fn new(op_type: OperatorType) -> Self {
        Self {
            op_type,
            ratio: if op_type.has_ratio() { Some(0.5) } else { None },
            height: if op_type.has_height() {
                Some(0.3)
            } else {
                None
            },
            nsides: None, // Will be set based on operator type if needed
            regular_faces: false,
            enabled: true,
        }
    }

    pub fn apply(&self, polyhedron: &mut Polyhedron) {
        if !self.enabled {
            return;
        }

        match self.op_type {
            OperatorType::Ambo => {
                polyhedron.ambo(self.ratio, false);
            }
            OperatorType::Bevel => {
                polyhedron.bevel(self.ratio, self.height, None, None, false);
            }
            OperatorType::Chamfer => {
                polyhedron.chamfer(self.ratio, false);
            }
            OperatorType::Dual => {
                polyhedron.dual(false);
            }
            OperatorType::Expand => {
                polyhedron.expand(None, false);
            }
            OperatorType::Gyro => {
                polyhedron.gyro(self.ratio, None, false);
            }
            OperatorType::Inset => {
                polyhedron.inset(self.ratio, None, false);
            }
            OperatorType::Join => {
                polyhedron.join(self.ratio, false);
            }
            OperatorType::Kis => {
                polyhedron.kis(self.height, None, None, None, false);
            }
            OperatorType::Meta => {
                polyhedron.meta(self.ratio, self.height, None, None, false);
            }
            OperatorType::Needle => {
                polyhedron.needle(None, None, None, false);
            }
            OperatorType::Ortho => {
                polyhedron.ortho(self.ratio, false);
            }
            OperatorType::Propellor => {
                polyhedron.propellor(self.ratio, false);
            }
            OperatorType::Quinto => {
                polyhedron.quinto(self.height, false);
            }
            OperatorType::Reflect => {
                polyhedron.reflect(false);
            }
            OperatorType::Snub => {
                polyhedron.snub(self.ratio, self.height, false);
            }
            OperatorType::Spherize => {
                polyhedron.spherize(self.ratio, false);
            }
            OperatorType::Truncate => {
                polyhedron.truncate(self.ratio, None, None, false);
            }
            OperatorType::Whirl => {
                polyhedron.whirl(self.ratio, self.height, false);
            }
            OperatorType::Zip => {
                polyhedron.zip(self.ratio, None, None, false);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BaseShape {
    Tetrahedron,
    Cube,
    Octahedron,
    Dodecahedron,
    Icosahedron,
    Prism(u8),
    Antiprism(u8),
}

impl BaseShape {
    pub fn name(&self) -> String {
        match self {
            Self::Tetrahedron => "Tetrahedron".to_string(),
            Self::Cube => "Cube".to_string(),
            Self::Octahedron => "Octahedron".to_string(),
            Self::Dodecahedron => "Dodecahedron".to_string(),
            Self::Icosahedron => "Icosahedron".to_string(),
            Self::Prism(n) => format!("{}-Prism", n),
            Self::Antiprism(n) => format!("{}-Antiprism", n),
        }
    }

    pub fn create(&self) -> Polyhedron {
        match self {
            Self::Tetrahedron => Polyhedron::tetrahedron(),
            Self::Cube => Polyhedron::hexahedron(),
            Self::Octahedron => Polyhedron::octahedron(),
            Self::Dodecahedron => Polyhedron::dodecahedron(),
            Self::Icosahedron => Polyhedron::icosahedron(),
            Self::Prism(n) => Polyhedron::prism(Some(*n as usize)),
            Self::Antiprism(n) => Polyhedron::antiprism(Some(*n as usize)),
        }
    }
}
