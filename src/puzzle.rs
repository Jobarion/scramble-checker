use std::fmt::{Debug, Display, Formatter};
use std::mem::swap;
use std::str::FromStr;

use itertools::Itertools;

use crate::rotate90;

pub struct CubeAlgorithm(pub Vec<CubeTurn>);

impl FromStr for CubeAlgorithm {
    type Err = std::fmt::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let vec: Result<Vec<CubeTurn>, Self::Err> = s.split_ascii_whitespace()
            .map(CubeTurn::from_str)
            .collect();
        Ok(CubeAlgorithm(vec?))
    }
}

pub struct CubeTurn {
    pub face: usize,
    pub layers: usize,
    pub clockwise_qt: usize
}

impl FromStr for CubeTurn {
    type Err = std::fmt::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut chars = s.chars().peekable();

        let layer_prefix = chars
            .take_while_ref(|x|x.is_ascii_digit())
            .collect::<String>();

        let face = chars.next().and_then(|c| match c {
                'U' => Some(UP),
                'D' => Some(DOWN),
                'F' => Some(FRONT),
                'B' => Some(BACK),
                'L' => Some(LEFT),
                'R' => Some(RIGHT),
                _ => None,
            })
            .ok_or(std::fmt::Error)?;
        let next = match chars.next() {
            Some(next) => next,
            None if !layer_prefix.is_empty() => return Err(std::fmt::Error),
            None => return Ok(CubeTurn {
                face,
                layers: 1,
                clockwise_qt: 1
            })
        };

        let (layers, next) = match (next, layer_prefix.as_str()) {
            ('w', "") => Ok((2usize, chars.next())),
            ('w', x) => usize::from_str(x).map(|x|(x, chars.next())).map_err(|_|std::fmt::Error),
            (_, "") => Ok((1, Some(next))),
            _ => Err(std::fmt::Error)
        }?;

        let qt = match next {
            None => Ok(1),
            Some('\'') => Ok(3),
            Some('2') => Ok(2),
            _ => Err(std::fmt::Error)
        }?;

        Ok(CubeTurn {
            face,
            layers,
            clockwise_qt: qt
        })
    }
}

impl Debug for CubeTurn {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.layers > 2 {
            write!(f, "{}", self.layers)?;
        }
        write!(f, "{}", FACE_NAMES[self.face])?;
        if self.layers > 1 {
            write!(f, "w")?;
        }
        match self.clockwise_qt {
            1 => Ok(()),
            3 => write!(f, "'"),
            n => write!(f, "{n}")
        }
    }
}

impl Display for CubeTurn {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Debug for CubeAlgorithm {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let x: String = self.0.iter()
            .map(|x|x.to_string())
            .intersperse(" ".to_string())
            .collect();
        write!(f, "{x}")
    }
}

impl Display for CubeAlgorithm {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Cube<const N: usize> {
    pub faces: [Face<N>; 6]
}

impl <const N: usize> Cube<N> {
    pub fn new_with_scheme(c: [usize; 6]) -> Self {
        Self {
            faces: [
                Face::<N>::new(c[0]),
                Face::<N>::new(c[1]),
                Face::<N>::new(c[2]),
                Face::<N>::new(c[3]),
                Face::<N>::new(c[4]),
                Face::<N>::new(c[5]),
            ]
        }
    }
}

impl <const N: usize> From<&CubeAlgorithm> for Cube<N> {
    fn from(value: &CubeAlgorithm) -> Self {
        let mut cube = Cube::<N>::default();
        for t in value.0.iter() {
            cube.turn(t);
        }
        cube
    }
}

impl <const N: usize> Default for Cube<N> {
    fn default() -> Self {
        Self::new_with_scheme([0, 1, 2, 3, 4, 5])
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Face<const N: usize> {
    pub stickers: [[Color; N]; N],
}

impl <const N: usize> Face<N> {
    pub fn new(color: usize) -> Self {
        Self {
            stickers: [[color; N]; N],
        }
    }
}

pub const UP: usize = 0;
pub const DOWN: usize = 1;
pub const FRONT: usize = 2;
pub const BACK: usize = 3;
pub const LEFT: usize = 4;
pub const RIGHT: usize = 5;
pub const FACE_NAMES: [char; 6] = ['U', 'D', 'F', 'B', 'L', 'R'];

type Color = usize;

impl <const N: usize> Cube<N> {
    pub fn turn_alg(&mut self, alg: &CubeAlgorithm) {
        for t in alg.0.iter() {
            self.turn(t);
        }
    }

    pub fn turn(&mut self, turn: &CubeTurn) {
        if turn.layers == 0 {
            return
        }
        let qt = turn.clockwise_qt % 4;
        for (ptf, qt) in Self::TURN_TABLE[turn.face] {
            for _ in 0..qt {
                rotate90(&mut self.faces[ptf].stickers)
            }
        }
        let tt = Self::TURN_TABLE[turn.face];
        let idx = [turn.face, tt[0].0, tt[1].0, tt[2].0, tt[3].0];
        let [face, a, b, c, d] = self.faces.get_many_mut(idx).unwrap();
        for _ in 0..qt {
            rotate90(&mut face.stickers);
            for layer in 0..turn.layers {
                swap(a.stickers.get_mut(layer).unwrap(), b.stickers.get_mut(layer).unwrap());
                swap(b.stickers.get_mut(layer).unwrap(), c.stickers.get_mut(layer).unwrap());
                swap(c.stickers.get_mut(layer).unwrap(), d.stickers.get_mut(layer).unwrap());
            }
        }
        for (ptf, qt) in Self::TURN_TABLE[turn.face] {
            let qt = (4 - qt) % 4;
            for _ in 0..qt {
                rotate90(&mut self.faces[ptf].stickers)
            }
        }
        if turn.layers >= N {
            let mut opposite_face = &mut self.faces[turn.face ^ 1];
            for _ in 0..(4 - qt) {
                rotate90(&mut opposite_face.stickers);
            }
        }
    }

    const TURN_TABLE: [[(usize, usize);4]; 6] = [
        [(FRONT, 0), (RIGHT, 0), (BACK, 0), (LEFT, 0)],
        [(FRONT, 2), (LEFT, 2), (BACK, 2), (RIGHT, 2)],
        [(UP, 2), (LEFT, 3), (DOWN, 0), (RIGHT, 1)],
        [(UP, 0), (RIGHT, 3), (DOWN, 2), (LEFT, 1)],
        [(UP, 1), (BACK, 3), (DOWN, 1), (FRONT, 1)],
        [(UP, 3), (FRONT, 3), (DOWN, 3), (BACK, 1)],
    ];
}

pub mod test {
    use std::str::FromStr;

    use crate::puzzle::{Cube, CubeAlgorithm};

    #[test]
    pub fn cube_test() {
        let mut cube = Cube::<3>::default();
        let alg = CubeAlgorithm::from_str("R' U' F").unwrap();
        for t in alg.0.iter() {
            cube.turn(t);
        }
        println!("{cube:?}");
    }
}