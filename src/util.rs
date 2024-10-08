use std::time::{Duration, Instant};

pub fn rotate90<T, const N: usize>(slice: &mut[[T; N]; N]) {
    for n in 0..N/2 {
        slice.swap(n, N - n - 1)
    }
    transpose(slice);
}

pub fn transpose<T, const N: usize>(slice: &mut[[T; N]; N]) {
    for n in 0..N {
        for m in 0..n {
            let (x, y) = slice.split_at_mut(n);
            std::mem::swap(&mut y[0][m], &mut x[m][n]);
        }
    }
}


pub struct Stopwatch(Instant);

impl Default for Stopwatch {
    fn default() -> Self {
        Self(Instant::now())
    }
}

impl Stopwatch {
    pub fn elapsed_and_reset(&mut self) -> Duration {
        let elapsed = self.0.elapsed();
        self.reset();
        elapsed
    }

    pub fn reset(&mut self) {
        self.0 = Instant::now()
    }
}