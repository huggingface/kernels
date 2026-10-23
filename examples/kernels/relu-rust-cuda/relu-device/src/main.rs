use cuda_device::{DisjointSlice, kernel, thread};

#[kernel]
pub fn relu(x: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(slot) = out.get_mut(idx) {
        let v = x[i];
        *slot = if v > 0.0 { v } else { 0.0 };
    }
}

fn main() {}
