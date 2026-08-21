use cuda_device::{kernel, thread, DisjointSlice};

#[kernel]
pub fn relu_fwd(x: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = if x[i] > 0.0 { x[i] } else { 0.0 };
    }
}

fn main() {}
