// While an async replay is in flight, the graph's output handle must be
// unreachable: `launch()` borrows the graph mutably for as long as the
// launch (or its future) lives, so `outputs()` — which would hand out the
// buffers the replay is concurrently writing — must not compile.
use cutile::cuda_async::cuda_graph::CudaGraph;
use cutile::prelude::*;
use std::future::IntoFuture;

fn main() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let a = api::ones::<f32>(&[4]).sync_on(&stream)?;
    let out = api::zeros::<f32>(&[4]).sync_on(&stream)?;
    let mut graph = CudaGraph::scope(&stream, move |s| {
        let mut out = out;
        s.record(api::memcpy(&mut out, &a))?;
        Ok(out)
    })?;
    let replay = graph.launch().into_future();
    let _peek = graph.outputs();
    drop(replay);
    Ok(())
}
