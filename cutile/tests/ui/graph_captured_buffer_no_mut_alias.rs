// A buffer a `CudaGraph` writes is mutably borrowed by the graph for its
// whole lifetime: taking another `&mut` to it (here, to enqueue an unrelated
// write) while the graph can still replay must not compile — the write
// could race with a replay in flight.
use cutile::cuda_async::cuda_graph::CudaGraph;
use cutile::prelude::*;

fn main() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let a = api::ones::<f32>(&[4]).sync_on(&stream)?;
    let mut out = api::zeros::<f32>(&[4]).sync_on(&stream)?;
    let mut graph = CudaGraph::scope(&stream, |s| {
        s.record(api::memcpy(&mut out, &a))?;
        Ok(())
    })?;
    api::memcpy(&mut out, &a).sync_on(&stream)?;
    graph.launch().sync()?;
    Ok(())
}
