// A `CudaGraph` borrows every buffer its captured ops touch for as long as
// it lives: the instantiated executable bakes in their device addresses, so
// dropping a captured buffer and replaying would be a use-after-free.
// Dropping `out` while the graph can still launch must not compile.
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
    drop(out);
    graph.launch().sync()?;
    Ok(())
}
