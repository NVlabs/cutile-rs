// Moving a captured buffer while its `CudaGraph` can still replay must not
// compile: the executable keeps the capture-time device address, and a move
// lets the buffer be dropped (freed) through the new owner.
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
    let stolen = out;
    graph.launch().sync()?;
    let _ = stolen;
    Ok(())
}
