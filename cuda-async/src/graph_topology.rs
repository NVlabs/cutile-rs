// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Read-only inspection of a captured CUDA graph.
//!
//! Stream capture records whatever the driver observed, which is not always
//! what the caller intended. A [`Topology`] answers questions about the
//! recorded DAG without exposing the underlying `CUgraph`: how many nodes it
//! has, what kinds of work they represent, how many independent entry points
//! there are, and what the whole thing looks like as a Graphviz drawing.
//!
//! Obtain one from [`CudaGraph::topology`](crate::cuda_graph::CudaGraph::topology).
//! Every handle produced here borrows the graph, so none of them can outlive
//! the [`CudaGraph`](crate::cuda_graph::CudaGraph) that owns it.
//!
//! # Examples
//!
//! Assert that capture recorded only kernel launches, with no host callbacks
//! (which serialize replay) and no memcpys (which usually mean a stray
//! host-to-device copy landed inside the captured region):
//!
//! ```rust,ignore
//! let topology = graph.topology();
//! for node in topology.nodes()? {
//!     assert_eq!(node.kind()?, NodeKind::Kernel);
//! }
//! ```
//!
//! Check that two independent branches were captured as independent, rather
//! than accidentally serialized by a shared stream:
//!
//! ```rust,ignore
//! assert_eq!(graph.topology().root_node_count()?, 2);
//! ```

use crate::error::DeviceError;
use cuda_core::{sys, IntoResult};
use std::collections::BTreeMap;
use std::ffi::CString;
use std::marker::PhantomData;
use std::path::Path;

/// The kind of work a graph node performs.
///
/// Mirrors `CUgraphNodeType`. Values the driver introduces after this crate
/// was built surface as [`NodeKind::Other`] rather than failing, so a newer
/// driver cannot break inspection of an older graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeKind {
    /// A kernel launch.
    Kernel,
    /// A memory copy.
    Memcpy,
    /// A memory set.
    Memset,
    /// A host-side callback. These serialize replay: the GPU waits for the
    /// CPU to run the callback before dependent nodes proceed.
    Host,
    /// A nested child graph.
    ChildGraph,
    /// An empty node, used purely to express dependencies.
    Empty,
    /// A wait on an event.
    WaitEvent,
    /// An event record.
    EventRecord,
    /// An external semaphore signal.
    ExternalSemaphoreSignal,
    /// An external semaphore wait.
    ExternalSemaphoreWait,
    /// A memory allocation owned by the graph.
    MemAlloc,
    /// A free of a graph-owned allocation.
    MemFree,
    /// A batch memory operation.
    BatchMemOp,
    /// A conditional node carrying data-dependent control flow.
    Conditional,
    /// A node type this crate does not know about, carrying the raw
    /// `CUgraphNodeType` value.
    Other(u32),
}

impl NodeKind {
    fn from_raw(raw: sys::CUgraphNodeType) -> Self {
        match raw {
            sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL => Self::Kernel,
            sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMCPY => Self::Memcpy,
            sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMSET => Self::Memset,
            sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_HOST => Self::Host,
            sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_GRAPH => Self::ChildGraph,
            sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_EMPTY => Self::Empty,
            sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_WAIT_EVENT => Self::WaitEvent,
            sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_EVENT_RECORD => Self::EventRecord,
            sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL => {
                Self::ExternalSemaphoreSignal
            }
            sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT => {
                Self::ExternalSemaphoreWait
            }
            sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEM_ALLOC => Self::MemAlloc,
            sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEM_FREE => Self::MemFree,
            sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP => Self::BatchMemOp,
            sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_CONDITIONAL => Self::Conditional,
            other => Self::Other(other),
        }
    }

    /// Whether replay of this node requires the host to make progress.
    ///
    /// Host callbacks block dependent GPU work until the CPU runs them, which
    /// defeats much of the point of capturing a graph. Use this to lint a
    /// captured graph for accidental host participation.
    pub fn blocks_on_host(self) -> bool {
        matches!(self, Self::Host)
    }
}

/// A borrowed view of one node in a captured graph.
///
/// Carries the lifetime of the graph it came from, so it cannot dangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Node<'g> {
    node: sys::CUgraphNode,
    _graph: PhantomData<&'g ()>,
}

impl<'g> Node<'g> {
    fn new(node: sys::CUgraphNode) -> Self {
        Self {
            node,
            _graph: PhantomData,
        }
    }

    /// Returns the kind of work this node performs.
    pub fn kind(&self) -> Result<NodeKind, DeviceError> {
        let mut raw: sys::CUgraphNodeType = 0;
        unsafe { sys::cuGraphNodeGetType(self.node, &mut raw).result() }?;
        Ok(NodeKind::from_raw(raw))
    }
}

// Note: whether a node is enabled is deliberately absent here. The driver
// exposes it as `cuGraphNodeGetEnabled(hGraphExec, hNode, ..)` - it is state
// on the *executable*, not on the graph, so two execs instantiated from one
// graph can disagree. Surfacing it on a graph-borrowed `Node` would imply the
// graph owns it. It belongs on an exec-scoped view instead.

/// A read-only view of a captured graph's structure.
///
/// Created by [`CudaGraph::topology`](crate::cuda_graph::CudaGraph::topology).
/// This borrows the graph, and every [`Node`] it yields borrows it too, so no
/// handle obtained here can outlive the owning graph.
#[derive(Debug, Clone, Copy)]
pub struct Topology<'g> {
    graph: sys::CUgraph,
    _owner: PhantomData<&'g ()>,
}

impl<'g> Topology<'g> {
    /// Wraps a raw graph handle borrowed from its owner.
    ///
    /// # Safety
    ///
    /// `graph` must be a live `CUgraph` that outlives `'g`.
    pub(crate) unsafe fn from_raw(graph: sys::CUgraph) -> Self {
        Self {
            graph,
            _owner: PhantomData,
        }
    }

    /// Returns the number of nodes in the graph.
    ///
    /// Cheaper than [`nodes`](Topology::nodes), which allocates.
    pub fn node_count(&self) -> Result<usize, DeviceError> {
        let mut count: usize = 0;
        unsafe { sys::cuGraphGetNodes(self.graph, std::ptr::null_mut(), &mut count).result() }?;
        Ok(count)
    }

    /// Returns every node in the graph, in no particular order.
    pub fn nodes(&self) -> Result<Vec<Node<'g>>, DeviceError> {
        let count = self.node_count()?;
        Self::collect_nodes(count, |buf, len| unsafe {
            sys::cuGraphGetNodes(self.graph, buf, len).result()
        })
    }

    /// Returns the number of root nodes: nodes with no dependencies.
    ///
    /// This is how many independent entry points replay starts from, and so a
    /// direct measure of whether capture preserved the parallelism you
    /// expressed. A graph you expected to fan out that reports one root was
    /// serialized somewhere.
    pub fn root_node_count(&self) -> Result<usize, DeviceError> {
        let mut count: usize = 0;
        unsafe { sys::cuGraphGetRootNodes(self.graph, std::ptr::null_mut(), &mut count).result() }?;
        Ok(count)
    }

    /// Returns the graph's root nodes: those with no dependencies.
    pub fn root_nodes(&self) -> Result<Vec<Node<'g>>, DeviceError> {
        let count = self.root_node_count()?;
        Self::collect_nodes(count, |buf, len| unsafe {
            sys::cuGraphGetRootNodes(self.graph, buf, len).result()
        })
    }

    /// Returns how many nodes of each kind the graph contains.
    ///
    /// Useful as a structural assertion in tests: a capture that starts
    /// recording an unintended memcpy or host callback shows up as a changed
    /// count without needing to inspect the DAG by hand.
    pub fn kind_counts(&self) -> Result<BTreeMap<NodeKind, usize>, DeviceError> {
        let mut counts = BTreeMap::new();
        for node in self.nodes()? {
            *counts.entry(node.kind()?).or_insert(0) += 1;
        }
        Ok(counts)
    }

    /// Returns whether any node requires the host to make progress during
    /// replay.
    ///
    /// A captured graph containing a host callback serializes against the CPU
    /// on every replay. This is almost always accidental.
    pub fn has_host_nodes(&self) -> Result<bool, DeviceError> {
        for node in self.nodes()? {
            if node.kind()?.blocks_on_host() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Returns the driver's unique identifier for this graph.
    ///
    /// Matches the graph id that Nsight Systems and Nsight Compute report, so
    /// it can be logged to correlate a replay with profiler output.
    pub fn id(&self) -> Result<u32, DeviceError> {
        let mut id: std::os::raw::c_uint = 0;
        unsafe { sys::cuGraphGetId(self.graph, &mut id).result() }?;
        Ok(id)
    }

    /// Writes the graph to `path` as a Graphviz `dot` drawing.
    ///
    /// Render it with, for example, `dot -Tsvg graph.dot -o graph.svg`.
    ///
    /// Pass `verbose` to include node parameters (kernel names, dimensions,
    /// copy sizes) rather than just the DAG shape.
    pub fn write_dot(&self, path: impl AsRef<Path>, verbose: bool) -> Result<(), DeviceError> {
        let path = path.as_ref();
        let text = path.to_str().ok_or_else(|| {
            DeviceError::Internal(format!(
                "graph dot path is not valid UTF-8: {}",
                path.display()
            ))
        })?;
        let c_path = CString::new(text).map_err(|_| {
            DeviceError::Internal(format!(
                "graph dot path contains an interior NUL byte: {}",
                path.display()
            ))
        })?;
        let flags = if verbose {
            sys::CUgraphDebugDot_flags_enum_CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE
        } else {
            0
        };
        unsafe { sys::cuGraphDebugDotPrint(self.graph, c_path.as_ptr(), flags).result() }?;
        Ok(())
    }

    /// Runs the driver's two-call count-then-fill protocol for a node query.
    ///
    /// The driver writes at most `count` entries and reports how many it
    /// actually wrote, which can be fewer if the graph changed between the
    /// two calls. Truncate to what was written rather than trusting `count`.
    fn collect_nodes(
        count: usize,
        fill: impl FnOnce(*mut sys::CUgraphNode, *mut usize) -> Result<(), cuda_core::DriverError>,
    ) -> Result<Vec<Node<'g>>, DeviceError> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut raw: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); count];
        let mut written = count;
        fill(raw.as_mut_ptr(), &mut written)?;
        raw.truncate(written.min(count));
        Ok(raw.into_iter().map(Node::new).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_kind_maps_every_known_driver_value() {
        let cases = [
            (
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL,
                NodeKind::Kernel,
            ),
            (
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMCPY,
                NodeKind::Memcpy,
            ),
            (
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMSET,
                NodeKind::Memset,
            ),
            (
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_HOST,
                NodeKind::Host,
            ),
            (
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_GRAPH,
                NodeKind::ChildGraph,
            ),
            (
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_EMPTY,
                NodeKind::Empty,
            ),
            (
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_WAIT_EVENT,
                NodeKind::WaitEvent,
            ),
            (
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_EVENT_RECORD,
                NodeKind::EventRecord,
            ),
            (
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEM_ALLOC,
                NodeKind::MemAlloc,
            ),
            (
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEM_FREE,
                NodeKind::MemFree,
            ),
            (
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_CONDITIONAL,
                NodeKind::Conditional,
            ),
        ];
        for (raw, expected) in cases {
            assert_eq!(NodeKind::from_raw(raw), expected, "raw value {raw}");
        }
    }

    #[test]
    fn unknown_node_kind_is_preserved_not_lost() {
        // A driver newer than this crate must not break inspection.
        let unknown = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_RESERVED_16;
        assert_eq!(NodeKind::from_raw(unknown), NodeKind::Other(unknown));
    }

    #[test]
    fn only_host_nodes_block_on_host() {
        assert!(NodeKind::Host.blocks_on_host());
        for kind in [
            NodeKind::Kernel,
            NodeKind::Memcpy,
            NodeKind::Memset,
            NodeKind::ChildGraph,
            NodeKind::Empty,
            NodeKind::WaitEvent,
            NodeKind::EventRecord,
            NodeKind::MemAlloc,
            NodeKind::MemFree,
            NodeKind::BatchMemOp,
            NodeKind::Conditional,
            NodeKind::Other(9999),
        ] {
            assert!(!kind.blocks_on_host(), "{kind:?} must not block on host");
        }
    }

    #[test]
    fn zero_node_count_allocates_nothing_and_never_calls_the_driver() {
        let nodes = Topology::collect_nodes(0, |_, _| panic!("driver must not be called")).unwrap();
        assert!(nodes.is_empty());
    }

    #[test]
    fn short_write_truncates_to_what_the_driver_reported() {
        // The driver may write fewer entries than the count from the first
        // call. The tail would otherwise be exposed as null node handles.
        let nodes = Topology::collect_nodes(4, |buf, len| {
            unsafe {
                *len = 2;
                *buf = 1usize as sys::CUgraphNode;
                *buf.add(1) = 2usize as sys::CUgraphNode;
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(nodes.len(), 2);
    }

    #[test]
    fn over_report_is_clamped_to_the_allocation() {
        // A driver reporting more than it was given must not make us read
        // past the buffer we allocated.
        let nodes = Topology::collect_nodes(2, |_, len| {
            unsafe { *len = 99 };
            Ok(())
        })
        .unwrap();
        assert_eq!(nodes.len(), 2);
    }
}
