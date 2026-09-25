/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Scanner idle policy: how long the completion scanner spins over armed slots
//! before falling back to yielding.
//!
//! The scanner must stay hot while completions are imminent — a waker fired
//! promptly is the whole point of the reactor — but a long kernel or a faulted
//! stream leaves it spinning against a set that will not change for a while.
//! A single fixed pass count cannot serve both: tuned for latency it burns a
//! core during long waits, tuned for CPU it adds latency to every fast
//! completion.
//!
//! [`SpinBudget::Adaptive`] tracks which regime the scanner is in. Progress
//! (a completion landed) grows the budget toward `max`, because completions are
//! flowing and spinning keeps latency low. A full budget with no progress
//! halves it toward `min`, because the scanner is waiting on something slow and
//! should stop competing with the threads that feed it.

/// Passes to spin while slots are armed but nothing has landed, before yielding.
#[derive(Clone, Copy, Debug)]
pub enum SpinBudget {
    /// Always spin this many passes, then yield. The pre-existing behavior.
    Fixed { passes: u32 },
    /// Grow toward `max` on progress, halve toward `min` on a fruitless budget.
    Adaptive {
        min: u32,
        max: u32,
        budget: u32,
        step: u32,
    },
}

/// Default fixed budget: the value the reactor shipped with.
pub const DEFAULT_FIXED_PASSES: u32 = 10_000;
/// Adaptive floor: enough passes to catch a completion that is already landing.
pub const ADAPTIVE_MIN: u32 = 64;
/// Adaptive ceiling: never spin longer than the old fixed budget.
pub const ADAPTIVE_MAX: u32 = DEFAULT_FIXED_PASSES;

impl SpinBudget {
    pub fn adaptive() -> Self {
        SpinBudget::Adaptive {
            min: ADAPTIVE_MIN,
            max: ADAPTIVE_MAX,
            budget: ADAPTIVE_MIN,
            step: ADAPTIVE_MIN,
        }
    }

    /// Current number of passes to spin before yielding.
    pub fn limit(&self) -> u32 {
        match *self {
            SpinBudget::Fixed { passes } => passes,
            SpinBudget::Adaptive { budget, .. } => budget,
        }
    }

    /// A completion landed: completions are flowing, stay responsive.
    pub fn on_progress(&mut self) {
        if let SpinBudget::Adaptive {
            max, budget, step, ..
        } = self
        {
            let grown = budget.saturating_add(*step);
            *budget = grown.min(*max);
        }
    }

    /// A full budget expired with nothing landed: assume a slow wait and back
    /// off so the scanner stops competing with the threads feeding it.
    pub fn on_stall(&mut self) {
        if let SpinBudget::Adaptive { min, budget, .. } = self {
            *budget = (*budget / 2).max(*min);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_never_moves() {
        let mut b = SpinBudget::Fixed { passes: 123 };
        assert_eq!(b.limit(), 123);
        b.on_progress();
        assert_eq!(b.limit(), 123);
        b.on_stall();
        assert_eq!(b.limit(), 123);
    }

    #[test]
    fn adaptive_grows_on_progress_and_halves_on_stall() {
        let mut b = SpinBudget::adaptive();
        assert_eq!(b.limit(), ADAPTIVE_MIN);
        b.on_progress();
        assert_eq!(b.limit(), ADAPTIVE_MIN * 2);
        b.on_progress();
        assert_eq!(b.limit(), ADAPTIVE_MIN * 3);
        // Saturates at the ceiling, never above the old fixed budget. The step
        // is `ADAPTIVE_MIN`, so reaching the ceiling takes more than a handful.
        for _ in 0..(ADAPTIVE_MAX / ADAPTIVE_MIN + 1) {
            b.on_progress();
        }
        assert_eq!(b.limit(), ADAPTIVE_MAX);
        b.on_progress();
        assert_eq!(b.limit(), ADAPTIVE_MAX, "growth is capped");
        // Stalls walk it back down, floored at the minimum.
        for _ in 0..64 {
            b.on_stall();
        }
        assert_eq!(b.limit(), ADAPTIVE_MIN);
    }

    #[test]
    fn adaptive_reaches_the_floor_quickly_from_the_ceiling() {
        let mut b = SpinBudget::adaptive();
        for _ in 0..(ADAPTIVE_MAX / ADAPTIVE_MIN + 1) {
            b.on_progress();
        }
        let mut stalls = 0;
        while b.limit() > ADAPTIVE_MIN {
            b.on_stall();
            stalls += 1;
        }
        // Halving from 10_000 to 64 takes 8 steps; a slow wait must back off
        // fast enough to matter for a long kernel.
        assert!(stalls <= 9, "took {stalls} stalls to reach the floor");
    }
}

/// Perf A/B for the idle policy, following the crate's `ab_bench` convention: an
/// ignored test rather than a benchmark target, so the measurement ships next to
/// the policy it justifies.
///
/// Run: `cargo test -p cuda-async --release --lib ab_spin_budget -- --nocapture --ignored`
///
/// Model: a strict ping-pong between a modelled device and the scanner, one
/// completion in flight. The device waits until the previous completion was
/// observed, sleeps `gap`, then writes that slot's flag; the scanner runs the
/// reactor's spin/yield loop and records how much of its budget it burned. Two
/// regimes: `fast` (10 us -- spinning is right) and `slow` (1 ms -- a long
/// kernel, where spinning wastes a core). The metric is spin passes, a
/// deterministic proxy for CPU burn, plus observe latency.
#[cfg(all(test, not(loom), not(miri)))]
mod ab_bench {
    use super::*;
    use crate::slot_table::{FlagArray, SlotTable};
    use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering as O};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    struct Flags(Vec<AtomicU32>);

    impl FlagArray for Flags {
        fn flag(&self, slot: usize) -> &AtomicU32 {
            &self.0[slot]
        }
    }

    // The table takes its flag backend by value; wrapping it in an `Arc` lets
    // the modelled device write the same words the scanner reads.
    impl FlagArray for Arc<Flags> {
        fn flag(&self, slot: usize) -> &AtomicU32 {
            &self.0[slot]
        }
    }

    fn now_ns() -> u64 {
        static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        START.get_or_init(Instant::now).elapsed().as_nanos() as u64
    }

    struct Outcome {
        spins: u64,
        yields: u64,
        lats: Vec<u64>,
    }

    fn run(policy: SpinBudget, ops: usize, gap: Duration) -> Outcome {
        const SLOTS: usize = 256;
        let flags = Arc::new(Flags((0..SLOTS).map(|_| AtomicU32::new(0)).collect()));
        let table = Arc::new(SlotTable::<usize, Arc<Flags>>::new(SLOTS, flags.clone()));
        let queue: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::new(AtomicUsize::new(0));
        let t_land = Arc::new(AtomicU64::new(0));

        // Device: gap-paced, one completion at a time.
        let device = {
            let (f, q, obs, tl) = (
                flags.clone(),
                queue.clone(),
                observed.clone(),
                t_land.clone(),
            );
            thread::spawn(move || {
                for i in 0..ops {
                    let slot = loop {
                        if let Some(&s) = q.lock().unwrap().get(i) {
                            break s;
                        }
                        std::hint::spin_loop();
                    };
                    // Do not overwrite the previous landing's timestamp.
                    while obs.load(O::Acquire) < i {
                        std::hint::spin_loop();
                    }
                    thread::sleep(gap);
                    f.flag(slot).store(1, O::Release);
                    tl.store(now_ns(), O::Release);
                }
            })
        };

        let mut spins = 0u64;
        let mut yields = 0u64;
        let mut lats = Vec::with_capacity(ops);
        let mut budget = policy;
        let mut idle_passes: u32 = 0;
        let mut woken: Vec<usize> = Vec::new();

        for i in 0..ops {
            let slot = loop {
                match table.claim() {
                    Some(s) => break s,
                    None => std::hint::spin_loop(),
                }
            };
            table.reset_flag(slot);
            table.publish(slot, slot);
            queue.lock().unwrap().push(slot);

            // The reactor's scan loop, with this policy.
            loop {
                table.scan_once(&mut woken);
                if !woken.is_empty() {
                    let landed_at = t_land.load(O::Acquire);
                    lats.push(now_ns().saturating_sub(landed_at));
                    woken.clear();
                    budget.on_progress();
                    idle_passes = 0;
                    observed.store(i + 1, O::Release);
                    break;
                }
                idle_passes += 1;
                if idle_passes < budget.limit() {
                    spins += 1;
                    std::hint::spin_loop();
                    continue;
                }
                budget.on_stall();
                yields += 1;
                thread::yield_now();
            }
        }
        device.join().unwrap();
        Outcome {
            spins,
            yields,
            lats,
        }
    }

    fn pct(sorted: &[u64], p: f64) -> u64 {
        if sorted.is_empty() {
            return 0;
        }
        sorted[((sorted.len() - 1) as f64 * p).round() as usize]
    }

    #[test]
    #[ignore = "perf A/B; run explicitly with --release --nocapture --ignored"]
    fn ab_spin_budget() {
        let ops: usize = 200;
        eprintln!("\n| workload | policy | spins | yields | lat p50 | lat p95 |");
        eprintln!("|---|---|---|---|---|---|");
        for (name, gap) in [
            ("fast (10us)", Duration::from_micros(10)),
            ("slow (1ms)", Duration::from_millis(1)),
        ] {
            for (policy_name, policy) in [
                (
                    "fixed 10000",
                    SpinBudget::Fixed {
                        passes: DEFAULT_FIXED_PASSES,
                    },
                ),
                ("adaptive", SpinBudget::adaptive()),
            ] {
                let mut o = run(policy, ops, gap);
                o.lats.sort_unstable();
                eprintln!(
                    "| {name} | {policy_name} | {} | {} | {} ns | {} ns |",
                    o.spins,
                    o.yields,
                    pct(&o.lats, 0.50),
                    pct(&o.lats, 0.95)
                );
            }
        }
    }
}
