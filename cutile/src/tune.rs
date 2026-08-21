/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Kernel autotuning: declared search spaces, pluggable samplers, and
//! persisted, provenance-checked results.
//!
//! **Experimental.** This module is gated behind the `experimental-tune`
//! Cargo feature; enabling it opts into an API that may change in breaking
//! ways between releases.
//!
//! The vocabulary follows the tools users already know: a [`Config`] is one
//! candidate configuration (Triton's `Config`), a [`Sampler`] decides the
//! visit order (Optuna's word), [`GridSampler`] is the default exhaustive
//! sampler, and each measured candidate produces a [`Trial`]. Measurement
//! runs through [`crate::bench::do_bench`] (CUDA events, warmup, L2
//! clearing, medians).
//!
//! Principles, in order:
//! - **Explicit opt-in, no magic.** Nothing tunes behind the programmer's
//!   back; the search space is declared, the objective is programmer-written,
//!   and results apply only when a program explicitly loads an artifact.
//! - **Invalid candidates are data.** A candidate rejected by launch checks
//!   or the correctness gate records [`Outcome::Invalid`] with its message;
//!   it never aborts the search.
//! - **Persistence is checked.** The trial log records the tuner's name and
//!   a hash of its search space and refuses to resume from a log that does
//!   not match; the artifact store built on top extends the same discipline
//!   with full provenance (source hash, toolchain fingerprint, arch).

use crate::bench::{do_bench, BenchOptions, Measurement};
use crate::error::Error;
use cuda_core::Stream;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ── Config ──────────────────────────────────────────────────────────────────

/// One candidate configuration: named integer/string parameters.
///
/// Parameters cover both user axes (tile sizes, split counts — read by the
/// launch closure to pick a monomorphization) and compiler knobs (applied by
/// the launch closure via `CompileOptions`). Keeping them in one ordered map
/// makes a `Config` fully serializable for artifacts and logs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Config {
    /// Stable identity within a search space; artifact and log records key
    /// on it. [`Config::new`] derives it from the parameters — construct
    /// through `new` so the two can never disagree.
    pub id: String,
    /// Ordered parameter map.
    pub params: BTreeMap<String, ParamValue>,
}

/// A parameter value: integer or string.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParamValue {
    Int(i64),
    Str(String),
}

impl Config {
    /// Builds a config from parameters, deriving a stable id of the form
    /// `k1=v1,k2=v2` (keys in sorted order).
    pub fn new<I, K>(params: I) -> Self
    where
        I: IntoIterator<Item = (K, ParamValue)>,
        K: Into<String>,
    {
        let params: BTreeMap<String, ParamValue> =
            params.into_iter().map(|(k, v)| (k.into(), v)).collect();
        // String values are JSON-encoded (quoted + escaped) so ids cannot
        // alias: `A=1` (int) differs from `A="1"` (string), and a string
        // containing `,` or `=` cannot imitate additional parameters. Keys
        // holding a separator (or a quote) are JSON-encoded the same way; a
        // raw key can never start with `"`, so the two forms cannot collide.
        let id = params
            .iter()
            .map(|(k, v)| {
                let key = if k.contains(['=', ',', '"']) {
                    serde_json::to_string(k).unwrap_or_else(|_| format!("{k:?}"))
                } else {
                    k.clone()
                };
                match v {
                    ParamValue::Int(i) => format!("{key}={i}"),
                    ParamValue::Str(s) => format!(
                        "{key}={}",
                        serde_json::to_string(s).unwrap_or_else(|_| format!("{s:?}"))
                    ),
                }
            })
            .collect::<Vec<_>>()
            .join(",");
        Self { id, params }
    }

    /// Integer parameter accessor.
    pub fn int(&self, key: &str) -> Option<i64> {
        match self.params.get(key) {
            Some(ParamValue::Int(i)) => Some(*i),
            _ => None,
        }
    }

    /// String parameter accessor.
    pub fn str(&self, key: &str) -> Option<&str> {
        match self.params.get(key) {
            Some(ParamValue::Str(s)) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// Order-independent fingerprint of a candidate set. The trial log and
/// artifacts record it so that resume/apply against a *different* search
/// space is detected instead of silently trusted.
pub fn space_hash(configs: &[Config]) -> String {
    // FNV-1a over the sorted ids; stability matters, cryptography does not.
    let mut ids: Vec<&str> = configs.iter().map(|c| c.id.as_str()).collect();
    ids.sort_unstable();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for id in ids {
        for b in id.as_bytes().iter().chain(&[0u8]) {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    }
    format!("{h:016x}")
}

// ── Trial ───────────────────────────────────────────────────────────────────

/// The result of visiting one candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Trial {
    pub config_id: String,
    pub outcome: Outcome,
}

/// What happened when a candidate was visited.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Outcome {
    /// Correctness gate passed; timing measured.
    Measured {
        median_ms: f32,
        min_ms: f32,
        reps: usize,
    },
    /// Rejected — by a launch check, the correctness gate, or a compile
    /// error. The message is recorded; the search continues.
    Invalid { reason: String },
}

impl Trial {
    /// The median time, if measured.
    pub fn median_ms(&self) -> Option<f32> {
        match &self.outcome {
            Outcome::Measured { median_ms, .. } => Some(*median_ms),
            Outcome::Invalid { .. } => None,
        }
    }
}

// ── Oracle ──────────────────────────────────────────────────────────────────

/// What a [`Sampler`] searches through: a finite candidate list and a way to
/// measure one candidate. The library implements this; users supply the
/// launch and gate closures via [`Autotuner`].
pub trait Oracle {
    /// The declared candidates, pruned, in declaration order.
    fn configs(&self) -> &[Config];
    /// Visits candidate `index`: correctness gate, then timing. Failures
    /// become [`Outcome::Invalid`]; this never panics for a bad candidate.
    fn measure(&mut self, index: usize) -> Trial;
    /// Remaining wall-clock budget, if one was set.
    fn budget_remaining(&self) -> Option<Duration>;
}

// ── Sampler ─────────────────────────────────────────────────────────────────

/// Decides which candidates to visit and in what order.
///
/// Implementations must treat the oracle's budget as authoritative and must
/// tolerate [`Outcome::Invalid`] trials. The library ships [`GridSampler`]
/// (the default); a TPE sampler is planned as an explicit opt-in.
pub trait Sampler {
    /// Runs the search, returning every trial visited (in visit order).
    fn search(&mut self, oracle: &mut dyn Oracle) -> Vec<Trial>;
}

/// Exhaustive sampler: visits every candidate once, in declaration order,
/// skipping candidates whose trials were supplied by [`resume`] and stopping
/// early only when the oracle's budget runs out.
///
/// [`resume`]: GridSampler::resume
#[derive(Default)]
pub struct GridSampler {
    known: Vec<Trial>,
}

impl GridSampler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Seeds already-known trials (e.g. parsed from a previous run's log);
    /// their candidates are not re-measured.
    pub fn resume(mut self, known: Vec<Trial>) -> Self {
        self.known = known;
        self
    }
}

impl Sampler for GridSampler {
    fn search(&mut self, oracle: &mut dyn Oracle) -> Vec<Trial> {
        // Resumed trials count only when (a) their config still exists in the
        // current space — a removed or renamed candidate's history must not
        // decide this search — and (b) they actually measured: an Invalid may
        // have been transient (poisoned context, OOM next door), so it is
        // retried; genuinely invalid candidates fail again cheaply.
        let current: std::collections::BTreeSet<&str> =
            oracle.configs().iter().map(|c| c.id.as_str()).collect();
        let mut trials: Vec<Trial> = std::mem::take(&mut self.known)
            .into_iter()
            .filter(|t| current.contains(t.config_id.as_str()) && t.median_ms().is_some())
            .collect();
        let visited: std::collections::BTreeSet<String> =
            trials.iter().map(|t| t.config_id.clone()).collect();
        let todo: Vec<usize> = (0..oracle.configs().len())
            .filter(|i| !visited.contains(&oracle.configs()[*i].id))
            .collect();
        for index in todo {
            if oracle.budget_remaining() == Some(Duration::ZERO) {
                break;
            }
            trials.push(oracle.measure(index));
        }
        trials
    }
}

/// Selects the winner among trials: the measured candidate with the lowest
/// median. `None` when nothing measured successfully.
pub fn best_config<'a>(configs: &'a [Config], trials: &[Trial]) -> Option<&'a Config> {
    let mut best: Option<(&'a Config, f32)> = None;
    for t in trials {
        let Some(ms) = t.median_ms() else { continue };
        if !ms.is_finite() {
            continue;
        }
        // A trial whose config is no longer in the space (a stale resumed
        // record) must not decide the winner — and must not shadow a valid
        // one either, so it is skipped rather than looked up and lost.
        let Some(config) = configs.iter().find(|c| c.id == t.config_id) else {
            continue;
        };
        if best.is_none_or(|(_, b)| ms < b) {
            best = Some((config, ms));
        }
    }
    best.map(|(c, _)| c)
}

// ── Autotuner ───────────────────────────────────────────────────────────────

/// The assembled tuner for one kernel and one shape class.
///
/// ```rust,ignore
/// let outcome = Autotuner::new("fmha_decode")
///     .configs(configs)
///     .prune(|c| c.int("BN").unwrap() <= pp)
///     .budget(Duration::from_secs(300))
///     .run(&stream, |stream, config| {
///         // launch with `config`, verify against a reference, then return
///         // the closure do_bench will time. Err(..) => Outcome::Invalid.
///         ...
///     })?;
/// ```
/// A candidate-filter predicate (see [`Autotuner::prune`]).
type PrunePredicate = Box<dyn Fn(&Config) -> bool>;

pub struct Autotuner {
    /// Name recorded in the trial log header; a resume against a log written
    /// by a different tuner is refused.
    pub name: String,
    configs: Vec<Config>,
    prune: Vec<PrunePredicate>,
    budget: Option<Duration>,
    bench: BenchOptions,
    log_path: Option<PathBuf>,
}

/// The outcome of a tuning run: all trials plus the winning config.
#[non_exhaustive]
pub struct TuneOutcome {
    pub trials: Vec<Trial>,
    pub best: Option<Config>,
}

impl Autotuner {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            configs: Vec::new(),
            prune: Vec::new(),
            budget: None,
            bench: BenchOptions::default(),
            log_path: None,
        }
    }

    /// Declares the candidate list (Triton's `configs=[...]`).
    pub fn configs(mut self, configs: Vec<Config>) -> Self {
        self.configs = configs;
        self
    }

    /// Filters candidates (Triton's `prune_configs_by`); rejected candidates
    /// are never visited. Predicates are applied when the search runs, so
    /// `.prune(..)` and `.configs(..)` compose in either order.
    pub fn prune(mut self, keep: impl Fn(&Config) -> bool + 'static) -> Self {
        self.prune.push(Box::new(keep));
        self
    }

    /// Wall-clock budget for the whole search.
    pub fn budget(mut self, budget: Duration) -> Self {
        self.budget = Some(budget);
        self
    }

    /// Measurement options ([`BenchOptions`]: `warmup`/`rep` budgets, L2
    /// clearing).
    pub fn bench(mut self, bench: BenchOptions) -> Self {
        self.bench = bench;
        self
    }

    /// Appends every trial to a JSONL log at `path` (created if missing) and
    /// seeds the sampler from any trials already in it — which is what makes
    /// an interrupted exhaustive run resumable.
    pub fn log(mut self, path: impl Into<PathBuf>) -> Self {
        self.log_path = Some(path.into());
        self
    }

    /// Runs the search with [`GridSampler`] (the default sampler), resuming
    /// from the trial log when one is configured.
    ///
    /// `setup` is called once per candidate. It applies the config (picks the
    /// monomorphization, builds `CompileOptions`), runs the programmer's
    /// correctness gate, and returns the closure to be timed — or an error,
    /// which records the candidate as [`Outcome::Invalid`] and moves on.
    ///
    /// After the search, the two best candidates are re-measured
    /// head-to-head with [`crate::bench::do_bench_paired`] and the winner is
    /// decided from that contemporaneous comparison — sequential (or resumed)
    /// medians never pick the winner on their own, since clock and thermal
    /// drift between measurements can exceed the margin between two good
    /// configurations. The runoff is skipped when the budget is already
    /// exhausted; a finalist that fails its runoff setup forfeits to the
    /// other; and if the paired measurement itself fails, the sequential
    /// medians decide.
    pub fn run<S, F>(mut self, stream: &Arc<Stream>, setup: S) -> Result<TuneOutcome, Error>
    where
        S: FnMut(&Arc<Stream>, &Config) -> Result<F, Error>,
        F: FnMut(&Arc<Stream>) -> Result<(), Error>,
    {
        self.apply_prune();
        let mut log = TrialLog::open(
            self.log_path.as_deref(),
            &self.name,
            &space_hash(&self.configs),
        )?;
        let sampler = GridSampler::new().resume(log.existing_trials());
        self.run_sampler(sampler, stream, setup, &mut log)
    }

    /// Runs the search with an explicit [`Sampler`]. The trial log still
    /// records every trial, but resume semantics are the sampler's concern.
    pub fn run_with<S, F>(
        mut self,
        sampler: impl Sampler,
        stream: &Arc<Stream>,
        setup: S,
    ) -> Result<TuneOutcome, Error>
    where
        S: FnMut(&Arc<Stream>, &Config) -> Result<F, Error>,
        F: FnMut(&Arc<Stream>) -> Result<(), Error>,
    {
        self.apply_prune();
        let mut log = TrialLog::open(
            self.log_path.as_deref(),
            &self.name,
            &space_hash(&self.configs),
        )?;
        self.run_sampler(sampler, stream, setup, &mut log)
    }

    fn apply_prune(&mut self) {
        let prune = std::mem::take(&mut self.prune);
        self.configs.retain(|c| prune.iter().all(|keep| keep(c)));
    }

    fn run_sampler<S, F>(
        mut self,
        mut sampler: impl Sampler,
        stream: &Arc<Stream>,
        setup: S,
        log: &mut TrialLog,
    ) -> Result<TuneOutcome, Error>
    where
        S: FnMut(&Arc<Stream>, &Config) -> Result<F, Error>,
        F: FnMut(&Arc<Stream>) -> Result<(), Error>,
    {
        let mut oracle = BenchOracle {
            configs: std::mem::take(&mut self.configs),
            stream: stream.clone(),
            setup,
            bench: self.bench.clone(),
            deadline: self.budget.map(|b| Instant::now() + b),
            log,
        };
        let mut trials = sampler.search(&mut oracle);

        // Paired runoff between the two best. Rationale in run()'s docs. An
        // exhausted budget skips it: the sequential medians decide, and no
        // further GPU work runs.
        let best = match top_two(&oracle.configs, &trials) {
            None => None,
            Some((only, None)) => Some(only.clone()),
            Some((a, Some(b))) => {
                let (a, b) = (a.clone(), b.clone());
                if oracle.budget_remaining() == Some(Duration::ZERO) {
                    Some(a)
                } else {
                    let result = oracle.runoff(&a, &b);
                    Some(runoff_verdict(a, b, result, &mut trials, oracle.log))
                }
            }
        };
        Ok(TuneOutcome { trials, best })
    }
}

/// The two best measured candidates present in `configs`, by median.
fn top_two<'a>(
    configs: &'a [Config],
    trials: &[Trial],
) -> Option<(&'a Config, Option<&'a Config>)> {
    let mut ranked: Vec<(&Config, f32)> = Vec::new();
    for t in trials {
        let Some(ms) = t.median_ms() else { continue };
        if !ms.is_finite() {
            continue;
        }
        if let Some(c) = configs.iter().find(|c| c.id == t.config_id) {
            // Keep the best median per config (runoff entries may duplicate).
            match ranked.iter_mut().find(|(rc, _)| rc.id == c.id) {
                Some(entry) => entry.1 = entry.1.min(ms),
                None => ranked.push((c, ms)),
            }
        }
    }
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut it = ranked.into_iter();
    let first = it.next()?.0;
    Some((first, it.next().map(|(c, _)| c)))
}

/// The library-owned oracle: applies a config via the user's setup closure,
/// then times it with [`do_bench`].
struct BenchOracle<'l, S> {
    configs: Vec<Config>,
    stream: Arc<Stream>,
    setup: S,
    bench: BenchOptions,
    deadline: Option<Instant>,
    log: &'l mut TrialLog,
}

impl<S, F> Oracle for BenchOracle<'_, S>
where
    S: FnMut(&Arc<Stream>, &Config) -> Result<F, Error>,
    F: FnMut(&Arc<Stream>) -> Result<(), Error>,
{
    fn configs(&self) -> &[Config] {
        &self.configs
    }

    fn measure(&mut self, index: usize) -> Trial {
        let config = &self.configs[index];
        let outcome = match (self.setup)(&self.stream, config) {
            Err(e) => Outcome::Invalid {
                reason: e.to_string(),
            },
            Ok(mut f) => match do_bench(&self.stream, &self.bench, |s| f(s)) {
                Err(e) => Outcome::Invalid {
                    reason: e.to_string(),
                },
                Ok(m) => measured(&m),
            },
        };
        let trial = Trial {
            config_id: config.id.clone(),
            outcome,
        };
        self.log.append(&trial);
        trial
    }

    fn budget_remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|d| d.saturating_duration_since(Instant::now()))
    }
}

impl<S, F> BenchOracle<'_, S>
where
    S: FnMut(&Arc<Stream>, &Config) -> Result<F, Error>,
    F: FnMut(&Arc<Stream>) -> Result<(), Error>,
{
    /// Contemporaneous A/B/A/B re-measurement of two finalists.
    fn runoff(
        &mut self,
        a: &Config,
        b: &Config,
    ) -> Result<(Measurement, Measurement), RunoffError> {
        let mut fa = (self.setup)(&self.stream, a).map_err(|error| RunoffError::Setup {
            b_failed: false,
            error,
        })?;
        let mut fb = (self.setup)(&self.stream, b).map_err(|error| RunoffError::Setup {
            b_failed: true,
            error,
        })?;
        crate::bench::do_bench_paired(&self.stream, &self.bench, |s| fa(s), |s| fb(s))
            .map_err(RunoffError::Bench)
    }
}

/// Why a runoff could not produce a verdict.
enum RunoffError {
    /// One finalist failed its runoff setup (compile, launch check, or the
    /// correctness gate): that finalist is invalid now, whatever its earlier
    /// median said.
    Setup { b_failed: bool, error: Error },
    /// The paired measurement failed after both setups succeeded —
    /// environmental, not attributable to either finalist.
    #[allow(dead_code)] // payload kept for debugging; not attributed to a config
    Bench(Error),
}

/// Decides the winner from a runoff attempt. `a` is the sequential leader:
/// it wins ties and unattributable failures. A finalist that fails its own
/// runoff setup forfeits to the other, and the failure is recorded (and
/// logged) against the finalist that actually failed.
fn runoff_verdict(
    a: Config,
    b: Config,
    result: Result<(Measurement, Measurement), RunoffError>,
    trials: &mut Vec<Trial>,
    log: &mut TrialLog,
) -> Config {
    match result {
        Err(RunoffError::Setup { b_failed, error }) => {
            let (loser, winner) = if b_failed { (b, a) } else { (a, b) };
            let t = Trial {
                config_id: loser.id.clone(),
                outcome: Outcome::Invalid {
                    reason: format!("runoff setup failed: {error}"),
                },
            };
            log.append(&t);
            trials.push(t);
            winner
        }
        // Neither finalist is marked Invalid: the failure cannot be pinned
        // on one of them, and the sequential medians remain the recorded
        // decision basis.
        Err(RunoffError::Bench(_)) => a,
        Ok((ma, mb)) => {
            let (oa, ob) = (measured(&ma), measured(&mb));
            for (cfg, o) in [(&a, &oa), (&b, &ob)] {
                let t = Trial {
                    config_id: cfg.id.clone(),
                    outcome: o.clone(),
                };
                log.append(&t);
                trials.push(t);
            }
            // An Invalid or non-finite runoff median can never win.
            let key = |o: &Outcome| match o {
                Outcome::Measured { median_ms, .. } if median_ms.is_finite() => *median_ms,
                _ => f32::INFINITY,
            };
            if key(&oa) <= key(&ob) {
                a
            } else {
                b
            }
        }
    }
}

fn measured(m: &Measurement) -> Outcome {
    if m.reps() == 0 {
        // Reachable via BenchOptions { min_reps: 0 } with a zero budget;
        // median of nothing would panic, and Oracle::measure never panics.
        return Outcome::Invalid {
            reason: "no timed reps (check BenchOptions)".into(),
        };
    }
    Outcome::Measured {
        median_ms: m.median_ms(),
        min_ms: m.min_ms(),
        reps: m.reps(),
    }
}

// ── Trial log (JSONL) ───────────────────────────────────────────────────────

/// Append-only JSONL record of every trial, headed by an identity record;
/// parsing it back is what makes interrupted runs resumable.
///
/// The first line identifies the tuner and its search space. A log whose
/// header does not match the running search is REFUSED — resuming kernel B
/// from kernel A's log (or from a differently-shaped space with coincident
/// parameter names) silently adopts foreign timings, which is exactly the
/// under-keyed-persistence failure this module exists to prevent.
///
/// Because logging is explicitly requested (`.log(path)`), failures to open
/// or head the file are hard errors, not silent no-ops. Per-trial append
/// failures after a successful open are best-effort.
#[derive(Debug)]
struct TrialLog {
    file: Option<std::fs::File>,
    existing: Vec<Trial>,
}

#[derive(Serialize, Deserialize, PartialEq)]
struct LogHeader {
    log_schema: u32,
    tuner: String,
    space: String,
}

impl TrialLog {
    fn open(path: Option<&Path>, tuner: &str, space: &str) -> Result<Self, Error> {
        let Some(path) = path else {
            return Ok(Self {
                file: None,
                existing: Vec::new(),
            });
        };
        let expected = LogHeader {
            log_schema: 1,
            tuner: tuner.to_string(),
            space: space.to_string(),
        };
        let mut existing = Vec::new();
        let mut needs_newline = false;
        // Fresh means "no valid header on disk": a missing file, an empty
        // file, or a whitespace-only one. Fresh logs are truncated and headed;
        // anything else must present a matching header or be refused.
        let mut fresh = true;
        match std::fs::read_to_string(path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(crate::error::tensor_error(&format!(
                    "trial log {} is unreadable: {e}",
                    path.display()
                )));
            }
            Ok(contents) if contents.trim().is_empty() => {}
            Ok(contents) => {
                let mut lines = contents.lines();
                let header: LogHeader = lines
                    .next()
                    .and_then(|l| serde_json::from_str(l).ok())
                    .ok_or_else(|| {
                        crate::error::tensor_error(&format!(
                            "trial log {} has no valid header; delete it or point .log() elsewhere",
                            path.display()
                        ))
                    })?;
                if header != expected {
                    return Err(crate::error::tensor_error(&format!(
                        "trial log {} belongs to tuner {:?} (space {}), not {:?} (space {}); delete it or point .log() elsewhere",
                        path.display(),
                        header.tuner,
                        header.space,
                        expected.tuner,
                        expected.space,
                    )));
                }
                existing = lines
                    .filter_map(|l| serde_json::from_str::<Trial>(l).ok())
                    .collect();
                // A crash mid-write can leave a torn final line without a
                // newline; appending directly would corrupt the next record.
                needs_newline = !contents.ends_with('\n');
                fresh = false;
            }
        }
        let mut opts = std::fs::OpenOptions::new();
        if fresh {
            // Truncate so stray whitespace can't precede the header.
            opts.create(true).write(true).truncate(true);
        } else {
            opts.append(true);
        }
        let mut file = opts.open(path).map_err(|e| {
            crate::error::tensor_error(&format!(
                "trial log {} cannot be opened for append: {e}",
                path.display()
            ))
        })?;
        if needs_newline {
            let _ = writeln!(file);
        }
        // Head a fresh log with the identity record.
        if fresh {
            if let Ok(line) = serde_json::to_string(&expected) {
                let _ = writeln!(file, "{line}");
            }
        }
        Ok(Self {
            file: Some(file),
            existing,
        })
    }

    fn existing_trials(&self) -> Vec<Trial> {
        self.existing.clone()
    }

    fn append(&mut self, trial: &Trial) {
        if let (Some(file), Ok(line)) = (self.file.as_mut(), serde_json::to_string(trial)) {
            let _ = writeln!(file, "{line}");
        }
    }
}

// ── Artifact ────────────────────────────────────────────────────────────────

/// Artifact record-format version; bump on breaking changes.
const ARTIFACT_SCHEMA: u32 = 1;

/// A persisted, provenance-checked record of tuning winners: one entry per
/// shape-class bucket, serialized as human-diffable pretty JSON intended to
/// be committed next to the code it tunes.
///
/// Staleness is enforced at load, not documented: [`Artifact::load_verified`]
/// refuses entries whose provenance no longer matches the running workspace,
/// so a stale winner cannot silently apply. The strong check is the stored
/// winner's persistent-cache key: recomputed via
/// `Specialization::l2_cache_key()` (or `KernelCompiler::l2_cache_key()`),
/// it covers the serialized bytecode — dependencies included — and the
/// toolchain fingerprint, closing the known gap of `source_hash` (which
/// covers only the kernel's own module).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// Record-format version; bump on breaking changes.
    pub schema: u32,
    /// Kernel (or tuner) name this artifact belongs to.
    pub kernel: String,
    /// The kernel module's `_SOURCE_HASH` at tune time.
    pub source_hash: String,
    /// cutile crate version at tune time.
    pub cutile_version: String,
    /// `tileiras --version` fingerprint at tune time.
    pub tileiras_fingerprint: String,
    /// Target architecture (e.g. `sm_120`). Artifacts are per-arch; loading
    /// on a different arch is refused, never approximated.
    pub arch: String,
    /// Hostname the tuning ran on (informational).
    pub machine: String,
    /// Unix seconds at creation (informational).
    pub created_unix_secs: u64,
    /// Fingerprint of the candidate set the winners were chosen from.
    #[serde(default)]
    pub space_hash: Option<String>,
    /// Free-form revision tag for the correctness gate / objective; drift is
    /// surfaced as a warning (a stronger gate may invalidate old winners).
    #[serde(default)]
    pub gate: Option<String>,
    /// Winners, one per bucket.
    pub entries: Vec<ArtifactEntry>,
}

/// One bucket's winner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactEntry {
    /// Shape-class bucket this winner applies to (e.g. `"tg<=512"`).
    pub bucket: String,
    /// The winning configuration.
    pub config: Config,
    /// Its measured median, milliseconds.
    pub median_ms: f32,
    /// Timed reps behind the median.
    pub samples: usize,
    /// The winner's persistent-cache (L2) key at tune time, when the caller
    /// recorded one. Enables the strong staleness check at load.
    pub l2_key: Option<String>,
}

/// The provenance the loader checks an artifact against.
pub struct Workspace {
    /// Kernel/tuner name the caller is loading FOR. Two kernels in one
    /// module share a source hash, so this is its own refusal axis.
    pub kernel: String,
    pub source_hash: String,
    pub arch: String,
    pub tileiras_fingerprint: String,
    /// Fingerprint of the CURRENT candidate set (see [`space_hash`]); when
    /// both sides carry one, a mismatch refuses — a winner chosen from a
    /// different space is not a winner here.
    pub space_hash: Option<String>,
}

impl Artifact {
    /// Starts an artifact with the given provenance; machine name and
    /// timestamp are captured from the environment.
    pub fn new(ws: &Workspace) -> Self {
        Self {
            schema: ARTIFACT_SCHEMA,
            kernel: ws.kernel.clone(),
            source_hash: ws.source_hash.clone(),
            cutile_version: env!("CARGO_PKG_VERSION").to_string(),
            tileiras_fingerprint: ws.tileiras_fingerprint.clone(),
            arch: ws.arch.clone(),
            machine: hostname(),
            created_unix_secs: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            space_hash: ws.space_hash.clone(),
            gate: None,
            entries: Vec::new(),
        }
    }

    /// Inserts or replaces the winner for `bucket`. Replacement is in place,
    /// preserving entry order (committed artifacts should diff cleanly).
    pub fn insert(&mut self, entry: ArtifactEntry) {
        match self.entries.iter_mut().find(|e| e.bucket == entry.bucket) {
            Some(slot) => *slot = entry,
            None => self.entries.push(entry),
        }
    }

    /// The winner for `bucket`, if recorded.
    pub fn get(&self, bucket: &str) -> Option<&ArtifactEntry> {
        self.entries.iter().find(|e| e.bucket == bucket)
    }

    /// Writes pretty JSON (stable field order — committed artifacts should
    /// diff cleanly).
    pub fn save(&self, path: &Path) -> Result<(), Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| crate::error::tensor_error(&format!("artifact serialize: {e}")))?;
        std::fs::write(path, json)
            .map_err(|e| crate::error::tensor_error(&format!("artifact write: {e}")))
    }

    /// Loads without verification. Prefer [`load_verified`](Self::load_verified)
    /// anywhere the entries will actually be applied.
    pub fn load(path: &Path) -> Result<Self, Error> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| crate::error::tensor_error(&format!("artifact read: {e}")))?;
        serde_json::from_str(&contents)
            .map_err(|e| crate::error::tensor_error(&format!("artifact parse: {e}")))
    }

    /// Loads and verifies against the running workspace.
    ///
    /// REFUSES (errors) on: schema, kernel, arch, or `source_hash` mismatch;
    /// a `space_hash` mismatch when both sides carry one; duplicate buckets;
    /// and — when the toolchain fingerprints MATCH — a stored winner
    /// `l2_key` that differs from the recomputed one (the strong,
    /// dependency-inclusive check).
    ///
    /// WARNS (returned, never silent) on: toolchain-fingerprint drift (the
    /// stored l2 keys embed the old fingerprint, so recomputation cannot
    /// match and is skipped — configs remain valid, timings may not); gate
    /// tag drift; entries without a stored key; and entries whose key the
    /// verifier declined to recompute (`Ok(None)`).
    ///
    /// `verify_l2` receives each keyed entry and returns the key the CURRENT
    /// workspace derives for its config — typically
    /// `launcher.specialize()?.l2_cache_key()` or
    /// `KernelCompiler::...l2_cache_key()`.
    pub fn load_verified(
        path: &Path,
        ws: &Workspace,
        mut verify_l2: impl FnMut(&ArtifactEntry) -> Result<Option<String>, Error>,
    ) -> Result<(Self, Vec<String>), Error> {
        let artifact = Self::load(path)?;
        let refuse = |what: &str, stored: &str, current: &str| {
            Err(crate::error::tensor_error(&format!(
                "stale tuning artifact at {}: {what} mismatch (artifact: {stored}, workspace: {current}); re-tune or delete it",
                path.display(),
            )))
        };
        if artifact.schema != ARTIFACT_SCHEMA {
            return refuse(
                "schema",
                &artifact.schema.to_string(),
                &ARTIFACT_SCHEMA.to_string(),
            );
        }
        if artifact.kernel != ws.kernel {
            return refuse("kernel", &artifact.kernel, &ws.kernel);
        }
        if artifact.arch != ws.arch {
            return refuse("arch", &artifact.arch, &ws.arch);
        }
        if artifact.source_hash != ws.source_hash {
            return refuse("source_hash", &artifact.source_hash, &ws.source_hash);
        }
        if let (Some(stored), Some(current)) = (&artifact.space_hash, &ws.space_hash) {
            if stored != current {
                return refuse("space_hash", stored, current);
            }
        }
        {
            let mut seen = std::collections::BTreeSet::new();
            for e in &artifact.entries {
                if !seen.insert(e.bucket.as_str()) {
                    return Err(crate::error::tensor_error(&format!(
                        "tuning artifact at {} has duplicate entries for bucket {:?}; fix or re-tune it",
                        path.display(),
                        e.bucket,
                    )));
                }
                // A config's id is derived from its params; a stored entry
                // where the two disagree has been hand-edited or corrupted,
                // and downstream consumers key on the id.
                let derived = Config::new(e.config.params.clone()).id;
                if e.config.id != derived {
                    return refuse(
                        &format!("config id for bucket {:?}", e.bucket),
                        &e.config.id,
                        &derived,
                    );
                }
            }
        }

        let mut warnings = Vec::new();
        if artifact.space_hash.is_none() && ws.space_hash.is_some() {
            // The workspace records a space hash, so an artifact without one
            // predates the field or has had it stripped; either way the
            // same-space check silently cannot run.
            warnings.push(
                "tuning artifact carries no space_hash; the search-space match was not checked"
                    .to_string(),
            );
        }
        let fingerprint_matches = artifact.tileiras_fingerprint == ws.tileiras_fingerprint;
        if !fingerprint_matches {
            // The stored keys embed the old fingerprint: recomputing under
            // the new toolchain CANNOT match, so the strong check is skipped
            // rather than misread as staleness. Ordering matters here.
            warnings.push(format!(
                "tuning artifact was produced by a different tileiras ({} vs {}); configs remain valid but timings may have shifted and per-entry key verification was skipped — consider re-tuning",
                artifact.tileiras_fingerprint, ws.tileiras_fingerprint,
            ));
        } else {
            for entry in &artifact.entries {
                match &entry.l2_key {
                    None => warnings.push(format!(
                        "bucket {:?} carries no l2 key; only source-level staleness checks applied",
                        entry.bucket
                    )),
                    Some(stored) => match verify_l2(entry)? {
                        None => warnings.push(format!(
                            "bucket {:?}: verifier declined to recompute the l2 key; stored key not checked",
                            entry.bucket
                        )),
                        Some(current) => {
                            if &current != stored {
                                return refuse(
                                    &format!("l2 key for bucket {:?}", entry.bucket),
                                    stored,
                                    &current,
                                );
                            }
                        }
                    },
                }
            }
        }
        if artifact.cutile_version != env!("CARGO_PKG_VERSION") {
            warnings.push(format!(
                "tuning artifact was produced by cutile {} (running {})",
                artifact.cutile_version,
                env!("CARGO_PKG_VERSION"),
            ));
        }
        Ok((artifact, warnings))
    }
}

fn hostname() -> String {
    std::fs::read_to_string("/etc/hostname")
        .map(|s| s.trim().to_string())
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok())
        .unwrap_or_else(|| "unknown".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(bn: i64, splits: i64) -> Config {
        Config::new([
            ("BN", ParamValue::Int(bn)),
            ("SPLITS", ParamValue::Int(splits)),
        ])
    }

    struct FakeOracle {
        configs: Vec<Config>,
        cost: fn(&Config) -> Option<f32>,
        measured: Vec<String>,
        budget: Option<Duration>,
    }

    impl Oracle for FakeOracle {
        fn configs(&self) -> &[Config] {
            &self.configs
        }
        fn measure(&mut self, index: usize) -> Trial {
            let c = &self.configs[index];
            self.measured.push(c.id.clone());
            let outcome = match (self.cost)(c) {
                Some(ms) => Outcome::Measured {
                    median_ms: ms,
                    min_ms: ms,
                    reps: 3,
                },
                None => Outcome::Invalid {
                    reason: "gate failed".into(),
                },
            };
            Trial {
                config_id: c.id.clone(),
                outcome,
            }
        }
        fn budget_remaining(&self) -> Option<Duration> {
            self.budget
        }
    }

    #[test]
    fn config_ids_are_stable_and_param_order_independent() {
        let a = Config::new([("SPLITS", ParamValue::Int(8)), ("BN", ParamValue::Int(64))]);
        let b = cfg(64, 8);
        assert_eq!(a.id, b.id);
        assert_eq!(a.id, "BN=64,SPLITS=8");
        assert_eq!(a.int("BN"), Some(64));
        assert_eq!(a.int("missing"), None);
    }

    #[test]
    fn grid_sampler_visits_everything_once_and_picks_best() {
        let configs = vec![cfg(32, 2), cfg(64, 4), cfg(128, 8)];
        let mut oracle = FakeOracle {
            configs: configs.clone(),
            cost: |c| Some(c.int("BN").unwrap() as f32), // smaller BN = faster
            measured: Vec::new(),
            budget: None,
        };
        let trials = GridSampler::new().search(&mut oracle);
        assert_eq!(oracle.measured.len(), 3);
        assert_eq!(trials.len(), 3);
        let best = best_config(&configs, &trials).unwrap();
        assert_eq!(best.int("BN"), Some(32));
    }

    #[test]
    fn invalid_candidates_are_recorded_not_fatal() {
        let configs = vec![cfg(32, 2), cfg(64, 4)];
        let mut oracle = FakeOracle {
            configs: configs.clone(),
            cost: |c| (c.int("BN") != Some(32)).then_some(1.0), // 32 invalid
            measured: Vec::new(),
            budget: None,
        };
        let trials = GridSampler::new().search(&mut oracle);
        assert_eq!(trials.len(), 2);
        assert!(matches!(trials[0].outcome, Outcome::Invalid { .. }));
        let best = best_config(&configs, &trials).unwrap();
        assert_eq!(best.int("BN"), Some(64), "invalid one never wins");
    }

    #[test]
    fn resume_skips_known_trials() {
        let configs = vec![cfg(32, 2), cfg(64, 4), cfg(128, 8)];
        let known = vec![Trial {
            config_id: configs[1].id.clone(),
            outcome: Outcome::Measured {
                median_ms: 0.5,
                min_ms: 0.5,
                reps: 3,
            },
        }];
        let mut oracle = FakeOracle {
            configs: configs.clone(),
            cost: |_| Some(9.0),
            measured: Vec::new(),
            budget: None,
        };
        let trials = GridSampler::new().resume(known).search(&mut oracle);
        assert_eq!(oracle.measured.len(), 2, "known candidate not re-measured");
        assert_eq!(trials.len(), 3, "known trial still in the result set");
        let best = best_config(&configs, &trials).unwrap();
        assert_eq!(best.int("BN"), Some(64), "resumed trial can win");
    }

    #[test]
    fn exhausted_budget_stops_the_search() {
        let configs = vec![cfg(32, 2), cfg(64, 4), cfg(128, 8)];
        let mut oracle = FakeOracle {
            configs,
            cost: |_| Some(1.0),
            measured: Vec::new(),
            budget: Some(Duration::ZERO),
        };
        let trials = GridSampler::new().search(&mut oracle);
        assert!(trials.is_empty(), "zero budget measures nothing");
    }

    fn ws() -> Workspace {
        Workspace {
            kernel: "fmha_decode".into(),
            source_hash: "abc123".into(),
            arch: "sm_120".into(),
            tileiras_fingerprint: "release 13.3, V13.3.36".into(),
            space_hash: None,
        }
    }

    fn artifact_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "cutile_artifact_{label}_{}.json",
            std::process::id()
        ))
    }

    #[test]
    fn artifact_roundtrips_and_verifies() {
        let path = artifact_path("roundtrip");
        let mut a = Artifact::new(&ws());
        a.insert(ArtifactEntry {
            bucket: "tg<=512".into(),
            config: cfg(64, 8),
            median_ms: 1.25,
            samples: 12,
            l2_key: Some("f".repeat(64)),
        });
        a.save(&path).unwrap();

        let (loaded, warnings) =
            Artifact::load_verified(&path, &ws(), |e| Ok(e.l2_key.clone())).unwrap();
        assert!(warnings.is_empty());
        let entry = loaded.get("tg<=512").unwrap();
        assert_eq!(entry.config.int("BN"), Some(64));
        assert_eq!(entry.samples, 12);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn artifact_refuses_source_hash_and_arch_mismatch() {
        let path = artifact_path("refuse");
        Artifact::new(&ws()).save(&path).unwrap();

        let mut other = ws();
        other.source_hash = "different".into();
        let err = Artifact::load_verified(&path, &other, |_| Ok(None)).unwrap_err();
        assert!(err.to_string().contains("source_hash mismatch"));
        assert!(err.to_string().contains("re-tune"));

        let mut other = ws();
        other.arch = "sm_100".into();
        let err = Artifact::load_verified(&path, &other, |_| Ok(None)).unwrap_err();
        assert!(err.to_string().contains("arch mismatch"));

        let mut other = ws();
        other.kernel = "other_kernel".into();
        let err = Artifact::load_verified(&path, &other, |_| Ok(None)).unwrap_err();
        assert!(err.to_string().contains("kernel mismatch"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn artifact_refuses_space_mismatch_and_duplicate_buckets() {
        let path = artifact_path("space");
        let mut with_space = ws();
        with_space.space_hash = Some(space_hash(&[cfg(64, 8), cfg(128, 8)]));
        Artifact::new(&with_space).save(&path).unwrap();

        // Same kernel, different candidate set: refused when both carry one.
        let mut other = ws();
        other.space_hash = Some(space_hash(&[cfg(64, 8)]));
        let err = Artifact::load_verified(&path, &other, |_| Ok(None)).unwrap_err();
        assert!(err.to_string().contains("space_hash mismatch"));
        // Loader without an expectation: accepted (no false refusals).
        let (_, _) = Artifact::load_verified(&path, &ws(), |_| Ok(None)).unwrap();

        // Duplicate buckets (hand-edited/merge-resolved artifact): refused.
        let mut dup = Artifact::new(&ws());
        for _ in 0..2 {
            dup.entries.push(ArtifactEntry {
                bucket: "b".into(),
                config: cfg(64, 8),
                median_ms: 1.0,
                samples: 3,
                l2_key: None,
            });
        }
        dup.save(&path).unwrap();
        let err = Artifact::load_verified(&path, &ws(), |_| Ok(None)).unwrap_err();
        assert!(err.to_string().contains("duplicate entries for bucket"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn artifact_refuses_l2_key_drift_and_warns_on_fingerprint_drift() {
        let path = artifact_path("l2");
        let mut a = Artifact::new(&ws());
        a.insert(ArtifactEntry {
            bucket: "b".into(),
            config: cfg(64, 8),
            median_ms: 1.0,
            samples: 5,
            l2_key: Some("a".repeat(64)),
        });
        a.save(&path).unwrap();

        // Recomputed key differs => refuse (the dependency-inclusive check).
        let err = Artifact::load_verified(&path, &ws(), |_| Ok(Some("b".repeat(64)))).unwrap_err();
        assert!(err.to_string().contains("l2 key for bucket"));

        // Verifier declines (None) => provenance fields alone decide; a
        // fingerprint drift is a warning, not a refusal.
        let mut drifted = ws();
        drifted.tileiras_fingerprint = "release 13.4, V13.4.1".into();
        let (_, warnings) = Artifact::load_verified(&path, &drifted, |_| Ok(None)).unwrap();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("different tileiras"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn artifact_insert_replaces_bucket_winner() {
        let mut a = Artifact::new(&ws());
        for (bn, ms) in [(64, 2.0), (128, 1.0)] {
            a.insert(ArtifactEntry {
                bucket: "b".into(),
                config: cfg(bn, 4),
                median_ms: ms,
                samples: 3,
                l2_key: None,
            });
        }
        assert_eq!(a.entries.len(), 1, "one winner per bucket");
        assert_eq!(a.get("b").unwrap().config.int("BN"), Some(128));
    }

    #[test]
    fn artifact_refuses_schema_and_id_param_mismatch() {
        let path = artifact_path("schema");
        let mut a = Artifact::new(&ws());
        a.schema = ARTIFACT_SCHEMA + 1;
        a.save(&path).unwrap();
        let err = Artifact::load_verified(&path, &ws(), |_| Ok(None)).unwrap_err();
        assert!(err.to_string().contains("schema mismatch"));

        // A hand-edited entry whose id disagrees with its params: refused.
        let mut a = Artifact::new(&ws());
        let mut config = cfg(64, 8);
        config.id = "BN=128,SPLITS=8".into();
        a.insert(ArtifactEntry {
            bucket: "b".into(),
            config,
            median_ms: 1.0,
            samples: 3,
            l2_key: None,
        });
        a.save(&path).unwrap();
        let err = Artifact::load_verified(&path, &ws(), |_| Ok(None)).unwrap_err();
        assert!(err.to_string().contains("config id for bucket"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn artifact_without_space_hash_warns_when_workspace_expects_one() {
        let path = artifact_path("nospace");
        Artifact::new(&ws()).save(&path).unwrap(); // artifact: space_hash None
        let mut expecting = ws();
        expecting.space_hash = Some(space_hash(&[cfg(64, 8)]));
        let (_, warnings) = Artifact::load_verified(&path, &expecting, |_| Ok(None)).unwrap();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("no space_hash"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fingerprint_drift_skips_l2_verification_instead_of_refusing() {
        // Ordering pin: under toolchain drift the recomputed key CANNOT match
        // the stored one, so a mismatching recomputation must be ignored
        // (drift warning), never misread as staleness.
        let path = artifact_path("driftorder");
        let mut a = Artifact::new(&ws());
        a.insert(ArtifactEntry {
            bucket: "b".into(),
            config: cfg(64, 8),
            median_ms: 1.0,
            samples: 5,
            l2_key: Some("a".repeat(64)),
        });
        a.save(&path).unwrap();
        let mut drifted = ws();
        drifted.tileiras_fingerprint = "release 13.4, V13.4.1".into();
        let mut called = false;
        let (_, warnings) = Artifact::load_verified(&path, &drifted, |_| {
            called = true;
            Ok(Some("b".repeat(64))) // would refuse if it were consulted
        })
        .unwrap();
        assert!(!called, "verifier must not run under fingerprint drift");
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("skipped"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn artifact_version_drift_warns() {
        let path = artifact_path("version");
        let mut a = Artifact::new(&ws());
        a.cutile_version = "0.0.0-elsewhere".into();
        a.save(&path).unwrap();
        let (_, warnings) = Artifact::load_verified(&path, &ws(), |_| Ok(None)).unwrap();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("produced by cutile 0.0.0-elsewhere"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn trial_log_roundtrips_and_resumes() {
        let dir = std::env::temp_dir().join(format!("cutile_tune_log_{}", std::process::id()));
        let _ = std::fs::remove_file(&dir);
        {
            let mut log = TrialLog::open(Some(dir.as_path()), "t", "s").unwrap();
            log.append(&Trial {
                config_id: "BN=64".into(),
                outcome: Outcome::Measured {
                    median_ms: 1.5,
                    min_ms: 1.4,
                    reps: 5,
                },
            });
            log.append(&Trial {
                config_id: "BN=128".into(),
                outcome: Outcome::Invalid {
                    reason: "launch check".into(),
                },
            });
        }
        let log = TrialLog::open(Some(dir.as_path()), "t", "s").unwrap();
        let existing = log.existing_trials();
        assert_eq!(existing.len(), 2);
        assert_eq!(existing[0].median_ms(), Some(1.5));
        assert!(existing[1].median_ms().is_none());

        // Wrong tuner or space: refused loudly, not silently adopted.
        let err = TrialLog::open(Some(dir.as_path()), "other", "s").unwrap_err();
        assert!(err.to_string().contains("belongs to tuner"));
        let err = TrialLog::open(Some(dir.as_path()), "t", "different").unwrap_err();
        assert!(err.to_string().contains("belongs to tuner"));

        // Torn final line: next append still yields a parseable record.
        {
            use std::io::Write as _;
            let mut f = std::fs::OpenOptions::new().append(true).open(&dir).unwrap();
            write!(f, "{{\"config_id\":\"torn").unwrap();
        }
        {
            let mut log = TrialLog::open(Some(dir.as_path()), "t", "s").unwrap();
            log.append(&Trial {
                config_id: "BN=256".into(),
                outcome: Outcome::Measured {
                    median_ms: 2.0,
                    min_ms: 2.0,
                    reps: 3,
                },
            });
        }
        let log = TrialLog::open(Some(dir.as_path()), "t", "s").unwrap();
        assert_eq!(
            log.existing_trials().len(),
            3,
            "torn line dropped, new record intact"
        );
        let _ = std::fs::remove_file(&dir);
    }

    #[test]
    fn config_ids_do_not_alias_across_types_or_separators() {
        let int1 = Config::new([("A", ParamValue::Int(1))]);
        let str1 = Config::new([("A", ParamValue::Str("1".into()))]);
        assert_ne!(int1.id, str1.id, "int 1 and string \"1\" must differ");
        let sneaky = Config::new([("x", ParamValue::Str("1,y=2".into()))]);
        let honest = Config::new([
            ("x", ParamValue::Str("1".into())),
            ("y", ParamValue::Int(2)),
        ]);
        assert_ne!(sneaky.id, honest.id, "separator injection must not alias");
    }

    #[test]
    fn stale_resumed_trials_neither_win_nor_block_a_winner() {
        // A resumed trial for a config no longer in the space: dropped, and
        // the remaining valid winner is still selected (not None).
        let configs = vec![cfg(64, 4)];
        let stale = Trial {
            config_id: "BN=16,SPLITS=2".into(),
            outcome: Outcome::Measured {
                median_ms: 0.1,
                min_ms: 0.1,
                reps: 3,
            },
        };
        let mut oracle = FakeOracle {
            configs: configs.clone(),
            cost: |_| Some(1.0),
            measured: Vec::new(),
            budget: None,
        };
        let trials = GridSampler::new().resume(vec![stale]).search(&mut oracle);
        assert_eq!(trials.len(), 1, "stale trial dropped from results");
        let best = best_config(&configs, &trials).expect("valid winner survives");
        assert_eq!(best.int("BN"), Some(64));
    }

    #[test]
    fn resumed_invalid_trials_are_retried() {
        let configs = vec![cfg(64, 4)];
        let invalid = Trial {
            config_id: configs[0].id.clone(),
            outcome: Outcome::Invalid {
                reason: "transient".into(),
            },
        };
        let mut oracle = FakeOracle {
            configs: configs.clone(),
            cost: |_| Some(1.0),
            measured: Vec::new(),
            budget: None,
        };
        let trials = GridSampler::new().resume(vec![invalid]).search(&mut oracle);
        assert_eq!(oracle.measured.len(), 1, "previously-Invalid retried");
        assert!(trials.iter().any(|t| t.median_ms() == Some(1.0)));
    }

    #[test]
    fn prune_composes_with_configs_in_either_order() {
        let mk = || vec![cfg(32, 2), cfg(64, 4)];
        // prune BEFORE configs — must still apply (predicates run at search).
        let mut a = Autotuner::new("t")
            .prune(|c| c.int("BN") != Some(32))
            .configs(mk());
        a.apply_prune();
        assert_eq!(a.configs.len(), 1);
        assert_eq!(a.configs[0].int("BN"), Some(64));
        // And after, identically.
        let mut b = Autotuner::new("t")
            .configs(mk())
            .prune(|c| c.int("BN") != Some(32));
        b.apply_prune();
        assert_eq!(b.configs.len(), 1);
        assert_eq!(b.configs[0].int("BN"), Some(64));
    }

    #[test]
    fn best_config_skips_non_finite_medians() {
        let configs = vec![cfg(64, 4), cfg(128, 8)];
        let trials = vec![
            Trial {
                config_id: configs[0].id.clone(),
                outcome: Outcome::Measured {
                    median_ms: f32::NAN,
                    min_ms: f32::NAN,
                    reps: 3,
                },
            },
            Trial {
                config_id: configs[1].id.clone(),
                outcome: Outcome::Measured {
                    median_ms: 2.0,
                    min_ms: 2.0,
                    reps: 3,
                },
            },
        ];
        let best = best_config(&configs, &trials).unwrap();
        assert_eq!(best.int("BN"), Some(128), "NaN never wins");
    }

    fn meas(times: &[f32]) -> Measurement {
        Measurement::from_times_ms(times.to_vec())
    }

    fn silent_log() -> TrialLog {
        TrialLog::open(None, "t", "s").unwrap()
    }

    #[test]
    fn runoff_setup_failure_forfeits_to_the_other_finalist() {
        let (a, b) = (cfg(32, 2), cfg(64, 4));
        for (b_failed, winner_id, loser_id) in [
            (false, b.id.clone(), a.id.clone()),
            (true, a.id.clone(), b.id.clone()),
        ] {
            let mut trials = Vec::new();
            let err = RunoffError::Setup {
                b_failed,
                error: crate::error::tensor_error("boom"),
            };
            let winner = runoff_verdict(
                a.clone(),
                b.clone(),
                Err(err),
                &mut trials,
                &mut silent_log(),
            );
            assert_eq!(winner.id, winner_id);
            assert_eq!(trials.len(), 1);
            assert_eq!(
                trials[0].config_id, loser_id,
                "failure blamed on the failing finalist"
            );
            assert!(matches!(
                &trials[0].outcome,
                Outcome::Invalid { reason } if reason.contains("runoff setup failed")
            ));
        }
    }

    #[test]
    fn runoff_bench_failure_keeps_the_sequential_leader() {
        let (a, b) = (cfg(32, 2), cfg(64, 4));
        let mut trials = Vec::new();
        let winner = runoff_verdict(
            a.clone(),
            b,
            Err(RunoffError::Bench(crate::error::tensor_error(
                "stream died",
            ))),
            &mut trials,
            &mut silent_log(),
        );
        assert_eq!(winner.id, a.id);
        assert!(
            trials.is_empty(),
            "unattributable failures mark nobody Invalid"
        );
    }

    #[test]
    fn non_finite_runoff_median_never_wins() {
        let (a, b) = (cfg(32, 2), cfg(64, 4));
        // Under a plain `<=`, `ms_a <= NaN` is false and NaN-b would win.
        let winner = runoff_verdict(
            a.clone(),
            b.clone(),
            Ok((meas(&[2.0, 2.0, 2.0]), meas(&[f32::NAN, f32::NAN]))),
            &mut Vec::new(),
            &mut silent_log(),
        );
        assert_eq!(winner.id, a.id);
        let winner = runoff_verdict(
            a,
            b.clone(),
            Ok((meas(&[f32::NAN]), meas(&[3.0]))),
            &mut Vec::new(),
            &mut silent_log(),
        );
        assert_eq!(winner.id, b.id);
    }

    #[test]
    fn runoff_trials_carry_real_measurement_metadata() {
        let (a, b) = (cfg(32, 2), cfg(64, 4));
        let mut trials = Vec::new();
        let winner = runoff_verdict(
            a.clone(),
            b,
            Ok((meas(&[1.0, 2.0, 3.0]), meas(&[4.0, 5.0, 6.0]))),
            &mut trials,
            &mut silent_log(),
        );
        assert_eq!(winner.id, a.id);
        assert_eq!(trials.len(), 2);
        match &trials[0].outcome {
            Outcome::Measured {
                median_ms,
                min_ms,
                reps,
            } => {
                assert_eq!(*median_ms, 2.0);
                assert_eq!(*min_ms, 1.0);
                assert_eq!(*reps, 3, "reps reflect the actual paired measurement");
            }
            other => panic!("expected Measured, got {other:?}"),
        }
    }

    #[test]
    fn config_ids_do_not_alias_across_key_separators() {
        // A key containing separators must not imitate two parameters.
        let smuggled = Config::new([("A=1,B", ParamValue::Int(2))]);
        let honest = Config::new([("A", ParamValue::Int(1)), ("B", ParamValue::Int(2))]);
        assert_ne!(smuggled.id, honest.id);
        // Nor can a key with literal quotes imitate an encoded one.
        let quoted = Config::new([("\"A\"", ParamValue::Int(1))]);
        let plain = Config::new([("A", ParamValue::Int(1))]);
        assert_ne!(quoted.id, plain.id);
    }

    #[test]
    fn whitespace_only_log_is_headed_and_resumable() {
        let dir = std::env::temp_dir().join(format!("cutile_tune_ws_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("trials.jsonl");
        std::fs::write(&path, "\n").unwrap();
        {
            let mut log = TrialLog::open(Some(&path), "t", "s").unwrap();
            log.append(&Trial {
                config_id: cfg(1, 1).id,
                outcome: Outcome::Invalid { reason: "x".into() },
            });
        }
        // Reopening must find a valid header, not refuse the log.
        let log = TrialLog::open(Some(&path), "t", "s").unwrap();
        assert_eq!(log.existing_trials().len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
