/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Kernel autotuning: declared search spaces, pluggable samplers, and
//! persisted, provenance-checked results.
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
//! - **Persistence is checked.** Artifacts carry provenance (source hash,
//!   toolchain fingerprint, arch, machine, date) and an optional per-entry
//!   verification key; a loader that detects a mismatch refuses the entry
//!   rather than silently applying a stale winner.

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
pub struct Config {
    /// Stable identity within a search space; artifact and log records key
    /// on it. [`Config::new`] derives it from the parameters.
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
        let id = params
            .iter()
            .map(|(k, v)| match v {
                ParamValue::Int(i) => format!("{k}={i}"),
                ParamValue::Str(s) => format!("{k}={s}"),
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

// ── Trial ───────────────────────────────────────────────────────────────────

/// The result of visiting one candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trial {
    pub config_id: String,
    pub outcome: Outcome,
}

/// What happened when a candidate was visited.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
        let mut trials = std::mem::take(&mut self.known);
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
    let mut best: Option<(&str, f32)> = None;
    for t in trials {
        if let Some(ms) = t.median_ms() {
            if best.is_none_or(|(_, b)| ms < b) {
                best = Some((t.config_id.as_str(), ms));
            }
        }
    }
    let (id, _) = best?;
    configs.iter().find(|c| c.id == id)
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
pub struct Autotuner {
    /// Name for logs and (future) artifact records.
    pub name: String,
    configs: Vec<Config>,
    budget: Option<Duration>,
    bench: BenchOptions,
    log_path: Option<PathBuf>,
}

/// The outcome of a tuning run: all trials plus the winning config.
pub struct TuneOutcome {
    pub trials: Vec<Trial>,
    pub best: Option<Config>,
}

impl Autotuner {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            configs: Vec::new(),
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

    /// Filters candidates at space construction (Triton's
    /// `prune_configs_by`); rejected candidates are never visited.
    pub fn prune<F: Fn(&Config) -> bool>(mut self, keep: F) -> Self {
        self.configs.retain(|c| keep(c));
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
    pub fn run<S, F>(self, stream: &Arc<Stream>, setup: S) -> Result<TuneOutcome, Error>
    where
        S: FnMut(&Arc<Stream>, &Config) -> Result<F, Error>,
        F: FnMut(&Arc<Stream>) -> Result<(), Error>,
    {
        let log = TrialLog::open(self.log_path.as_deref())?;
        let sampler = GridSampler::new().resume(log.existing_trials());
        self.run_sampler(sampler, stream, setup, log)
    }

    /// Runs the search with an explicit [`Sampler`]. The trial log still
    /// records every trial, but resume semantics are the sampler's concern.
    pub fn run_with<S, F>(
        self,
        sampler: impl Sampler,
        stream: &Arc<Stream>,
        setup: S,
    ) -> Result<TuneOutcome, Error>
    where
        S: FnMut(&Arc<Stream>, &Config) -> Result<F, Error>,
        F: FnMut(&Arc<Stream>) -> Result<(), Error>,
    {
        let log = TrialLog::open(self.log_path.as_deref())?;
        self.run_sampler(sampler, stream, setup, log)
    }

    fn run_sampler<S, F>(
        mut self,
        mut sampler: impl Sampler,
        stream: &Arc<Stream>,
        setup: S,
        mut log: TrialLog,
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
            log: &mut log,
        };
        let trials = sampler.search(&mut oracle);
        let best = best_config(&oracle.configs, &trials).cloned();
        Ok(TuneOutcome { trials, best })
    }
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

fn measured(m: &Measurement) -> Outcome {
    Outcome::Measured {
        median_ms: m.median_ms(),
        min_ms: m.min_ms(),
        reps: m.reps(),
    }
}

// ── Trial log (JSONL) ───────────────────────────────────────────────────────

/// Append-only JSONL record of every trial; parsing it back is what makes
/// interrupted runs resumable. Best-effort: log I/O failures never abort a
/// search.
struct TrialLog {
    file: Option<std::fs::File>,
    existing: Vec<Trial>,
}

impl TrialLog {
    fn open(path: Option<&Path>) -> Result<Self, Error> {
        let Some(path) = path else {
            return Ok(Self {
                file: None,
                existing: Vec::new(),
            });
        };
        let existing = match std::fs::read_to_string(path) {
            Ok(contents) => contents
                .lines()
                .filter_map(|l| serde_json::from_str::<Trial>(l).ok())
                .collect(),
            Err(_) => Vec::new(),
        };
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok();
        Ok(Self { file, existing })
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

    #[test]
    fn trial_log_roundtrips_and_resumes() {
        let dir = std::env::temp_dir().join(format!("cutile_tune_log_{}", std::process::id()));
        let _ = std::fs::remove_file(&dir);
        {
            let mut log = TrialLog::open(Some(dir.as_path())).unwrap();
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
        let log = TrialLog::open(Some(dir.as_path())).unwrap();
        let existing = log.existing_trials();
        assert_eq!(existing.len(), 2);
        assert_eq!(existing[0].median_ms(), Some(1.5));
        assert!(existing[1].median_ms().is_none());
        let _ = std::fs::remove_file(&dir);
    }
}
