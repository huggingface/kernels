//! Concurrent uploads to a version branch against an in-memory Hub. Every Hub
//! call is a scheduling point, so the tests run every order of the calls (or
//! random ones) and check the branch after each run.

use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    fs,
    path::{Path, PathBuf},
    sync::{Condvar, Mutex},
};

use eyre::Result;
use hf_hub::{
    repository::{AddSource, CommitOperation},
    HFError,
};

use super::{
    plan_upload, prepare_version_branch, CreateBranch, PinnedCommit, UploadHub, MAIN_BRANCH,
};

const BRANCH: &str = "v2";
const MAIN_CONTENT: &str = "main";
const EARLIER_CONTENT: &str = "earlier";

thread_local! {
    static JOB: Cell<usize> = const { Cell::new(usize::MAX) };
}

/// Branches a Hub call reads and writes. Only the order of calls that touch
/// the same branch, with at least one write, is explored.
#[derive(Clone, Debug, Default)]
struct Access {
    reads_all: bool,
    reads: BTreeSet<String>,
    writes: BTreeSet<String>,
}

impl Access {
    fn read_all() -> Self {
        Access {
            reads_all: true,
            ..Access::default()
        }
    }

    /// Commit ids are immutable, so reading one does not depend on any branch.
    fn read(revision: &str) -> Self {
        let is_commit_id =
            revision.starts_with('c') && revision[1..].chars().all(|c| c.is_ascii_digit());
        Access {
            reads: if is_commit_id {
                BTreeSet::new()
            } else {
                BTreeSet::from([revision.to_owned()])
            },
            ..Access::default()
        }
    }

    fn write(branch: &str) -> Self {
        Access {
            writes: BTreeSet::from([branch.to_owned()]),
            ..Access::default()
        }
    }

    fn depends_on(&self, other: &Access) -> bool {
        let writes_used_by = |a: &Access, b: &Access| {
            (b.reads_all && !a.writes.is_empty())
                || a.writes
                    .iter()
                    .any(|w| b.reads.contains(w) || b.writes.contains(w))
        };
        writes_used_by(self, other) || writes_used_by(other, self)
    }
}

#[derive(Default)]
struct SchedState {
    live: BTreeSet<usize>,
    waiting: BTreeMap<usize, Access>,
    running: Option<usize>,
    crash: HashSet<usize>,
    choices: Option<Choices>,
    /// The rest of the run is equivalent to an earlier one, so the remaining
    /// jobs are crashed to end it.
    stopped: bool,
}

/// Runs one job at a time, switching only at Hub calls.
#[derive(Default)]
struct Scheduler {
    state: Mutex<SchedState>,
    cv: Condvar,
}

impl Scheduler {
    /// Returns `false` if the job should crash at this call.
    fn yield_now(&self, access: Access) -> bool {
        let job = JOB.get();
        let mut state = self.state.lock().unwrap();
        state.running = None;
        state.waiting.insert(job, access);
        self.schedule(&mut state);
        while state.running != Some(job) {
            state = self.cv.wait(state).unwrap();
        }
        !state.crash.contains(&job)
    }

    fn finish(&self) {
        let job = JOB.get();
        let mut state = self.state.lock().unwrap();
        state.live.remove(&job);
        state.running = None;
        self.schedule(&mut state);
    }

    /// The job that yields picks the next one, so there is no scheduler thread.
    fn schedule(&self, state: &mut SchedState) {
        if state.running.is_some() || state.waiting.len() != state.live.len() {
            return;
        }
        let ready: Vec<(usize, Access)> = state
            .waiting
            .iter()
            .map(|(job, access)| (*job, access.clone()))
            .collect();
        let Some(&(first, _)) = ready.first() else {
            return;
        };
        let picked = if state.stopped {
            None
        } else {
            state.choices.as_mut().unwrap().pick(ready)
        };
        let (job, crash) = picked.unwrap_or_else(|| {
            state.stopped = true;
            (first, true)
        });
        state.waiting.remove(&job);
        if crash {
            state.crash.insert(job);
        }
        state.running = Some(job);
        self.cv.notify_all();
    }
}

#[derive(Clone)]
struct Commit {
    parent: Option<String>,
    files: BTreeMap<String, String>,
}

struct RepoState {
    commits: HashMap<String, Commit>,
    branches: BTreeMap<String, String>,
    next_id: usize,
    /// Whether deleting a missing file fails the commit, the Hub may do either.
    strict_deletes: bool,
    /// Pinned commits to refuse even though the parent matches, like a 409
    /// for another reason.
    spurious_refusals: usize,
    log: Vec<String>,
}

struct FakeHub {
    sched: Scheduler,
    repo: Mutex<RepoState>,
}

fn crashed() -> HFError {
    HFError::CacheNotEnabled
}

impl FakeHub {
    fn new(main_files: BTreeMap<String, String>, options: Options) -> Self {
        let files = main_files;
        let strict_deletes = options.strict_deletes;
        let hub = FakeHub {
            sched: Scheduler::default(),
            repo: Mutex::new(RepoState {
                commits: HashMap::from([(
                    "c0".to_owned(),
                    Commit {
                        parent: None,
                        files,
                    },
                )]),
                branches: BTreeMap::from([(MAIN_BRANCH.to_owned(), "c0".to_owned())]),
                next_id: 1,
                strict_deletes,
                spurious_refusals: options.spurious_refusals,
                log: Vec::new(),
            }),
        };
        {
            let mut repo = hub.repo.lock().unwrap();
            match options.start {
                Start::NewBranch => {}
                Start::AtMainCommit => {
                    repo.branches.insert(BRANCH.to_owned(), "c0".to_owned());
                }
                Start::OwnCommits => {
                    repo.commits.insert(
                        "c1".to_owned(),
                        Commit {
                            parent: Some("c0".to_owned()),
                            files: earlier_upload_files(),
                        },
                    );
                    repo.branches.insert(BRANCH.to_owned(), "c1".to_owned());
                    repo.next_id = 2;
                }
            }
        }
        hub
    }

    fn call<R>(
        &self,
        access: Access,
        what: impl FnOnce() -> String,
        f: impl FnOnce(&mut RepoState) -> std::result::Result<R, HFError>,
        outcome: impl FnOnce(&RepoState, &R) -> String,
    ) -> std::result::Result<R, HFError> {
        let alive = self.sched.yield_now(access);
        let mut repo = self.repo.lock().unwrap();
        let job = JOB.get();
        if !alive {
            repo.log.push(format!("job {job}: crashed"));
            return Err(crashed());
        }
        let what = what();
        let result = f(&mut repo);
        let outcome = match &result {
            Ok(r) => outcome(&repo, r),
            Err(err) => format!("error: {err}"),
        };
        repo.log.push(format!("job {job}: {what:<40} -> {outcome}"));
        result
    }

    fn branch_files(&self, branch: &str) -> BTreeMap<String, String> {
        let repo = self.repo.lock().unwrap();
        repo.commits[&repo.branches[branch]].files.clone()
    }

    fn log(&self) -> String {
        self.repo.lock().unwrap().log.join("\n")
    }

    /// Unpinned, like the upload commits.
    fn commit(
        &self,
        branch: &str,
        operations: Vec<CommitOperation>,
    ) -> std::result::Result<(), HFError> {
        self.call(
            Access::write(branch),
            || format!("commit {} to `{branch}`", op_counts(&operations)),
            |repo| repo.apply(branch, &operations),
            |repo, _| format!("committed {}", repo.branches[branch]),
        )
    }
}

impl RepoState {
    fn resolve(&self, revision: &str) -> Option<String> {
        self.branches.get(revision).cloned().or_else(|| {
            self.commits
                .contains_key(revision)
                .then(|| revision.to_owned())
        })
    }

    fn not_found(revision: &str) -> HFError {
        HFError::RevisionNotFound {
            repo_id: "fake".to_owned(),
            revision: revision.to_owned(),
            context: None,
        }
    }

    fn apply(
        &mut self,
        branch: &str,
        operations: &[CommitOperation],
    ) -> std::result::Result<(), HFError> {
        let head = self
            .resolve(branch)
            .ok_or_else(|| Self::not_found(branch))?;
        let mut files = self.commits[&head].files.clone();
        for op in operations {
            match op {
                CommitOperation::Add {
                    path_in_repo,
                    source,
                } => {
                    let content = match source {
                        AddSource::File(path) => fs::read_to_string(path).unwrap(),
                        AddSource::Bytes(bytes) => String::from_utf8_lossy(bytes).into_owned(),
                        _ => unimplemented!("unsupported add source"),
                    };
                    files.insert(path_in_repo.clone(), content);
                }
                CommitOperation::Delete { path_in_repo } => {
                    if files.remove(path_in_repo).is_none() && self.strict_deletes {
                        return Err(HFError::EntryNotFound {
                            path: path_in_repo.clone(),
                            repo_id: "fake".to_owned(),
                            context: None,
                        });
                    }
                }
            }
        }
        let id = format!("c{}", self.next_id);
        self.next_id += 1;
        self.commits.insert(
            id.clone(),
            Commit {
                parent: Some(head),
                files,
            },
        );
        self.branches.insert(branch.to_owned(), id);
        Ok(())
    }
}

fn op_counts(operations: &[CommitOperation]) -> String {
    let adds = operations
        .iter()
        .filter(|op| matches!(op, CommitOperation::Add { .. }))
        .count();
    format!("+{adds} -{}", operations.len() - adds)
}

impl UploadHub for FakeHub {
    fn branch_heads(&self) -> std::result::Result<BTreeMap<String, String>, HFError> {
        self.call(
            Access::read_all(),
            || "list refs".to_owned(),
            |repo| Ok(repo.branches.clone()),
            |_, heads| {
                heads
                    .iter()
                    .map(|(branch, head)| format!("{branch} at {head}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            },
        )
    }

    fn create_branch(
        &self,
        branch: &str,
        revision: Option<&str>,
    ) -> std::result::Result<CreateBranch, HFError> {
        let mut access = Access::write(branch);
        access.reads = Access::read(revision.unwrap_or(MAIN_BRANCH)).reads;
        self.call(
            access,
            || format!("create `{branch}` from {}", revision.unwrap_or(MAIN_BRANCH)),
            |repo| {
                if repo.branches.contains_key(branch) {
                    return Ok(CreateBranch::AlreadyExists);
                }
                let revision = revision.unwrap_or(MAIN_BRANCH);
                let commit = repo
                    .resolve(revision)
                    .ok_or_else(|| RepoState::not_found(revision))?;
                repo.branches.insert(branch.to_owned(), commit);
                Ok(CreateBranch::Created)
            },
            |_, created| match created {
                CreateBranch::Created => "created".to_owned(),
                CreateBranch::AlreadyExists => "already exists (409)".to_owned(),
            },
        )
    }

    fn commit_history(&self, revision: &str) -> std::result::Result<HashSet<String>, HFError> {
        self.call(
            Access::read(revision),
            || format!("list commits of `{revision}`"),
            |repo| {
                let mut history = HashSet::new();
                let mut next = repo.resolve(revision);
                while let Some(id) = next {
                    next = repo.commits[&id].parent.clone();
                    history.insert(id);
                }
                Ok(history)
            },
            |repo, history| {
                repo.branches
                    .iter()
                    .filter(|(branch, _)| *branch != MAIN_BRANCH)
                    .map(|(branch, head)| {
                        if history.contains(head) {
                            format!("{branch} at {head} has no commits of its own")
                        } else {
                            format!("{branch} at {head} has its own commits")
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            },
        )
    }

    fn files(&self, revision: &str) -> std::result::Result<BTreeSet<String>, HFError> {
        self.call(
            Access::read(revision),
            || format!("list files of `{revision}`"),
            |repo| {
                let id = repo
                    .resolve(revision)
                    .ok_or_else(|| RepoState::not_found(revision))?;
                Ok(repo.commits[&id].files.keys().cloned().collect())
            },
            |_, files: &BTreeSet<String>| format!("{} files", files.len()),
        )
    }

    fn commit_pinned(
        &self,
        branch: &str,
        operations: Vec<CommitOperation>,
        _commit_message: &str,
        parent_commit: &str,
    ) -> std::result::Result<PinnedCommit, HFError> {
        self.call(
            Access::write(branch),
            || {
                format!(
                    "commit {} to `{branch}` pinned to {parent_commit}",
                    op_counts(&operations)
                )
            },
            |repo| {
                if repo.branches.get(branch).map(String::as_str) != Some(parent_commit) {
                    return Ok(PinnedCommit::Refused);
                }
                if repo.spurious_refusals > 0 {
                    repo.spurious_refusals -= 1;
                    return Ok(PinnedCommit::Refused);
                }
                repo.apply(branch, &operations)?;
                Ok(PinnedCommit::Committed)
            },
            |repo, committed| {
                let head = &repo.branches[branch];
                match committed {
                    PinnedCommit::Committed => format!("committed {head}"),
                    PinnedCommit::Refused if head == parent_commit => {
                        "refused, although the head matches".to_owned()
                    }
                    PinnedCommit::Refused => format!("refused, head moved to {head}"),
                }
            },
        )
    }
}

struct Job {
    kernel_dir: PathBuf,
    build_dir: PathBuf,
    variants: Vec<PathBuf>,
    files: BTreeMap<String, String>,
}

impl Job {
    fn new(root: &Path, name: &str, variants: &[&str]) -> Job {
        let kernel_dir = root.join(name);
        let build_dir = kernel_dir.join("build");
        let mut files = BTreeMap::new();
        let mut variant_dirs = Vec::new();
        for variant in variants {
            let dir = build_dir.join(variant);
            fs::create_dir_all(&dir).unwrap();
            for file in ["__init__.py", "_ops.py", "kernel.so"] {
                let content = format!("{name}:{variant}/{file}");
                fs::write(dir.join(file), &content).unwrap();
                files.insert(format!("build/{variant}/{file}"), content);
            }
            variant_dirs.push(dir);
        }
        fs::write(build_dir.join("CARD.md"), "card").unwrap();
        Job {
            kernel_dir,
            build_dir,
            variants: variant_dirs,
            files,
        }
    }
}

fn upload(hub: &FakeHub, job: &Job) -> Result<()> {
    upload_to(hub, job, BRANCH)
}

/// Mirrors a direct upload in `run_upload_typed`.
fn upload_to(hub: &FakeHub, job: &Job, branch: &str) -> Result<()> {
    prepare_version_branch(hub, "fake", branch, false)?;
    let operations_by_branch = plan_upload(
        hub,
        &job.kernel_dir,
        &job.build_dir,
        &job.variants,
        branch,
        str::to_owned,
    )?;
    for (branch, operations) in operations_by_branch {
        if !operations.is_empty() {
            hub.commit(&branch, operations)?;
        }
    }
    Ok(())
}

fn move_main(hub: &FakeHub) -> Result<()> {
    hub.commit(
        MAIN_BRANCH,
        vec![CommitOperation::Add {
            path_in_repo: "README.md".to_owned(),
            source: AddSource::Bytes(b"updated".to_vec().into()),
        }],
    )?;
    Ok(())
}

/// `old-a` and `old-b` are not built anymore and must not reach the branch.
fn main_files() -> BTreeMap<String, String> {
    let mut files = BTreeMap::from([
        ("README.md".to_owned(), "readme".to_owned()),
        (
            "benchmarks/benchmark_x.py".to_owned(),
            MAIN_CONTENT.to_owned(),
        ),
    ]);
    for variant in ["torch-a", "torch-b", "old-a", "old-b"] {
        for file in ["__init__.py", "_ops.py", "kernel.so", "stale.so"] {
            files.insert(format!("build/{variant}/{file}"), MAIN_CONTENT.to_owned());
        }
    }
    files
}

/// `torch-a` has a file the new build lacks, and `torch-c` is not rebuilt so
/// it must be kept.
fn earlier_upload_files() -> BTreeMap<String, String> {
    let mut files = BTreeMap::from([("README.md".to_owned(), "readme".to_owned())]);
    for variant in ["torch-a", "torch-c"] {
        for file in ["__init__.py", "_ops.py", "kernel.so", "stale.so"] {
            files.insert(
                format!("build/{variant}/{file}"),
                EARLIER_CONTENT.to_owned(),
            );
        }
    }
    files
}

struct Node {
    ready: Vec<(usize, Access)>,
    /// Jobs whose runs from here are equivalent to ones already explored.
    sleep: BTreeSet<usize>,
    tried: BTreeSet<usize>,
    current: usize,
}

impl Node {
    fn access(&self, job: usize) -> &Access {
        &self.ready.iter().find(|(j, _)| *j == job).unwrap().1
    }

    fn untried(&self) -> Option<usize> {
        self.ready
            .iter()
            .map(|(job, _)| *job)
            .find(|job| !self.sleep.contains(job) && !self.tried.contains(job))
    }
}

/// Depth-first search over call orders, re-running the uploads for each. Sleep
/// sets (Godefroid) skip orders that only swap independent calls.
#[derive(Default)]
struct Explorer {
    stack: Vec<Node>,
    depth: usize,
}

impl Explorer {
    /// `None` if the rest of this run is equivalent to an explored one.
    fn pick(&mut self, ready: Vec<(usize, Access)>) -> Option<usize> {
        let depth = self.depth;
        self.depth += 1;
        if let Some(node) = self.stack.get(depth) {
            debug_assert_eq!(
                node.ready.iter().map(|(job, _)| *job).collect::<Vec<_>>(),
                ready.iter().map(|(job, _)| *job).collect::<Vec<_>>(),
                "uploads are not deterministic"
            );
            return Some(node.current);
        }

        let sleep = match self.stack.last() {
            None => BTreeSet::new(),
            Some(parent) => {
                let ran = parent.access(parent.current);
                parent
                    .sleep
                    .iter()
                    .chain(&parent.tried)
                    .copied()
                    .filter(|job| *job != parent.current)
                    .filter(|job| !parent.access(*job).depends_on(ran))
                    .collect()
            }
        };
        let current = ready
            .iter()
            .map(|(job, _)| *job)
            .find(|job| !sleep.contains(job))?;
        self.stack.push(Node {
            ready,
            sleep,
            tried: BTreeSet::new(),
            current,
        });
        Some(current)
    }

    /// Returns `false` when every order is done.
    fn backtrack(&mut self) -> bool {
        self.depth = 0;
        while let Some(node) = self.stack.last_mut() {
            node.tried.insert(node.current);
            if let Some(next) = node.untried() {
                node.current = next;
                return true;
            }
            self.stack.pop();
        }
        false
    }
}

/// A job to run its next Hub call, or to crash at it.
#[derive(Clone, Copy)]
enum Step {
    Run(usize),
    Crash(usize),
}

enum Choices {
    All(Explorer),
    /// Runs the steps in order, skipping those of finished jobs, then the
    /// remaining jobs by id.
    Script(VecDeque<Step>),
    /// Crashes a job at a Hub call with odds `1 / crash_odds`.
    Random {
        rng: u64,
        crash_odds: u64,
    },
}

impl Choices {
    fn next_u64(rng: &mut u64) -> u64 {
        // xorshift64*
        *rng ^= *rng >> 12;
        *rng ^= *rng << 25;
        *rng ^= *rng >> 27;
        rng.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn pick(&mut self, ready: Vec<(usize, Access)>) -> Option<(usize, bool)> {
        match self {
            Choices::All(explorer) => explorer.pick(ready).map(|job| (job, false)),
            Choices::Random { rng, crash_odds } => {
                let job = ready[(Self::next_u64(rng) % ready.len() as u64) as usize].0;
                let crash = *crash_odds > 0 && Self::next_u64(rng) % *crash_odds == 0;
                Some((job, crash))
            }
            Choices::Script(steps) => {
                while let Some(step) = steps.pop_front() {
                    let (job, crash) = match step {
                        Step::Run(job) => (job, false),
                        Step::Crash(job) => (job, true),
                    };
                    if ready.iter().any(|(j, _)| *j == job) {
                        return Some((job, crash));
                    }
                }
                Some((ready[0].0, false))
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Options {
    start: Start,
    move_main: bool,
    strict_deletes: bool,
    spurious_refusals: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Start {
    NewBranch,
    /// Points to a `main` commit, so it is cleaned like a new branch.
    AtMainCommit,
    OwnCommits,
}

impl Options {
    fn all(start: Start, move_main: bool) -> [Options; 2] {
        [true, false].map(|strict_deletes| Options {
            start,
            move_main,
            strict_deletes,
            spurious_refusals: 0,
        })
    }
}

/// Returns `false` if the run was stopped as equivalent to an earlier one.
fn run_once(jobs: &[Job], options: Options, choices: &mut Choices) -> bool {
    let (hub, results, stopped) = run(jobs, options, choices);
    if !stopped {
        check(&hub, jobs, options, &results);
    }
    !stopped
}

fn run(jobs: &[Job], options: Options, choices: &mut Choices) -> (FakeHub, Vec<Result<()>>, bool) {
    let hub = FakeHub::new(main_files(), options);
    let n_threads = jobs.len() + usize::from(options.move_main);
    {
        let mut state = hub.sched.state.lock().unwrap();
        state.live = (0..n_threads).collect();
        state.choices = Some(std::mem::replace(
            choices,
            Choices::Random {
                rng: 0,
                crash_odds: 0,
            },
        ));
    }

    let results: Vec<Result<()>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..n_threads)
            .map(|id| {
                let hub = &hub;
                scope.spawn(move || {
                    JOB.set(id);
                    let result = match jobs.get(id) {
                        Some(job) => upload(hub, job),
                        None => move_main(hub),
                    };
                    hub.sched.finish();
                    result
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let stopped = {
        let mut state = hub.sched.state.lock().unwrap();
        *choices = state.choices.take().unwrap();
        state.stopped
    };
    (hub, results, stopped)
}

fn check(hub: &FakeHub, jobs: &[Job], options: Options, results: &[Result<()>]) {
    let fail = |msg: String| -> ! {
        panic!(
            "{msg}\n\nWith {options:?}, Hub calls in the order they ran:\n{}",
            hub.log()
        );
    };

    let crashed: HashSet<usize> = hub.sched.state.lock().unwrap().crash.clone();
    for (id, result) in results.iter().enumerate() {
        if let Err(err) = result {
            if !crashed.contains(&id) {
                fail(format!("job {id} failed: {err:?}"));
            }
        }
    }

    let succeeded: Vec<usize> = (0..jobs.len())
        .filter(|id| results[*id].is_ok() && !crashed.contains(id))
        .collect();
    if succeeded.is_empty() {
        return;
    }
    let files = hub.branch_files(BRANCH);

    for (path, content) in &files {
        let under_build = path.starts_with("build/") || path.starts_with("benchmarks/");
        if under_build && content == MAIN_CONTENT {
            fail(format!(
                "`{path}` inherited from `{MAIN_BRANCH}` is still on `{BRANCH}`"
            ));
        }
    }
    // Rebuilt variants hold only uploaded files, the others are kept.
    let built: BTreeSet<&str> = succeeded
        .iter()
        .flat_map(|id| jobs[*id].files.keys())
        .filter_map(|path| path.split('/').nth(1))
        .collect();
    for path in files.keys() {
        let variant = path
            .strip_prefix("build/")
            .and_then(|p| p.split('/').next());
        if variant.is_some_and(|v| built.contains(v))
            && !jobs.iter().any(|job| job.files.contains_key(path))
        {
            fail(format!("stale `{path}` is still on `{BRANCH}`"));
        }
    }
    if options.start == Start::OwnCommits {
        for (path, content) in earlier_upload_files() {
            let variant = path
                .strip_prefix("build/")
                .and_then(|p| p.split('/').next());
            let replaced = variant.is_some_and(|v| built.contains(v)) || path == "README.md";
            if !replaced && files.get(&path) != Some(&content) {
                fail(format!(
                    "`{path}` of the earlier upload was changed: {:?}",
                    files.get(&path)
                ));
            }
        }
    }

    for id in succeeded {
        for (path, content) in &jobs[id].files {
            // Another job may upload the same variant, e.g. a re-run.
            let uploaded_by_some_job = files.get(path).is_some_and(|actual| {
                actual == content || jobs.iter().any(|job| job.files.get(path) == Some(actual))
            });
            if !uploaded_by_some_job {
                fail(format!(
                    "`{path}` of job {id} is missing or wrong: {:?}",
                    files.get(path)
                ));
            }
        }
    }
}

/// Returns the number of runs that were checked.
fn run_all_orders(jobs: &[Job], options: Options) -> usize {
    let mut choices = Choices::All(Explorer::default());
    let mut checked = 0;
    loop {
        checked += usize::from(run_once(jobs, options, &mut choices));
        let Choices::All(explorer) = &mut choices else {
            unreachable!()
        };
        if !explorer.backtrack() {
            return checked;
        }
    }
}

fn run_random_orders(jobs: &[Job], options: Options, runs: u64, crash_odds: u64) {
    for seed in 1..=runs {
        let mut choices = Choices::Random {
            rng: seed.wrapping_mul(0x9e37_79b9_7f4a_7c15),
            crash_odds,
        };
        run_once(jobs, options, &mut choices);
    }
}

fn for_all_options(start: Start, move_main: bool, f: impl Fn(Options) + Sync) {
    std::thread::scope(|scope| {
        for options in Options::all(start, move_main) {
            let f = &f;
            scope.spawn(move || f(options));
        }
    });
}

#[test]
fn two_uploads_create_branch_concurrently_in_every_order() {
    let tmp = tempfile::tempdir().unwrap();
    let jobs = [
        Job::new(tmp.path(), "job0", &["torch-a"]),
        Job::new(tmp.path(), "job1", &["torch-b"]),
    ];
    for_all_options(Start::NewBranch, false, |options| {
        let runs = run_all_orders(&jobs, options);
        assert!(runs > 100, "only {runs} orders were run with {options:?}");
    });
}

#[test]
fn two_uploads_while_main_moves_in_every_order() {
    let tmp = tempfile::tempdir().unwrap();
    let jobs = [
        Job::new(tmp.path(), "job0", &["torch-a"]),
        Job::new(tmp.path(), "job1", &["torch-b"]),
    ];
    for_all_options(Start::NewBranch, true, |options| {
        run_all_orders(&jobs, options);
    });
}

#[test]
fn three_uploads_with_rerun_and_crashes_in_random_orders() {
    let tmp = tempfile::tempdir().unwrap();
    let jobs = [
        Job::new(tmp.path(), "job0", &["torch-a"]),
        Job::new(tmp.path(), "job1", &["torch-b"]),
        // A re-run of job 1.
        Job::new(tmp.path(), "job2", &["torch-b"]),
    ];
    for_all_options(Start::NewBranch, true, |options| {
        run_random_orders(&jobs, options, 3000, 25)
    });
}

#[test]
fn two_uploads_to_branch_with_earlier_upload_in_every_order() {
    let tmp = tempfile::tempdir().unwrap();
    let jobs = [
        Job::new(tmp.path(), "job0", &["torch-a"]),
        Job::new(tmp.path(), "job1", &["torch-b"]),
    ];
    for_all_options(Start::OwnCommits, true, |options| {
        run_all_orders(&jobs, options);
    });
}

#[test]
fn two_uploads_to_branch_at_main_commit_in_every_order() {
    let tmp = tempfile::tempdir().unwrap();
    let jobs = [
        Job::new(tmp.path(), "job0", &["torch-a"]),
        Job::new(tmp.path(), "job1", &["torch-b"]),
    ];
    for_all_options(Start::AtMainCommit, false, |options| {
        run_all_orders(&jobs, options);
    });
}

fn run_alone<R>(hub: &FakeHub, f: impl FnOnce() -> R) -> R {
    {
        let mut state = hub.sched.state.lock().unwrap();
        state.live = BTreeSet::from([0]);
        state.choices = Some(Choices::Random {
            rng: 1,
            crash_odds: 0,
        });
    }
    JOB.set(0);
    let result = f();
    hub.sched.finish();
    result
}

#[test]
fn upload_to_main_keeps_other_variants() {
    let tmp = tempfile::tempdir().unwrap();
    let job = Job::new(tmp.path(), "job0", &["torch-a"]);
    for options in Options::all(Start::NewBranch, false) {
        let hub = FakeHub::new(main_files(), options);
        run_alone(&hub, || upload_to(&hub, &job, MAIN_BRANCH)).unwrap();

        let files = hub.branch_files(MAIN_BRANCH);
        for (path, content) in main_files() {
            if path.starts_with("build/torch-a/") || path == "README.md" {
                continue;
            }
            assert_eq!(
                files.get(&path),
                Some(&content),
                "`{path}` on `{MAIN_BRANCH}` was changed by an upload of `torch-a` \
                 with {options:?}\n\nHub calls:\n{}",
                hub.log()
            );
        }
        for (path, content) in &job.files {
            assert_eq!(files.get(path), Some(content), "`{path}` was not uploaded");
        }
        assert!(
            !files.contains_key("build/torch-a/stale.so"),
            "stale `build/torch-a/stale.so` is still on `{MAIN_BRANCH}`"
        );
    }
}

/// The Hub may refuse a cleanup for another reason than a sibling's commit.
#[test]
fn upload_after_spurious_cleanup_refusal() {
    let tmp = tempfile::tempdir().unwrap();
    let jobs = [Job::new(tmp.path(), "job0", &["torch-a"])];
    for start in [Start::NewBranch, Start::AtMainCommit] {
        for mut options in Options::all(start, false) {
            options.spurious_refusals = 1;
            let hub = FakeHub::new(main_files(), options);
            let result = run_alone(&hub, || upload(&hub, &jobs[0]));
            check(&hub, &jobs, options, &[result]);
        }
    }
}

#[test]
fn two_uploads_after_spurious_cleanup_refusal_in_every_order() {
    let tmp = tempfile::tempdir().unwrap();
    let jobs = [
        Job::new(tmp.path(), "job0", &["torch-a"]),
        Job::new(tmp.path(), "job1", &["torch-b"]),
    ];
    for_all_options(Start::NewBranch, false, |mut options| {
        options.spurious_refusals = 1;
        run_all_orders(&jobs, options);
    });
}

/// Where the files of each directory on `branch` come from.
fn describe_branch(hub: &FakeHub, branch: &str) -> String {
    let mut origins: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    for (path, content) in hub.branch_files(branch) {
        let Some((dir, _)) = path.rsplit_once('/') else {
            continue;
        };
        let origin = match content.as_str() {
            MAIN_CONTENT => "main <- left behind".to_owned(),
            EARLIER_CONTENT => "earlier upload".to_owned(),
            content => content.split(':').next().unwrap_or(content).to_owned(),
        };
        *origins
            .entry(format!("{dir}/"))
            .or_default()
            .entry(origin)
            .or_default() += 1;
    }
    origins
        .iter()
        .map(|(dir, origins)| {
            let origins: Vec<_> = origins
                .iter()
                .map(|(origin, n)| format!("{n} from {origin}"))
                .collect();
            format!("  {dir:<20} {}", origins.join(", "))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Prints how a few races resolve, run with `--nocapture` to see them.
#[test]
fn traced_scenarios() {
    use Step::{Crash, Run};

    let tmp = tempfile::tempdir().unwrap();
    let jobs = [
        Job::new(tmp.path(), "job0", &["torch-a"]),
        Job::new(tmp.path(), "job1", &["torch-b"]),
    ];
    let options = Options::all(Start::NewBranch, false)[1];
    let scenarios: Vec<(&str, &[Job], Options, Vec<Step>)> = vec![
        (
            "job 1 gets a 409 and commits before job 0, which created the branch",
            &jobs,
            options,
            [1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0]
                .into_iter()
                .map(Run)
                .chain([Run(0); 20])
                .collect(),
        ),
        (
            "the Hub refuses the cleanup for another reason than a commit",
            &jobs[..1],
            Options {
                spurious_refusals: 1,
                ..options
            },
            vec![],
        ),
        (
            "job 0 creates the branch and crashes before its cleanup",
            &jobs,
            options,
            vec![Run(1), Run(0), Run(0), Crash(0)],
        ),
    ];

    let mut failed = vec![];
    for (name, jobs, options, steps) in scenarios {
        let mut choices = Choices::Script(steps.into());
        let (hub, results, _) = run(jobs, options, &mut choices);
        let outcomes: Vec<_> = results
            .iter()
            .enumerate()
            .map(|(id, result)| match result {
                Ok(()) => format!("job {id} ok"),
                Err(_) => format!("job {id} failed"),
            })
            .collect();
        println!(
            "=== {name}\n{}\n{}\n`{BRANCH}` after the uploads:\n{}\n",
            hub.log(),
            outcomes.join(", "),
            describe_branch(&hub, BRANCH)
        );
        let checked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check(&hub, jobs, options, &results)
        }));
        if checked.is_err() {
            failed.push(name);
        }
    }
    assert!(failed.is_empty(), "failed: {failed:#?}");
}
