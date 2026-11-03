# histdb-rs v1.0 — Architecture Review

Review of the original `ARCHITECTURE.md` (branch `claude/code-review-GOrhf`, commit `607113b`,
7,868 lines), done 2026-11-03. Line references (`L1234`) point into that original file.
The split v1 docs carry `> **Review:**` callouts linking back to the findings here.

**Status of findings:** each finding below is a *proposal*. The v1 docs were split
faithfully and only P2P sync was removed (per decision). Nothing else in the design was
rewritten yet — accept/reject each finding and the docs get updated accordingly.

---

## Verdict

The doc is thorough, but most of its effort goes to problems this tool doesn't have, and
it gets the one hot path that matters wrong.

- **~80% of the text is speculative code.** The git history shows five rounds of
  "fix N consistency issues". Those rounds made the snippets agree with each other, but
  never questioned the scope. "Still open: none — ready for implementation" (L7843)
  appears in a doc that has no non-goals and no v1 cut.
- **The workload is tiny.** A person runs a command every few seconds. Even 100 open
  shells produce a few hundred small inserts per minute, and SQLite in WAL mode handles
  tens of thousands of inserts per second. The lock-free MPSC queue, left-right index,
  writer supervisor, overflow queue and bounded-staleness model all address a contention
  problem that does not exist here.
- **The real latency lives in the shell hooks, and that's where the design is weakest**
  ([F1](#f1), [F2](#f2)). Every prompt forks 4–8 processes, and the hand-built JSON
  breaks on multi-line commands.
- **P2P sync was ~40% of the doc.** It has real correctness bugs ([F10](#f10)), and it
  can't reach the ephemeral machines it was designed for. It is now deferred to
  [deferred/sync.md](deferred/sync.md). Note that zsh-histdb *already* has a sync story
  (git + `histdb-merge` 3-way merge). v1 keeps that for parity.

### Recommended v1 shape (proposal)

| Area | Original | Proposed |
|---|---|---|
| Daemon runtime | tokio multi-thread, crossbeam `ArrayQueue`, writer thread + supervisor, overflow queue | tokio `current_thread`; bounded `tokio::sync::mpsc` → one writer task that batches (e.g. 10 ms / 64 rows) into one rusqlite connection; restarts left to systemd/launchd |
| Reads | left-right in-memory index (1,000 rows) merged with SQLite | plain SQLite with indexes on `start_time`, `(host, dir)`, `session`; FTS5 for text |
| Consistency | "bounded staleness ≤10 ms" | read-your-writes: queries go through the daemon, which flushes pending writes before answering |
| Shell → daemon | `$(histdb-cli …)` forks + hand-built JSON | fork-free zsh (`zsocket`, `$EPOCHREALTIME`, NUL-separated fields); bash via bash-preexec + one persistent `coproc` |
| Overflow / daemon down | three different mechanisms (sync retry, async overflow, spool) | one: if the socket write fails, append to the spool |
| Durability | ack before persist by default | durable by default (cheap at this write rate) or state the loss window explicitly |
| Retention | 365-day default age limit | unlimited by default; limits opt-in |

---

## Findings (most severe first)

<a id="f1"></a>
### F1. Shell hooks fork 4–8 processes per command

L3319, L3361, L3363, L3380, L3414, L3445, L3676 — every prompt runs `$(histdb-cli uuid)`,
`$(date +%N | cut -c1-3)` twice, `histdb-cli hlc-now` (now removed with sync), and in
bash/nu also `histdb-cli send`. Each fork+exec costs roughly 1–3 ms, so the hooks spend
~5–20 ms per prompt. That dwarfs the "<100 ns queue push" the daemon design optimises for.
The table at L746–753 measures the wrong side.

**Proposal:**
- zsh: `zmodload zsh/datetime zsh/net/socket` and use `$EPOCHREALTIME` for timestamps.
  Send with `zsocket` + `print -u`, which costs zero forks.
- IDs: either let the daemon assign `history_id` (the shell only needs it for
  `FinishCommand`, and can use a per-session counter instead), or generate UUIDv7 in pure
  zsh from `$EPOCHREALTIME` plus `$RANDOM`/`/dev/urandom` read once per session.
- bash: use bash-preexec (or ble.sh) instead of a raw DEBUG trap, and keep one
  long-lived `coproc` helper so there is one fork per session, not per command.
- Add a benchmark gate: p99 hook time < 1 ms in zsh, measured with `zsh/zprof` or
  `EPOCHREALTIME` deltas.

<a id="f2"></a>
### F2. Hand-built JSON corrupts or spoofs fields

L3365–3369 and L963: `'"argv":"'${cmd//\"/\\\"}'"'` escapes only `"`. Backslashes,
newlines (multi-line commands are common), tabs and other control characters produce
invalid JSON, so those commands are silently lost. A crafted command or directory name
can also inject fields (`"dir": …`). This contradicts the doc's own "no injection" goal.

**Proposal:** don't build JSON in shell. Send a tiny framed record instead:
`START\0<id>\0<session>\0<start_ms>\0<dir>\0<argv>\n` with a length prefix, since argv
may contain anything except NUL. Alternatively pass the fields as argv to a helper, which
needs no escaping at all. Property-test the framing with arbitrary byte strings.

<a id="f3"></a>
### F3. Backpressure: two contradictory designs, both broken

- L755–817 sync retry calls `std::thread::sleep` inside an `async fn` handler, which
  blocks a tokio worker. The "Decided" list (L7782) picks this one.
- L819–984 async overflow: `PendingCommand` derives `Clone` (L211) but contains
  `Option<oneshot::Sender<_>>`, which is not `Clone`, so `cmd.clone()` doesn't compile.
  The retry loop also never gives up, so commands can sit in memory indefinitely.
- Responses `AcceptedOverflow` / `QueueFull` / `Timeout` require the shell to *read* a
  reply. That contradicts "fire-and-forget, shell ignores response" (L3323).

**Proposal:** one bounded channel. If it is full, the daemon writes the record to the
spool itself, so it is never dropped and the shell never waits. Remove `AcceptedOverflow`.

<a id="f4"></a>
### F4. ID and timestamp ownership is contradictory

- The "Decided" entry says "Shell tracks returned ID from StartCommand" (L7788). The
  protocol (L3157+) and spool design say the shell generates the UUID and never reads
  replies.
- `handle_start_with_id` / `Response::CommandStarted` (L703, L3269) is dead code under
  the chosen design.
- The daemon stamps `start_time: Utc::now()` on receipt (L689). Spool replay therefore
  records the *replay* time, not the original. Only the HLC (now removed) carried the
  original time. **The split v1 docs add `start_time_ms` to `StartCommand`** as the
  minimum fix.

**Proposal:** the shell owns `history_id`, `session_id` and `start_time_ms`, and the
daemon never invents them. Delete the start-with-id path.

<a id="f5"></a>
### F5. "Bounded staleness" is not a guarantee

L986–1167: a commit can be delayed past 10 ms by fsync, a WAL checkpoint or a
`SQLITE_BUSY` from a CLI reader. The tests that assert visibility within 15 ms
(L1147) will be flaky in CI. `Instant::from_nanos` (L1059) doesn't exist in std.

**Proposal:** replace this with read-your-writes. `Query` requests go through the daemon,
which drains the channel and commits before running the query. That costs ~ms, which is
invisible next to a human typing. Drop the timing-based tests.

<a id="f6"></a>
### F6. The left-right index adds complexity and returns wrong results

L1197–1884:
- **Wrong results:** the hybrid query takes index hits first, then DB hits, and dedupes
  by ID. That is only correct for "most recent first" with no other ordering. It is wrong
  for `histdb top`/frequency ordering and for `--host`/`--dir` filters, where the newest
  1,000 rows aren't the newest *matching* rows across the whole DB, and for `limit`
  semantics.
- **Duplicates the page cache:** SQLite already keeps hot pages in memory. Indexed
  "latest N matching" queries over ~1M rows take ~1 ms.
- **Unreachable from the CLI:** standalone `histdb` CLI invocations are separate
  processes and can't use the daemon's in-memory index anyway.

**Proposal:** drop `read/` and the `left-right`/`crossbeam-queue` dependencies. If
profiling ever shows a need, add an SQLite covering index first.

<a id="f7"></a>
### F7. The 365-day default retention silently deletes history

L7257+: `max_age = 365d`, `per_host = 100k`, `global = 1M` by default. History gets more
valuable as it ages, and zsh-histdb never deletes anything. One million rows is only
~100–200 MB.

**Proposal:** all limits off by default. `histdb prune --dry-run` shows what a policy
would delete before it is enabled.

<a id="f8"></a>
### F8. The default durability mode loses acknowledged commands

L4497–4627: the default "fast" mode acks before persisting, so a daemon crash loses up
to one batch of *acknowledged* commands. The shell spool doesn't cover this window: the
shell already got "Accepted". The table at L4604 claims "≤10 ms" loss, but the loss is
whatever is in the batch, not a time.

**Proposal:** default to durable (commit before ack). At a few writes per second that is
free. Otherwise, document the window honestly.

<a id="f9"></a>
### F9. Wire format and versioning claims

- `bincode` 1.x is not self-describing, so "forward-compatible messages via
  `#[serde(default)]`" (L7832) doesn't hold: an added field breaks older readers.
- The schema comment says the operation column is "CBOR-encoded" (L6370), but the code
  uses bincode (L5316).

These were sync-only and moved to [deferred/sync.md](deferred/sync.md). For v1 IPC:
put a version byte or field on every frame, and keep the format human-debuggable.

<a id="f10"></a>
### F10. P2P sync correctness bugs (deferred; recorded for later)

1. **"Merge as single node" loses writes** (L6804–6906). Two Codespaces sharing one
   identity run *independent* HLCs, so both can mint `(origin, wall, logical) = (X, t, 0)`.
   Those collide on the `operation_log` primary key and one write is dropped.
2. **Delta sync permanently skips late ops.** Sync uses a watermark per origin
   (`vector_clock: NodeId → HLC`, L5269). If instance B's ops reach a peer *after*
   instance A's later ops, the watermark has already passed them and they are never
   requested again. That breaks the claimed strong eventual consistency.
3. **mDNS can't reach the target machines.** Codespaces and K8s pods are behind NAT on
   different networks (L6690, "Codespace reconnects to Desktop via mDNS discovery").
4. **Leader election buys nothing.** The data is a grow-only set plus delete markers.
   Monotone operations need no coordination (the CALM theorem), so the Bully election
   and its split-brain handling (L6636–6760) is machinery with no payoff.
5. **The value is unclear.** Per the product decision, P2P sync is out of v1 until there
   is a clear user story. See the direction notes in [deferred/sync.md](deferred/sync.md).

### F11. Process

- Every question is marked "decided", but the doc has no non-goals, no v1 scope and no
  success metrics. [README.md](README.md) now has a v1 scope and non-goals list.
- The doc is mostly code. Future revisions should keep the design docs to interfaces,
  invariants and data formats, and leave implementations to the code.

---

## Should we use Hydro?

**What it is:** [Hydro](https://hydro.run) (UC Berkeley + AWS) is a Rust framework for
writing distributed services as one program.
- Code is placed on typed *locations* (`Process`, `Cluster`).
- Network sends are typed and checked.
- It detects ordering and retry hazards at compile time and ships a simulator.
- The design rests on the CALM/lattice research line.
- Hydro Deploy places the processes.

The site says "production ready", but the crates are 0.x with breaking releases
(e.g. `hydro_deploy` 0.15, `lattices` 0.6).

**Verdict: don't depend on it.**
- v1 has no distribution at all.
- For any future sync, our data model is a grow-only set of immutable entries plus
  delete markers. That is too simple to justify a distributed-dataflow compiler.
- Hydro also assumes you deploy and place the processes. It doesn't target a laptop that
  is offline half the time.

**What to borrow:**
- *CALM*: monotone ops need no coordination, so the leader election goes away.
- *Simulation-first testing*: the same idea as deterministic simulation testing (see
  below).
- If we ever need merge algebra, the standalone `lattices` crate is small enough to read
  for ideas.

## Should we use Turso?

**What it is:** [Turso](https://github.com/tursodatabase/turso) (crate `turso`) is a
from-scratch Rust rewrite of SQLite. Its own docs report the following:
- Not yet 1.0. The README recommends keeping independent backups.
- Targets SQLite 3.50.x file-format and SQL compatibility, with gaps listed in
  `COMPAT.md`.
- `BEGIN CONCURRENT` (MVCC multi-writer) and io_uring async I/O.
- CDC and vector search.
- Full-text search (tantivy-based) is **experimental**.
- Multi-process WAL is **experimental**.
- Encryption at rest is **experimental**.

**Verdict: use rusqlite with bundled SQLite for v1.**
- v1 depends on exactly the experimental parts: FTS, and multiple processes reading
  while the daemon writes.
- Our write rate doesn't need MVCC.
- Keep the option open: all SQL lives in `histdb-core/src/db/` behind a small `Store`
  API, with no rusqlite types leaking into `ops/`, so we can swap later.
- Revisit at Turso 1.0. Its **CDC + sync engine** is the most credible way to implement
  the deferred sync without hand-rolling a replication protocol.

---

## Other things worth reusing or learning from

**Verified this session:**
- **[Atuin](https://github.com/atuinsh/atuin)** is the closest prior art: shell
  history in Rust, with sync.
  - It has an optional daemon (gRPC over a Unix socket) and a **record store**: tagged,
    append-only records per host.
  - Sync compares the max index per `(host, tag)` and ships the missing records. Records
    are end-to-end encrypted, so the server is a dumb store. History is
    `Create(History)` / `Delete(HistoryId)` records "built" into `history.db`.
  - It is the blueprint for deferred sync (see [deferred/sync.md](deferred/sync.md)), and
    its importers for zsh, bash, fish, nu and others are worth studying or reusing for
    the `ImportBackend` trait.
  - Its forum notes that all machines must run the same sync version. Version the
    record format from day one.

**From prior knowledge, not re-verified this session** (a background research pass was
lost to a container restart). Check before adopting:
- **iroh** (n0): QUIC connections dialled by public key, with relay fallback and hole
  punching. If P2P ever returns, this replaces Noise + mDNS + static addresses, and it
  works for Codespaces behind NAT.
- **SQLite session extension** (`rusqlite` `session` feature): built-in changesets and
  patchsets, a cheap way to produce "what changed since X" for a git- or file-based sync.
- **Deterministic simulation testing**: `turmoil` (tokio network simulation), `madsim`,
  `shuttle`/`loom` for concurrency permutation testing. Turso itself is built this way.
- **nucleo** (helix's fuzzy matcher) + **ratatui**: a built-in interactive search UI,
  instead of shelling out to fzf with the escaping bugs that brings.
- **reedline / Nushell's SQLite history**: nu already keeps its history in SQLite. It is
  an import source, and worth checking before adding a nu hook.
- **postcard**: a compact serde format if a binary format is ever needed. JSON or framed
  text is fine for v1 IPC.
