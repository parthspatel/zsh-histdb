# histdb-rs v1.0 — Architecture

A Rust rewrite of zsh-histdb focusing on **safety**, **performance**, and **correctness**.
Multi-shell support for **ZSH**, **BASH**, and **Nushell**.

> **Provenance.** This directory is the original `ARCHITECTURE.md` (branch
> `claude/code-review-GOrhf`, commit `607113b`) split into topic files. Text is carried
> over verbatim except:
> 1. P2P sync was removed from v1 and moved to [deferred/sync.md](deferred/sync.md).
> 2. `> **Review:**` callouts link to findings in [REVIEW.md](REVIEW.md).
>
> Everything cut out of a v1 section is preserved in the appendix of `deferred/sync.md`.

## Documents

| File | Contents |
|---|---|
| [REVIEW.md](REVIEW.md) | Architecture review: findings, a simpler v1 shape, Hydro / Turso / prior-art assessment |
| [01-project-structure.md](01-project-structure.md) | Workspace and crate layout |
| [02-daemon-and-ingest.md](02-daemon-and-ingest.md) | Write path, concurrency (queue, writer, left-right), daemon lifecycle, shutdown, crash recovery |
| [03-storage.md](03-storage.md) | SQLite connection, schema, migrations, models, data retention & pruning |
| [04-query-and-search.md](04-query-and-search.md) | Record / query / 3-way merge operations, pluggable search backend (FTS5) |
| [05-shell-integration.md](05-shell-integration.md) | IPC protocol, zsh / bash / nu hooks, offline disk spool, shell-facing CLI helpers, session IDs |
| [06-cli-config-import.md](06-cli-config-import.md) | Configuration, CLI, import interface, forget |
| [07-errors-testing-deps.md](07-errors-testing-deps.md) | Error types, testing strategy, dependencies |
| [deferred/sync.md](deferred/sync.md) | **Not in v1.** P2P sync design (HLC, CRDTs, log replication, Noise, topology, compaction, versioning) |

## v1 scope

**In scope**
- Record every command (argv, dir, host, session, start time, duration, exit status)
  from zsh, bash and nushell, with the shell never blocking on histdb.
- Local SQLite store, with schema migrations and import from the existing zsh-histdb database.
- Query, `top`, full-text search (FTS5), interactive search, and forget.
- Survive the daemon being down, using the disk spool and idempotent replay.
- Parity with zsh-histdb's existing **git-based sync** (`histdb sync` + 3-way merge).

**Non-goals for v1**
- P2P / live multi-machine sync, peer discovery, leader election, transport encryption
  between machines. The user value is unproven, so these are deferred to
  [deferred/sync.md](deferred/sync.md).
- Encryption at rest. The design stays "encryption-ready" only.
- Replacing SQLite (Turso, for example). Revisit after Turso 1.0; see [REVIEW.md](REVIEW.md#should-we-use-turso).

**Success criteria (proposed)**
- p99 shell hook overhead < 1 ms in zsh, < 3 ms in bash.
- Zero lost commands across daemon restart, crash and spool replay.
- Arbitrary bytes in argv/dir round-trip exactly (property-tested).

---

## Goals

### Safety
- **No SQL injection**: All queries use parameterized statements
- **Type-safe schema**: Compile-time checked database operations
- **Proper error handling**: All errors propagated via `Result<T, E>`
- **Transaction safety**: ACID guarantees for all multi-step operations
- **Input validation**: All user input validated before use
- **Encryption-ready**: Architecture supports future encryption at rest

### Performance
- **Lock-free ingestion**: Wait-free MPSC queue for concurrent shell sessions
- **Left-Right reads**: Wait-free reads via left-right concurrency pattern
- **Zero-copy where possible**: Minimize allocations in hot paths
- **Batched writes**: Coalesce multiple commands into single transactions
- **Prepared statements**: Pre-compiled queries for hot paths
- **Efficient merging**: Optimized 3-way merge algorithm
- **Minimal shell overhead**: Fire-and-forget IPC, sub-millisecond latency

### Correctness
- **Comprehensive tests**: Unit, integration, and property-based testing
- **Migration safety**: Atomic migrations with automatic rollback
- **Concurrent access**: Proper WAL handling for multi-shell scenarios

### Extensibility
- **Pluggable search**: SQLite FTS5 default, extensible for alternatives
- **Multi-shell**: ZSH, BASH, Nushell with shell-agnostic IPC
- **Future encryption**: Design accommodates SQLCipher or similar

---

## Migration Path

1. **Phase 1**: Core library + CLI (can coexist with zsh version)
2. **Phase 2**: Daemon + zsh integration
3. **Phase 3**: Import tool for existing databases
4. **Phase 4**: Deprecate zsh version

---

## Decision log

The original doc listed every question as "decided". After the review, they fall into
three groups.

### Decided for v1 (unchanged)
- ✅ **Daemon vs embedded**: Daemon (socket-activated), with shell-initiated fallback
- ✅ **Async runtime**: Tokio for socket handling; SQLite writes on a dedicated writer
- ✅ **Config format**: TOML with environment variable overrides
- ✅ **FTS**: SQLite FTS5 default, extensible via `SearchBackend` trait
- ✅ **Encryption**: Design supports future SQLCipher (deferred implementation)
- ✅ **Shell support**: ZSH, BASH, Nushell with shell-agnostic IPC
- ✅ **Daemon lifecycle**: systemd/launchd socket activation with shell-initiated fallback
   - Primary: Socket activation (zero resources until first connection)
   - Fallback: Shell starts daemon if socket missing (Alpine, BSDs)
   - Graceful shutdown drains queue and flushes SQLite
- ✅ **Import tool**: Interface-only with fresh instance implementation
   - `ImportBackend` trait for future extensibility
   - Default `NewInstance` implementation creates empty database
   - No import formats in initial release (deferred). The review suggests the zsh-histdb DB plus Atuin's importers as first candidates.
- ✅ **Shell offline queue**: Disk spool with atomic rename when daemon unavailable
   - Shell queues to `~/.local/share/histdb/spool/` via atomic rename
   - Daemon replays on startup/connect (processes in timestamp order)
- ✅ **Session ID generation**: Shell generates UUIDv7 (works with offline queue, time-ordered)
- ✅ **Sync (v1)**: Legacy git-based `histdb sync` + 3-way merge only. **No P2P.**

### Reopened by the review
- ⚠️ **Concurrency**: Lock-free MPSC queue for writes, left-right for reads. A bounded channel plus plain SQLite is proposed instead ([F6](REVIEW.md#f6)).
- ⚠️ **Backpressure strategy**: Retry with exponential backoff (100μs, 200μs, 400μs). This blocks the async runtime; spool-on-full is proposed ([F3](REVIEW.md#f3)).
- ⚠️ **History ID correlation**: "Shell tracks returned ID from StartCommand" contradicts shell-generated IDs and fire-and-forget ([F4](REVIEW.md#f4)).
- ⚠️ **Crash recovery**: "≤10ms data loss" actually means one batch of already-acknowledged commands. Durable by default is proposed ([F8](REVIEW.md#f8)).
- ⚠️ **Shell hook implementation**: Fork-free hooks and safe framing are proposed ([F1](REVIEW.md#f1), [F2](REVIEW.md#f2)).
- ⚠️ **Data retention**: Configurable limits with smart pruning strategies. Defaults (365 days / 100k per host / 1M) would delete history silently; unlimited by default is proposed ([F7](REVIEW.md#f7)).
   - Strategies: oldest, LRU, LFU, smart (weighted scoring)
   - Protected commands: starred, long-running, pattern-matched
   - Usage tracking (use_count, last_used) for smart pruning

### Deferred with P2P sync ([deferred/sync.md](deferred/sync.md))
- ⏸ **Sync**: Hybrid (Raft-style log replication + CRDT semantics), P2P cluster with operation-based CRDTs, HLC causal ordering, mDNS discovery
- ⏸ **Sync security**: Noise Protocol with Ed25519 keys
- ⏸ **Sync topology**: Dual-mode (P2P mesh + hub-spoke with Bully leader election)
- ⏸ **Ephemeral machine support**: Portable identity + token enrollment
- ⏸ **Log compaction**: Tiered strategy with peer-class awareness
- ⏸ **Protocol versioning**: Semantic versioning with ±1 major compatibility (sync protocol)

### Still open
- Accept or reject each review finding ([REVIEW.md](REVIEW.md)).
- What user problem would justify live multi-machine sync beyond git sync? (See [deferred/sync.md](deferred/sync.md).)

---

## Summary

This architecture prioritizes:

- **Safety**: Parameterized queries, type-safe models, proper error handling
- **Performance**: Fork-free shell hooks, batched SQLite writes (see review for the read path)
- **Correctness**: Transaction safety, comprehensive testing, atomic migrations
- **Extensibility**: Pluggable search backends, multi-shell support, encryption-ready
- **Maintainability**: Clean separation of concerns, well-defined interfaces

### Key Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Write path | Lock-free MPSC queue (⚠️ review: bounded channel) | Sub-millisecond shell latency |
| Read path | Left-right pattern (⚠️ review: plain SQLite) | Wait-free queries |
| Storage | SQLite + WAL (rusqlite, bundled) | Reliable, portable, well-understood |
| Search | FTS5 (extensible) | Fast full-text, graceful fallback |
| Sync | v1: git + 3-way merge; P2P deferred | User value of P2P unproven |
| Config | TOML | Human-readable, well-supported |
| Shells | ZSH, BASH, Nushell | Cover majority of users |
