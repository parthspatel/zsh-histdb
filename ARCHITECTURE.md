# histdb-rs: Rust Rewrite Architecture

## Overview

A Rust rewrite of zsh-histdb focusing on **safety**, **performance**, and **correctness**.
Multi-shell support for **ZSH**, **BASH**, and **Nushell**.

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

## Project Structure

```
histdb-rs/
├── Cargo.toml                    # Workspace manifest
├── crates/
│   ├── histdb-core/              # Core library
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── db/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── connection.rs # Connection management (encryption-ready)
│   │   │   │   ├── schema.rs     # Type-safe schema definitions
│   │   │   │   ├── queries.rs    # Prepared query definitions
│   │   │   │   └── migrations.rs # Schema migrations
│   │   │   ├── ingest/           # Lock-free ingestion pipeline
│   │   │   │   ├── mod.rs
│   │   │   │   ├── queue.rs      # Lock-free MPSC queue (writes)
│   │   │   │   ├── writer.rs     # Single-consumer writer thread
│   │   │   │   └── batch.rs      # Batch accumulation logic
│   │   │   ├── read/             # Left-Right read path
│   │   │   │   ├── mod.rs
│   │   │   │   ├── left_right.rs # Left-right data structure
│   │   │   │   ├── index.rs      # In-memory recent history index
│   │   │   │   └── reader.rs     # Wait-free reader handle
│   │   │   ├── search/           # Extensible search backend
│   │   │   │   ├── mod.rs
│   │   │   │   ├── traits.rs     # SearchBackend trait
│   │   │   │   ├── fts5.rs       # SQLite FTS5 implementation
│   │   │   │   └── simple.rs     # GLOB/LIKE fallback
│   │   │   ├── models/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── command.rs    # Command model
│   │   │   │   ├── place.rs      # Place (host+dir) model
│   │   │   │   ├── history.rs    # History entry model
│   │   │   │   └── session.rs    # Session model
│   │   │   ├── ops/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── record.rs     # Record new history (via queue)
│   │   │   │   ├── query.rs      # Query history (via left-right)
│   │   │   │   ├── forget.rs     # Delete history
│   │   │   │   └── stats.rs      # Statistics
│   │   │   ├── sync/             # Hybrid sync (log replication + CRDTs)
│   │   │   │   ├── mod.rs
│   │   │   │   ├── hlc.rs        # Hybrid Logical Clock
│   │   │   │   ├── crdt.rs       # CRDT types (G-Set, LWW-Register)
│   │   │   │   ├── log.rs        # Operation log (append-only)
│   │   │   │   ├── protocol.rs   # Sync protocol messages
│   │   │   │   └── peer.rs       # Peer connection management
│   │   │   ├── import/           # Import backends (extensible)
│   │   │   │   ├── mod.rs
│   │   │   │   ├── traits.rs     # ImportBackend trait
│   │   │   │   └── new.rs        # NewInstance (fresh DB)
│   │   │   ├── config.rs         # TOML configuration
│   │   │   └── error.rs          # Error types
│   │   └── Cargo.toml
│   │
│   ├── histdb-cli/               # Command-line interface
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── commands/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── query.rs      # histdb query command
│   │   │   │   ├── top.rs        # histdb-top command
│   │   │   │   ├── sync.rs       # histdb-sync command
│   │   │   │   ├── forget.rs     # histdb --forget
│   │   │   │   ├── import.rs     # Import from shell history files
│   │   │   │   └── export.rs     # Export to JSON/CSV
│   │   │   ├── output.rs         # Output formatting
│   │   │   └── args.rs           # Argument parsing (clap)
│   │   └── Cargo.toml
│   │
│   └── histdb-daemon/            # Shell-agnostic daemon
│       ├── src/
│       │   ├── main.rs           # Daemon entry point
│       │   ├── ipc.rs            # Unix socket IPC
│       │   ├── protocol.rs       # Wire protocol (JSON)
│       │   └── handlers.rs       # Request handlers
│       └── Cargo.toml
│
├── shell/                        # Shell integrations
│   ├── zsh/
│   │   ├── histdb.zsh            # ZSH plugin
│   │   └── histdb-isearch.zsh    # Interactive search widget
│   ├── bash/
│   │   ├── histdb.bash           # BASH plugin
│   │   └── histdb-search.bash    # Interactive search (fzf-based)
│   └── nu/
│       ├── histdb.nu             # Nushell module
│       └── histdb-search.nu      # Interactive search
│
├── migrations/                   # SQL migration files
│   ├── 001_initial.sql
│   ├── 002_add_indexes.sql
│   ├── 003_add_fts5.sql          # Optional FTS5 virtual table
│   └── 004_add_sync.sql          # UUID, origin_host, sync_state
│
├── config/
│   └── histdb.example.toml       # Example configuration
│
└── tests/                        # Integration tests
    ├── migration_tests.rs
    ├── merge_tests.rs
    ├── left_right_tests.rs
    └── concurrent_tests.rs
```

---

## Concurrency Architecture (Lock-Free Design)

The daemon uses a **lock-free, wait-free** architecture to handle concurrent shell sessions with minimal latency.

### Design Goals
- Shell hooks return in **< 1ms** (fire-and-forget)
- Support **100+ concurrent shell sessions** without contention
- **No mutex locks** on the hot path (command recording)
- Queries see **eventually consistent** data (typically < 10ms delay)

### Architecture Overview

```
┌─────────┐ ┌─────────┐ ┌─────────┐
│ Shell 1 │ │ Shell 2 │ │ Shell N │   Multiple producers (shells)
└────┬────┘ └────┬────┘ └────┬────┘
     │           │           │
     ▼           ▼           ▼
┌────────────────────────────────────┐
│   Lock-Free MPSC Ring Buffer       │   Wait-free enqueue
│   (crossbeam-queue ArrayQueue)     │   Bounded capacity (e.g., 4096)
└─────────────────┬──────────────────┘
                  │
                  ▼ (single consumer)
┌─────────────────────────────────────┐
│         Writer Thread               │
│  ┌─────────────────────────────┐   │
│  │  Batch Accumulator          │   │   Collects entries for batching
│  │  (VecDeque, drain every     │   │
│  │   10ms or 64 entries)       │   │
│  └──────────────┬──────────────┘   │
│                 ▼                   │
│  ┌─────────────────────────────┐   │
│  │  SQLite Transaction         │   │   Single writer, batched inserts
│  │  (prepared statements)      │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

### Data Structures

#### 1. Lock-Free Command Queue

```rust
// ingest/queue.rs
use crossbeam_queue::ArrayQueue;

/// Lock-free bounded queue for incoming commands
/// - Wait-free push (returns error if full, never blocks)
/// - Wait-free pop (returns None if empty)
pub struct CommandQueue {
    queue: ArrayQueue<PendingCommand>,
    // Atomic counter for monitoring
    pending_count: AtomicU64,
}

#[derive(Clone)]
pub struct PendingCommand {
    pub kind: CommandKind,
    pub timestamp: Instant,  // For latency tracking
}

pub enum CommandKind {
    Start {
        session_id: u64,
        argv: Box<str>,        // Owned, no lifetime issues
        dir: Box<Path>,
        host: Box<str>,
        start_time: i64,
        response_tx: Option<oneshot::Sender<HistoryId>>,
    },
    Finish {
        history_id: HistoryId,
        exit_status: i32,
        duration_ms: u64,
    },
}

impl CommandQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: ArrayQueue::new(capacity),
            pending_count: AtomicU64::new(0),
        }
    }

    /// Wait-free enqueue. Returns Err if queue is full.
    #[inline]
    pub fn push(&self, cmd: PendingCommand) -> Result<(), PendingCommand> {
        self.queue.push(cmd).map(|_| {
            self.pending_count.fetch_add(1, Ordering::Relaxed);
        })
    }

    /// Wait-free dequeue for the writer thread.
    #[inline]
    pub fn pop(&self) -> Option<PendingCommand> {
        self.queue.pop().map(|cmd| {
            self.pending_count.fetch_sub(1, Ordering::Relaxed);
            cmd
        })
    }

    /// Drain up to `max` entries into a Vec (for batching)
    pub fn drain_batch(&self, max: usize) -> Vec<PendingCommand> {
        let mut batch = Vec::with_capacity(max);
        while batch.len() < max {
            match self.pop() {
                Some(cmd) => batch.push(cmd),
                None => break,
            }
        }
        batch
    }
}
```

#### 2. Writer Thread (Single Consumer)

```rust
// ingest/writer.rs
pub struct WriterThread {
    queue: Arc<CommandQueue>,
    conn: Connection,
    // Pre-compiled statements for hot path
    insert_command_stmt: Statement,
    insert_place_stmt: Statement,
    insert_history_stmt: Statement,
    update_finish_stmt: Statement,
}

impl WriterThread {
    pub fn spawn(queue: Arc<CommandQueue>, db_path: &Path) -> JoinHandle<()> {
        let conn = Connection::open(db_path).expect("Failed to open database");
        // Optimize for write throughput
        conn.execute_batch("
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA wal_autocheckpoint = 1000;
            PRAGMA busy_timeout = 5000;
        ").unwrap();

        std::thread::spawn(move || {
            let mut writer = WriterThread::new(queue, conn);
            writer.run();
        })
    }

    fn run(&mut self) {
        let mut batch = Vec::with_capacity(64);
        let batch_interval = Duration::from_millis(10);
        let mut last_flush = Instant::now();

        loop {
            // Drain available commands (non-blocking)
            batch.extend(self.queue.drain_batch(64 - batch.len()));

            let should_flush = !batch.is_empty() && (
                batch.len() >= 64 ||                        // Batch full
                last_flush.elapsed() >= batch_interval      // Time limit
            );

            if should_flush {
                self.flush_batch(&mut batch);
                last_flush = Instant::now();
            } else if batch.is_empty() {
                // No work: park thread briefly to avoid busy-spin
                std::thread::park_timeout(Duration::from_millis(1));
            }
        }
    }

    fn flush_batch(&mut self, batch: &mut Vec<PendingCommand>) {
        if batch.is_empty() {
            return;
        }

        // Single transaction for entire batch
        let tx = self.conn.transaction().unwrap();

        for cmd in batch.drain(..) {
            match cmd.kind {
                CommandKind::Start { session_id, argv, dir, host, start_time, response_tx } => {
                    let id = self.insert_history_entry(&tx, session_id, &argv, &dir, &host, start_time);
                    if let Some(tx) = response_tx {
                        let _ = tx.send(id);  // Ignore if receiver dropped
                    }
                }
                CommandKind::Finish { history_id, exit_status, duration_ms } => {
                    self.update_finish(&tx, history_id, exit_status, duration_ms);
                }
            }
        }

        tx.commit().unwrap();
    }

    #[inline]
    fn insert_history_entry(
        &self,
        tx: &Transaction,
        session_id: u64,
        argv: &str,
        dir: &Path,
        host: &str,
        start_time: i64,
    ) -> HistoryId {
        // Uses prepared statements, parameterized queries
        // ... implementation
    }
}
```

#### 3. Fire-and-Forget IPC Handler

```rust
// daemon/handler.rs
pub struct ConnectionHandler {
    queue: Arc<CommandQueue>,
    session_id: u64,
    host: Box<str>,
}

impl ConnectionHandler {
    /// Handle incoming StartCommand - returns immediately
    pub async fn handle_start(&self, argv: String, dir: String) -> Response {
        let cmd = PendingCommand {
            kind: CommandKind::Start {
                session_id: self.session_id,
                argv: argv.into_boxed_str(),
                dir: PathBuf::from(dir).into_boxed_path(),
                host: self.host.clone(),
                start_time: Utc::now().timestamp(),
                response_tx: None,  // Fire-and-forget: no response channel
            },
            timestamp: Instant::now(),
        };

        match self.queue.push(cmd) {
            Ok(()) => Response::Accepted,
            Err(_) => Response::QueueFull,  // Backpressure signal
        }
    }

    /// Handle StartCommand with ID response (for finish correlation)
    pub async fn handle_start_with_id(&self, argv: String, dir: String) -> Response {
        let (tx, rx) = oneshot::channel();

        let cmd = PendingCommand {
            kind: CommandKind::Start {
                session_id: self.session_id,
                argv: argv.into_boxed_str(),
                dir: PathBuf::from(dir).into_boxed_path(),
                host: self.host.clone(),
                start_time: Utc::now().timestamp(),
                response_tx: Some(tx),
            },
            timestamp: Instant::now(),
        };

        if self.queue.push(cmd).is_err() {
            return Response::QueueFull;
        }

        // Wait for writer to process (typically < 10ms)
        match tokio::time::timeout(Duration::from_millis(100), rx).await {
            Ok(Ok(id)) => Response::CommandStarted { id: id.0 },
            _ => Response::Timeout,
        }
    }
}
```

### Latency Characteristics

| Operation | Expected Latency | Blocking? |
|-----------|-----------------|-----------|
| Shell → Queue (push) | < 100ns | **No** (wait-free) |
| Queue → SQLite (batch) | 1-10ms | Yes (writer only) |
| Shell hook total | < 1ms | No |
| Query (from SQLite) | 1-50ms | Yes (read lock) |

### Backpressure Handling

When the queue is full (extremely rare with 4096 capacity), use **retry with exponential backoff**:

```rust
// ingest/queue.rs
impl CommandQueue {
    /// Push with retry and exponential backoff
    /// Returns Ok(()) on success, Err after max retries
    pub fn push_with_retry(&self, cmd: PendingCommand, max_retries: u32) -> Result<(), PendingCommand> {
        let mut cmd = cmd;

        for attempt in 0..max_retries {
            match self.queue.push(cmd) {
                Ok(()) => {
                    self.pending_count.fetch_add(1, Ordering::Relaxed);
                    if attempt > 0 {
                        tracing::debug!("Queue push succeeded after {} retries", attempt);
                    }
                    return Ok(());
                }
                Err(returned) => {
                    cmd = returned;
                    if attempt < max_retries - 1 {
                        // Exponential backoff: 100μs, 200μs, 400μs
                        let delay = Duration::from_micros(100 << attempt);
                        std::thread::sleep(delay);
                    }
                }
            }
        }

        tracing::warn!("Queue full after {} retries, command may be lost", max_retries);
        Err(cmd)
    }
}

// daemon/handler.rs - Usage
impl ConnectionHandler {
    pub async fn handle_start(&self, argv: String, dir: String) -> Response {
        let cmd = PendingCommand { /* ... */ };

        // Retry up to 3 times with backoff (100μs, 200μs, 400μs)
        match self.queue.push_with_retry(cmd, 3) {
            Ok(()) => Response::Accepted,
            Err(_) => {
                // Final fallback: metrics + error response
                metrics::counter!("histdb.queue.drops").increment(1);
                Response::QueueFull
            }
        }
    }
}
```

**Backoff schedule:**
| Attempt | Delay | Cumulative |
|---------|-------|------------|
| 1 | 100μs | 100μs |
| 2 | 200μs | 300μs |
| 3 | 400μs | 700μs |

This keeps total worst-case latency under 1ms while giving the writer thread time to drain.

### Query Consistency

Queries read directly from SQLite, which may be slightly behind the queue:

```rust
// query.rs
impl QueryExecutor {
    pub fn query(&self, builder: QueryBuilder) -> Result<Vec<HistoryRow>, Error> {
        // Option 1: Read from SQLite only (simple, eventually consistent)
        self.conn.query(&builder.build_sql())

        // Option 2: Merge with pending queue (complex, strongly consistent)
        // let db_results = self.conn.query(&builder.build_sql())?;
        // let pending = self.queue.peek_matching(&builder);
        // merge_results(db_results, pending)
    }
}
```

**Recommendation**: Use eventual consistency (Option 1). The 10ms delay is imperceptible for interactive use, and strong consistency adds complexity.

### Thread Model Summary

```
┌──────────────────────────────────────────────────────────┐
│                     histdb-daemon                        │
├──────────────────────────────────────────────────────────┤
│  Thread 1: Async Runtime (tokio)                         │
│    - Accept Unix socket connections                      │
│    - Parse JSON requests                                 │
│    - Push to lock-free queue (wait-free)                 │
│    - Handle queries via left-right reader (wait-free)    │
├──────────────────────────────────────────────────────────┤
│  Thread 2: Writer Thread                                 │
│    - Drain queue (single consumer)                       │
│    - Batch writes to SQLite                              │
│    - Update left-right index after commit                │
│    - Own the write connection                            │
├──────────────────────────────────────────────────────────┤
│  Shared State:                                           │
│    - Arc<CommandQueue>       (lock-free write queue)     │
│    - Arc<LeftRight<Index>>   (wait-free read index)      │
│    - Atomic counters for monitoring                      │
└──────────────────────────────────────────────────────────┘
```

---

## Left-Right Concurrency (Wait-Free Reads)

The **left-right pattern** provides wait-free reads while maintaining consistency with writes.
This balances the write-heavy ingestion with read-heavy queries.

### How Left-Right Works

```
                    ┌─────────────────────────────────────────────┐
                    │            Left-Right Structure             │
                    ├─────────────────────────────────────────────┤
   Readers ────────▶│  ┌─────────┐         ┌─────────┐           │
   (many, wait-free)│  │  LEFT   │         │  RIGHT  │           │
                    │  │  Index  │         │  Index  │           │
                    │  └─────────┘         └─────────┘           │
                    │       ▲                   ▲                 │
                    │       │                   │                 │
                    │  ┌────┴───────────────────┴────┐           │
                    │  │     Active Side Pointer     │           │
                    │  │        (atomic swap)        │           │
                    │  └─────────────────────────────┘           │
                    │                   ▲                        │
   Writer ──────────┼───────────────────┘                        │
   (one, exclusive) │  1. Write to inactive side                 │
                    │  2. Swap pointer                           │
                    │  3. Wait for old readers to finish         │
                    │  4. Write to other side (now inactive)     │
                    └─────────────────────────────────────────────┘
```

### Key Properties

| Property | Guarantee |
|----------|-----------|
| Reader latency | **Wait-free** (no locks, no retries) |
| Writer latency | Bounded (waits for reader epoch) |
| Consistency | Readers see consistent snapshot |
| Memory | 2x index size (both sides) |

### Implementation

```rust
// read/left_right.rs
use left_right::{ReadHandle, WriteHandle};

/// In-memory index of recent history for fast queries
pub struct HistoryIndex {
    /// Recent commands (LRU, bounded size)
    recent: Vec<RecentEntry>,
    /// Command → entry IDs (for dedup/lookup)
    by_command: HashMap<Box<str>, Vec<EntryId>>,
    /// Directory → entry IDs (for --in/--at queries)
    by_dir: HashMap<Box<Path>, Vec<EntryId>>,
    /// Last N entries per session (for isearch)
    by_session: HashMap<u64, VecDeque<EntryId>>,
}

#[derive(Clone)]
pub struct RecentEntry {
    pub id: EntryId,
    pub session: u64,
    pub argv: Box<str>,
    pub dir: Box<Path>,
    pub host: Box<str>,
    pub start_time: i64,
    pub exit_status: Option<i32>,
}

/// Operations that can be applied to the index
pub enum IndexOp {
    Insert(RecentEntry),
    UpdateExitStatus { id: EntryId, status: i32, duration: u64 },
    Evict { before: i64 },  // Evict entries older than timestamp
}

impl Absorb<IndexOp> for HistoryIndex {
    fn absorb_first(&mut self, op: &mut IndexOp, _: &Self) {
        match op {
            IndexOp::Insert(entry) => {
                self.by_command
                    .entry(entry.argv.clone())
                    .or_default()
                    .push(entry.id);
                self.by_dir
                    .entry(entry.dir.clone())
                    .or_default()
                    .push(entry.id);
                self.by_session
                    .entry(entry.session)
                    .or_default()
                    .push_back(entry.id);
                self.recent.push(entry.clone());
            }
            IndexOp::UpdateExitStatus { id, status, duration } => {
                if let Some(entry) = self.recent.iter_mut().find(|e| e.id == *id) {
                    entry.exit_status = Some(*status);
                }
            }
            IndexOp::Evict { before } => {
                self.recent.retain(|e| e.start_time >= *before);
                // Also clean up maps...
            }
        }
    }

    fn absorb_second(&mut self, op: IndexOp, other: &Self) {
        // Same logic, can optimize by copying from `other`
        self.absorb_first(&mut op, other);
    }
}

/// Writer handle (owned by writer thread)
pub struct IndexWriter {
    write: WriteHandle<HistoryIndex, IndexOp>,
}

impl IndexWriter {
    pub fn insert(&mut self, entry: RecentEntry) {
        self.write.append(IndexOp::Insert(entry));
    }

    pub fn update_exit_status(&mut self, id: EntryId, status: i32, duration: u64) {
        self.write.append(IndexOp::UpdateExitStatus { id, status, duration });
    }

    /// Publish pending changes to readers
    pub fn publish(&mut self) {
        self.write.publish();
    }
}

/// Reader handle (cloneable, used by query handlers)
#[derive(Clone)]
pub struct IndexReader {
    read: ReadHandle<HistoryIndex>,
}

impl IndexReader {
    /// Wait-free read access
    pub fn read(&self) -> impl Deref<Target = HistoryIndex> + '_ {
        self.read.enter().unwrap()
    }

    /// Query recent history by pattern
    pub fn search_recent(&self, pattern: &str, limit: usize) -> Vec<RecentEntry> {
        let guard = self.read();
        guard.recent
            .iter()
            .rev()  // Most recent first
            .filter(|e| e.argv.contains(pattern))
            .take(limit)
            .cloned()
            .collect()
    }

    /// Query by directory (for --in flag)
    pub fn search_in_dir(&self, dir: &Path, limit: usize) -> Vec<RecentEntry> {
        let guard = self.read();
        guard.by_dir
            .iter()
            .filter(|(d, _)| d.starts_with(dir))
            .flat_map(|(_, ids)| ids.iter())
            .filter_map(|id| guard.recent.iter().find(|e| e.id == *id))
            .take(limit)
            .cloned()
            .collect()
    }
}
```

### Integration with Write Path

```rust
// ingest/writer.rs - Updated
impl WriterThread {
    fn flush_batch(&mut self, batch: &mut Vec<PendingCommand>) {
        let tx = self.conn.transaction().unwrap();
        let mut index_ops = Vec::with_capacity(batch.len());

        for cmd in batch.drain(..) {
            match cmd.kind {
                CommandKind::Start { session_id, argv, dir, host, start_time, response_tx } => {
                    let id = self.insert_history_entry(&tx, session_id, &argv, &dir, &host, start_time);

                    // Queue index update
                    index_ops.push(IndexOp::Insert(RecentEntry {
                        id,
                        session: session_id,
                        argv: argv.clone(),
                        dir: dir.clone(),
                        host: host.clone(),
                        start_time,
                        exit_status: None,
                    }));

                    if let Some(tx) = response_tx {
                        let _ = tx.send(id);
                    }
                }
                CommandKind::Finish { history_id, exit_status, duration_ms } => {
                    self.update_finish(&tx, history_id, exit_status, duration_ms);
                    index_ops.push(IndexOp::UpdateExitStatus {
                        id: history_id,
                        status: exit_status,
                        duration: duration_ms,
                    });
                }
            }
        }

        tx.commit().unwrap();

        // Apply index updates and publish to readers
        for op in index_ops {
            self.index_writer.append(op);
        }
        self.index_writer.publish();  // Readers now see new entries
    }
}
```

### Query Path (Wait-Free)

```rust
// ops/query.rs
pub struct QueryExecutor {
    /// Wait-free reader for recent entries
    index: IndexReader,
    /// SQLite connection for older entries
    conn: Connection,
    /// Search backend (FTS5 or GLOB)
    search: Box<dyn SearchBackend>,
}

impl QueryExecutor {
    pub fn query(&self, builder: &QueryBuilder) -> Result<Vec<HistoryRow>, Error> {
        // 1. Fast path: check in-memory index first (wait-free)
        let recent = self.index.search_recent(
            builder.pattern.as_deref().unwrap_or(""),
            builder.limit.unwrap_or(100),
        );

        if recent.len() >= builder.limit.unwrap_or(100) {
            // Index had enough results, skip SQLite
            return Ok(recent.into_iter().map(Into::into).collect());
        }

        // 2. Slow path: query SQLite for older entries
        let from_db = self.search.search(&self.conn, builder)?;

        // 3. Merge and deduplicate
        Ok(merge_results(recent, from_db, builder.limit))
    }
}
```

### Memory Budget

```rust
// config.rs
pub struct IndexConfig {
    /// Maximum entries in memory (default: 10,000)
    pub max_entries: usize,
    /// Evict entries older than this (default: 7 days)
    pub max_age: Duration,
    /// Memory limit for index (default: 50MB)
    pub memory_limit: usize,
}
```

---

## Core Components

### 1. Database Layer (`histdb-core/src/db/`)

#### Connection Management
```rust
// connection.rs
pub struct ConnectionPool {
    pool: r2d2::Pool<SqliteConnectionManager>,
}

impl ConnectionPool {
    pub fn new(path: &Path) -> Result<Self, Error>;
    pub fn get(&self) -> Result<PooledConnection, Error>;
}
```

**Key decisions:**
- Use `rusqlite` for SQLite bindings
- `r2d2` connection pool for multi-threaded access
- WAL mode enabled by default
- Busy timeout configured for concurrent access

#### Schema (Type-Safe)
```rust
// schema.rs
pub struct Command {
    pub id: i64,
    pub argv: String,
}

pub struct Place {
    pub id: i64,
    pub host: String,
    pub dir: PathBuf,
}

pub struct HistoryEntry {
    pub id: i64,
    pub session: i64,
    pub command_id: i64,
    pub place_id: i64,
    pub exit_status: Option<i32>,
    pub start_time: DateTime<Utc>,
    pub duration: Option<Duration>,
}
```

#### Migrations
```rust
// migrations.rs
pub struct Migrator {
    conn: Connection,
    migrations_dir: PathBuf,
}

impl Migrator {
    /// Run migrations with automatic rollback on failure
    pub fn migrate(&mut self) -> Result<(), MigrationError>;

    /// Create backup before destructive migrations
    pub fn backup(&self, suffix: &str) -> Result<PathBuf, Error>;
}
```

**Key safety features:**
- All migrations run in a transaction
- Automatic rollback on failure
- Backup created before schema changes
- Version tracking via `user_version` pragma

---

### 2. Models Layer (`histdb-core/src/models/`)

Type-safe models with validation:

```rust
// command.rs
pub struct NewCommand<'a> {
    argv: &'a str,
}

impl<'a> NewCommand<'a> {
    pub fn new(argv: &'a str) -> Result<Self, ValidationError> {
        // Validate: non-empty, no null bytes
        if argv.is_empty() {
            return Err(ValidationError::EmptyCommand);
        }
        if argv.contains('\0') {
            return Err(ValidationError::NullByte);
        }
        Ok(Self { argv })
    }
}
```

---

### 3. Operations Layer (`histdb-core/src/ops/`)

#### Recording History
```rust
// record.rs
pub struct Recorder<'a> {
    conn: &'a Connection,
    session_id: i64,
    host: &'a str,
}

impl<'a> Recorder<'a> {
    /// Record a new command execution (called on zshaddhistory)
    pub fn start_command(
        &self,
        argv: &str,
        dir: &Path,
    ) -> Result<HistoryId, Error>;

    /// Update with exit status (called on precmd)
    pub fn finish_command(
        &self,
        id: HistoryId,
        exit_status: i32,
    ) -> Result<(), Error>;
}
```

**Safety**: Uses a single transaction for the insert sequence:
```rust
pub fn start_command(&self, argv: &str, dir: &Path) -> Result<HistoryId, Error> {
    let tx = self.conn.transaction()?;

    // All inserts in one transaction - no race conditions
    tx.execute(
        "INSERT OR IGNORE INTO commands (argv) VALUES (?1)",
        params![argv],
    )?;

    tx.execute(
        "INSERT OR IGNORE INTO places (host, dir) VALUES (?1, ?2)",
        params![self.host, dir.display().to_string()],
    )?;

    tx.execute(
        "INSERT INTO history (session, command_id, place_id, start_time)
         SELECT ?1, c.id, p.id, ?2
         FROM commands c, places p
         WHERE c.argv = ?3 AND p.host = ?4 AND p.dir = ?5",
        params![self.session_id, Utc::now().timestamp(), argv, self.host, dir.display().to_string()],
    )?;

    let id = tx.last_insert_rowid();
    tx.commit()?;
    Ok(HistoryId(id))
}
```

#### Querying History
```rust
// query.rs
pub struct QueryBuilder {
    host_filter: Option<String>,
    dir_filter: Option<DirFilter>,
    session_filter: Option<i64>,
    time_range: Option<TimeRange>,
    pattern: Option<String>,
    limit: Option<usize>,
    order: Order,
}

impl QueryBuilder {
    pub fn new() -> Self;
    pub fn host(self, host: &str) -> Self;
    pub fn in_dir(self, dir: &Path) -> Self;      // dir and subdirs
    pub fn at_dir(self, dir: &Path) -> Self;      // exact dir
    pub fn session(self, session: i64) -> Self;
    pub fn from(self, from: DateTime<Utc>) -> Self;
    pub fn until(self, until: DateTime<Utc>) -> Self;
    pub fn pattern(self, pattern: &str) -> Self;  // LIKE pattern
    pub fn glob(self, glob: &str) -> Self;        // GLOB pattern (escaped!)
    pub fn limit(self, n: usize) -> Self;
    pub fn descending(self) -> Self;

    /// Execute query with parameterized SQL (no injection possible)
    pub fn execute(&self, conn: &Connection) -> Result<Vec<HistoryRow>, Error>;
}
```

**Key safety**: The `glob()` method properly escapes GLOB special characters:
```rust
pub fn escape_glob(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    for c in input.chars() {
        match c {
            '*' | '?' | '[' | ']' => {
                result.push('[');
                result.push(c);
                result.push(']');
            }
            _ => result.push(c),
        }
    }
    result
}
```

#### 3-Way Merge
```rust
// merge.rs
pub struct Merger {
    ancestor: PathBuf,
    ours: PathBuf,
    theirs: PathBuf,
}

impl Merger {
    pub fn new(ancestor: PathBuf, ours: PathBuf, theirs: PathBuf) -> Self;

    /// Perform 3-way merge with conflict detection
    pub fn merge(&self) -> Result<MergeResult, MergeError>;
}

pub enum MergeResult {
    Success { entries_added: usize },
    VersionMismatch { ours: u32, theirs: u32 },
}
```

---

### 4. Configuration (`histdb-core/src/config.rs`)

```rust
#[derive(Debug, Deserialize)]
pub struct Config {
    pub database: DatabaseConfig,
    pub daemon: DaemonConfig,
    pub index: IndexConfig,
    pub search: SearchConfig,
    pub ignore: IgnoreConfig,
}

#[derive(Debug, Deserialize)]
pub struct DatabaseConfig {
    /// Database file path (default: ~/.histdb/history.db)
    pub path: PathBuf,
    /// Hostname override (default: system hostname)
    pub hostname: Option<String>,
    /// Enable encryption (future: requires SQLCipher)
    pub encrypted: bool,
}

#[derive(Debug, Deserialize)]
pub struct DaemonConfig {
    /// Unix socket path
    pub socket_path: PathBuf,
    /// Write batch size
    pub batch_size: usize,
    /// Write batch interval (ms)
    pub batch_interval_ms: u64,
    /// Command queue capacity
    pub queue_capacity: usize,
}

#[derive(Debug, Deserialize)]
pub struct IndexConfig {
    /// Maximum entries in left-right index
    pub max_entries: usize,
    /// Evict entries older than (days)
    pub max_age_days: u32,
    /// Memory limit (bytes)
    pub memory_limit: usize,
}

#[derive(Debug, Deserialize)]
pub struct SearchConfig {
    /// Search backend: "fts5" | "glob" | "like"
    pub backend: SearchBackendType,
    /// FTS5 tokenizer (if using fts5)
    pub fts5_tokenizer: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct IgnoreConfig {
    /// Regex patterns for commands to ignore
    pub patterns: Vec<String>,
    /// Respect shell's histignorespace
    pub space_prefix: bool,
}

impl Config {
    /// Load from ~/.config/histdb/config.toml with env overrides
    pub fn load() -> Result<Self, ConfigError> {
        let path = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("~/.config"))
            .join("histdb/config.toml");

        let mut config: Config = if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            toml::from_str(&content)?
        } else {
            Config::default()
        };

        // Environment variable overrides
        if let Ok(db_path) = std::env::var("HISTDB_DATABASE") {
            config.database.path = PathBuf::from(db_path);
        }
        if let Ok(socket) = std::env::var("HISTDB_SOCKET") {
            config.daemon.socket_path = PathBuf::from(socket);
        }

        Ok(config)
    }
}
```

**Config file format** (`~/.config/histdb/config.toml`):
```toml
# histdb configuration

[database]
path = "~/.histdb/history.db"
hostname = "my-machine"          # Optional: override system hostname
encrypted = false                # Future: SQLCipher encryption

[daemon]
socket_path = "/run/user/1000/histdb.sock"  # Or use $XDG_RUNTIME_DIR
batch_size = 64                  # Commands per batch
batch_interval_ms = 10           # Max wait before flush
queue_capacity = 4096            # Lock-free queue size

[index]
max_entries = 10000              # Entries in left-right index
max_age_days = 7                 # Evict older entries
memory_limit = 52428800          # 50MB

[search]
backend = "fts5"                 # "fts5" | "glob" | "like"
fts5_tokenizer = "unicode61"     # FTS5 tokenizer

[ignore]
space_prefix = true              # Ignore commands starting with space
patterns = [
    "^ls$",
    "^cd$",
    "^histdb",
    "^top$",
    "^htop$",
    "^exit$",
    "^clear$",
]
```

---

### 5. CLI (`histdb-cli/`)

Using `clap` for argument parsing:

```rust
#[derive(Parser)]
#[command(name = "histdb")]
pub enum Cli {
    /// Query command history
    Query(QueryArgs),

    /// Show most frequent commands/directories
    Top(TopArgs),

    /// Sync history via git
    Sync(SyncArgs),

    /// Import from .zsh_history
    Import(ImportArgs),

    /// Export to JSON/CSV
    Export(ExportArgs),

    /// Start the daemon for zsh integration
    Daemon(DaemonArgs),
}

#[derive(Args)]
pub struct QueryArgs {
    /// Search pattern
    pattern: Option<String>,

    #[arg(long)]
    host: Option<String>,

    #[arg(long, name = "DIR")]
    in_dir: Option<PathBuf>,

    #[arg(long, name = "DIR")]
    at_dir: Option<PathBuf>,

    #[arg(short)]
    session: Option<i64>,

    #[arg(long)]
    from: Option<String>,

    #[arg(long)]
    until: Option<String>,

    #[arg(long, default_value = "25")]
    limit: usize,

    #[arg(long)]
    desc: bool,

    #[arg(long)]
    detail: bool,

    #[arg(long)]
    forget: bool,

    #[arg(long, short = 'y')]
    yes: bool,

    #[arg(long)]
    exact: bool,
}
```

---

### 6. Import Interface (`histdb-core/src/import/`)

Trait-based design for future extensibility, with a minimal initial implementation:

```rust
// import/traits.rs
use crate::db::Connection;
use crate::error::Error;

/// Import backend trait - allows plugging in different import sources
pub trait ImportBackend: Send + Sync {
    /// Human-readable name for this importer
    fn name(&self) -> &'static str;

    /// Check if this backend can import from the given path
    fn can_import(&self, path: &Path) -> bool;

    /// Import history entries from source into the database
    fn import(
        &self,
        source: &Path,
        conn: &mut Connection,
        progress: Option<&dyn ImportProgress>,
    ) -> Result<ImportStats, Error>;
}

/// Progress callback for long-running imports
pub trait ImportProgress: Send + Sync {
    fn on_progress(&self, imported: usize, total: Option<usize>);
    fn on_skip(&self, reason: &str);
}

/// Statistics from an import operation
#[derive(Debug, Default)]
pub struct ImportStats {
    pub entries_imported: usize,
    pub entries_skipped: usize,
    pub duplicates_found: usize,
}

// import/new_instance.rs - Default implementation
/// Creates a fresh database instance (no import)
pub struct NewInstance;

impl ImportBackend for NewInstance {
    fn name(&self) -> &'static str {
        "new"
    }

    fn can_import(&self, _path: &Path) -> bool {
        false // Never imports from existing files
    }

    fn import(
        &self,
        _source: &Path,
        conn: &mut Connection,
        _progress: Option<&dyn ImportProgress>,
    ) -> Result<ImportStats, Error> {
        // Just ensure schema is initialized
        crate::db::migrations::run_migrations(conn)?;
        Ok(ImportStats::default())
    }
}

/// Factory to create import backend (only NewInstance for now)
pub fn create_backend(_format: Option<&str>) -> Box<dyn ImportBackend> {
    // Future: match on format to return ZshHistdb, BashHistory, Atuin, etc.
    Box::new(NewInstance)
}
```

Future import backends (deferred):
- `ZshHistdbImport` — Import from existing zsh-histdb SQLite
- `BashHistoryImport` — Import from `~/.bash_history`
- `ZshHistoryImport` — Import from `~/.zsh_history`
- `AtuinImport` — Import from atuin SQLite

---

### 7. Extensible Search Backend (`histdb-core/src/search/`)

The search system uses a trait-based design for extensibility:

```rust
// search/traits.rs
pub trait SearchBackend: Send + Sync {
    /// Search commands matching pattern
    fn search(
        &self,
        conn: &Connection,
        query: &QueryBuilder,
    ) -> Result<Vec<HistoryRow>, Error>;

    /// Initialize backend (create indexes, virtual tables, etc.)
    fn initialize(&self, conn: &Connection) -> Result<(), Error>;

    /// Check if backend is available
    fn is_available(&self, conn: &Connection) -> bool;
}

// search/fts5.rs - SQLite FTS5 implementation
pub struct Fts5Backend {
    tokenizer: String,
}

impl Fts5Backend {
    pub fn new(tokenizer: &str) -> Self {
        Self { tokenizer: tokenizer.to_string() }
    }
}

impl SearchBackend for Fts5Backend {
    fn initialize(&self, conn: &Connection) -> Result<(), Error> {
        conn.execute_batch(&format!(r#"
            CREATE VIRTUAL TABLE IF NOT EXISTS commands_fts
            USING fts5(argv, tokenize='{}');

            -- Trigger to keep FTS in sync
            CREATE TRIGGER IF NOT EXISTS commands_fts_insert
            AFTER INSERT ON commands BEGIN
                INSERT INTO commands_fts(rowid, argv) VALUES (NEW.id, NEW.argv);
            END;
        "#, self.tokenizer))?;
        Ok(())
    }

    fn search(
        &self,
        conn: &Connection,
        query: &QueryBuilder,
    ) -> Result<Vec<HistoryRow>, Error> {
        let pattern = query.pattern.as_deref().unwrap_or("");
        // FTS5 query with proper escaping
        let fts_query = format!("\"{}\"", pattern.replace('"', "\"\""));

        conn.prepare(&format!(r#"
            SELECT h.*, c.argv, p.host, p.dir
            FROM commands_fts fts
            JOIN commands c ON c.id = fts.rowid
            JOIN history h ON h.command_id = c.id
            JOIN places p ON h.place_id = p.id
            WHERE commands_fts MATCH ?1
            {}
            ORDER BY h.start_time DESC
            LIMIT ?2
        "#, query.build_where_clause()))?
        .query_map(params![fts_query, query.limit], |row| /* ... */)
    }

    fn is_available(&self, conn: &Connection) -> bool {
        conn.query_row(
            "SELECT 1 FROM pragma_compile_options WHERE compile_options = 'ENABLE_FTS5'",
            [],
            |_| Ok(()),
        ).is_ok()
    }
}

// search/simple.rs - GLOB fallback
pub struct GlobBackend;

impl SearchBackend for GlobBackend {
    fn initialize(&self, _conn: &Connection) -> Result<(), Error> {
        Ok(()) // No setup needed
    }

    fn search(
        &self,
        conn: &Connection,
        query: &QueryBuilder,
    ) -> Result<Vec<HistoryRow>, Error> {
        let pattern = format!("*{}*", escape_glob(query.pattern.as_deref().unwrap_or("")));
        // Standard GLOB query...
    }

    fn is_available(&self, _conn: &Connection) -> bool {
        true // Always available
    }
}

/// Factory function to create search backend from config
pub fn create_backend(config: &SearchConfig, conn: &Connection) -> Box<dyn SearchBackend> {
    match config.backend {
        SearchBackendType::Fts5 => {
            let backend = Fts5Backend::new(
                config.fts5_tokenizer.as_deref().unwrap_or("unicode61")
            );
            if backend.is_available(conn) {
                return Box::new(backend);
            }
            tracing::warn!("FTS5 not available, falling back to GLOB");
        }
        SearchBackendType::Like => return Box::new(LikeBackend),
        SearchBackendType::Glob => {}
    }
    Box::new(GlobBackend)
}
```

---

### 7. Multi-Shell Integration (`shell/`)

**Architecture**: Shell-agnostic daemon with thin shell wrappers

```
┌─────────────┐
│     ZSH     │──┐
└─────────────┘  │
┌─────────────┐  │   Unix Socket      ┌─────────────────┐
│    BASH     │──┼──────────────────►│  histdb-daemon  │
└─────────────┘  │   JSON protocol    │  (Rust binary)  │
┌─────────────┐  │                    └────────┬────────┘
│   Nushell   │──┘                             │
└─────────────┘                                ▼
                                      ┌─────────────────┐
                                      │    SQLite DB    │
                                      └─────────────────┘
```

#### Protocol (`daemon/protocol.rs`)
```rust
#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Request {
    /// Register a new shell session
    Register {
        shell: ShellType,
        pid: u32,
    },
    /// Record command start
    StartCommand {
        argv: String,
        dir: String,
        session_id: Option<u64>,
    },
    /// Record command completion
    FinishCommand {
        id: i64,
        exit_status: i32,
    },
    /// Query history
    Query {
        pattern: Option<String>,
        limit: usize,
        offset: usize,
        filters: QueryFilters,
    },
    /// Graceful shutdown
    Shutdown,
}

#[derive(Serialize, Deserialize)]
pub enum ShellType {
    Zsh,
    Bash,
    Nu,
    Other(String),
}

#[derive(Serialize, Deserialize)]
pub enum Response {
    Registered { session_id: u64 },
    Accepted,
    CommandStarted { id: i64 },
    CommandFinished,
    QueryResults { entries: Vec<HistoryEntry> },
    Error { message: String },
}
```

#### ZSH Integration (`shell/zsh/histdb.zsh`)
```zsh
# histdb ZSH integration - thin wrapper over daemon
autoload -U add-zsh-hook

typeset -g HISTDB_SOCKET="${HISTDB_SOCKET:-${XDG_RUNTIME_DIR:-/tmp}/histdb-${UID}.sock}"
typeset -g HISTDB_SESSION=""
typeset -g HISTDB_LAST_ID=""

# Use zsh/net/socket if available, fall back to socat
if zmodload zsh/net/socket 2>/dev/null; then
    _histdb_send() {
        local fd
        zsocket $HISTDB_SOCKET && fd=$REPLY
        [[ -n $fd ]] || return 1
        print -u $fd -- "$1"
        local response
        read -u $fd response
        exec {fd}>&-
        print -- "$response"
    }
else
    _histdb_send() {
        print -- "$1" | socat -t1 - UNIX-CONNECT:$HISTDB_SOCKET 2>/dev/null
    }
fi

_histdb_init() {
    [[ -n $HISTDB_SESSION ]] && return
    local resp=$(_histdb_send '{"type":"Register","shell":"Zsh","pid":'$$'}')
    HISTDB_SESSION=$(print -- "$resp" | jq -r '.session_id // empty')
}

_histdb_addhistory() {
    _histdb_init
    local cmd="${1[1,-2]}"  # Remove trailing newline
    [[ -z "$cmd" ]] && return 0
    local json='{"type":"StartCommand","argv":"'${cmd//\"/\\\"}'",'
    json+='"dir":"'${PWD//\"/\\\"}'","session_id":'$HISTDB_SESSION'}'
    local resp=$(_histdb_send "$json")
    HISTDB_LAST_ID=$(print -- "$resp" | jq -r '.id // empty')
    return 0
}

_histdb_precmd() {
    local status=$?
    [[ -z "$HISTDB_LAST_ID" ]] && return
    _histdb_send '{"type":"FinishCommand","id":'$HISTDB_LAST_ID',"exit_status":'$status'}' &!
    HISTDB_LAST_ID=""
}

add-zsh-hook zshaddhistory _histdb_addhistory
add-zsh-hook precmd _histdb_precmd
```

#### BASH Integration (`shell/bash/histdb.bash`)
```bash
# histdb BASH integration
HISTDB_SOCKET="${HISTDB_SOCKET:-${XDG_RUNTIME_DIR:-/tmp}/histdb-$(id -u).sock}"
HISTDB_SESSION=""
HISTDB_LAST_ID=""
HISTDB_LAST_CMD=""

_histdb_send() {
    echo "$1" | socat -t1 - UNIX-CONNECT:"$HISTDB_SOCKET" 2>/dev/null
}

_histdb_init() {
    [[ -n "$HISTDB_SESSION" ]] && return
    local resp
    resp=$(_histdb_send '{"type":"Register","shell":"Bash","pid":'"$$"'}')
    HISTDB_SESSION=$(echo "$resp" | jq -r '.session_id // empty')
}

_histdb_preexec() {
    _histdb_init
    local cmd="$1"
    [[ -z "$cmd" ]] && return
    # Escape for JSON
    cmd="${cmd//\\/\\\\}"
    cmd="${cmd//\"/\\\"}"
    cmd="${cmd//$'\n'/\\n}"
    local json='{"type":"StartCommand","argv":"'"$cmd"'",'
    json+='"dir":"'"${PWD//\"/\\\"}"'","session_id":'"$HISTDB_SESSION"'}'
    local resp
    resp=$(_histdb_send "$json")
    HISTDB_LAST_ID=$(echo "$resp" | jq -r '.id // empty')
}

_histdb_precmd() {
    local status=$?
    [[ -z "$HISTDB_LAST_ID" ]] && return
    _histdb_send '{"type":"FinishCommand","id":'"$HISTDB_LAST_ID"',"exit_status":'"$status"'}' &
    HISTDB_LAST_ID=""
}

# BASH doesn't have preexec, use DEBUG trap
_histdb_debug() {
    [[ "$BASH_COMMAND" == "$PROMPT_COMMAND" ]] && return
    [[ "$BASH_COMMAND" == "_histdb_precmd" ]] && return
    [[ "$HISTDB_LAST_CMD" == "$BASH_COMMAND" ]] && return
    HISTDB_LAST_CMD="$BASH_COMMAND"
    _histdb_preexec "$BASH_COMMAND"
}

trap '_histdb_debug' DEBUG
PROMPT_COMMAND="_histdb_precmd${PROMPT_COMMAND:+;$PROMPT_COMMAND}"
```

#### Nushell Integration (`shell/nu/histdb.nu`)
```nu
# histdb Nushell integration

let-env HISTDB_SOCKET = ($env.HISTDB_SOCKET? | default $"($env.XDG_RUNTIME_DIR? | default '/tmp')/histdb-($env.USER).sock")
mut histdb_session = null
mut histdb_last_id = null

def histdb-send [msg: string] {
    echo $msg | socat -t1 - $"UNIX-CONNECT:($env.HISTDB_SOCKET)" | from json
}

def histdb-init [] {
    if $histdb_session == null {
        let resp = (histdb-send $'{"type":"Register","shell":"Nu","pid":($nu.pid)}')
        $histdb_session = $resp.session_id
    }
}

# Hook into command execution
$env.config.hooks.pre_execution = {||
    histdb-init
    let cmd = (commandline)
    if ($cmd | is-empty) { return }
    let json = {
        type: "StartCommand"
        argv: $cmd
        dir: $env.PWD
        session_id: $histdb_session
    } | to json
    let resp = (histdb-send $json)
    $histdb_last_id = $resp.id?
}

$env.config.hooks.env_change = {
    PWD: {|before, after|
        # Could track directory changes if needed
    }
}

# Note: Nushell doesn't expose exit status in hooks yet
# This is a limitation we document
```

#### Shell Feature Matrix

| Feature | ZSH | BASH | Nushell |
|---------|-----|------|---------|
| Command recording | ✅ | ✅ | ✅ |
| Exit status | ✅ | ✅ | ⚠️ Limited |
| Command duration | ✅ | ✅ | ✅ |
| Native socket | ✅ (zsh/net/socket) | ❌ (socat) | ❌ (socat) |
| Interactive search | ✅ (widget) | ✅ (fzf) | ✅ (menu) |
| Async send | ✅ (&!) | ✅ (&) | ⚠️ |

#### Shell Offline Queue

When the daemon is unavailable, shells queue commands locally for later replay.
This provides **near 100% reliability** - no commands are lost.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Shell Offline Queue                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Shell executes command                                                 │
│          │                                                               │
│          ▼                                                               │
│   ┌──────────────┐     success    ┌─────────────────┐                   │
│   │ Try daemon   │ ─────────────► │ Command recorded │                   │
│   │ socket       │                │ (normal path)    │                   │
│   └──────────────┘                └─────────────────┘                   │
│          │ fail                                                          │
│          ▼                                                               │
│   ┌──────────────────────────────────────┐                              │
│   │ Queue to local SQLite                │                              │
│   │ ~/.local/share/histdb/offline.db     │                              │
│   └──────────────────────────────────────┘                              │
│          │                                                               │
│          ▼ (on next successful daemon connect)                          │
│   ┌──────────────────────────────────────┐                              │
│   │ Replay queued commands to daemon     │                              │
│   │ (in order, with original timestamps) │                              │
│   └──────────────────────────────────────┘                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Offline database schema:**

```sql
-- ~/.local/share/histdb/offline.db
CREATE TABLE IF NOT EXISTS offline_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    queued_at INTEGER NOT NULL,          -- When queued (for ordering)
    request_type TEXT NOT NULL,          -- 'start' or 'finish'
    request_json TEXT NOT NULL,          -- Full request payload
    original_timestamp INTEGER NOT NULL  -- Original command timestamp
);

CREATE INDEX IF NOT EXISTS offline_queue_order ON offline_queue(queued_at);
```

**Shell implementation (ZSH example):**

```zsh
# histdb.zsh - with offline queue support
typeset -g HISTDB_OFFLINE_DB="${XDG_DATA_HOME:-$HOME/.local/share}/histdb/offline.db"

_histdb_ensure_offline_db() {
    [[ -f $HISTDB_OFFLINE_DB ]] && return
    mkdir -p "${HISTDB_OFFLINE_DB:h}"
    sqlite3 "$HISTDB_OFFLINE_DB" "
        CREATE TABLE IF NOT EXISTS offline_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            queued_at INTEGER NOT NULL,
            request_type TEXT NOT NULL,
            request_json TEXT NOT NULL,
            original_timestamp INTEGER NOT NULL
        );
    "
}

_histdb_queue_offline() {
    local request_type="$1" json="$2" timestamp="$3"
    _histdb_ensure_offline_db
    sqlite3 "$HISTDB_OFFLINE_DB" "
        INSERT INTO offline_queue (queued_at, request_type, request_json, original_timestamp)
        VALUES ($(date +%s), '$request_type', '${json//\'/\'\'}', $timestamp);
    "
}

_histdb_replay_offline() {
    [[ -f $HISTDB_OFFLINE_DB ]] || return
    local count=$(sqlite3 "$HISTDB_OFFLINE_DB" "SELECT COUNT(*) FROM offline_queue")
    [[ $count -eq 0 ]] && return

    # Replay in order
    sqlite3 -json "$HISTDB_OFFLINE_DB" "SELECT * FROM offline_queue ORDER BY queued_at" | \
    jq -c '.[]' | while read -r row; do
        local id=$(echo "$row" | jq -r '.id')
        local json=$(echo "$row" | jq -r '.request_json')

        if _histdb_send "$json" >/dev/null 2>&1; then
            sqlite3 "$HISTDB_OFFLINE_DB" "DELETE FROM offline_queue WHERE id = $id"
        else
            break  # Daemon still unavailable
        fi
    done
}

_histdb_send_or_queue() {
    local json="$1" request_type="$2" timestamp="${3:-$(date +%s)}"

    # Try daemon first
    local resp=$(_histdb_send "$json" 2>/dev/null)
    if [[ -n "$resp" ]]; then
        # Success - also try to replay any queued commands
        _histdb_replay_offline &!
        print -- "$resp"
        return 0
    fi

    # Queue for later
    _histdb_queue_offline "$request_type" "$json" "$timestamp"
    return 1
}

# Updated hook to use offline queue
_histdb_addhistory() {
    _histdb_init
    local cmd="${1[1,-2]}"
    [[ -z "$cmd" ]] && return 0

    local timestamp=$(date +%s)
    local json='{"type":"StartCommand","argv":"'${cmd//\"/\\\"}'",'
    json+='"dir":"'${PWD//\"/\\\"}'","session_id":'$HISTDB_SESSION',"timestamp":'$timestamp'}'

    local resp=$(_histdb_send_or_queue "$json" "start" "$timestamp")
    HISTDB_LAST_ID=$(print -- "$resp" | jq -r '.id // empty')
    return 0
}
```

**Replay behavior:**
- Commands replayed in original order (by `queued_at`)
- Original timestamps preserved (for correct history ordering)
- Partial replay safe (stops on first failure, resumes later)
- Background replay doesn't block shell

---

## Daemon Lifecycle

### Startup Mechanisms

**Primary**: Socket activation (systemd/launchd)
- Zero resource usage until first shell connects
- Automatic restart on crash
- Proper dependency ordering

**Fallback**: Shell-initiated startup
- For Alpine/OpenRC, BSDs, or systems without service manager
- Shell plugin checks socket, starts daemon if missing
- PID file prevents duplicate daemons

#### Platform Support

| Platform | Init System | Socket Activation | Service File |
|----------|-------------|-------------------|--------------|
| Debian/Ubuntu | systemd | ✅ | `~/.config/systemd/user/histdb.service` |
| Arch | systemd | ✅ | `~/.config/systemd/user/histdb.service` |
| NixOS | systemd | ✅ | Via Home Manager |
| Alpine | OpenRC | ❌ | Shell-initiated fallback |
| macOS | launchd | ✅ | `~/Library/LaunchAgents/histdb.plist` |

#### systemd User Service

```ini
# ~/.config/systemd/user/histdb.service
[Unit]
Description=histdb shell history daemon
Documentation=https://github.com/user/histdb-rs

[Service]
Type=notify
ExecStart=%h/.cargo/bin/histdb-daemon
Restart=on-failure
RestartSec=1

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=%h/.local/share/histdb %t/histdb

[Install]
WantedBy=default.target
```

```ini
# ~/.config/systemd/user/histdb.socket
[Unit]
Description=histdb socket

[Socket]
ListenStream=%t/histdb.sock
SocketMode=0600

[Install]
WantedBy=sockets.target
```

#### launchd Service (macOS)

```xml
<!-- ~/Library/LaunchAgents/com.histdb.daemon.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.histdb.daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/histdb-daemon</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>Sockets</key>
    <dict>
        <key>histdb</key>
        <dict>
            <key>SockPathName</key>
            <string>/tmp/histdb.sock</string>
            <key>SockPathMode</key>
            <integer>384</integer>
        </dict>
    </dict>
</dict>
</plist>
```

### Shutdown Handling

#### Graceful Shutdown (SIGTERM/SIGINT)

```
Signal received
     │
     ▼
┌─────────────────────────┐
│ 1. Stop accepting new   │  ← Close listener socket
│    connections          │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 2. Drain command queue  │  ← Process remaining PendingCommands
│    (timeout: 500ms)     │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 3. Flush final batch    │  ← Commit to SQLite
│    to SQLite            │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 4. Best-effort sync     │  ← Push pending ops to peers
│    (timeout: 1s)        │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 5. Cleanup              │  ← WAL checkpoint, remove socket/pidfile
└───────────┬─────────────┘
            ▼
        Exit(0)
```

```rust
// daemon/shutdown.rs
pub struct GracefulShutdown {
    shutdown_rx: broadcast::Receiver<()>,
    drain_timeout: Duration,
}

impl GracefulShutdown {
    pub async fn run(
        &self,
        queue: Arc<CommandQueue>,
        writer: &mut WriterThread,
        sync: Option<&PeerSync>,
    ) -> Result<(), Error> {
        // 1. Drain queue with timeout
        let drain_result = tokio::time::timeout(
            Duration::from_millis(500),
            Self::drain_queue(&queue, writer)
        ).await;

        if drain_result.is_err() {
            tracing::warn!(
                "Queue drain timed out, {} commands may be lost",
                queue.pending_count.load(Ordering::Relaxed)
            );
        }

        // 2. Final SQLite flush
        writer.flush_final()?;

        // 3. Best-effort sync (don't block shutdown)
        if let Some(sync) = sync {
            let _ = tokio::time::timeout(
                Duration::from_secs(1),
                sync.push_pending()
            ).await;
        }

        Ok(())
    }
}
```

#### Non-Graceful Shutdown (SIGKILL, Power Loss, Crash)

| Component | Data at Risk | Recovery |
|-----------|--------------|----------|
| Command queue | Commands not yet batched (≤10ms) | **Lost** |
| Current batch | Commands being written | SQLite rollback (safe) |
| SQLite WAL | Committed transactions | Auto-recover on open |
| Operation log | Persisted operations | Intact |
| Left-right index | In-memory cache | Rebuild from SQLite |

**Bounded data loss**: At most ~10ms of commands (one batch interval)

### Crash Recovery

```rust
// daemon/recovery.rs
impl Daemon {
    pub fn recover_on_startup(&mut self) -> Result<(), Error> {
        // 1. Check for stale socket
        Self::check_stale_socket(&self.socket_path)?;

        // 2. Acquire PID file
        self.pidfile = PidFile::acquire(&self.pidfile_path)?;

        // 3. Open SQLite (auto-recovers WAL)
        let conn = Connection::open(&self.db_path)?;

        // 4. Rebuild left-right index from recent history
        let recent = conn.prepare("
            SELECT * FROM history
            WHERE start_time > ?1
            ORDER BY start_time DESC
            LIMIT ?2
        ")?.query_map(
            params![now() - self.config.index.max_age, self.config.index.max_entries],
            |row| Ok(RecentEntry::from(row))
        )?;

        for entry in recent {
            self.index_writer.insert(entry?);
        }
        self.index_writer.publish();

        // 5. Rebuild vector clock from operation log
        let vector_clock: HashMap<NodeId, u64> = conn.prepare("
            SELECT origin, MAX(seq) FROM operation_log GROUP BY origin
        ")?.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<Result<_, _>>()?;

        *self.log.vector_clock.write() = vector_clock;

        tracing::info!("Recovery complete: {} entries in index", self.index_writer.len());
        Ok(())
    }

    fn check_stale_socket(path: &Path) -> Result<(), Error> {
        if path.exists() {
            match UnixStream::connect(path) {
                Ok(_) => return Err(Error::DaemonAlreadyRunning),
                Err(_) => {
                    std::fs::remove_file(path)?;
                    tracing::info!("Removed stale socket");
                }
            }
        }
        Ok(())
    }
}
```

### Reliability Guarantees

| Scenario | Data Loss | Recovery Time |
|----------|-----------|---------------|
| Graceful shutdown | None | Instant |
| SIGKILL | ≤10ms of commands | ~100ms (index rebuild) |
| Power loss | ≤10ms of commands | ~100ms |
| Daemon crash | ≤10ms of commands | Auto-restart via systemd |
| Corrupt database | All local data | Restore from sync peer |

### Optional Durable Mode

For users requiring zero data loss (at cost of latency):

```toml
# ~/.config/histdb/config.toml
[daemon]
# "fast" = ack before persist (default, <1ms, may lose ≤10ms on crash)
# "durable" = ack after persist (~5-10ms, crash-safe)
ack_mode = "fast"
```

---

## Synchronization Architecture (Hybrid: Raft Log Replication + CRDT Semantics)

A hybrid approach combining **Raft-style log replication** for efficient message passing with **CRDT semantics** for conflict-free offline operation. Every machine is a full peer—no coordination server required.

### Why Hybrid?

| Feature | Pure Raft | Pure CRDTs | **Hybrid** |
|---------|-----------|------------|------------|
| Local writes | ❌ Need quorum | ✅ Always | **✅ Always** |
| Propagation | Log replication | Gossip | **Log replication** |
| Conflict resolution | Consensus | Auto-merge | **Auto-merge (CRDT)** |
| Ordering | Total (linearizable) | Partial | **Causal** |
| Offline support | Queue only | ✅ Full | **✅ Full** |
| Coordination server | ❌ | ❌ | **❌** |
| Convergence | Immediate | Eventually | **Strong eventual** |

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Hybrid Sync Cluster (No Leader Required)              │
│                                                                          │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │
│   │   Machine A     │    │   Machine B     │    │   Machine C     │    │
│   │                 │    │                 │    │                 │    │
│   │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │    │
│   │ │ Local Log   │◄├────┼─┤ Local Log   │◄├────┼─┤ Local Log   │ │    │
│   │ │ (append-only)│─┼────►│ (append-only)│─┼────►│ (append-only)│ │    │
│   │ └──────┬──────┘ │    │ └──────┬──────┘ │    │ └──────┬──────┘ │    │
│   │        │ apply  │    │        │ apply  │    │        │ apply  │    │
│   │ ┌──────▼──────┐ │    │ ┌──────▼──────┐ │    │ ┌──────▼──────┐ │    │
│   │ │   SQLite    │ │    │ │   SQLite    │ │    │ │   SQLite    │ │    │
│   │ │ (CRDT state)│ │    │ │ (CRDT state)│ │    │ │ (CRDT state)│ │    │
│   │ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │    │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘    │
│            ▲                      ▲                      ▲              │
│            │                      │                      │              │
│     ┌──────┴──────┐        ┌──────┴──────┐        ┌──────┴──────┐      │
│     │   Shells    │        │   Shells    │        │   Shells    │      │
│     └─────────────┘        └─────────────┘        └─────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘

Writes: Shell → Local Log → Apply to SQLite (immediate) → Replicate to peers (async)
Reads:  Shell → Local SQLite (always available)
Sync:   Bi-directional log exchange, CRDT merge on apply
```

### Key Properties

| Property | Guarantee |
|----------|-----------|
| **Consistency** | Strong eventual (all nodes converge) |
| **Availability** | Always (writes always succeed locally) |
| **Partition tolerance** | Full (works offline indefinitely) |
| **Convergence** | Automatic via CRDT semantics |
| **Ordering** | Causal (via Hybrid Logical Clocks) |

### Core Design: Operation-Based CRDTs with Log Shipping

```rust
// sync/hlc.rs - Hybrid Logical Clock for causal ordering
use std::sync::atomic::{AtomicU64, Ordering};

/// Hybrid Logical Clock: combines wall clock with logical counter
/// Provides causal ordering across distributed nodes
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct HLC {
    /// Wall clock time (milliseconds since epoch)
    pub wall_time: u64,
    /// Logical counter for events at same wall time
    pub logical: u32,
    /// Node ID that created this timestamp
    pub node_id: NodeId,
}

impl HLC {
    /// Generate a new timestamp, ensuring it's greater than any seen
    pub fn now(&self, node_id: NodeId) -> HLC {
        let wall = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let (new_wall, new_logical) = if wall > self.wall_time {
            (wall, 0)
        } else {
            (self.wall_time, self.logical + 1)
        };

        HLC {
            wall_time: new_wall,
            logical: new_logical,
            node_id,
        }
    }

    /// Update clock after receiving a remote timestamp
    pub fn receive(&mut self, remote: &HLC, local_node: NodeId) -> HLC {
        let wall = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let (new_wall, new_logical) = if wall > self.wall_time && wall > remote.wall_time {
            (wall, 0)
        } else if self.wall_time > remote.wall_time {
            (self.wall_time, self.logical + 1)
        } else if remote.wall_time > self.wall_time {
            (remote.wall_time, remote.logical + 1)
        } else {
            (self.wall_time, self.logical.max(remote.logical) + 1)
        };

        self.wall_time = new_wall;
        self.logical = new_logical;

        HLC {
            wall_time: new_wall,
            logical: new_logical,
            node_id: local_node,
        }
    }
}
```

### CRDT Data Types for History

```rust
// sync/crdt.rs

/// History entry with CRDT semantics
/// Uses UUIDv7 as unique identifier (globally unique, time-ordered)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrdtEntry {
    /// Globally unique ID (UUIDv7 - contains timestamp + random)
    pub id: Uuid,
    /// HLC timestamp for causal ordering
    pub hlc: HLC,
    /// Origin node
    pub origin: NodeId,
    /// Entry state (G-Set semantics - once added, never removed from set)
    pub data: EntryData,
    /// Tombstone for deletion (once set, wins over any update)
    pub deleted: Option<HLC>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntryData {
    pub session: u64,
    pub argv: String,
    pub dir: String,
    pub host: String,
    pub start_time: i64,
    /// LWW-Register: last-writer-wins for mutable fields
    pub exit_status: Option<LWWRegister<i32>>,
    pub duration: Option<LWWRegister<u64>>,
}

/// Last-Writer-Wins Register for mutable fields
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LWWRegister<T> {
    pub value: T,
    pub timestamp: HLC,
}

impl<T: Clone> LWWRegister<T> {
    pub fn new(value: T, timestamp: HLC) -> Self {
        Self { value, timestamp }
    }

    /// Merge with another register - highest timestamp wins
    pub fn merge(&mut self, other: &Self) {
        if other.timestamp > self.timestamp {
            self.value = other.value.clone();
            self.timestamp = other.timestamp;
        }
    }
}

/// Operations that can be applied (operation-based CRDT)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Operation {
    /// Insert new entry (G-Set add)
    Insert(CrdtEntry),

    /// Update exit status (LWW-Register)
    UpdateStatus {
        entry_id: Uuid,
        exit_status: LWWRegister<i32>,
        duration: LWWRegister<u64>,
    },

    /// Delete entry (tombstone)
    Delete {
        entry_id: Uuid,
        timestamp: HLC,
    },
}

impl Operation {
    /// Operations are idempotent and commutative
    pub fn apply(&self, state: &mut HistoryState) {
        match self {
            Operation::Insert(entry) => {
                // G-Set semantics: insert if not exists
                state.entries.entry(entry.id).or_insert_with(|| entry.clone());
            }

            Operation::UpdateStatus { entry_id, exit_status, duration } => {
                if let Some(entry) = state.entries.get_mut(entry_id) {
                    // LWW merge for mutable fields
                    match &mut entry.data.exit_status {
                        Some(existing) => existing.merge(exit_status),
                        None => entry.data.exit_status = Some(exit_status.clone()),
                    }
                    match &mut entry.data.duration {
                        Some(existing) => existing.merge(duration),
                        None => entry.data.duration = Some(duration.clone()),
                    }
                }
            }

            Operation::Delete { entry_id, timestamp } => {
                if let Some(entry) = state.entries.get_mut(entry_id) {
                    // Tombstone wins if timestamp is newer
                    match &entry.deleted {
                        Some(existing) if existing >= timestamp => {}
                        _ => entry.deleted = Some(*timestamp),
                    }
                }
            }
        }
    }
}
```

### Log Replication (Raft-style, but without consensus)

```rust
// sync/log.rs

/// Local operation log - append-only, replicated to peers
pub struct OperationLog {
    /// SQLite-backed persistent log
    conn: Connection,
    /// Local node ID
    node_id: NodeId,
    /// Hybrid logical clock
    hlc: Mutex<HLC>,
    /// Vector clock tracking what we've seen from each peer
    vector_clock: RwLock<HashMap<NodeId, u64>>,
}

/// Log entry with sequence number for replication
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogEntry {
    /// Sequence number (per-node, monotonic)
    pub seq: u64,
    /// Node that created this entry
    pub origin: NodeId,
    /// The operation
    pub operation: Operation,
    /// HLC timestamp
    pub timestamp: HLC,
}

impl OperationLog {
    /// Append a new operation (local write)
    pub fn append(&self, op: Operation) -> Result<LogEntry, Error> {
        let mut hlc = self.hlc.lock();
        let timestamp = hlc.now(self.node_id);
        *hlc = timestamp;

        let seq = self.next_seq()?;

        let entry = LogEntry {
            seq,
            origin: self.node_id,
            operation: op,
            timestamp,
        };

        // Persist to local log
        self.conn.execute(
            "INSERT INTO operation_log (seq, origin, timestamp_wall, timestamp_logical, operation)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                entry.seq,
                entry.origin,
                entry.timestamp.wall_time,
                entry.timestamp.logical,
                bincode::serialize(&entry.operation)?,
            ],
        )?;

        // Update our own vector clock
        self.vector_clock.write().insert(self.node_id, seq);

        Ok(entry)
    }

    /// Get entries after a given vector clock (for replication)
    pub fn entries_since(&self, vector_clock: &HashMap<NodeId, u64>) -> Result<Vec<LogEntry>, Error> {
        let mut entries = Vec::new();

        // Get entries from each node that are newer than the given vector clock
        for (node_id, last_seq) in self.vector_clock.read().iter() {
            let peer_last = vector_clock.get(node_id).copied().unwrap_or(0);
            if *last_seq > peer_last {
                let node_entries: Vec<LogEntry> = self.conn
                    .prepare("SELECT * FROM operation_log WHERE origin = ?1 AND seq > ?2 ORDER BY seq")?
                    .query_map(params![node_id, peer_last], |row| /* deserialize */)?
                    .collect();
                entries.extend(node_entries);
            }
        }

        Ok(entries)
    }

    /// Apply entries from a remote peer
    pub fn apply_remote(&self, entries: Vec<LogEntry>, state: &mut HistoryState) -> Result<(), Error> {
        let mut hlc = self.hlc.lock();

        for entry in entries {
            // Update HLC with remote timestamp
            *hlc = hlc.receive(&entry.timestamp, self.node_id);

            // Check if we already have this entry (idempotent)
            let existing_seq = self.vector_clock.read()
                .get(&entry.origin)
                .copied()
                .unwrap_or(0);

            if entry.seq > existing_seq {
                // Persist to local log
                self.conn.execute(
                    "INSERT OR IGNORE INTO operation_log (seq, origin, timestamp_wall, timestamp_logical, operation)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        entry.seq,
                        entry.origin,
                        entry.timestamp.wall_time,
                        entry.timestamp.logical,
                        bincode::serialize(&entry.operation)?,
                    ],
                )?;

                // Apply to CRDT state (idempotent)
                entry.operation.apply(state);

                // Update vector clock
                self.vector_clock.write().insert(entry.origin, entry.seq);
            }
        }

        Ok(())
    }
}
```

### Sync Protocol (Raft-style RPCs, CRDT semantics)

```rust
// sync/protocol.rs

/// Sync messages between peers
#[derive(Clone, Serialize, Deserialize)]
pub enum SyncMessage {
    /// Request entries since a vector clock
    SyncRequest {
        from_node: NodeId,
        vector_clock: HashMap<NodeId, u64>,
    },

    /// Response with new entries
    SyncResponse {
        entries: Vec<LogEntry>,
        vector_clock: HashMap<NodeId, u64>,
    },

    /// Push new entries (eager replication)
    PushEntries {
        entries: Vec<LogEntry>,
    },

    /// Acknowledge received entries
    Ack {
        vector_clock: HashMap<NodeId, u64>,
    },
}

/// Peer connection for sync
pub struct PeerSync {
    peers: Arc<RwLock<HashMap<NodeId, PeerConnection>>>,
    log: Arc<OperationLog>,
    state: Arc<RwLock<HistoryState>>,
}

impl PeerSync {
    /// Push new local entry to all peers (eager replication)
    pub async fn broadcast(&self, entry: LogEntry) {
        let peers = self.peers.read().clone();

        for (peer_id, conn) in peers {
            let entry = entry.clone();
            tokio::spawn(async move {
                if let Err(e) = conn.send(SyncMessage::PushEntries {
                    entries: vec![entry],
                }).await {
                    tracing::warn!("Failed to push to peer {}: {}", peer_id, e);
                }
            });
        }
    }

    /// Periodic anti-entropy sync with a peer
    pub async fn sync_with_peer(&self, peer_id: NodeId) -> Result<usize, Error> {
        let conn = self.peers.read()
            .get(&peer_id)
            .cloned()
            .ok_or(Error::PeerNotFound)?;

        // Send our vector clock
        let our_vc = self.log.vector_clock.read().clone();
        conn.send(SyncMessage::SyncRequest {
            from_node: self.log.node_id,
            vector_clock: our_vc.clone(),
        }).await?;

        // Receive their entries
        let response = conn.receive().await?;
        let SyncMessage::SyncResponse { entries, vector_clock: their_vc } = response else {
            return Err(Error::UnexpectedMessage);
        };

        // Apply their entries
        let mut state = self.state.write();
        self.log.apply_remote(entries.clone(), &mut state)?;

        // Send entries they're missing
        let entries_for_them = self.log.entries_since(&their_vc)?;
        if !entries_for_them.is_empty() {
            conn.send(SyncMessage::PushEntries {
                entries: entries_for_them,
            }).await?;
        }

        Ok(entries.len())
    }

    /// Handle incoming sync message
    pub async fn handle_message(&self, msg: SyncMessage, from: NodeId) -> Option<SyncMessage> {
        match msg {
            SyncMessage::SyncRequest { vector_clock, .. } => {
                let entries = self.log.entries_since(&vector_clock).ok()?;
                let our_vc = self.log.vector_clock.read().clone();
                Some(SyncMessage::SyncResponse {
                    entries,
                    vector_clock: our_vc,
                })
            }

            SyncMessage::PushEntries { entries } => {
                let mut state = self.state.write();
                self.log.apply_remote(entries, &mut state).ok()?;
                let our_vc = self.log.vector_clock.read().clone();
                Some(SyncMessage::Ack { vector_clock: our_vc })
            }

            SyncMessage::SyncResponse { entries, .. } => {
                let mut state = self.state.write();
                self.log.apply_remote(entries, &mut state).ok()?;
                None
            }

            SyncMessage::Ack { .. } => None,
        }
    }
}
```

### Daemon Integration

```rust
// daemon/main.rs
pub struct Daemon {
    // Local write queue (for shell latency)
    command_queue: Arc<CommandQueue>,
    index_reader: IndexReader,

    // CRDT sync
    log: Arc<OperationLog>,
    state: Arc<RwLock<HistoryState>>,
    sync: Arc<PeerSync>,
    node_id: NodeId,
}

impl Daemon {
    /// Handle shell command - always succeeds locally
    pub async fn handle_command(&self, cmd: PendingCommand) -> Result<Uuid, Error> {
        let uuid = Uuid::now_v7();

        // 1. Create CRDT entry
        let entry = CrdtEntry {
            id: uuid,
            hlc: self.log.hlc.lock().now(self.node_id),
            origin: self.node_id,
            data: EntryData {
                session: cmd.session_id,
                argv: cmd.argv.to_string(),
                dir: cmd.dir.to_string_lossy().to_string(),
                host: cmd.host.to_string(),
                start_time: cmd.start_time,
                exit_status: None,
                duration: None,
            },
            deleted: None,
        };

        // 2. Append to local log (always succeeds)
        let log_entry = self.log.append(Operation::Insert(entry.clone()))?;

        // 3. Apply to local state immediately
        self.state.write().entries.insert(uuid, entry);

        // 4. Broadcast to peers (async, fire-and-forget)
        let sync = self.sync.clone();
        tokio::spawn(async move {
            sync.broadcast(log_entry).await;
        });

        Ok(uuid)
    }

    /// Update exit status - LWW semantics
    pub async fn finish_command(&self, uuid: Uuid, exit_status: i32, duration: u64) -> Result<(), Error> {
        let timestamp = self.log.hlc.lock().now(self.node_id);

        let op = Operation::UpdateStatus {
            entry_id: uuid,
            exit_status: LWWRegister::new(exit_status, timestamp),
            duration: LWWRegister::new(duration, timestamp),
        };

        let log_entry = self.log.append(op.clone())?;

        // Apply locally
        op.apply(&mut self.state.write());

        // Broadcast
        let sync = self.sync.clone();
        tokio::spawn(async move {
            sync.broadcast(log_entry).await;
        });

        Ok(())
    }
}
```

### Schema Updates for Hybrid Sync

```sql
-- migrations/004_add_hybrid_sync.sql

-- Add CRDT columns to history
ALTER TABLE history ADD COLUMN uuid TEXT UNIQUE;
ALTER TABLE history ADD COLUMN origin_node INTEGER;
ALTER TABLE history ADD COLUMN hlc_wall INTEGER;
ALTER TABLE history ADD COLUMN hlc_logical INTEGER;
ALTER TABLE history ADD COLUMN deleted_at_wall INTEGER;
ALTER TABLE history ADD COLUMN deleted_at_logical INTEGER;

-- Operation log (append-only, replicated)
CREATE TABLE IF NOT EXISTS operation_log (
    seq INTEGER NOT NULL,
    origin INTEGER NOT NULL,
    timestamp_wall INTEGER NOT NULL,
    timestamp_logical INTEGER NOT NULL,
    operation BLOB NOT NULL,
    PRIMARY KEY (origin, seq)
);

-- Vector clock state
CREATE TABLE IF NOT EXISTS vector_clock (
    node_id INTEGER PRIMARY KEY,
    last_seq INTEGER NOT NULL
);

-- Known peers
CREATE TABLE IF NOT EXISTS peers (
    node_id INTEGER PRIMARY KEY,
    addr TEXT NOT NULL,
    last_seen INTEGER
);

-- Indexes
CREATE INDEX IF NOT EXISTS operation_log_origin ON operation_log(origin, seq);
CREATE INDEX IF NOT EXISTS history_uuid ON history(uuid);
CREATE INDEX IF NOT EXISTS history_hlc ON history(hlc_wall, hlc_logical);
```

### Configuration

```toml
# ~/.config/histdb/config.toml

[sync]
enabled = true
# Unique node ID (auto-generated from machine-id if not set)
# node_id = 1

# Listen address for sync
listen_addr = "0.0.0.0:4242"

[sync.replication]
# Eager push to peers on write
eager_push = true
# Anti-entropy sync interval (seconds)
sync_interval = 30
# Max entries per sync batch
batch_size = 1000

[sync.peers]
# Static peer list (in addition to mDNS discovery)
bootstrap = [
    "192.168.1.100:4242",
    "192.168.1.101:4242",
]

[sync.discovery]
# Enable mDNS for automatic peer discovery on LAN
mdns_enabled = true
# Service name for mDNS
mdns_service = "_histdb._tcp.local"
```

### Consistency Guarantees

| Scenario | Behavior |
|----------|----------|
| **Local write** | Always succeeds, immediately visible locally |
| **Concurrent writes** | Both succeed, merge via CRDT rules |
| **Network partition** | Both sides continue writing, merge on reconnect |
| **Exit status conflict** | Last-writer-wins (by HLC timestamp) |
| **Delete vs update** | Delete tombstone wins |
| **New node joins** | Catches up via anti-entropy sync |

### Convergence Properties

```
Node A writes: cmd1, cmd2, cmd3
Node B writes: cmd4, cmd5
(partition)

After reconnect:
- Both nodes have: cmd1, cmd2, cmd3, cmd4, cmd5
- Order may differ locally, but final state is identical
- UUIDs ensure no duplicates
- HLC ensures causal ordering for queries
```

---

## Error Handling

Comprehensive error types:

```rust
// error.rs
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Migration failed: {0}")]
    Migration(#[from] MigrationError),

    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("Validation error: {0}")]
    Validation(#[from] ValidationError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Merge conflict: our version {ours}, their version {theirs}")]
    MergeVersionConflict { ours: u32, theirs: u32 },
}

#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("Command cannot be empty")]
    EmptyCommand,

    #[error("Command contains null byte")]
    NullByte,

    #[error("Invalid time format: {0}")]
    InvalidTimeFormat(String),

    #[error("Invalid glob pattern: {0}")]
    InvalidGlob(String),
}
```

---

## Testing Strategy

```rust
// Unit tests for each module
#[cfg(test)]
mod tests {
    #[test]
    fn test_sql_escape_glob() {
        assert_eq!(escape_glob("foo*bar"), "foo[*]bar");
        assert_eq!(escape_glob("test[0]"), "test[[]0[]]");
    }

    #[test]
    fn test_command_validation() {
        assert!(NewCommand::new("ls -la").is_ok());
        assert!(NewCommand::new("").is_err());
        assert!(NewCommand::new("foo\0bar").is_err());
    }
}

// Integration tests with in-memory SQLite
#[test]
fn test_record_and_query() {
    let conn = Connection::open_in_memory().unwrap();
    setup_schema(&conn).unwrap();

    let recorder = Recorder::new(&conn, 1, "test-host");
    let id = recorder.start_command("echo hello", Path::new("/tmp")).unwrap();
    recorder.finish_command(id, 0).unwrap();

    let results = QueryBuilder::new()
        .pattern("echo")
        .execute(&conn)
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].argv, "echo hello");
}

// Property-based tests
#[test]
fn prop_escape_glob_roundtrip() {
    proptest!(|(s: String)| {
        let escaped = escape_glob(&s);
        // Escaped string should match original literally
        let pattern = format!("*{}*", escaped);
        // ... verify GLOB matches
    });
}
```

---

## Dependencies

```toml
[workspace.dependencies]
# Database
rusqlite = { version = "0.31", features = ["bundled", "backup"] }

# Lock-free concurrency
crossbeam-queue = "0.3"       # Lock-free ArrayQueue
crossbeam-channel = "0.5"     # For oneshot-style responses
left-right = "0.11"           # Wait-free reads via left-right pattern
parking_lot = "0.12"          # Fast mutexes (for non-hot paths only)

# Hybrid sync (CRDT + log replication)
bincode = "1.3"               # Efficient serialization for operation log

# Async runtime
tokio = { version = "1", features = ["rt-multi-thread", "net", "io-util", "time", "sync"] }

# CLI & Serialization
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Error handling & utilities
thiserror = "1"
chrono = { version = "0.4", features = ["serde"] }
toml = "0.8"
regex = "1"
uuid = { version = "1", features = ["v7"] }  # Time-ordered UUIDs for entries

# Observability
tracing = "0.1"
tracing-subscriber = "0.3"

# Testing
proptest = "1"                # Property-based testing
criterion = "0.5"             # Benchmarking
```

---

## Migration Path

1. **Phase 1**: Core library + CLI (can coexist with zsh version)
2. **Phase 2**: Daemon + zsh integration
3. **Phase 3**: Import tool for existing databases
4. **Phase 4**: Deprecate zsh version

---

## Open Questions for Review

### Decided
- ✅ **Daemon vs embedded**: Daemon with lock-free queue (for minimal latency)
- ✅ **Async runtime**: Tokio for socket handling, dedicated writer thread for SQLite
- ✅ **Concurrency**: Lock-free MPSC queue for writes, left-right for reads
- ✅ **Config format**: TOML with environment variable overrides
- ✅ **FTS**: SQLite FTS5 default, extensible via `SearchBackend` trait
- ✅ **Encryption**: Design supports future SQLCipher (deferred implementation)
- ✅ **Shell support**: ZSH, BASH, Nushell with shell-agnostic IPC
- ✅ **Sync**: Hybrid (Raft-style log replication + CRDT semantics)
- ✅ **Backpressure strategy**: Retry with exponential backoff (100μs, 200μs, 400μs)
- ✅ **Sync deployment**: P2P cluster with operation-based CRDTs
   - Raft-style log shipping for efficient propagation
   - CRDT merge semantics (always-writable, auto-converge)
   - Hybrid Logical Clocks for causal ordering
   - mDNS for automatic peer discovery on LAN
- ✅ **History ID correlation**: Shell tracks returned ID from StartCommand
   - More reliable than session + heuristic approach
   - Explicit correlation prevents wrong entry updates
   - Shell stores ID in variable, passes back to FinishCommand
- ✅ **Daemon lifecycle**: systemd/launchd socket activation with shell-initiated fallback
   - Primary: Socket activation (zero resources until first connection)
   - Fallback: Shell starts daemon if socket missing (Alpine, BSDs)
   - Graceful shutdown drains queue, flushes SQLite, syncs peers
   - Crash recovery: ≤10ms data loss, auto-restart, index rebuild from SQLite
- ✅ **Import tool**: Interface-only with fresh instance implementation
   - `ImportBackend` trait for future extensibility
   - Default `NewInstance` implementation creates empty database
   - No import formats in initial release (deferred)
- ✅ **Shell offline queue**: Local SQLite buffer when daemon unavailable
   - Shell queues commands to `~/.local/share/histdb/offline.db`
   - Replay to daemon on reconnect
   - Near 100% reliability (no lost commands)

### Still Open

(No critical questions remaining - ready for implementation review)

---

## Summary

This architecture prioritizes:

- **Safety**: Parameterized queries, type-safe models, proper error handling
- **Performance**: Lock-free writes, wait-free reads (left-right), batched SQLite operations
- **Correctness**: Transaction safety, comprehensive testing, atomic migrations
- **Extensibility**: Pluggable search backends, multi-shell support, encryption-ready
- **Sync**: Offline-first hybrid sync (Raft log replication + CRDT semantics)
- **Maintainability**: Clean separation of concerns, well-defined interfaces

### Key Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Write path | Lock-free MPSC queue | Sub-millisecond shell latency |
| Read path | Left-right pattern | Wait-free queries |
| Storage | SQLite + WAL | Reliable, portable, well-understood |
| Search | FTS5 (extensible) | Fast full-text, graceful fallback |
| Sync | Hybrid (log replication + CRDTs) | Always-writable, auto-converge, P2P |
| Config | TOML | Human-readable, well-supported |
| Shells | ZSH, BASH, Nushell | Cover majority of users |
