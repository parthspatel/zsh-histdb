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
use uuid::Uuid;

/// Type aliases for UUID-based identifiers
/// All IDs use UUIDv7 for global uniqueness and time-ordering
pub type SessionId = Uuid;
pub type HistoryId = Uuid;
pub type EntryId = Uuid;

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
        session_id: SessionId,  // UUIDv7 - globally unique, time-ordered
        argv: Box<str>,         // Owned, no lifetime issues
        dir: Box<Path>,
        host: Box<str>,
        start_time: i64,
        response_tx: Option<oneshot::Sender<HistoryId>>,
    },
    Finish {
        history_id: HistoryId,  // UUIDv7 - correlates with Start response
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

#### 2. Writer Thread with Supervisor Pattern

```rust
// ingest/supervisor.rs

/// Supervisor that manages and restarts the writer thread on failure
pub struct WriterSupervisor {
    queue: Arc<CommandQueue>,
    db_path: PathBuf,
    health: Arc<WriterHealth>,
    config: SupervisorConfig,
    /// Channel to signal supervisor shutdown
    shutdown_tx: broadcast::Sender<()>,
}

#[derive(Clone, Debug)]
pub struct SupervisorConfig {
    /// Maximum restart attempts before giving up
    pub max_restarts: u32,
    /// Time window for counting restarts
    pub restart_window: Duration,
    /// Delay before restarting after failure
    pub restart_delay: Duration,
    /// Backoff multiplier for repeated failures
    pub backoff_multiplier: f64,
    /// Maximum restart delay
    pub max_restart_delay: Duration,
}

impl Default for SupervisorConfig {
    fn default() -> Self {
        Self {
            max_restarts: 5,
            restart_window: Duration::from_secs(60),
            restart_delay: Duration::from_millis(100),
            backoff_multiplier: 2.0,
            max_restart_delay: Duration::from_secs(10),
        }
    }
}

/// Tracks restart history for rate limiting
struct RestartHistory {
    timestamps: VecDeque<Instant>,
    current_delay: Duration,
}

impl WriterSupervisor {
    pub fn new(
        queue: Arc<CommandQueue>,
        db_path: PathBuf,
        health: Arc<WriterHealth>,
        config: SupervisorConfig,
    ) -> (Self, broadcast::Receiver<()>) {
        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let supervisor = Self {
            queue,
            db_path,
            health,
            config,
            shutdown_tx,
        };
        (supervisor, shutdown_rx)
    }

    /// Start the supervisor (runs until shutdown signal)
    pub async fn run(self) {
        let mut history = RestartHistory {
            timestamps: VecDeque::new(),
            current_delay: self.config.restart_delay,
        };

        loop {
            // Spawn writer thread
            let handle = WriterThread::spawn(
                Arc::clone(&self.queue),
                &self.db_path,
                Arc::clone(&self.health),
            );

            tracing::info!("Writer thread started");

            // Wait for thread to exit (it shouldn't under normal operation)
            let exit_reason = tokio::task::spawn_blocking(move || handle.join()).await;

            // Check if we should shutdown
            if self.shutdown_tx.receiver_count() == 0 {
                tracing::info!("Supervisor shutting down");
                break;
            }

            // Thread exited unexpectedly - handle restart
            match exit_reason {
                Ok(Ok(())) => {
                    tracing::warn!("Writer thread exited normally (unexpected)");
                }
                Ok(Err(_)) => {
                    tracing::error!("Writer thread panicked");
                }
                Err(e) => {
                    tracing::error!("Failed to join writer thread: {}", e);
                }
            }

            // Rate limit restarts
            let now = Instant::now();
            history.timestamps.push_back(now);

            // Remove old timestamps outside the window
            while let Some(&ts) = history.timestamps.front() {
                if now.duration_since(ts) > self.config.restart_window {
                    history.timestamps.pop_front();
                } else {
                    break;
                }
            }

            // Check if too many restarts
            if history.timestamps.len() as u32 >= self.config.max_restarts {
                tracing::error!(
                    "Writer thread restarted {} times in {:?}, giving up",
                    self.config.max_restarts,
                    self.config.restart_window
                );
                self.health.mark_error("Max restarts exceeded - supervisor stopped");
                break;
            }

            // Apply backoff delay
            tracing::info!("Restarting writer thread in {:?}", history.current_delay);
            tokio::time::sleep(history.current_delay).await;

            // Increase delay for next failure (exponential backoff)
            history.current_delay = Duration::from_secs_f64(
                (history.current_delay.as_secs_f64() * self.config.backoff_multiplier)
                    .min(self.config.max_restart_delay.as_secs_f64())
            );
        }
    }

    /// Request supervisor shutdown
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }
}

// ingest/writer.rs
/// Writer thread health status
pub struct WriterHealth {
    /// True if writer is healthy, false if experiencing errors
    healthy: AtomicBool,
    /// Count of consecutive errors
    error_count: AtomicU64,
    /// Last error message (for diagnostics)
    last_error: parking_lot::RwLock<Option<String>>,
    /// Count of restarts (set by supervisor)
    restart_count: AtomicU32,
}

impl WriterHealth {
    pub fn new() -> Self {
        Self {
            healthy: AtomicBool::new(true),
            error_count: AtomicU64::new(0),
            last_error: parking_lot::RwLock::new(None),
            restart_count: AtomicU32::new(0),
        }
    }

    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }

    pub fn mark_healthy(&self) {
        self.healthy.store(true, Ordering::Relaxed);
        self.error_count.store(0, Ordering::Relaxed);
    }

    pub fn mark_error(&self, err: &str) {
        self.healthy.store(false, Ordering::Relaxed);
        self.error_count.fetch_add(1, Ordering::Relaxed);
        *self.last_error.write() = Some(err.to_string());
    }

    pub fn error_count(&self) -> u64 {
        self.error_count.load(Ordering::Relaxed)
    }

    pub fn restart_count(&self) -> u32 {
        self.restart_count.load(Ordering::Relaxed)
    }

    pub fn increment_restart(&self) {
        self.restart_count.fetch_add(1, Ordering::Relaxed);
    }
}

pub struct WriterThread {
    queue: Arc<CommandQueue>,
    conn: Connection,
    health: Arc<WriterHealth>,
    // Pre-compiled statements for hot path
    insert_command_stmt: Statement,
    insert_place_stmt: Statement,
    insert_history_stmt: Statement,
    update_finish_stmt: Statement,
}

impl WriterThread {
    /// Spawn writer thread with health monitoring
    pub fn spawn(
        queue: Arc<CommandQueue>,
        db_path: &Path,
        health: Arc<WriterHealth>,
    ) -> JoinHandle<()> {
        let db_path = db_path.to_owned();

        std::thread::spawn(move || {
            // Catch panics to prevent silent thread death
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                match Self::init_and_run(queue, &db_path, Arc::clone(&health)) {
                    Ok(()) => {}
                    Err(e) => {
                        health.mark_error(&format!("Writer thread error: {}", e));
                        tracing::error!("Writer thread exited with error: {}", e);
                    }
                }
            }));

            if let Err(panic) = result {
                let msg = panic
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "Unknown panic".to_string());

                health.mark_error(&format!("Writer thread panicked: {}", msg));
                tracing::error!("Writer thread panicked: {}", msg);
            }
        })
    }

    fn init_and_run(
        queue: Arc<CommandQueue>,
        db_path: &Path,
        health: Arc<WriterHealth>,
    ) -> Result<(), rusqlite::Error> {
        let conn = Connection::open(db_path)?;

        // Optimize for write throughput
        conn.execute_batch("
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA wal_autocheckpoint = 1000;
            PRAGMA busy_timeout = 5000;
        ")?;

        let mut writer = WriterThread::new(queue, conn, health);
        writer.run()
    }

    fn run(&mut self) -> Result<(), rusqlite::Error> {
        let mut batch = Vec::with_capacity(64);
        let batch_interval = Duration::from_millis(10);
        let mut last_flush = Instant::now();
        let mut retry_backoff = Duration::from_millis(10);
        const MAX_BACKOFF: Duration = Duration::from_secs(5);

        loop {
            // Drain available commands (non-blocking)
            batch.extend(self.queue.drain_batch(64 - batch.len()));

            let should_flush = !batch.is_empty() && (
                batch.len() >= 64 ||                        // Batch full
                last_flush.elapsed() >= batch_interval      // Time limit
            );

            if should_flush {
                match self.flush_batch(&mut batch) {
                    Ok(()) => {
                        self.health.mark_healthy();
                        retry_backoff = Duration::from_millis(10);  // Reset backoff
                        last_flush = Instant::now();
                    }
                    Err(e) => {
                        self.health.mark_error(&e.to_string());
                        tracing::warn!("Batch flush failed: {}, retrying in {:?}", e, retry_backoff);

                        // Exponential backoff before retry
                        std::thread::sleep(retry_backoff);
                        retry_backoff = std::cmp::min(retry_backoff * 2, MAX_BACKOFF);

                        // Re-attempt: if database is locked/busy, retry will succeed
                        // Commands remain in batch for retry
                    }
                }
            } else if batch.is_empty() {
                // No work: park thread briefly to avoid busy-spin
                std::thread::park_timeout(Duration::from_millis(1));
            }
        }
    }

    fn flush_batch(&mut self, batch: &mut Vec<PendingCommand>) -> Result<(), rusqlite::Error> {
        if batch.is_empty() {
            return Ok(());
        }

        // Single transaction for entire batch
        let tx = self.conn.transaction()?;

        for cmd in batch.drain(..) {
            match cmd.kind {
                CommandKind::Start { session_id, argv, dir, host, start_time, response_tx } => {
                    let id = self.insert_history_entry(&tx, session_id, &argv, &dir, &host, start_time)?;
                    if let Some(tx) = response_tx {
                        let _ = tx.send(id);  // Ignore if receiver dropped
                    }
                }
                CommandKind::Finish { history_id, exit_status, duration_ms } => {
                    self.update_finish(&tx, history_id, exit_status, duration_ms)?;
                }
            }
        }

        tx.commit()?;
        Ok(())
    }

    #[inline]
    fn insert_history_entry(
        &self,
        tx: &Transaction,
        session_id: SessionId,
        argv: &str,
        dir: &Path,
        host: &str,
        start_time: i64,
    ) -> Result<HistoryId, rusqlite::Error> {
        // Generate new UUIDv7 for history entry
        let history_id = Uuid::now_v7();
        // Uses prepared statements, parameterized queries
        // ... implementation stores history_id as TEXT in SQLite
        Ok(history_id)
    }

    fn update_finish(
        &self,
        tx: &Transaction,
        history_id: HistoryId,
        exit_status: i32,
        duration_ms: u64,
    ) -> Result<(), rusqlite::Error> {
        // Update history entry with exit status and duration
        // ... implementation
        Ok(())
    }
}
```

#### 3. Fire-and-Forget IPC Handler

```rust
// daemon/handler.rs
pub struct ConnectionHandler {
    queue: Arc<CommandQueue>,
    session_id: SessionId,  // UUIDv7 from shell
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
            Ok(Ok(id)) => Response::CommandStarted { id },  // id is HistoryId (Uuid)
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

### Async Backpressure Handling (Shell Never Waits)

The shell should **never block** waiting for the daemon. Instead of synchronous retries,
the daemon "watches" the shell asynchronously:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Async Backpressure Flow                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Shell                    Daemon                       Writer Thread         │
│    │                        │                              │                 │
│    │  send_async(cmd)       │                              │                 │
│    │───────────────────────►│                              │                 │
│    │                        │  try_push (non-blocking)     │                 │
│    │  ack (immediate)       │◄─────────────────────────────│                 │
│    │◄───────────────────────│                              │                 │
│    │                        │                              │                 │
│    │  (shell continues)     │  if QueueFull:               │                 │
│    │                        │    spawn async retry task    │                 │
│    │                        │    │                         │                 │
│    │                        │    │  wait_for_space()       │                 │
│    │                        │    │────────────────────────►│                 │
│    │                        │    │                         │  drain()        │
│    │                        │    │  space_available        │                 │
│    │                        │    │◄────────────────────────│                 │
│    │                        │    │                         │                 │
│    │                        │    │  retry_push(cmd)        │                 │
│    │                        │    │────────────────────────►│                 │
│    │                        │                              │                 │
│    ▼                        ▼                              ▼                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation:**

```rust
// daemon/handler.rs
use tokio::sync::mpsc;

/// Overflow queue for commands when main queue is full
struct OverflowHandler {
    /// Channel for commands that couldn't be queued immediately
    overflow_tx: mpsc::Sender<PendingCommand>,
    /// Task handle for async retry processor
    retry_task: JoinHandle<()>,
}

impl OverflowHandler {
    /// Spawn async overflow handler
    pub fn new(queue: Arc<CommandQueue>, capacity: usize) -> Self {
        let (overflow_tx, overflow_rx) = mpsc::channel(capacity);

        let retry_task = tokio::spawn(Self::retry_loop(queue, overflow_rx));

        Self { overflow_tx, retry_task }
    }

    /// Async retry loop - watches for space in main queue
    async fn retry_loop(
        queue: Arc<CommandQueue>,
        mut overflow_rx: mpsc::Receiver<PendingCommand>,
    ) {
        while let Some(cmd) = overflow_rx.recv().await {
            // Wait for queue space with exponential backoff
            let mut delay = Duration::from_micros(100);
            let max_delay = Duration::from_millis(100);

            loop {
                match queue.try_push(cmd.clone()) {
                    Ok(()) => {
                        tracing::debug!("Overflow command successfully queued");
                        break;
                    }
                    Err(_) => {
                        // Queue still full - async wait
                        tokio::time::sleep(delay).await;
                        delay = (delay * 2).min(max_delay);
                    }
                }
            }
        }
    }
}

impl ConnectionHandler {
    pub async fn handle_start(&self, argv: String, dir: String) -> Response {
        let cmd = PendingCommand { /* ... */ };

        // Non-blocking push attempt
        match self.queue.try_push(cmd.clone()) {
            Ok(()) => Response::Accepted,
            Err(_) => {
                // Queue full - hand off to async overflow handler
                // Shell gets immediate response, daemon retries async
                match self.overflow.overflow_tx.try_send(cmd) {
                    Ok(()) => {
                        metrics::counter!("histdb.queue.overflow").increment(1);
                        Response::AcceptedOverflow  // Tell shell "we got it, processing async"
                    }
                    Err(_) => {
                        // Even overflow is full - extreme backpressure
                        metrics::counter!("histdb.queue.overflow_full").increment(1);
                        Response::QueueFull
                    }
                }
            }
        }
    }
}
```

**Shell side - non-blocking send:**

```zsh
# zsh integration - async send
_histdb_send_async() {
    local json="$1" request_type="$2"
    local hlc_json=$(_histdb_get_hlc)

    # Background send - shell never waits
    {
        local resp
        resp=$(_histdb_send "$json" 2>/dev/null)

        case "$resp" in
            *"Accepted"*|*"AcceptedOverflow"*)
                # Success - daemon is handling it
                ;;
            *"QueueFull"*|"")
                # Daemon overloaded or unreachable - queue locally
                _histdb_queue_offline "$request_type" "$json" "$hlc_json"
                ;;
        esac
    } &!  # Disown: shell doesn't wait, no zombie processes
}

# Updated hooks use async send
_histdb_addhistory() {
    _histdb_init
    local cmd="${1[1,-2]}"
    [[ -z "$cmd" ]] && return 0

    HISTDB_LAST_HISTORY_ID=$(_histdb_generate_uuid)
    local json='{"type":"StartCommand","history_id":"'$HISTDB_LAST_HISTORY_ID'",...}'

    # Async send - returns immediately
    _histdb_send_async "$json" "start"
    return 0
}
```

**Response types:**

| Response | Meaning | Shell Action |
|----------|---------|--------------|
| `Accepted` | Queued in main queue | Done |
| `AcceptedOverflow` | Queued in overflow, daemon retrying | Done |
| `QueueFull` | Both queues full (extreme load) | Queue offline |
| (timeout) | Daemon unreachable | Queue offline |

**Benefits:**
- Shell hooks complete in <1μs (just fork a background process)
- Daemon handles backpressure asynchronously with retries
- Extreme backpressure falls back to offline queue
- No shell latency even under high load

### Query Consistency Model: Bounded Staleness

histdb uses a **bounded staleness** consistency model:

> **Guarantee**: Queries may not see commands written within the last `STALENESS_BOUND_MS`
> (default: 10ms). After this window, all written commands are guaranteed to be visible.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Bounded Staleness Timeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  t=0ms        t=5ms          t=10ms         t=15ms                          │
│    │            │              │              │                              │
│    ▼            ▼              ▼              ▼                              │
│  ┌────┐      ┌────┐        ┌────┐         ┌────┐                            │
│  │CMD │      │    │        │CMD │         │CMD │                            │
│  │SENT│      │    │        │IN  │         │VISIBLE│                         │
│  └────┘      └────┘        │DB  │         │TO     │                         │
│                            └────┘         │QUERY  │                         │
│                                           └────┘                            │
│  ◄────── Staleness Window ──────►                                           │
│          (configurable)                                                      │
│                                                                              │
│  During this window:          After this window:                            │
│  - Command MAY not be visible - Command GUARANTEED visible                  │
│  - Depends on batch timing    - Strong consistency                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Configuration**:
```toml
[daemon]
# Bounded staleness window (milliseconds)
# Queries may not see commands newer than this
staleness_bound_ms = 10
```

**Implementation**:
```rust
// query.rs

/// Configuration for query consistency
pub struct ConsistencyConfig {
    /// Maximum staleness for queries (default: 10ms)
    pub staleness_bound: Duration,
}

impl Default for ConsistencyConfig {
    fn default() -> Self {
        Self {
            staleness_bound: Duration::from_millis(10),
        }
    }
}

impl QueryExecutor {
    pub fn query(&self, builder: QueryBuilder) -> Result<Vec<HistoryRow>, Error> {
        // Read from SQLite - bounded staleness guarantees visibility after staleness_bound
        self.conn.query(&builder.build_sql())
    }

    /// Query with explicit freshness requirement
    /// Blocks until staleness bound has elapsed since last write
    pub fn query_fresh(&self, builder: QueryBuilder) -> Result<Vec<HistoryRow>, Error> {
        // Wait for any pending writes to complete
        self.wait_for_staleness_bound()?;
        self.conn.query(&builder.build_sql())
    }

    fn wait_for_staleness_bound(&self) -> Result<(), Error> {
        let last_write = self.last_write_time.load(Ordering::Acquire);
        let elapsed = Instant::now().duration_since(Instant::from_nanos(last_write));

        if elapsed < self.config.staleness_bound {
            std::thread::sleep(self.config.staleness_bound - elapsed);
        }
        Ok(())
    }
}
```

**Why Bounded Staleness?**
- **Predictable**: Users know exactly how stale results can be
- **Testable**: Can write tests that verify the bound
- **Simple**: No complex merge logic between queue and database
- **Fast**: No blocking on writes for normal queries

**Test Cases for Bounded Staleness**:
```rust
#[cfg(test)]
mod consistency_tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Test that commands become visible within staleness bound
    #[test]
    fn test_bounded_staleness_visibility() {
        let daemon = TestDaemon::new();
        let staleness_bound = Duration::from_millis(10);

        // Write a command
        let cmd = "echo test";
        daemon.record_command(cmd);

        // Immediately query - may not be visible
        let results_immediate = daemon.query_recent(10);

        // Wait for staleness bound + buffer
        std::thread::sleep(staleness_bound + Duration::from_millis(5));

        // Query again - must be visible
        let results_after = daemon.query_recent(10);
        assert!(results_after.iter().any(|r| r.argv == cmd),
            "Command must be visible after staleness bound");
    }

    /// Test that staleness bound is actually bounded
    #[test]
    fn test_staleness_bound_maximum() {
        let daemon = TestDaemon::new();
        let staleness_bound = Duration::from_millis(10);

        // Record 100 commands rapidly
        for i in 0..100 {
            daemon.record_command(&format!("cmd_{}", i));
        }

        // Wait for staleness bound
        std::thread::sleep(staleness_bound + Duration::from_millis(5));

        // All commands must be visible
        let results = daemon.query_recent(100);
        assert_eq!(results.len(), 100, "All commands visible after staleness bound");
    }

    /// Measure actual staleness to verify bound
    #[test]
    fn test_measure_actual_staleness() {
        let daemon = TestDaemon::new();
        let mut max_staleness = Duration::ZERO;

        for _ in 0..100 {
            let cmd = format!("test_{}", uuid::Uuid::new_v4());
            let write_time = Instant::now();
            daemon.record_command(&cmd);

            // Poll until visible
            loop {
                let results = daemon.query_recent(1);
                if results.iter().any(|r| r.argv == cmd) {
                    let staleness = write_time.elapsed();
                    max_staleness = max_staleness.max(staleness);
                    break;
                }
                std::thread::sleep(Duration::from_micros(100));
            }
        }

        println!("Max observed staleness: {:?}", max_staleness);
        assert!(max_staleness < Duration::from_millis(15),
            "Staleness should be bounded by ~10ms + jitter");
    }

    /// Test query_fresh blocks appropriately
    #[test]
    fn test_query_fresh_blocks() {
        let daemon = TestDaemon::new();

        let cmd = "fresh_test";
        daemon.record_command(cmd);

        let start = Instant::now();
        let results = daemon.query_fresh(1);  // Should block
        let elapsed = start.elapsed();

        assert!(results.iter().any(|r| r.argv == cmd));
        assert!(elapsed >= Duration::from_millis(8),
            "query_fresh should wait for staleness bound");
    }
}
```

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
use std::mem::size_of;

/// Memory bounds configuration for the left-right index
#[derive(Clone, Debug)]
pub struct IndexBounds {
    /// Maximum number of entries
    pub max_entries: usize,
    /// Maximum memory usage (bytes, approximate)
    pub max_memory_bytes: usize,
    /// Maximum age of entries (seconds)
    pub max_age_secs: u64,
    /// Maximum entries per session
    pub max_per_session: usize,
}

impl Default for IndexBounds {
    fn default() -> Self {
        Self {
            // Default to 1000 entries - user-configurable via [index].max_entries
            // Hybrid query falls back to SQLite for entries beyond this limit
            max_entries: 1_000,
            max_memory_bytes: 50 * 1024 * 1024,  // 50MB
            max_age_secs: 7 * 24 * 60 * 60,      // 7 days
            max_per_session: 500,
        }
    }
}

/// In-memory index of recent history for fast queries
pub struct HistoryIndex {
    /// Recent commands (LRU, bounded size)
    recent: Vec<RecentEntry>,
    /// Command → entry IDs (for dedup/lookup)
    by_command: HashMap<Box<str>, Vec<EntryId>>,
    /// Directory → entry IDs (for --in/--at queries)
    by_dir: HashMap<Box<Path>, Vec<EntryId>>,
    /// Last N entries per session (for isearch)
    by_session: HashMap<SessionId, VecDeque<EntryId>>,  // SessionId is Uuid
    /// Memory bounds configuration
    bounds: IndexBounds,
    /// Current estimated memory usage
    estimated_memory: usize,
}

impl HistoryIndex {
    pub fn new(bounds: IndexBounds) -> Self {
        Self {
            recent: Vec::with_capacity(bounds.max_entries.min(1000)),
            by_command: HashMap::new(),
            by_dir: HashMap::new(),
            by_session: HashMap::new(),
            bounds,
            estimated_memory: 0,
        }
    }

    /// Estimate memory used by an entry
    fn entry_memory_size(entry: &RecentEntry) -> usize {
        size_of::<RecentEntry>()
            + entry.argv.len()
            + entry.dir.as_os_str().len()
            + entry.host.len()
            + 16 * 3  // UUID storage overhead in maps
    }

    /// Check if eviction is needed
    pub fn needs_eviction(&self) -> bool {
        self.recent.len() >= self.bounds.max_entries
            || self.estimated_memory >= self.bounds.max_memory_bytes
    }

    /// Evict entries to stay within bounds
    pub fn evict_if_needed(&mut self) {
        let now = chrono::Utc::now().timestamp();
        let age_threshold = now - self.bounds.max_age_secs as i64;

        // Phase 1: Evict by age
        let before_count = self.recent.len();
        self.evict_older_than(age_threshold);

        // Phase 2: Evict by count if still over
        while self.recent.len() > self.bounds.max_entries {
            self.evict_oldest_entry();
        }

        // Phase 3: Evict by memory if still over
        while self.estimated_memory > self.bounds.max_memory_bytes && !self.recent.is_empty() {
            self.evict_oldest_entry();
        }

        if self.recent.len() < before_count {
            tracing::debug!(
                "Evicted {} entries, now at {} entries, ~{} bytes",
                before_count - self.recent.len(),
                self.recent.len(),
                self.estimated_memory
            );
        }
    }

    fn evict_older_than(&mut self, threshold: i64) {
        let to_remove: Vec<_> = self.recent.iter()
            .filter(|e| e.start_time < threshold)
            .map(|e| e.id)
            .collect();

        for id in to_remove {
            self.remove_entry(id);
        }
    }

    fn evict_oldest_entry(&mut self) {
        if let Some(oldest) = self.recent.iter().min_by_key(|e| e.start_time) {
            let id = oldest.id;
            self.remove_entry(id);
        }
    }

    fn remove_entry(&mut self, id: EntryId) {
        if let Some(pos) = self.recent.iter().position(|e| e.id == id) {
            let entry = self.recent.remove(pos);
            self.estimated_memory -= Self::entry_memory_size(&entry);

            // Clean up maps
            if let Some(ids) = self.by_command.get_mut(&entry.argv) {
                ids.retain(|&eid| eid != id);
            }
            if let Some(ids) = self.by_dir.get_mut(&entry.dir) {
                ids.retain(|&eid| eid != id);
            }
            if let Some(ids) = self.by_session.get_mut(&entry.session) {
                ids.retain(|&eid| eid != id);
            }
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> IndexStats {
        IndexStats {
            entry_count: self.recent.len(),
            estimated_memory_bytes: self.estimated_memory,
            unique_commands: self.by_command.len(),
            unique_directories: self.by_dir.len(),
            active_sessions: self.by_session.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct IndexStats {
    pub entry_count: usize,
    pub estimated_memory_bytes: usize,
    pub unique_commands: usize,
    pub unique_directories: usize,
    pub active_sessions: usize,
}

#[derive(Clone)]
pub struct RecentEntry {
    pub id: EntryId,          // UUIDv7
    pub session: SessionId,   // UUIDv7
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
    EvictIfNeeded,          // Check bounds and evict as needed
}

impl Absorb<IndexOp> for HistoryIndex {
    fn absorb_first(&mut self, op: &mut IndexOp, _: &Self) {
        match op {
            IndexOp::Insert(entry) => {
                // Track memory
                self.estimated_memory += Self::entry_memory_size(entry);

                self.by_command
                    .entry(entry.argv.clone())
                    .or_default()
                    .push(entry.id);
                self.by_dir
                    .entry(entry.dir.clone())
                    .or_default()
                    .push(entry.id);

                // Enforce per-session limit
                let session_entries = self.by_session
                    .entry(entry.session)
                    .or_default();
                if session_entries.len() >= self.bounds.max_per_session {
                    // Evict oldest from this session
                    if let Some(old_id) = session_entries.pop_front() {
                        self.remove_entry(old_id);
                    }
                }
                session_entries.push_back(entry.id);

                self.recent.push(entry.clone());

                // Check bounds after insert
                if self.needs_eviction() {
                    self.evict_if_needed();
                }
            }
            IndexOp::UpdateExitStatus { id, status, .. } => {
                if let Some(entry) = self.recent.iter_mut().find(|e| e.id == *id) {
                    entry.exit_status = Some(*status);
                }
            }
            IndexOp::Evict { before } => {
                self.evict_older_than(*before);
            }
            IndexOp::EvictIfNeeded => {
                self.evict_if_needed();
            }
        }
    }

    fn absorb_second(&mut self, op: IndexOp, other: &Self) {
        // Same logic, can optimize by copying from `other`
        self.absorb_first(&mut { op }, other);
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

/// Hybrid query executor: in-memory index + SQLite fallback
///
/// The in-memory index holds the N most recent commands (configurable, default 1000).
/// Queries first check the index (wait-free), then fall back to SQLite for older
/// entries or when the index doesn't satisfy the query fully.
pub struct QueryExecutor {
    /// Wait-free reader for recent entries (bounded to N entries)
    index: IndexReader,
    /// SQLite connection pool for older entries
    pool: ConnectionPool,
    /// Search backend (FTS5 or GLOB fallback)
    search: Box<dyn SearchBackend>,
    /// Index bounds configuration (for knowing coverage)
    index_bounds: IndexBounds,
}

/// Result source tracking for debugging/metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultSource {
    /// All results from in-memory index (fastest)
    IndexOnly,
    /// Results from SQLite only (index empty or query forced DB)
    DatabaseOnly,
    /// Merged results from both sources
    Hybrid,
}

/// Query result with metadata
pub struct QueryResult {
    pub rows: Vec<HistoryRow>,
    pub source: ResultSource,
    pub index_hits: usize,
    pub db_hits: usize,
}

impl QueryExecutor {
    pub fn new(
        index: IndexReader,
        pool: ConnectionPool,
        search: Box<dyn SearchBackend>,
        index_bounds: IndexBounds,
    ) -> Self {
        Self { index, pool, search, index_bounds }
    }

    /// Primary query method - uses hybrid index+SQLite approach
    pub fn query(&self, builder: &QueryBuilder) -> Result<Vec<HistoryRow>, Error> {
        self.query_with_metadata(builder).map(|r| r.rows)
    }

    /// Query with metadata about result sources (for debugging/metrics)
    pub fn query_with_metadata(&self, builder: &QueryBuilder) -> Result<QueryResult, Error> {
        let requested_limit = builder.limit.unwrap_or(100);

        // 1. Fast path: check in-memory index first (wait-free, O(1) reader acquisition)
        let recent = self.index.search(builder)?;
        let index_hits = recent.len();

        // 2. Determine if we need SQLite fallback
        let needs_db_fallback = self.needs_database_fallback(builder, &recent);

        if !needs_db_fallback && recent.len() >= requested_limit {
            // Index fully satisfied the query - fastest path
            return Ok(QueryResult {
                rows: recent.into_iter().take(requested_limit).map(Into::into).collect(),
                source: ResultSource::IndexOnly,
                index_hits,
                db_hits: 0,
            });
        }

        // 3. Query SQLite for older/additional entries
        let conn = self.pool.get()?;

        // Adjust DB query to exclude IDs already found in index
        let exclude_ids: HashSet<_> = recent.iter().map(|e| e.id).collect();
        let db_builder = builder.clone()
            .exclude_ids(exclude_ids)
            .limit(requested_limit.saturating_sub(recent.len()));

        let from_db = self.search.search(&conn, &db_builder)?;
        let db_hits = from_db.len();

        // 4. Merge results: index entries first (more recent), then DB entries
        let merged = self.merge_results(recent, from_db, requested_limit);

        let source = match (index_hits > 0, db_hits > 0) {
            (true, true) => ResultSource::Hybrid,
            (true, false) => ResultSource::IndexOnly,
            (false, _) => ResultSource::DatabaseOnly,
        };

        Ok(QueryResult {
            rows: merged,
            source,
            index_hits,
            db_hits,
        })
    }

    /// Determine if query requires database fallback
    fn needs_database_fallback(&self, builder: &QueryBuilder, index_results: &[RecentEntry]) -> bool {
        // Always need DB if:
        // 1. Query requests more results than index can hold
        if builder.limit.unwrap_or(100) > self.index_bounds.max_entries {
            return true;
        }

        // 2. Query has time range that extends beyond index coverage
        if let Some(ref time_range) = builder.time_range {
            let oldest_in_index = self.index.oldest_timestamp();
            if let Some(oldest) = oldest_in_index {
                if time_range.start.map_or(true, |s| s < oldest) {
                    return true;
                }
            }
        }

        // 3. Index results are fewer than requested and query has no constraints
        //    that would limit results to recent entries only
        let requested = builder.limit.unwrap_or(100);
        if index_results.len() < requested && !builder.is_recent_only() {
            return true;
        }

        false
    }

    /// Merge index and database results, preserving order and deduplicating
    fn merge_results(
        &self,
        index_results: Vec<RecentEntry>,
        db_results: Vec<HistoryRow>,
        limit: usize,
    ) -> Vec<HistoryRow> {
        let mut seen: HashSet<Uuid> = HashSet::new();
        let mut merged = Vec::with_capacity(limit);

        // Add index results first (more recent)
        for entry in index_results {
            if seen.insert(entry.id) {
                merged.push(entry.into());
                if merged.len() >= limit {
                    return merged;
                }
            }
        }

        // Add DB results (older entries, already filtered to exclude index IDs)
        for row in db_results {
            if seen.insert(row.id) {
                merged.push(row);
                if merged.len() >= limit {
                    return merged;
                }
            }
        }

        merged
    }

    /// Force query through SQLite only (for testing/debugging)
    pub fn query_db_only(&self, builder: &QueryBuilder) -> Result<Vec<HistoryRow>, Error> {
        let conn = self.pool.get()?;
        self.search.search(&conn, builder)
    }
}

/// Extension trait for QueryBuilder to support hybrid queries
impl QueryBuilder {
    /// Check if query is constrained to recent entries only
    pub fn is_recent_only(&self) -> bool {
        // Query is recent-only if it has:
        // - session filter (sessions are bounded in index)
        // - time range starting recently
        // - explicit recent flag
        self.session.is_some() || self.recent_only
    }
}
```

### Memory Budget

The in-memory index is bounded to prevent unbounded growth. Default configuration
keeps the last 1000 commands in memory - queries for older entries use hybrid
query fallback to SQLite.

```rust
// config.rs
pub struct IndexConfig {
    /// Maximum entries in memory (default: 1,000 - user configurable)
    /// Higher values = faster recent queries, more memory
    /// Lower values = less memory, more SQLite fallback
    pub max_entries: usize,
    /// Evict entries older than this (default: 7 days)
    pub max_age: Duration,
    /// Memory limit for index (default: 50MB)
    pub memory_limit: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            max_entries: 1_000,  // Hybrid query handles overflow
            max_age: Duration::from_secs(7 * 24 * 60 * 60),
            memory_limit: 50 * 1024 * 1024,
        }
    }
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

/// Migration direction for rollback support
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MigrationDirection {
    Up,
    Down,
}

/// A database migration with up and down SQL
#[derive(Debug)]
pub struct Migration {
    /// Migration version number (sequential)
    pub version: u32,
    /// Human-readable name
    pub name: &'static str,
    /// SQL to apply the migration
    pub up: &'static str,
    /// SQL to rollback the migration (required for reversibility)
    pub down: &'static str,
    /// Whether this migration is destructive (requires backup)
    pub destructive: bool,
}

/// Migration error types
#[derive(Debug, thiserror::Error)]
pub enum MigrationError {
    #[error("Migration {0} failed: {1}")]
    ExecutionFailed(u32, String),

    #[error("Rollback of migration {0} failed: {1}")]
    RollbackFailed(u32, String),

    #[error("Backup failed: {0}")]
    BackupFailed(String),

    #[error("Database version {current} is newer than supported {supported}")]
    VersionTooNew { current: u32, supported: u32 },

    #[error("Missing rollback SQL for migration {0}")]
    NoRollbackSql(u32),

    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
}

/// Migration result with rollback information
#[derive(Debug)]
pub struct MigrationResult {
    pub from_version: u32,
    pub to_version: u32,
    pub migrations_applied: Vec<u32>,
    pub backup_path: Option<PathBuf>,
}

pub struct Migrator {
    conn: Connection,
    migrations: Vec<Migration>,
    backup_dir: PathBuf,
}

impl Migrator {
    pub fn new(conn: Connection, backup_dir: PathBuf) -> Self {
        Self {
            conn,
            migrations: Self::all_migrations(),
            backup_dir,
        }
    }

    /// Get current database version
    pub fn current_version(&self) -> Result<u32, MigrationError> {
        let version: i32 = self.conn.pragma_query_value(None, "user_version", |r| r.get(0))?;
        Ok(version as u32)
    }

    /// Get latest supported version
    pub fn latest_version(&self) -> u32 {
        self.migrations.last().map(|m| m.version).unwrap_or(0)
    }

    /// Run all pending migrations (up direction)
    pub fn migrate(&mut self) -> Result<MigrationResult, MigrationError> {
        let current = self.current_version()?;
        let latest = self.latest_version();

        // Check for newer database
        if current > latest {
            return Err(MigrationError::VersionTooNew {
                current,
                supported: latest,
            });
        }

        if current == latest {
            return Ok(MigrationResult {
                from_version: current,
                to_version: current,
                migrations_applied: vec![],
                backup_path: None,
            });
        }

        // Check if any pending migration is destructive
        let has_destructive = self.migrations.iter()
            .filter(|m| m.version > current)
            .any(|m| m.destructive);

        // Create backup if needed
        let backup_path = if has_destructive {
            Some(self.backup(&format!("pre_migration_v{}", current))?)
        } else {
            None
        };

        // Apply migrations in order
        let mut applied = Vec::new();
        for migration in self.migrations.iter().filter(|m| m.version > current) {
            match self.apply_migration(migration, MigrationDirection::Up) {
                Ok(()) => {
                    applied.push(migration.version);
                    tracing::info!("Applied migration {}: {}", migration.version, migration.name);
                }
                Err(e) => {
                    // Rollback applied migrations
                    tracing::error!("Migration {} failed: {}, rolling back", migration.version, e);
                    self.rollback_migrations(&applied)?;
                    return Err(e);
                }
            }
        }

        Ok(MigrationResult {
            from_version: current,
            to_version: latest,
            migrations_applied: applied,
            backup_path,
        })
    }

    /// Rollback to a specific version
    pub fn rollback_to(&mut self, target_version: u32) -> Result<MigrationResult, MigrationError> {
        let current = self.current_version()?;

        if target_version >= current {
            return Ok(MigrationResult {
                from_version: current,
                to_version: current,
                migrations_applied: vec![],
                backup_path: None,
            });
        }

        // Create backup before rollback
        let backup_path = self.backup(&format!("pre_rollback_v{}", current))?;

        // Collect migrations to rollback (in reverse order)
        let to_rollback: Vec<_> = self.migrations.iter()
            .filter(|m| m.version > target_version && m.version <= current)
            .collect();

        let mut rolled_back = Vec::new();
        for migration in to_rollback.into_iter().rev() {
            if migration.down.is_empty() {
                return Err(MigrationError::NoRollbackSql(migration.version));
            }

            match self.apply_migration(migration, MigrationDirection::Down) {
                Ok(()) => {
                    rolled_back.push(migration.version);
                    tracing::info!("Rolled back migration {}: {}", migration.version, migration.name);
                }
                Err(e) => {
                    tracing::error!("Rollback of migration {} failed: {}", migration.version, e);
                    return Err(MigrationError::RollbackFailed(migration.version, e.to_string()));
                }
            }
        }

        Ok(MigrationResult {
            from_version: current,
            to_version: target_version,
            migrations_applied: rolled_back,
            backup_path: Some(backup_path),
        })
    }

    /// Apply a single migration in the specified direction
    fn apply_migration(&self, migration: &Migration, direction: MigrationDirection) -> Result<(), MigrationError> {
        let sql = match direction {
            MigrationDirection::Up => migration.up,
            MigrationDirection::Down => migration.down,
        };

        // Run in transaction
        let tx = self.conn.transaction()?;

        tx.execute_batch(sql)
            .map_err(|e| MigrationError::ExecutionFailed(migration.version, e.to_string()))?;

        // Update version
        let new_version = match direction {
            MigrationDirection::Up => migration.version,
            MigrationDirection::Down => migration.version - 1,
        };
        tx.pragma_update(None, "user_version", new_version)?;

        tx.commit()?;
        Ok(())
    }

    /// Rollback a list of migrations (in reverse order)
    fn rollback_migrations(&self, versions: &[u32]) -> Result<(), MigrationError> {
        for &version in versions.iter().rev() {
            if let Some(migration) = self.migrations.iter().find(|m| m.version == version) {
                self.apply_migration(migration, MigrationDirection::Down)?;
            }
        }
        Ok(())
    }

    /// Create a backup of the database
    pub fn backup(&self, suffix: &str) -> Result<PathBuf, MigrationError> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let backup_name = format!("history_{}_{}.db", timestamp, suffix);
        let backup_path = self.backup_dir.join(&backup_name);

        // Use SQLite's backup API
        self.conn.backup(DatabaseName::Main, &backup_path, None)
            .map_err(|e| MigrationError::BackupFailed(e.to_string()))?;

        tracing::info!("Created backup: {}", backup_path.display());
        Ok(backup_path)
    }

    /// Define all migrations
    fn all_migrations() -> Vec<Migration> {
        vec![
            Migration {
                version: 1,
                name: "initial_schema",
                up: include_str!("../migrations/001_initial.up.sql"),
                down: include_str!("../migrations/001_initial.down.sql"),
                destructive: false,
            },
            Migration {
                version: 2,
                name: "add_fts5_search",
                up: include_str!("../migrations/002_fts5.up.sql"),
                down: include_str!("../migrations/002_fts5.down.sql"),
                destructive: false,
            },
            Migration {
                version: 3,
                name: "add_uuid_columns",
                up: include_str!("../migrations/003_uuid.up.sql"),
                down: include_str!("../migrations/003_uuid.down.sql"),
                destructive: false,
            },
            Migration {
                version: 4,
                name: "add_sync_tables",
                up: include_str!("../migrations/004_sync.up.sql"),
                down: include_str!("../migrations/004_sync.down.sql"),
                destructive: false,
            },
        ]
    }
}
```

**Key safety features:**
- All migrations run in a transaction (atomic apply/rollback)
- Every migration has both `up` and `down` SQL
- Automatic rollback on failure during migration sequence
- Backup created before destructive migrations
- Version tracking via `user_version` pragma
- Explicit `rollback_to(version)` for manual rollback
- Version compatibility check (prevents running against newer DB)

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

    // Generate UUIDv7 for this history entry
    let history_id = Uuid::now_v7();

    // All inserts in one transaction - no race conditions
    tx.execute(
        "INSERT OR IGNORE INTO commands (argv) VALUES (?1)",
        params![argv],
    )?;

    tx.execute(
        "INSERT OR IGNORE INTO places (host, dir) VALUES (?1, ?2)",
        params![self.host, dir.display().to_string()],
    )?;

    // INSERT OR IGNORE provides idempotent insert - UUID uniqueness acts as
    // deduplication key for offline queue replay and sync operations
    tx.execute(
        "INSERT OR IGNORE INTO history (uuid, session, command_id, place_id, start_time)
         SELECT ?1, ?2, c.id, p.id, ?3
         FROM commands c, places p
         WHERE c.argv = ?4 AND p.host = ?5 AND p.dir = ?6",
        params![history_id.to_string(), self.session_id.to_string(), Utc::now().timestamp(),
                argv, self.host, dir.display().to_string()],
    )?;

    tx.commit()?;
    Ok(history_id)
}
```

#### Querying History
```rust
// query.rs
pub struct QueryBuilder {
    host_filter: Option<String>,
    dir_filter: Option<DirFilter>,
    session_filter: Option<SessionId>,  // UUID filter
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
    /// Maximum entries in in-memory index (default: 1000)
    /// Queries beyond this limit fall back to SQLite (hybrid query)
    #[serde(default = "default_max_entries")]
    pub max_entries: usize,
    /// Evict entries older than (days)
    pub max_age_days: u32,
    /// Memory limit (bytes)
    pub memory_limit: usize,
}

fn default_max_entries() -> usize { 1000 }

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

/// Configuration validation errors
#[derive(Debug, thiserror::Error)]
pub enum ConfigValidationError {
    #[error("database.path: parent directory does not exist: {0}")]
    DatabasePathInvalid(PathBuf),

    #[error("daemon.socket_path: parent directory does not exist: {0}")]
    SocketPathInvalid(PathBuf),

    #[error("daemon.batch_size: must be between 1 and 1000, got {0}")]
    BatchSizeInvalid(usize),

    #[error("daemon.batch_interval_ms: must be between 1 and 1000, got {0}")]
    BatchIntervalInvalid(u64),

    #[error("daemon.queue_capacity: must be power of 2 between 64 and 65536, got {0}")]
    QueueCapacityInvalid(usize),

    #[error("index.max_entries: must be between 100 and 1000000, got {0}")]
    MaxEntriesInvalid(usize),

    #[error("index.memory_limit: must be at least 1MB, got {0} bytes")]
    MemoryLimitTooSmall(usize),

    #[error("ignore.patterns: invalid regex at index {0}: {1}")]
    InvalidIgnorePattern(usize, String),

    #[error("search.backend: unknown backend type: {0}")]
    UnknownSearchBackend(String),
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

        // Validate configuration
        config.validate()?;

        Ok(config)
    }

    /// Validate all configuration values
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        self.validate_database()?;
        self.validate_daemon()?;
        self.validate_index()?;
        self.validate_search()?;
        self.validate_ignore()?;
        Ok(())
    }

    fn validate_database(&self) -> Result<(), ConfigValidationError> {
        // Expand ~ in path
        let expanded = shellexpand::tilde(&self.database.path.to_string_lossy());
        let path = PathBuf::from(expanded.as_ref());

        // Check parent directory exists or can be created
        if let Some(parent) = path.parent() {
            if !parent.exists() && parent != Path::new("") {
                // Try to create parent directory
                if std::fs::create_dir_all(parent).is_err() {
                    return Err(ConfigValidationError::DatabasePathInvalid(
                        parent.to_path_buf()
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_daemon(&self) -> Result<(), ConfigValidationError> {
        // Validate socket path parent
        if let Some(parent) = self.daemon.socket_path.parent() {
            if !parent.exists() && parent != Path::new("") {
                return Err(ConfigValidationError::SocketPathInvalid(
                    parent.to_path_buf()
                ));
            }
        }

        // Validate batch size (1-1000)
        if self.daemon.batch_size == 0 || self.daemon.batch_size > 1000 {
            return Err(ConfigValidationError::BatchSizeInvalid(self.daemon.batch_size));
        }

        // Validate batch interval (1-1000ms)
        if self.daemon.batch_interval_ms == 0 || self.daemon.batch_interval_ms > 1000 {
            return Err(ConfigValidationError::BatchIntervalInvalid(self.daemon.batch_interval_ms));
        }

        // Validate queue capacity (power of 2, 64-65536)
        let cap = self.daemon.queue_capacity;
        if cap < 64 || cap > 65536 || !cap.is_power_of_two() {
            return Err(ConfigValidationError::QueueCapacityInvalid(cap));
        }

        Ok(())
    }

    fn validate_index(&self) -> Result<(), ConfigValidationError> {
        // Validate max entries (100-1M)
        if self.index.max_entries < 100 || self.index.max_entries > 1_000_000 {
            return Err(ConfigValidationError::MaxEntriesInvalid(self.index.max_entries));
        }

        // Validate memory limit (at least 1MB)
        if self.index.memory_limit < 1_048_576 {
            return Err(ConfigValidationError::MemoryLimitTooSmall(self.index.memory_limit));
        }

        Ok(())
    }

    fn validate_search(&self) -> Result<(), ConfigValidationError> {
        // Backend is already typed via SearchBackendType enum
        // Additional validation could check FTS5 tokenizer names
        Ok(())
    }

    fn validate_ignore(&self) -> Result<(), ConfigValidationError> {
        // Validate all regex patterns compile
        for (i, pattern) in self.ignore.patterns.iter().enumerate() {
            if let Err(e) = regex::Regex::new(pattern) {
                return Err(ConfigValidationError::InvalidIgnorePattern(i, e.to_string()));
            }
        }
        Ok(())
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
# Maximum entries in in-memory index (default: 1000)
# Queries for entries beyond this fall back to SQLite (hybrid query)
# Increase for faster queries over recent history, decrease to save memory
max_entries = 1000
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

    /// Queue a command to offline storage (shell helper)
    #[command(name = "offline-queue")]
    OfflineQueue(OfflineQueueArgs),

    /// Generate a new session ID (UUIDv7)
    #[command(name = "session-id")]
    SessionId,
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
    session: Option<Uuid>,  // SessionId is UUIDv7

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

/// Arguments for offline-queue command (shell helper)
#[derive(Args)]
pub struct OfflineQueueArgs {
    /// Path to offline queue database
    #[arg(long)]
    db: PathBuf,

    /// Request type ('start' or 'finish')
    #[arg(long, name = "TYPE")]
    r#type: String,

    /// JSON payload (safely bound, no injection possible)
    #[arg(long)]
    json: String,

    /// Original command timestamp
    #[arg(long)]
    timestamp: i64,
}
```

**Offline queue command implementation:**

```rust
// cli/offline.rs

pub fn handle_offline_queue(args: OfflineQueueArgs) -> Result<()> {
    let conn = Connection::open(&args.db)?;

    // Ensure schema exists
    conn.execute_batch("
        CREATE TABLE IF NOT EXISTS offline_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            queued_at INTEGER NOT NULL,
            request_type TEXT NOT NULL,
            request_json TEXT NOT NULL,
            original_timestamp INTEGER NOT NULL
        );
    ")?;

    // Safe parameterized insert - no SQL injection possible
    conn.execute(
        "INSERT INTO offline_queue (queued_at, request_type, request_json, original_timestamp)
         VALUES (?1, ?2, ?3, ?4)",
        params![
            Utc::now().timestamp(),
            &args.r#type,
            &args.json,
            args.timestamp
        ],
    )?;

    Ok(())
}

pub fn handle_session_id() {
    // Generate and print UUIDv7
    println!("{}", Uuid::now_v7());
}

// Note: Offline replay is handled by the daemon (OfflineQueueProcessor),
// not the CLI - see daemon/offline.rs
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

/// Pre-flight check for FTS5 availability
///
/// Called at daemon startup to check if FTS5 is available and provide
/// clear diagnostics if not. Returns availability status and diagnostic info.
pub fn check_fts5_availability(conn: &Connection) -> Fts5AvailabilityResult {
    // Check 1: SQLite compiled with FTS5 support
    let compile_check = conn.query_row(
        "SELECT 1 FROM pragma_compile_options WHERE compile_options = 'ENABLE_FTS5'",
        [],
        |_| Ok(()),
    ).is_ok();

    if !compile_check {
        return Fts5AvailabilityResult {
            available: false,
            reason: Fts5Unavailable::NotCompiled,
            sqlite_version: get_sqlite_version(conn),
            recommendation: "Install SQLite with FTS5 support or use backend = \"glob\" in config",
        };
    }

    // Check 2: Can create FTS5 virtual table
    let create_check = conn.execute_batch(
        "CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_test USING fts5(test); DROP TABLE _fts5_test;"
    ).is_ok();

    if !create_check {
        return Fts5AvailabilityResult {
            available: false,
            reason: Fts5Unavailable::CreateFailed,
            sqlite_version: get_sqlite_version(conn),
            recommendation: "FTS5 extension may not be loaded. Check SQLite configuration.",
        };
    }

    Fts5AvailabilityResult {
        available: true,
        reason: Fts5Unavailable::Available,
        sqlite_version: get_sqlite_version(conn),
        recommendation: "",
    }
}

#[derive(Debug)]
pub struct Fts5AvailabilityResult {
    pub available: bool,
    pub reason: Fts5Unavailable,
    pub sqlite_version: String,
    pub recommendation: &'static str,
}

#[derive(Debug)]
pub enum Fts5Unavailable {
    Available,
    NotCompiled,
    CreateFailed,
}

fn get_sqlite_version(conn: &Connection) -> String {
    conn.query_row("SELECT sqlite_version()", [], |row| row.get(0))
        .unwrap_or_else(|_| "unknown".to_string())
}

/// Factory function to create search backend from config
pub fn create_backend(config: &SearchConfig, conn: &Connection) -> Box<dyn SearchBackend> {
    match config.backend {
        SearchBackendType::Fts5 => {
            // Pre-flight check with detailed diagnostics
            let fts5_check = check_fts5_availability(conn);

            if fts5_check.available {
                let backend = Fts5Backend::new(
                    config.fts5_tokenizer.as_deref().unwrap_or("unicode61")
                );
                tracing::info!(
                    "FTS5 search enabled (SQLite {})",
                    fts5_check.sqlite_version
                );
                return Box::new(backend);
            }

            // FTS5 not available - log detailed warning
            tracing::warn!(
                "FTS5 not available (SQLite {}, reason: {:?}). {}. Falling back to GLOB.",
                fts5_check.sqlite_version,
                fts5_check.reason,
                fts5_check.recommendation
            );
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
│          ▼ (daemon handles replay on startup/connect)                   │
│   ┌──────────────────────────────────────┐                              │
│   │ Daemon scans offline queue           │                              │
│   │ - On startup                         │                              │
│   │ - On new shell connection            │                              │
│   │ - Periodic background scan           │                              │
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
    request_json TEXT NOT NULL,          -- Full request payload with shell-generated UUID
    -- HLC timestamp for CRDT causal ordering (preserved on replay)
    hlc_wall_time INTEGER NOT NULL,      -- HLC wall time (ms since epoch)
    hlc_logical INTEGER NOT NULL,        -- HLC logical counter
    hlc_node_id TEXT NOT NULL            -- HLC node ID (for CRDT)
);

CREATE INDEX IF NOT EXISTS offline_queue_order ON offline_queue(queued_at);
```

**Key Design Decisions:**
1. **Shell generates all UUIDs**: The shell generates `history_id` (UUIDv7) before sending to daemon. This prevents UUID collision between offline and online paths.
2. **HLC stored in offline queue**: Preserves causal ordering when replaying. Without this, replayed commands would get "wrong" timestamps and break CRDT merge.
3. **FinishCommand also queued**: Exit status is just as important as command start - queue both.

**Shell implementation (ZSH example):**

```zsh
# histdb.zsh - with offline queue support, shell-generated UUIDs, and HLC
typeset -g HISTDB_OFFLINE_DB="${XDG_DATA_HOME:-$HOME/.local/share}/histdb/offline.db"
typeset -g HISTDB_LAST_HISTORY_ID=""  # Shell-generated UUID for correlation

_histdb_ensure_offline_db() {
    [[ -f $HISTDB_OFFLINE_DB ]] && return
    mkdir -p "${HISTDB_OFFLINE_DB:h}"
    sqlite3 "$HISTDB_OFFLINE_DB" "
        CREATE TABLE IF NOT EXISTS offline_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            queued_at INTEGER NOT NULL,
            request_type TEXT NOT NULL,
            request_json TEXT NOT NULL,
            hlc_wall_time INTEGER NOT NULL,
            hlc_logical INTEGER NOT NULL,
            hlc_node_id TEXT NOT NULL
        );
    "
}

# Generate UUIDv7 using CLI helper (shell generates, not daemon)
_histdb_generate_uuid() {
    histdb-cli session-id  # Reuse session-id command for UUIDv7 generation
}

# Get current HLC from daemon (or generate locally if offline)
_histdb_get_hlc() {
    local hlc_json
    hlc_json=$(histdb-cli hlc-now 2>/dev/null)
    if [[ -n "$hlc_json" ]]; then
        print -- "$hlc_json"
    else
        # Offline: generate local HLC (wall time only, logical=0)
        local wall_ms=$(($(date +%s) * 1000))
        print -- '{"wall_time":'$wall_ms',"logical":0,"node_id":"'$HISTDB_SESSION'"}'
    fi
}

_histdb_queue_offline() {
    local request_type="$1" json="$2" hlc_json="$3"
    _histdb_ensure_offline_db
    # Use CLI helper for safe parameter binding (prevents SQL injection)
    histdb-cli offline-queue \
        --db "$HISTDB_OFFLINE_DB" \
        --type "$request_type" \
        --json "$json" \
        --hlc "$hlc_json"
}

_histdb_send_or_queue() {
    local json="$1" request_type="$2"

    # Get HLC for this operation
    local hlc_json=$(_histdb_get_hlc)

    # Try daemon first
    local resp=$(_histdb_send "$json" 2>/dev/null)
    if [[ -n "$resp" ]]; then
        print -- "$resp"
        return 0
    fi

    # Queue for later - daemon will replay with preserved HLC
    _histdb_queue_offline "$request_type" "$json" "$hlc_json"
    return 1
}

# Updated hook: shell generates history_id UUID
_histdb_addhistory() {
    _histdb_init
    local cmd="${1[1,-2]}"
    [[ -z "$cmd" ]] && return 0

    # Shell generates UUID for this history entry (prevents offline/online collision)
    HISTDB_LAST_HISTORY_ID=$(_histdb_generate_uuid)

    local json='{"type":"StartCommand",'
    json+='"history_id":"'$HISTDB_LAST_HISTORY_ID'",'  # Shell-generated UUID
    json+='"argv":"'${cmd//\"/\\\"}'",'
    json+='"dir":"'${PWD//\"/\\\"}'","session_id":"'$HISTDB_SESSION'"}'

    _histdb_send_or_queue "$json" "start"
    return 0
}

# FinishCommand also uses offline queue (previously fire-and-forget)
_histdb_precmd() {
    local status=$?
    [[ -z "$HISTDB_LAST_HISTORY_ID" ]] && return

    local json='{"type":"FinishCommand",'
    json+='"history_id":"'$HISTDB_LAST_HISTORY_ID'",'
    json+='"exit_status":'$status'}'

    # Queue FinishCommand too - don't lose exit status on daemon restart
    _histdb_send_or_queue "$json" "finish"
    HISTDB_LAST_HISTORY_ID=""
}

add-zsh-hook zshaddhistory _histdb_addhistory
add-zsh-hook precmd _histdb_precmd
```

**CLI helper updates:**

```rust
// cli/offline.rs - updated for HLC support

#[derive(Args)]
pub struct OfflineQueueArgs {
    #[arg(long)]
    db: PathBuf,

    #[arg(long, name = "TYPE")]
    r#type: String,

    #[arg(long)]
    json: String,

    /// HLC timestamp as JSON: {"wall_time":..., "logical":..., "node_id":"..."}
    #[arg(long)]
    hlc: String,
}

pub fn handle_offline_queue(args: OfflineQueueArgs) -> Result<()> {
    let conn = Connection::open(&args.db)?;

    // Parse HLC from JSON
    let hlc: HlcJson = serde_json::from_str(&args.hlc)?;

    conn.execute(
        "INSERT INTO offline_queue (queued_at, request_type, request_json, hlc_wall_time, hlc_logical, hlc_node_id)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            Utc::now().timestamp(),
            &args.r#type,
            &args.json,
            hlc.wall_time,
            hlc.logical,
            &hlc.node_id,
        ],
    )?;

    Ok(())
}

#[derive(Deserialize)]
struct HlcJson {
    wall_time: u64,
    logical: u32,
    node_id: String,
}

// New CLI command: get current HLC
pub fn handle_hlc_now() {
    // Connect to daemon and get HLC, or generate local one
    let hlc = match get_daemon_hlc() {
        Ok(hlc) => hlc,
        Err(_) => HLC::local_now(),
    };
    println!("{}", serde_json::to_string(&hlc).unwrap());
}
```

**Daemon-driven replay:**

The daemon is responsible for processing offline queues, not the shell. This ensures:
- Centralized control over replay timing and rate limiting
- No shell startup latency from replay
- Consistent handling across all shell types

```rust
// daemon/offline.rs

/// Offline queue processor - runs as background task in daemon
pub struct OfflineQueueProcessor {
    db_path: PathBuf,
    ingest_queue: Arc<CommandQueue>,
    scan_interval: Duration,
}

impl OfflineQueueProcessor {
    pub fn new(data_dir: &Path, ingest_queue: Arc<CommandQueue>) -> Self {
        Self {
            db_path: data_dir.join("offline.db"),
            ingest_queue,
            scan_interval: Duration::from_secs(30),
        }
    }

    /// Start background processing task
    pub fn spawn(self) -> JoinHandle<()> {
        tokio::spawn(async move {
            self.run().await;
        })
    }

    async fn run(&self) {
        // Process on startup
        self.process_queue().await;

        // Then periodically
        let mut interval = tokio::time::interval(self.scan_interval);
        loop {
            interval.tick().await;
            self.process_queue().await;
        }
    }

    async fn process_queue(&self) {
        if !self.db_path.exists() {
            return;
        }

        let conn = match Connection::open(&self.db_path) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("Failed to open offline queue: {}", e);
                return;
            }
        };

        // Get queued commands in order
        let entries: Vec<(i64, String, String, i64)> = {
            let mut stmt = match conn.prepare(
                "SELECT id, request_type, request_json, original_timestamp
                 FROM offline_queue ORDER BY queued_at LIMIT 100"
            ) {
                Ok(s) => s,
                Err(_) => return,
            };

            match stmt.query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            }) {
                Ok(rows) => rows.filter_map(|r| r.ok()).collect(),
                Err(_) => return,
            }
        };

        for (id, request_type, json, timestamp) in entries {
            // Parse and enqueue to ingest pipeline
            match self.replay_entry(&request_type, &json, timestamp).await {
                Ok(()) => {
                    // Successfully queued - delete from offline db
                    let _ = conn.execute(
                        "DELETE FROM offline_queue WHERE id = ?1",
                        params![id]
                    );
                    tracing::debug!("Replayed offline entry {}", id);
                }
                Err(e) => {
                    tracing::warn!("Failed to replay entry {}: {}", id, e);
                    // Don't delete - will retry next scan
                    break;  // Stop on first failure to maintain order
                }
            }
        }
    }

    async fn replay_entry(
        &self,
        request_type: &str,
        json: &str,
        timestamp: i64
    ) -> Result<(), Error> {
        // Parse JSON and create PendingCommand
        let cmd = parse_offline_request(request_type, json, timestamp)?;

        // Enqueue to normal ingest pipeline
        self.ingest_queue.push(cmd)
            .map_err(|_| Error::QueueFull)?;

        Ok(())
    }
}

/// Called when a new shell connects - triggers immediate queue scan
pub fn on_shell_connect(processor: &OfflineQueueProcessor) {
    // Notify processor to scan immediately
    processor.trigger_scan();
}
```

**Replay behavior:**
- Commands replayed in original order (by `queued_at`)
- Original timestamps preserved (for correct history ordering)
- Daemon scans on startup, shell connect, and every 30 seconds
- Rate limited to 100 entries per scan cycle

**Deduplication via UUID:**

Replayed entries use `INSERT OR IGNORE` with the history UUID as the uniqueness key:
- Shell generates UUID before queueing (see shell integration)
- Same UUID may be queued multiple times (retries, reconnects)
- `INSERT OR IGNORE` silently skips duplicates (no error, no overwrite)
- This makes replay **idempotent** - safe to replay same entry multiple times

```sql
-- Deduplication constraint (from migration)
ALTER TABLE history ADD COLUMN uuid TEXT UNIQUE;

-- Idempotent insert (used by replay and sync)
INSERT OR IGNORE INTO history (uuid, session, command_id, place_id, start_time)
SELECT ?, ?, c.id, p.id, ?
FROM commands c, places p
WHERE c.argv = ? AND p.host = ? AND p.dir = ?;
-- Returns 0 rows changed if UUID already exists (duplicate silently ignored)
```

**Why this matters:**
- Network retries won't create duplicate history entries
- Offline queue replay can be safely interrupted and resumed
- Sync from peers won't duplicate already-synced entries
- Crash recovery is safe (just re-process the queue)
- Partial replay safe (stops on first failure, resumes next scan)

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

/// Coordinates graceful shutdown across all daemon components
pub struct ShutdownCoordinator {
    /// Broadcast channel to notify all tasks
    notify_tx: broadcast::Sender<ShutdownPhase>,
    /// Tracks which components have completed shutdown
    completion: Arc<ShutdownCompletion>,
    /// Configuration
    config: ShutdownConfig,
}

#[derive(Clone, Copy, Debug)]
pub enum ShutdownPhase {
    /// Stop accepting new connections
    StopAccepting,
    /// Drain pending work from in-memory queue
    DrainQueues,
    /// Process any pending offline queue entries
    ProcessOfflineQueue,
    /// Flush to disk
    FlushStorage,
    /// Sync with peers (best effort)
    SyncPeers,
    /// Final cleanup
    Terminate,
}

#[derive(Debug)]
pub struct ShutdownConfig {
    /// Total shutdown timeout
    pub total_timeout: Duration,
    /// Time to drain queues
    pub drain_timeout: Duration,
    /// Time to process offline queue
    pub offline_queue_timeout: Duration,
    /// Time to flush storage
    pub flush_timeout: Duration,
    /// Time for peer sync (best effort)
    pub sync_timeout: Duration,
}

impl Default for ShutdownConfig {
    fn default() -> Self {
        Self {
            total_timeout: Duration::from_secs(10),
            drain_timeout: Duration::from_millis(500),
            offline_queue_timeout: Duration::from_secs(2),  // Process pending offline entries
            flush_timeout: Duration::from_secs(2),
            sync_timeout: Duration::from_secs(1),
        }
    }
}

/// Tracks completion of each component
struct ShutdownCompletion {
    socket_listener: AtomicBool,
    writer_thread: AtomicBool,
    offline_processor: AtomicBool,
    peer_sync: AtomicBool,
    index_writer: AtomicBool,
}

impl ShutdownCoordinator {
    pub fn new(config: ShutdownConfig) -> (Self, broadcast::Receiver<ShutdownPhase>) {
        let (notify_tx, notify_rx) = broadcast::channel(16);
        let coordinator = Self {
            notify_tx,
            completion: Arc::new(ShutdownCompletion {
                socket_listener: AtomicBool::new(false),
                writer_thread: AtomicBool::new(false),
                offline_processor: AtomicBool::new(false),
                peer_sync: AtomicBool::new(false),
                index_writer: AtomicBool::new(false),
            }),
            config,
        };
        (coordinator, notify_rx)
    }

    /// Get a shutdown handle for a component
    pub fn handle(&self) -> ShutdownHandle {
        ShutdownHandle {
            receiver: self.notify_tx.subscribe(),
        }
    }

    /// Execute coordinated shutdown
    pub async fn shutdown(
        &self,
        queue: Arc<CommandQueue>,
        writer_health: Arc<WriterHealth>,
        sync: Option<Arc<PeerSync>>,
    ) -> ShutdownReport {
        let start = Instant::now();
        let mut report = ShutdownReport::default();

        tracing::info!("Starting graceful shutdown...");

        // Phase 1: Stop accepting new connections
        let _ = self.notify_tx.send(ShutdownPhase::StopAccepting);
        tokio::time::sleep(Duration::from_millis(50)).await;
        tracing::debug!("Phase 1: Stopped accepting connections");

        // Phase 2: Drain queues
        let _ = self.notify_tx.send(ShutdownPhase::DrainQueues);
        let drain_start = Instant::now();
        let initial_pending = queue.pending_count.load(Ordering::Relaxed);

        loop {
            let pending = queue.pending_count.load(Ordering::Relaxed);
            if pending == 0 {
                report.commands_drained = initial_pending;
                break;
            }
            if drain_start.elapsed() > self.config.drain_timeout {
                report.commands_lost = pending;
                tracing::warn!("Queue drain timeout, {} commands lost", pending);
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        tracing::debug!("Phase 2: Drained {} commands", report.commands_drained);

        // Phase 3: Process offline queue
        // Process any queued offline entries before shutdown
        let _ = self.notify_tx.send(ShutdownPhase::ProcessOfflineQueue);
        let offline_result = tokio::time::timeout(
            self.config.offline_queue_timeout,
            Self::process_offline_queue(&offline_processor)
        ).await;

        match offline_result {
            Ok(Ok(processed)) => {
                report.offline_entries_processed = processed;
                tracing::debug!("Phase 3: Processed {} offline entries", processed);
            }
            Ok(Err(e)) => {
                tracing::warn!("Offline queue processing failed: {}", e);
            }
            Err(_) => {
                tracing::warn!("Offline queue processing timeout");
            }
        }

        // Phase 4: Flush storage
        let _ = self.notify_tx.send(ShutdownPhase::FlushStorage);
        let flush_result = tokio::time::timeout(
            self.config.flush_timeout,
            Self::wait_for_writer_idle(&writer_health)
        ).await;

        report.storage_flushed = flush_result.is_ok();
        if !report.storage_flushed {
            tracing::warn!("Storage flush timeout");
        }
        tracing::debug!("Phase 4: Storage flush complete");

        // Phase 5: Sync with peers (best effort)
        if let Some(ref sync) = sync {
            let _ = self.notify_tx.send(ShutdownPhase::SyncPeers);
            let sync_result = tokio::time::timeout(
                self.config.sync_timeout,
                sync.push_pending()
            ).await;
            report.peers_synced = sync_result.is_ok();
            tracing::debug!("Phase 5: Peer sync {:?}", if report.peers_synced { "complete" } else { "timeout" });
        }

        // Phase 6: Final termination
        let _ = self.notify_tx.send(ShutdownPhase::Terminate);

        report.duration = start.elapsed();
        tracing::info!("Shutdown complete in {:?}", report.duration);
        report
    }

    /// Process pending offline queue entries
    async fn process_offline_queue(
        processor: &OfflineQueueProcessor,
    ) -> Result<usize, Error> {
        let mut processed = 0;

        // Get all pending entries from offline queue
        let entries = processor.get_pending_entries().await?;

        for entry in entries {
            match processor.replay_entry(&entry).await {
                Ok(()) => {
                    processor.mark_processed(entry.id).await?;
                    processed += 1;
                }
                Err(e) => {
                    // Log but continue - don't fail entire shutdown for one entry
                    tracing::warn!("Failed to replay offline entry {}: {}", entry.id, e);
                }
            }
        }

        Ok(processed)
    }

    async fn wait_for_writer_idle(health: &WriterHealth) {
        // Wait for writer to become idle (no more work)
        for _ in 0..50 {
            if health.is_healthy() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(40)).await;
        }
    }
}

/// Handle given to each component to receive shutdown notifications
pub struct ShutdownHandle {
    receiver: broadcast::Receiver<ShutdownPhase>,
}

impl ShutdownHandle {
    /// Wait for a specific shutdown phase
    pub async fn wait_for(&mut self, phase: ShutdownPhase) {
        loop {
            match self.receiver.recv().await {
                Ok(p) if std::mem::discriminant(&p) == std::mem::discriminant(&phase) => return,
                Ok(_) => continue,
                Err(_) => return, // Channel closed
            }
        }
    }

    /// Check if shutdown has been initiated
    pub fn is_shutting_down(&mut self) -> bool {
        matches!(self.receiver.try_recv(), Ok(_))
    }
}

/// Report of shutdown execution
#[derive(Default, Debug)]
pub struct ShutdownReport {
    pub duration: Duration,
    pub commands_drained: u64,
    pub commands_lost: u64,
    pub storage_flushed: bool,
    pub peers_synced: bool,
}

impl ShutdownReport {
    pub fn is_clean(&self) -> bool {
        self.commands_lost == 0 && self.storage_flushed
    }
}

/// Install signal handlers for graceful shutdown
pub fn install_signal_handlers(coordinator: Arc<ShutdownCoordinator>) {
    // SIGTERM handler
    tokio::spawn(async move {
        let mut sigterm = tokio::signal::unix::signal(
            tokio::signal::unix::SignalKind::terminate()
        ).expect("Failed to install SIGTERM handler");

        let mut sigint = tokio::signal::unix::signal(
            tokio::signal::unix::SignalKind::interrupt()
        ).expect("Failed to install SIGINT handler");

        tokio::select! {
            _ = sigterm.recv() => {
                tracing::info!("Received SIGTERM, initiating shutdown");
            }
            _ = sigint.recv() => {
                tracing::info!("Received SIGINT, initiating shutdown");
            }
        }

        // Trigger shutdown (actual execution happens in main)
        coordinator.notify_tx.send(ShutdownPhase::StopAccepting).ok();
    });
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

/// Maximum allowed clock skew (1 hour by default)
/// Timestamps more than this far in the future are rejected
const MAX_CLOCK_SKEW_MS: u64 = 60 * 60 * 1000;

/// Error type for HLC operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HlcError {
    /// Remote timestamp is too far in the future
    ClockSkewExceeded { remote_time: u64, local_time: u64, skew_ms: u64 },
    /// Logical counter overflow (extremely unlikely)
    LogicalOverflow,
}

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
    /// Create an initial HLC (at epoch)
    pub fn new() -> Self {
        Self {
            wall_time: 0,
            logical: 0,
            node_id: NodeId::default(),
        }
    }

    /// Get current wall clock time
    fn current_wall_time() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    /// Generate a new timestamp, ensuring it's greater than any seen
    pub fn now(&self, node_id: NodeId) -> Result<HLC, HlcError> {
        let wall = Self::current_wall_time();

        let (new_wall, new_logical) = if wall > self.wall_time {
            (wall, 0)
        } else {
            // Check for logical counter overflow
            let new_logical = self.logical.checked_add(1)
                .ok_or(HlcError::LogicalOverflow)?;
            (self.wall_time, new_logical)
        };

        Ok(HLC {
            wall_time: new_wall,
            logical: new_logical,
            node_id,
        })
    }

    /// Update clock after receiving a remote timestamp
    /// Returns error if remote clock is too far ahead (clock skew exceeded)
    pub fn receive(&mut self, remote: &HLC, local_node: NodeId) -> Result<HLC, HlcError> {
        let wall = Self::current_wall_time();

        // Check for excessive clock skew
        if remote.wall_time > wall + MAX_CLOCK_SKEW_MS {
            return Err(HlcError::ClockSkewExceeded {
                remote_time: remote.wall_time,
                local_time: wall,
                skew_ms: remote.wall_time - wall,
            });
        }

        let (new_wall, new_logical) = if wall > self.wall_time && wall > remote.wall_time {
            (wall, 0)
        } else if self.wall_time > remote.wall_time {
            let logical = self.logical.checked_add(1)
                .ok_or(HlcError::LogicalOverflow)?;
            (self.wall_time, logical)
        } else if remote.wall_time > self.wall_time {
            let logical = remote.logical.checked_add(1)
                .ok_or(HlcError::LogicalOverflow)?;
            (remote.wall_time, logical)
        } else {
            let logical = self.logical.max(remote.logical).checked_add(1)
                .ok_or(HlcError::LogicalOverflow)?;
            (self.wall_time, logical)
        };

        self.wall_time = new_wall;
        self.logical = new_logical;

        Ok(HLC {
            wall_time: new_wall,
            logical: new_logical,
            node_id: local_node,
        })
    }

    /// Check if this timestamp is within acceptable skew of current time
    pub fn is_valid(&self) -> bool {
        let wall = Self::current_wall_time();
        self.wall_time <= wall + MAX_CLOCK_SKEW_MS
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
///
/// Merge semantics:
/// 1. Higher HLC timestamp wins
/// 2. If timestamps equal, higher node_id wins (deterministic tie-breaker)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LWWRegister<T> {
    pub value: T,
    pub timestamp: HLC,
}

impl<T: Clone> LWWRegister<T> {
    pub fn new(value: T, timestamp: HLC) -> Self {
        Self { value, timestamp }
    }

    /// Merge with another register
    ///
    /// Resolution order:
    /// 1. Higher HLC timestamp wins (wall_time, then logical counter)
    /// 2. If HLC equal, higher node_id wins (lexicographic comparison)
    ///
    /// This ensures deterministic convergence even with clock skew.
    pub fn merge(&mut self, other: &Self) {
        if self.should_take_other(&other.timestamp) {
            self.value = other.value.clone();
            self.timestamp = other.timestamp;
        }
    }

    /// Determine if other's timestamp should win
    fn should_take_other(&self, other_ts: &HLC) -> bool {
        // Primary: compare wall time
        match other_ts.wall_time.cmp(&self.timestamp.wall_time) {
            std::cmp::Ordering::Greater => return true,
            std::cmp::Ordering::Less => return false,
            std::cmp::Ordering::Equal => {}
        }

        // Secondary: compare logical counter
        match other_ts.logical.cmp(&self.timestamp.logical) {
            std::cmp::Ordering::Greater => return true,
            std::cmp::Ordering::Less => return false,
            std::cmp::Ordering::Equal => {}
        }

        // Tie-breaker: lexicographic node_id comparison
        // Higher node_id wins - ensures deterministic convergence
        other_ts.node_id > self.timestamp.node_id
    }
}

#[cfg(test)]
mod lww_tests {
    use super::*;

    #[test]
    fn test_lww_higher_timestamp_wins() {
        let mut reg1 = LWWRegister::new(
            42,
            HLC { wall_time: 100, logical: 0, node_id: "node-a".into() },
        );
        let reg2 = LWWRegister::new(
            99,
            HLC { wall_time: 200, logical: 0, node_id: "node-b".into() },
        );

        reg1.merge(&reg2);
        assert_eq!(reg1.value, 99);
    }

    #[test]
    fn test_lww_equal_time_node_id_tiebreaker() {
        // Same wall_time and logical, different node_id
        let mut reg1 = LWWRegister::new(
            42,
            HLC { wall_time: 100, logical: 5, node_id: "node-a".into() },
        );
        let reg2 = LWWRegister::new(
            99,
            HLC { wall_time: 100, logical: 5, node_id: "node-z".into() },
        );

        reg1.merge(&reg2);
        // node-z > node-a lexicographically, so reg2 wins
        assert_eq!(reg1.value, 99);
    }

    #[test]
    fn test_lww_convergence_is_deterministic() {
        // Both nodes start with different values, same timestamp
        let ts_a = HLC { wall_time: 100, logical: 0, node_id: "node-a".into() };
        let ts_b = HLC { wall_time: 100, logical: 0, node_id: "node-b".into() };

        let mut node_a = LWWRegister::new("value-a", ts_a.clone());
        let mut node_b = LWWRegister::new("value-b", ts_b.clone());

        // Cross-merge in any order
        let node_b_copy = node_b.clone();
        node_a.merge(&node_b_copy);

        let node_a_copy = LWWRegister::new("value-a", ts_a);
        node_b.merge(&node_a_copy);

        // Both nodes converge to same value (node-b wins, higher node_id)
        assert_eq!(node_a.value, node_b.value);
        assert_eq!(node_a.value, "value-b");
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

/// Configuration for peer connection retry behavior
#[derive(Clone, Debug)]
pub struct RetryConfig {
    /// Initial delay before first retry
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Multiplier for exponential backoff
    pub backoff_multiplier: f64,
    /// Maximum number of retry attempts (None = unlimited)
    pub max_attempts: Option<u32>,
    /// Jitter factor (0.0 - 1.0) to randomize delays
    pub jitter: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            max_attempts: Some(10),
            jitter: 0.1,
        }
    }
}

/// State of a peer connection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting { attempt: u32 },
    Failed { last_error: &'static str },
}

/// Peer connection with automatic retry and health monitoring
pub struct PeerConnection {
    peer_id: NodeId,
    addr: String,
    transport: Option<NoiseStream>,
    state: Arc<RwLock<ConnectionState>>,
    retry_config: RetryConfig,
    last_activity: Arc<AtomicU64>,
    metrics: ConnectionMetrics,
}

#[derive(Default)]
pub struct ConnectionMetrics {
    pub messages_sent: AtomicU64,
    pub messages_received: AtomicU64,
    pub bytes_sent: AtomicU64,
    pub bytes_received: AtomicU64,
    pub reconnect_count: AtomicU32,
    pub last_error: parking_lot::RwLock<Option<String>>,
}

impl PeerConnection {
    pub fn new(peer_id: NodeId, addr: String, retry_config: RetryConfig) -> Self {
        Self {
            peer_id,
            addr,
            transport: None,
            state: Arc::new(RwLock::new(ConnectionState::Disconnected)),
            retry_config,
            last_activity: Arc::new(AtomicU64::new(0)),
            metrics: ConnectionMetrics::default(),
        }
    }

    /// Connect with automatic retry
    pub async fn connect(&mut self) -> Result<(), Error> {
        let mut attempt = 0;
        let mut delay = self.retry_config.initial_delay;

        loop {
            *self.state.write() = if attempt == 0 {
                ConnectionState::Connecting
            } else {
                ConnectionState::Reconnecting { attempt }
            };

            match self.try_connect().await {
                Ok(()) => {
                    *self.state.write() = ConnectionState::Connected;
                    self.update_activity();
                    tracing::info!("Connected to peer {} after {} attempts", self.peer_id, attempt + 1);
                    return Ok(());
                }
                Err(e) => {
                    attempt += 1;
                    *self.metrics.last_error.write() = Some(e.to_string());

                    if let Some(max) = self.retry_config.max_attempts {
                        if attempt >= max {
                            *self.state.write() = ConnectionState::Failed {
                                last_error: "max retries exceeded",
                            };
                            tracing::error!(
                                "Failed to connect to peer {} after {} attempts: {}",
                                self.peer_id, attempt, e
                            );
                            return Err(Error::ConnectionFailed);
                        }
                    }

                    // Apply jitter to delay
                    let jittered_delay = self.apply_jitter(delay);
                    tracing::warn!(
                        "Connection to peer {} failed (attempt {}), retrying in {:?}: {}",
                        self.peer_id, attempt, jittered_delay, e
                    );

                    tokio::time::sleep(jittered_delay).await;

                    // Exponential backoff
                    delay = Duration::from_secs_f64(
                        (delay.as_secs_f64() * self.retry_config.backoff_multiplier)
                            .min(self.retry_config.max_delay.as_secs_f64())
                    );
                }
            }
        }
    }

    async fn try_connect(&mut self) -> Result<(), Error> {
        // Establish TCP connection
        let stream = TcpStream::connect(&self.addr).await?;

        // Perform Noise handshake
        let transport = NoiseHandshake::initiate(stream, &self.peer_id).await?;

        self.transport = Some(transport);
        Ok(())
    }

    fn apply_jitter(&self, delay: Duration) -> Duration {
        let jitter_range = delay.as_secs_f64() * self.retry_config.jitter;
        let jitter = rand::random::<f64>() * jitter_range * 2.0 - jitter_range;
        Duration::from_secs_f64((delay.as_secs_f64() + jitter).max(0.0))
    }

    fn update_activity(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.last_activity.store(now, Ordering::Relaxed);
    }

    /// Send message with automatic reconnect on failure
    pub async fn send(&mut self, msg: SyncMessage) -> Result<(), Error> {
        // Try to send
        match self.try_send(&msg).await {
            Ok(()) => {
                self.metrics.messages_sent.fetch_add(1, Ordering::Relaxed);
                self.update_activity();
                Ok(())
            }
            Err(e) => {
                tracing::warn!("Send to peer {} failed, attempting reconnect: {}", self.peer_id, e);
                self.metrics.reconnect_count.fetch_add(1, Ordering::Relaxed);

                // Attempt reconnect and retry
                self.connect().await?;
                self.try_send(&msg).await?;
                self.metrics.messages_sent.fetch_add(1, Ordering::Relaxed);
                self.update_activity();
                Ok(())
            }
        }
    }

    async fn try_send(&mut self, msg: &SyncMessage) -> Result<(), Error> {
        let transport = self.transport.as_mut().ok_or(Error::NotConnected)?;
        let bytes = bincode::serialize(msg)?;
        self.metrics.bytes_sent.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        transport.send(&bytes).await?;
        Ok(())
    }

    /// Receive message
    pub async fn receive(&mut self) -> Result<SyncMessage, Error> {
        let transport = self.transport.as_mut().ok_or(Error::NotConnected)?;
        let bytes = transport.receive().await?;
        self.metrics.bytes_received.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        self.metrics.messages_received.fetch_add(1, Ordering::Relaxed);
        self.update_activity();
        let msg = bincode::deserialize(&bytes)?;
        Ok(msg)
    }

    /// Check if connection is healthy
    pub fn is_healthy(&self) -> bool {
        matches!(*self.state.read(), ConnectionState::Connected)
    }

    /// Get connection state
    pub fn state(&self) -> ConnectionState {
        *self.state.read()
    }
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

### Sync Security (Noise Protocol + Ed25519)

All sync traffic is encrypted and authenticated using the **Noise Protocol Framework** with Ed25519 keys (same format as SSH keys).

#### Security Properties

| Property | Guaranteed |
|----------|------------|
| **Encryption** | ✅ ChaCha20-Poly1305 |
| **Mutual Authentication** | ✅ Both peers verified via Ed25519 |
| **Forward Secrecy** | ✅ Ephemeral key exchange |
| **Replay Protection** | ✅ Nonces |
| **Unknown Peer Rejection** | ✅ Must be in known peers list |

#### Implementation

```rust
// sync/crypto.rs
use snow::{Builder, TransportState};
use ed25519_dalek::{SigningKey, VerifyingKey};

/// Noise protocol pattern: KK (both sides know each other's static key)
const NOISE_PATTERN: &str = "Noise_KK_25519_ChaChaPoly_BLAKE2s";

/// Handshake timeout configuration
#[derive(Debug, Clone)]
pub struct HandshakeConfig {
    /// Total timeout for handshake (default: 10s)
    pub timeout: Duration,
    /// Per-message timeout within handshake (default: 5s)
    pub message_timeout: Duration,
}

impl Default for HandshakeConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(10),
            message_timeout: Duration::from_secs(5),
        }
    }
}

pub struct SecureChannel {
    transport: TransportState,
    peer_pubkey: VerifyingKey,
}

impl SecureChannel {
    /// Initiate connection to known peer with timeout
    pub async fn connect(
        our_key: &SigningKey,
        peer_pubkey: &VerifyingKey,
        stream: TcpStream,
        config: &HandshakeConfig,
    ) -> Result<Self, Error> {
        // Wrap entire handshake in timeout
        tokio::time::timeout(config.timeout, async {
            Self::connect_inner(our_key, peer_pubkey, stream, config).await
        })
        .await
        .map_err(|_| Error::HandshakeTimeout)?
    }

    async fn connect_inner(
        our_key: &SigningKey,
        peer_pubkey: &VerifyingKey,
        mut stream: TcpStream,
        config: &HandshakeConfig,
    ) -> Result<Self, Error> {
        let builder = Builder::new(NOISE_PATTERN.parse()?)
            .local_private_key(&our_key.to_bytes())
            .remote_public_key(&peer_pubkey.to_bytes());

        let mut handshake = builder.build_initiator()?;

        // Two-round handshake with per-message timeouts
        let mut buf = [0u8; 65535];

        // Round 1: Send -> e, es
        let len = handshake.write_message(&[], &mut buf)?;
        tokio::time::timeout(config.message_timeout, stream.write_all(&buf[..len]))
            .await
            .map_err(|_| Error::HandshakeTimeout)??;

        // Round 2: Recv <- e, ee
        let len = tokio::time::timeout(config.message_timeout, stream.read(&mut buf))
            .await
            .map_err(|_| Error::HandshakeTimeout)??;
        handshake.read_message(&buf[..len], &mut [])?;

        let transport = handshake.into_transport_mode()?;
        Ok(Self { transport, peer_pubkey: *peer_pubkey })
    }

    /// Accept connection with timeout, verify peer is in known_peers list
    pub async fn accept(
        our_key: &SigningKey,
        known_peers: &[VerifyingKey],
        stream: TcpStream,
        config: &HandshakeConfig,
    ) -> Result<Self, Error> {
        // Wrap entire handshake in timeout
        tokio::time::timeout(config.timeout, async {
            Self::accept_inner(our_key, known_peers, stream, config).await
        })
        .await
        .map_err(|_| Error::HandshakeTimeout)?
    }

    async fn accept_inner(
        our_key: &SigningKey,
        known_peers: &[VerifyingKey],
        mut stream: TcpStream,
        config: &HandshakeConfig,
    ) -> Result<Self, Error> {
        let builder = Builder::new(NOISE_PATTERN.parse()?)
            .local_private_key(&our_key.to_bytes());

        let mut handshake = builder.build_responder()?;

        let mut buf = [0u8; 65535];

        // Round 1: Recv <- e, es (with timeout)
        let len = tokio::time::timeout(config.message_timeout, stream.read(&mut buf))
            .await
            .map_err(|_| Error::HandshakeTimeout)??;
        handshake.read_message(&buf[..len], &mut [])?;

        // Verify peer is in known_peers list
        let peer_pubkey = handshake.get_remote_static().ok_or(Error::NoPeerKey)?;
        let peer_key = VerifyingKey::from_bytes(peer_pubkey.try_into()?)?;

        if !known_peers.contains(&peer_key) {
            return Err(Error::UnknownPeer);
        }

        // Round 2: Send -> e, ee (with timeout)
        let len = handshake.write_message(&[], &mut buf)?;
        tokio::time::timeout(config.message_timeout, stream.write_all(&buf[..len]))
            .await
            .map_err(|_| Error::HandshakeTimeout)??;

        let transport = handshake.into_transport_mode()?;
        Ok(Self { transport, peer_pubkey: peer_key })
    }
}

/// Handshake errors
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Handshake timed out")]
    HandshakeTimeout,

    #[error("No peer public key in handshake")]
    NoPeerKey,

    #[error("Unknown peer - not in known_peers list")]
    UnknownPeer,

    #[error("Noise protocol error: {0}")]
    Noise(#[from] snow::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
```

#### Key Management CLI

```bash
# Generate new identity
$ histdb keys init
Generated: ~/.config/histdb/identity.key
Public key: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGwZ...

# Or import from existing SSH key
$ histdb keys init --from-ssh ~/.ssh/id_ed25519

# Export for Codespaces/K8s secrets
$ histdb keys export --format=base64
HISTDB_IDENTITY=LS0tLS1CRUdJTi...

# Add trusted peer
$ histdb peers add laptop "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHxK..."

# List peers
$ histdb peers list
```

### Sync Topology

Two deployment models supported:

#### Model 1: Full P2P Mesh (Static Machines Only)

```
┌─────────────────────────────────────────────────────────────────┐
│      ┌──────────┐         ┌──────────┐         ┌──────────┐    │
│      │  Laptop  │◄───────►│  Desktop │◄───────►│  Server  │    │
│      │ (static) │         │ (static) │         │ (static) │    │
│      └────┬─────┘         └────┬─────┘         └────┬─────┘    │
│           └────────────────────┴────────────────────┘           │
│                    Full mesh (all in peers.toml)                │
└─────────────────────────────────────────────────────────────────┘
```

#### Model 2: Hub-Spoke with Leader Election (Static + Ephemeral)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│   Static Machines (leader election cluster)                                  │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│   │   Desktop   │◄──►│   Laptop    │◄──►│   Server    │  ← P2P mesh         │
│   │ priority=75 │    │ priority=50 │    │ priority=100│                     │
│   └─────────────┘    └─────────────┘    └──────┬──────┘                     │
│                                                │                             │
│                                    Current Leader                            │
│                                    (highest priority)                        │
│                                                │                             │
│              ┌─────────────────────────────────┼─────────────────┐          │
│              ▼                                 ▼                 ▼          │
│       ┌────────────┐                   ┌────────────┐    ┌────────────┐     │
│       │ Codespace  │                   │  K8s Pod   │    │  Temp VM   │     │
│       │ (ephemeral)│                   │ (ephemeral)│    │ (ephemeral)│     │
│       └────────────┘                   └────────────┘    └────────────┘     │
│                                                                              │
│   Failover: Server dies → Desktop (priority=75) becomes leader              │
│             Ephemeral machines auto-reconnect to new leader                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Leader Election

Simple Bully Algorithm for small clusters:

```rust
// sync/election.rs
pub struct Election {
    our_id: NodeId,
    our_priority: u32,
    peers: HashMap<NodeId, PeerState>,
    current_leader: Option<NodeId>,
}

impl Election {
    const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
    const LEADER_TIMEOUT: Duration = Duration::from_secs(15);

    /// We're leader if highest priority among reachable static peers
    pub fn should_be_leader(&self) -> bool {
        let dominated_by = self.peers.values()
            .filter(|p| p.is_static && p.last_seen.elapsed() < Self::LEADER_TIMEOUT)
            .any(|p| p.priority > self.our_priority);
        !dominated_by
    }

    pub fn on_heartbeat(&mut self, from: NodeId, msg: Heartbeat) {
        self.peers.insert(from, PeerState {
            priority: msg.priority,
            last_seen: Instant::now(),
            is_static: msg.is_static,
        });

        if self.should_be_leader() {
            self.current_leader = Some(self.our_id);
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Heartbeat {
    pub node_id: NodeId,
    pub priority: u32,
    pub is_leader: bool,
    pub is_static: bool,
    pub leader_addr: Option<SocketAddr>,
}
```

#### Failover Timeline

```
0s      Server (priority=100) is leader, Codespace connected
10s     Server crashes 💥
15s     Desktop/Laptop miss heartbeats, Desktop (priority=75) becomes leader
16s     Codespace reconnects to Desktop via mDNS discovery
17s     Sync continues (CRDTs merge any divergence)
```

#### Split-Brain Behavior (Availability > Consistency)

histdb explicitly chooses **availability over consistency** during network partitions.
This is the correct tradeoff for command history:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Split-Brain Scenario                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Before Partition:            During Partition:           After Reconnect:  │
│                                                                              │
│  ┌─────────┐                  ┌─────────┐                ┌─────────┐        │
│  │ Desktop │                  │ Desktop │                │ Desktop │        │
│  │ leader  │◄──────────────── │ leader  │                │ merged  │        │
│  └────┬────┘     Network      │ writes  │                └────┬────┘        │
│       │          Split        │ locally │                     │             │
│       │            ↓          └─────────┘    CRDT Merge  ┌────┴────┐        │
│  ┌────┴────┐                  ┌─────────┐        ↓       │ Desktop │        │
│  │ Laptop  │                  │ Laptop  │    ═══════►    │   +     │        │
│  │ follower│                  │ leader  │                │ Laptop  │        │
│  └─────────┘                  │ writes  │                │ entries │        │
│                               │ locally │                └─────────┘        │
│                               └─────────┘                                   │
│                                                                              │
│  Single leader               Two independent             All entries        │
│                              leaders (split-brain)       merged             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**What happens during split-brain:**

| Phase | Desktop Side | Laptop Side |
|-------|--------------|-------------|
| Normal operation | Leader, syncs to Laptop | Follower, receives from Desktop |
| Network partition | Continues as leader | Detects timeout, becomes leader |
| Both sides | Writes accepted locally | Writes accepted locally |
| Reconnect | Discovers Laptop has entries | Discovers Desktop has entries |
| Merge | CRDT merge (union of entries) | Same merge result |

**Why this is correct for histdb:**

1. **No data loss**: Both sides continue recording commands - nothing is dropped
2. **Automatic resolution**: CRDT merge requires no manual intervention
3. **User expectation**: Users expect their commands to be recorded regardless of network
4. **Eventual consistency**: After reconnect, all machines have the same complete history

**What CRDT merge does:**

- **History entries (G-Set)**: Union of all entries from both sides
- **Exit status (LWW-Register)**: Higher HLC timestamp wins, node_id as tie-breaker
- **No conflicts**: Every command has a globally unique UUID, no overwrites

**Documented behavior:**

> During network partitions, histdb may elect multiple leaders (split-brain).
> Each partition continues recording commands independently.
> On reconnect, all entries are merged via CRDT semantics.
> This is by design: availability and durability of command history
> is prioritized over strong consistency.

**Edge case - identical timestamps:**

If two machines record a command at exactly the same HLC (same wall_time,
same logical counter), the LWW-Register uses node_id as a deterministic
tie-breaker (see "CRDT LWW Tie-Breaker" section).

### Ephemeral Machine Support

#### Portable Identity (Personal Use)

Store identity key in secrets — same key across all Codespaces/K8s pods:

```yaml
# GitHub Codespaces: Settings → Secrets
HISTDB_IDENTITY: <base64 private key>

# K8s Secret
apiVersion: v1
kind: Secret
metadata:
  name: histdb-identity
data:
  identity.key: <base64 private key>
```

One peer entry covers all ephemeral machines:

```toml
# On static peers
[[sync.peers]]
name = "my-ephemeral"
public_key = "ssh-ed25519 AAAAC3..."
addresses = []  # Ephemeral connects OUT
role = "ephemeral"
```

#### Token Enrollment (Team Use)

```bash
# On static machine
$ histdb tokens create --name="new-dev" --expires=1h
Token: hdb_enroll_7f3a9b2c...

# On new machine
$ histdb sync enroll hdb_enroll_7f3a9b2c...
✓ Enrolled with home.example.com
```

### Configuration (Complete)

```toml
# ═══════════════════════════════════════════════════════════════════
# STATIC MACHINE (~/.config/histdb/config.toml)
# ═══════════════════════════════════════════════════════════════════

[sync]
enabled = true
mode = "static"
listen_addr = "0.0.0.0:4242"

[sync.identity]
private_key_path = "~/.config/histdb/identity.key"

[sync.cluster]
priority = 100  # Higher = more likely leader (100 for servers, 50 for laptops)

[sync.replication]
eager_push = true
sync_interval = 30
batch_size = 1000
heartbeat_interval = 5
leader_timeout = 15

[sync.discovery]
mdns_enabled = true
mdns_service = "_histdb._tcp.local"

[sync.ephemeral]
accept_ephemeral = true

# ─────────────────────────────────────────────────────────────────
# Peer definitions (or in ~/.config/histdb/peers.toml)
# ─────────────────────────────────────────────────────────────────

[[sync.peers]]
name = "laptop"
public_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHxK..."
addresses = ["192.168.1.100:4242", "laptop.local:4242"]
role = "static"
priority = 50

[[sync.peers]]
name = "desktop"
public_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPqR..."
addresses = ["192.168.1.101:4242"]
role = "static"
priority = 75

[[sync.peers]]
name = "my-ephemeral"
public_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGwZ..."
addresses = []
role = "ephemeral"
```

```toml
# ═══════════════════════════════════════════════════════════════════
# EPHEMERAL MACHINE (Codespaces, K8s)
# ═══════════════════════════════════════════════════════════════════

[sync]
enabled = true
mode = "ephemeral"

[sync.identity]
from_env = "HISTDB_IDENTITY"

[sync.connect]
bootstrap_peers = ["home.example.com:4242"]
mdns_service = "_histdb._tcp.local"
reconnect_delay = "1s"
max_reconnect_delay = "30s"
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

### Session ID Generation

Shells generate their own session IDs using **UUIDv7** — no daemon round-trip required.

```
┌─────────────────────────────────────────────────────────────────┐
│                    UUIDv7 Structure                              │
├─────────────────────────────────────────────────────────────────┤
│  48-bit timestamp (ms)  │  4-bit ver  │  12-bit random  │ ...  │
│  ──────────────────────   ──────────   ────────────────         │
│  Time-ordered for        Fixed "7"    Uniqueness even           │
│  efficient queries                    with shared identity      │
└─────────────────────────────────────────────────────────────────┘
```

**Why UUIDv7:**
- **Works with offline queue**: Shell generates ID without daemon
- **Shared identity safe**: Random component ensures uniqueness when multiple Codespaces share same identity key
- **Time-ordered**: Efficient for "most recent session" queries
- **Zero coordination**: No round-trip, no leader involvement

```rust
// Shell generates on startup (or shell plugin)
let session_id = Uuid::now_v7();

// In ZSH plugin
typeset -g HISTDB_SESSION=$(histdb-cli session-id)  # Generate UUIDv7 session ID
```

### Log Compaction

Tiered compaction strategy that doesn't let ephemeral peers block garbage collection.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Log Compaction Strategy                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Operation Log Entry                                                    │
│          │                                                               │
│          ▼                                                               │
│   ┌──────────────────┐                                                  │
│   │ Applied to local │                                                  │
│   │ SQLite state?    │                                                  │
│   └────────┬─────────┘                                                  │
│            │ yes                                                         │
│            ▼                                                             │
│   ┌──────────────────┐     no      ┌────────────────────┐               │
│   │ All STATIC peers │ ──────────► │ Keep in log        │               │
│   │ ACKed?           │             │ (still replicating)│               │
│   └────────┬─────────┘             └────────────────────┘               │
│            │ yes                                                         │
│            ▼                                                             │
│   ┌──────────────────┐     OR      ┌────────────────────┐               │
│   │ COMPACT          │ ◄────────── │ Age > max_log_age  │               │
│   │ (remove from log)│             │ (default: 90 days) │               │
│   └──────────────────┘             └────────────────────┘               │
│                                                                          │
│   Ephemeral peers: Ignored for compaction (sync on connect)             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Compaction rules:**

| Component | Rule | Default |
|-----------|------|---------|
| **Operation log** | Remove when all static peers ACK, OR older than max age | 90 days |
| **Tombstones** | Keep for TTL, then hard-delete | 30 days |
| **Entries** | Never compact (actual data) | — |
| **Ephemeral peer state** | Expire tracking after inactivity | 24 hours |

**Implementation:**

```rust
// sync/compaction.rs
impl CompactionStrategy {
    pub fn can_compact(&self, entry: &LogEntry, peer_acks: &HashMap<NodeId, u64>) -> bool {
        // Rule 1: All static peers have ACKed
        let static_peers_acked = self.static_peers.iter()
            .all(|peer| peer_acks.get(peer).copied().unwrap_or(0) >= entry.seq);

        if static_peers_acked {
            return true;
        }

        // Rule 2: Entry older than max_log_age (fallback for long-offline peers)
        let age = Utc::now() - entry.timestamp.to_datetime();
        if age > self.config.max_log_age {
            return true;
        }

        false // Ephemeral peers not considered
    }
}
```

**Long-offline peer reconnection:**

```rust
// If peer's vector clock is too old, trigger full state sync
pub async fn sync_with_peer(&self, peer: &PeerConnection) -> Result<SyncResult, Error> {
    let their_vc = peer.get_vector_clock().await?;
    let our_oldest_log = self.log.oldest_entry_seq()?;

    let needs_full_sync = their_vc.values().any(|&seq| seq < our_oldest_log);

    if needs_full_sync {
        tracing::info!("Peer too far behind, initiating full state sync");
        self.full_state_sync(peer).await
    } else {
        self.delta_sync(peer, &their_vc).await
    }
}
```

### Protocol Versioning

Semantic versioning with one-version backward compatibility window for rolling upgrades.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Protocol Version Strategy                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Version Format: MAJOR.MINOR                                           │
│                                                                          │
│   MAJOR (breaking):              MINOR (additive):                      │
│   - Changed message structure    - New optional fields                  │
│   - Removed fields               - New message types                    │
│   - Changed semantics            - New enum variants                    │
│                                                                          │
│   Compatibility: abs(our.major - their.major) <= 1                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Handshake with version negotiation:**

```rust
// sync/protocol.rs

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
}

impl ProtocolVersion {
    pub const CURRENT: Self = Self { major: 1, minor: 0 };

    pub fn compatible_with(&self, other: &Self) -> Compatibility {
        match (self.major as i32 - other.major as i32).abs() {
            0 => Compatibility::Full,
            1 => Compatibility::Degraded,
            _ => Compatibility::Incompatible,
        }
    }
}

pub enum Compatibility {
    Full,         // Same major, use all features
    Degraded,     // ±1 major, use common subset
    Incompatible, // Reject connection
}

/// First message after Noise handshake
#[derive(Serialize, Deserialize)]
pub struct Hello {
    pub protocol_version: ProtocolVersion,
    pub node_id: NodeId,
    pub capabilities: Capabilities,
}

#[derive(Serialize, Deserialize, Default)]
pub struct Capabilities {
    #[serde(default)]
    pub supports_compression: bool,
    #[serde(default)]
    pub supports_batched_sync: bool,
}
```

**Connection handling:**

```rust
impl PeerConnection {
    pub async fn negotiate_version(&mut self, their_hello: &Hello) -> Result<(), Error> {
        match ProtocolVersion::CURRENT.compatible_with(&their_hello.protocol_version) {
            Compatibility::Full => {
                self.negotiated_version = ProtocolVersion::CURRENT;
            }
            Compatibility::Degraded => {
                self.negotiated_version = their_hello.protocol_version
                    .min(ProtocolVersion::CURRENT);
                tracing::warn!(
                    "Peer using older protocol {:?}, degraded mode",
                    self.negotiated_version
                );
            }
            Compatibility::Incompatible => {
                return Err(Error::IncompatibleProtocol {
                    ours: ProtocolVersion::CURRENT,
                    theirs: their_hello.protocol_version,
                });
            }
        }
        Ok(())
    }
}
```

**Forward-compatible message design:**

```rust
// Always use Option + #[serde(default)] for new fields
#[derive(Serialize, Deserialize)]
pub struct SyncMessage {
    pub entries: Vec<LogEntry>,
    pub vector_clock: HashMap<NodeId, u64>,

    #[serde(default)]  // v1.1: older peers ignore
    pub compression: Option<CompressionType>,

    #[serde(default)]  // v1.2: older peers ignore
    pub checksum: Option<u64>,
}
```

**Configuration:**

```toml
[sync.protocol]
# Minimum peer version to accept (default: current.major - 1)
min_version = "0.1"
# Warn when connecting to older peers
warn_degraded = true
```

### Data Retention & Pruning

Configurable limits and strategies to manage history growth.

#### Retention Limits

```toml
# ~/.config/histdb/config.toml

[retention]
# ═══════════════════════════════════════════════════════════════════
# Limits (all optional, set to 0 or "unlimited" to disable)
# ═══════════════════════════════════════════════════════════════════

# Age-based: commands older than this are candidates for pruning
max_age = "365d"

# Per-host: prevents one busy machine from dominating history
max_commands_per_host = 100000

# Global: total commands across all hosts
max_commands_total = 1000000

# Size-based: hard cap on database size
max_database_size = "1GB"

# ═══════════════════════════════════════════════════════════════════
# Strategy: how to choose which commands to prune
# ═══════════════════════════════════════════════════════════════════
strategy = "smart"  # "oldest" | "lru" | "lfu" | "smart"

# ═══════════════════════════════════════════════════════════════════
# Protection: commands matching these criteria are never pruned
# ═══════════════════════════════════════════════════════════════════
preserve_starred = true       # User-starred commands
preserve_failed = false       # Commands with non-zero exit status
preserve_long_running = "5m"  # Commands that took longer than this
preserve_patterns = [         # Regex patterns to always keep
    "^git commit",
    "^docker run",
    "^kubectl apply",
]

# ═══════════════════════════════════════════════════════════════════
# Pruning behavior
# ═══════════════════════════════════════════════════════════════════
prune_interval = "1h"         # How often to check limits
prune_batch_size = 1000       # Commands to delete per batch
```

#### Retention Strategies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Retention Strategies                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐                                                            │
│  │   oldest    │  Prune by age only (simplest)                              │
│  └─────────────┘  Score = -start_time                                       │
│                   Good for: Simple setups, predictable behavior             │
│                                                                              │
│  ┌─────────────┐                                                            │
│  │     lru     │  Least Recently Used                                       │
│  └─────────────┘  Score = -last_used_time                                   │
│                   Good for: Keeping recently relevant commands              │
│                   Requires: tracking last search/execution time             │
│                                                                              │
│  ┌─────────────┐                                                            │
│  │     lfu     │  Least Frequently Used                                     │
│  └─────────────┘  Score = -use_count                                        │
│                   Good for: Keeping frequently used commands                │
│                   Requires: tracking execution/search count                 │
│                                                                              │
│  ┌─────────────┐                                                            │
│  │    smart    │  Weighted combination (recommended)                        │
│  └─────────────┘  Score = w1*age + w2*recency + w3*frequency + w4*length   │
│                   Good for: Balanced pruning based on actual value          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Smart Strategy Scoring

The `smart` strategy computes a **retention score** for each command:

```rust
// retention/strategy.rs

#[derive(Debug, Clone, Deserialize)]
pub struct SmartWeights {
    pub age: f64,           // How old the command is (default: 0.3)
    pub recency: f64,       // Time since last use (default: 0.3)
    pub frequency: f64,     // How often used (default: 0.25)
    pub complexity: f64,    // Command length/complexity (default: 0.15)
}

impl Default for SmartWeights {
    fn default() -> Self {
        Self { age: 0.3, recency: 0.3, frequency: 0.25, complexity: 0.15 }
    }
}

/// Lower score = more likely to be pruned
pub fn compute_retention_score(entry: &HistoryEntry, weights: &SmartWeights) -> f64 {
    let now = Utc::now();

    // Age factor: older = lower score (0.0 to 1.0)
    let age_days = (now - entry.start_time).num_days() as f64;
    let age_score = 1.0 / (1.0 + age_days / 365.0);  // Half-life of ~1 year

    // Recency factor: recently used = higher score
    let recency_days = entry.last_used
        .map(|t| (now - t).num_days() as f64)
        .unwrap_or(age_days);  // Fall back to age if never reused
    let recency_score = 1.0 / (1.0 + recency_days / 30.0);  // Half-life of ~1 month

    // Frequency factor: more uses = higher score
    let frequency_score = (entry.use_count as f64).ln_1p() / 10.0;  // Log scale

    // Complexity factor: longer/complex commands = higher score (worth keeping)
    let complexity_score = (entry.command.len() as f64).ln_1p() / 10.0;

    // Weighted sum
    weights.age * age_score
        + weights.recency * recency_score
        + weights.frequency * frequency_score
        + weights.complexity * complexity_score
}
```

**Configuration for smart strategy:**

```toml
[retention]
strategy = "smart"

[retention.smart_weights]
age = 0.3           # 30% weight on command age
recency = 0.3       # 30% weight on last use time
frequency = 0.25    # 25% weight on use count
complexity = 0.15   # 15% weight on command length
```

#### Usage Tracking

To support LRU/LFU/Smart strategies, track command usage:

```sql
-- Schema additions
ALTER TABLE history ADD COLUMN use_count INTEGER DEFAULT 1;
ALTER TABLE history ADD COLUMN last_used INTEGER;  -- Unix timestamp

CREATE INDEX history_use_count ON history(use_count);
CREATE INDEX history_last_used ON history(last_used);
```

```rust
// Update on search result selection or re-execution
pub fn record_command_use(&self, uuid: Uuid) -> Result<(), Error> {
    self.conn.execute("
        UPDATE history
        SET use_count = use_count + 1, last_used = ?1
        WHERE uuid = ?2
    ", params![Utc::now().timestamp(), uuid])?;

    // Also create CRDT operation to sync usage stats
    let op = Operation::UpdateUsage {
        entry_id: uuid,
        use_count: LWWRegister::new(/* ... */),
        last_used: LWWRegister::new(/* ... */),
    };
    self.log.append(op)?;

    Ok(())
}
```

#### Pruning Implementation

```rust
// retention/pruner.rs

pub struct Pruner {
    config: RetentionConfig,
    strategy: Box<dyn PruneStrategy>,
}

pub trait PruneStrategy: Send + Sync {
    fn get_prune_candidates(
        &self,
        conn: &Connection,
        limit: usize,
        protected: &ProtectionRules,
    ) -> Result<Vec<Uuid>, Error>;
}

impl Pruner {
    pub async fn run_prune_cycle(&self) -> Result<PruneStats, Error> {
        let mut stats = PruneStats::default();

        // Check if any limits exceeded
        let status = self.check_limits()?;
        if !status.needs_pruning {
            return Ok(stats);
        }

        // Get candidates based on strategy
        let candidates = self.strategy.get_prune_candidates(
            &self.conn,
            status.prune_target,
            &self.config.protection,
        )?;

        // Prune in batches
        for batch in candidates.chunks(self.config.prune_batch_size) {
            for uuid in batch {
                // Create tombstone (syncs to peers)
                let op = Operation::Delete {
                    entry_id: *uuid,
                    timestamp: self.hlc.now(),
                };
                self.log.append(op)?;
                stats.pruned += 1;
            }

            // Yield between batches
            tokio::task::yield_now().await;
        }

        tracing::info!(
            "Pruned {} commands (strategy: {:?})",
            stats.pruned,
            self.config.strategy
        );

        Ok(stats)
    }
}

// Strategy implementations
pub struct OldestStrategy;

impl PruneStrategy for OldestStrategy {
    fn get_prune_candidates(
        &self,
        conn: &Connection,
        limit: usize,
        protected: &ProtectionRules,
    ) -> Result<Vec<Uuid>, Error> {
        conn.prepare("
            SELECT uuid FROM history
            WHERE starred = 0
              AND (exit_status = 0 OR ?1 = 0)
              AND NOT matches_protected_pattern(command)
            ORDER BY start_time ASC
            LIMIT ?2
        ")?.query_map(
            params![protected.preserve_failed as i32, limit],
            |row| row.get(0)
        )?.collect()
    }
}

pub struct SmartStrategy {
    weights: SmartWeights,
}

impl PruneStrategy for SmartStrategy {
    fn get_prune_candidates(
        &self,
        conn: &Connection,
        limit: usize,
        protected: &ProtectionRules,
    ) -> Result<Vec<Uuid>, Error> {
        // Compute scores and get lowest-scored entries
        let mut entries: Vec<(Uuid, f64)> = conn.prepare("
            SELECT uuid, start_time, last_used, use_count, length(command)
            FROM history
            WHERE starred = 0
              AND (exit_status = 0 OR ?1 = 0)
        ")?.query_map(
            params![protected.preserve_failed as i32],
            |row| {
                let uuid: Uuid = row.get(0)?;
                let entry = HistoryEntry {
                    start_time: row.get(1)?,
                    last_used: row.get(2)?,
                    use_count: row.get(3)?,
                    command_len: row.get(4)?,
                };
                let score = compute_retention_score(&entry, &self.weights);
                Ok((uuid, score))
            }
        )?.collect::<Result<_, _>>()?;

        // Sort by score ascending (lowest = prune first)
        entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        Ok(entries.into_iter().take(limit).map(|(uuid, _)| uuid).collect())
    }
}
```

#### Starred Commands

Users can protect important commands from pruning:

```bash
# Star a command (by search)
$ histdb star "docker run"
Starred: docker run -v /var/run/docker.sock...

# Star by ID
$ histdb star --id 550e8400-e29b-41d4-a716-446655440000

# List starred
$ histdb list --starred

# Unstar
$ histdb unstar "docker run"
```

#### Sync Considerations

| Scenario | Behavior |
|----------|----------|
| **Different retention settings** | Deletes sync (most restrictive wins) |
| **Starred on host A, pruned on B** | Star should sync first; if not, conflict resolution needed |
| **Usage stats sync** | LWW-Register for use_count and last_used |
| **Offline host reconnects** | Receives tombstones, prunes locally |

**Conflict resolution for starred:**

```rust
// Starred flag uses LWW with bias toward preservation
pub fn merge_starred(local: &LWWRegister<bool>, remote: &LWWRegister<bool>) -> bool {
    if local.timestamp == remote.timestamp {
        // Tie-breaker: preserve (starred = true) wins
        local.value || remote.value
    } else if local.timestamp > remote.timestamp {
        local.value
    } else {
        remote.value
    }
}
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

# Sync security & networking
snow = "0.9"                  # Noise Protocol Framework
ed25519-dalek = "2"           # Ed25519 signatures (SSH key compatible)
x25519-dalek = "2"            # X25519 key exchange
mdns-sd = "0.10"              # mDNS for peer discovery
base64 = "0.21"               # Key encoding for secrets

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
- ✅ **Sync security**: Noise Protocol with Ed25519 keys
   - ChaCha20-Poly1305 encryption, mutual authentication
   - Forward secrecy via ephemeral key exchange
   - SSH key format compatible (can reuse `~/.ssh/id_ed25519`)
   - Only registered peers in `peers.toml` can connect
- ✅ **Sync topology**: Dual-mode (P2P mesh + Hub-spoke with leader election)
   - Static machines: Full P2P mesh, participate in leader election
   - Ephemeral machines: Connect to current leader only
   - Leader election: Bully algorithm, highest-priority static node wins
   - Automatic failover: ~15s detection, ephemeral auto-reconnect
- ✅ **Ephemeral machine support**: Portable identity + token enrollment
   - Portable identity: Store key in Codespaces/K8s secrets
   - Token enrollment: One-time tokens for team/shared environments
   - One peer entry covers all machines with same identity
- ✅ **Session ID generation**: Shell generates UUIDv7
   - Works with offline queue (no daemon round-trip needed)
   - Safe with shared portable identity (random component ensures uniqueness)
   - Time-ordered for efficient queries
- ✅ **Log compaction**: Tiered strategy with peer-class awareness
   - Compact when all static peers ACK, OR entry older than 90 days
   - Ephemeral peers don't block compaction
   - Long-offline peers get full state sync instead of delta
   - Tombstones expire after 30 days
- ✅ **Protocol versioning**: Semantic versioning with ±1 major compatibility
   - MAJOR.MINOR format, breaking vs additive changes
   - Support current + previous major version (rolling upgrades)
   - Graceful degradation for mixed-version clusters
   - Forward-compatible messages via `#[serde(default)]`
- ✅ **Data retention**: Configurable limits with smart pruning strategies
   - Per-host limit (default: 100k commands) prevents domination
   - Global limit (default: 1M commands) caps total storage
   - Age limit (default: 365 days) auto-expires old commands
   - Strategies: oldest, LRU, LFU, smart (weighted scoring)
   - Protected commands: starred, long-running, pattern-matched
   - Usage tracking (use_count, last_used) for smart pruning

### Still Open

(All questions decided - ready for implementation)

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
