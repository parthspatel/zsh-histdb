# Daemon, Ingest & Concurrency

*Part of the [histdb-rs v1.0 architecture](README.md). Source: original ARCHITECTURE.md L148–1884, L4086–4627.*

## Concurrency Architecture (Lock-Free Design)

> **Review:** the workload is a few inserts per second; this machinery targets contention that does not exist. A bounded channel + one writer task is proposed — see [REVIEW.md › Recommended v1 shape](REVIEW.md#recommended-v1-shape-proposal).

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
/// Note: SessionId and HistoryId are the only ID types - both shell-generated
pub type SessionId = Uuid;  // Shell session (generated on shell startup)
pub type HistoryId = Uuid;  // Individual command entry (generated per command)

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
        history_id: HistoryId,  // UUIDv7 - shell-generated for correlation
        session_id: SessionId,  // UUIDv7 - globally unique, time-ordered
        argv: Box<str>,         // Owned, no lifetime issues
        dir: Box<Path>,
        host: Box<str>,
        start_time: DateTime<Utc>,  // Standardized on DateTime<Utc>
        response_tx: Option<oneshot::Sender<HistoryId>>,
    },
    Finish {
        history_id: HistoryId,  // UUIDv7 - correlates with Start
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
        // busy_timeout aligned with query timeout (50ms) to prevent latency spikes
        // If locked longer, retry via queue rather than blocking
        conn.execute_batch("
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA wal_autocheckpoint = 1000;
            PRAGMA busy_timeout = 50;
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
                CommandKind::Start { history_id, session_id, argv, dir, host, start_time, response_tx } => {
                    self.insert_history_entry(&tx, history_id, session_id, &argv, &dir, &host, start_time)?;
                    if let Some(tx) = response_tx {
                        let _ = tx.send(history_id);  // Return shell-generated ID
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
        history_id: HistoryId,  // Shell-generated UUIDv7
        session_id: SessionId,
        argv: &str,
        dir: &Path,
        host: &str,
        start_time: DateTime<Utc>,  // Standardized on DateTime<Utc>
    ) -> Result<(), rusqlite::Error> {
        // history_id is shell-generated (UUIDv7) for:
        // 1. Offline queue correlation (shell tracks ID for FinishCommand)
        // 2. Idempotent replay (same ID = same entry, no duplicates)
        // Uses prepared statements, parameterized queries
        // Stores start_time.timestamp() as INTEGER in SQLite
        Ok(())
    }

    /// Update finish status - returns true if row was updated, false if not found
    fn update_finish(
        &self,
        tx: &Transaction,
        history_id: HistoryId,
        exit_status: i32,
        duration_ms: u64,
    ) -> bool {
        let rows_updated = tx.execute(
            "UPDATE history SET exit_status = ?1, duration_ms = ?2 WHERE uuid = ?3",
            params![exit_status, duration_ms, history_id.to_string()],
        ).unwrap_or(0);

        rows_updated > 0
    }

    /// Create placeholder entry for orphaned FinishCommand
    /// Allows reconstruction when StartCommand arrives later (spool replay)
    fn insert_placeholder(
        &self,
        tx: &Transaction,
        history_id: HistoryId,
        exit_status: i32,
        duration_ms: u64,
    ) -> Result<(), rusqlite::Error> {
        // Calculate start_time from duration: now - duration = approximate start
        // This gives a more accurate estimate than just using now()
        let now = Utc::now();
        let start_time = now - chrono::Duration::milliseconds(duration_ms as i64);

        tx.execute(
            "INSERT OR IGNORE INTO history (uuid, session_id, argv, dir, host, start_time, exit_status, duration_ms)
             VALUES (?1, NULL, NULL, NULL, ?2, ?3, ?4, ?5)",
            params![
                history_id.to_string(),
                self.host,
                start_time.timestamp(),  // Calculated from: now - duration_ms
                exit_status,
                duration_ms,
            ],
        )?;
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
    /// Handle incoming StartCommand - returns immediately (fire-and-forget)
    pub async fn handle_start(&self, history_id: HistoryId, argv: String, dir: String) -> FireAndForgetAck {
        let cmd = PendingCommand {
            kind: CommandKind::Start {
                history_id,  // Shell-generated
                session_id: self.session_id,
                argv: argv.into_boxed_str(),
                dir: PathBuf::from(dir).into_boxed_path(),
                host: self.host.clone(),
                start_time: Utc::now(),  // DateTime<Utc>
                response_tx: None,  // Fire-and-forget: no response channel
            },
            timestamp: Instant::now(),
        };

        match self.queue.push(cmd) {
            Ok(()) => FireAndForgetAck::Accepted,
            Err(_) => FireAndForgetAck::QueueFull,  // Backpressure signal
        }
    }

    /// Handle StartCommand with ID response (for finish correlation)
    /// This is a request-response pattern - shell waits for history_id confirmation
    pub async fn handle_start_with_id(&self, history_id: HistoryId, argv: String, dir: String) -> Response {
        let (tx, rx) = oneshot::channel();

        let cmd = PendingCommand {
            kind: CommandKind::Start {
                history_id,  // Shell-generated
                session_id: self.session_id,
                argv: argv.into_boxed_str(),
                dir: PathBuf::from(dir).into_boxed_path(),
                host: self.host.clone(),
                start_time: Utc::now(),  // DateTime<Utc>
                response_tx: Some(tx),
            },
            timestamp: Instant::now(),
        };

        if self.queue.push(cmd).is_err() {
            return Response::QueueFull;
        }

        // Wait for writer to process (typically < 10ms)
        // Distinguish timeout (transient, safe to retry) from channel error (fatal)
        match tokio::time::timeout(Duration::from_millis(100), rx).await {
            Ok(Ok(id)) => Response::CommandStarted { history_id: id },
            Ok(Err(_)) => {
                // Channel closed - writer thread crashed or shut down
                // Shell should fall back to offline queue, not retry
                tracing::error!("Writer channel closed unexpectedly");
                Response::Error {
                    code: ErrorCode::Internal,
                    message: "Writer unavailable".into(),
                }
            }
            Err(_) => {
                // Timeout - writer is slow but may still process
                // Shell can retry or queue offline
                Response::Timeout
            }
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

> **Review:** the two backpressure designs below contradict each other; one blocks the tokio runtime, the other does not compile (`PendingCommand: Clone` holds a `oneshot::Sender`). [F3](REVIEW.md#f3)

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
    pub async fn handle_start(&self, history_id: HistoryId, argv: String, dir: String) -> FireAndForgetAck {
        let cmd = PendingCommand { /* ... */ };

        // Retry up to 3 times with backoff (100μs, 200μs, 400μs)
        match self.queue.push_with_retry(cmd, 3) {
            Ok(()) => FireAndForgetAck::Accepted,
            Err(_) => {
                // Final fallback: metrics + error response
                metrics::counter!("histdb.queue.drops").increment(1);
                FireAndForgetAck::QueueFull
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
    pub async fn handle_start(&self, history_id: HistoryId, argv: String, dir: String) -> FireAndForgetAck {
        let cmd = PendingCommand { /* ... */ };

        // Non-blocking push attempt
        match self.queue.try_push(cmd.clone()) {
            Ok(()) => FireAndForgetAck::Accepted,
            Err(_) => {
                // Queue full - hand off to async overflow handler
                // Shell gets immediate response, daemon retries async
                match self.overflow.overflow_tx.try_send(cmd) {
                    Ok(()) => {
                        metrics::counter!("histdb.queue.overflow").increment(1);
                        FireAndForgetAck::AcceptedOverflow  // Tell shell "we got it, processing async"
                    }
                    Err(_) => {
                        // Even overflow is full - extreme backpressure
                        metrics::counter!("histdb.queue.overflow_full").increment(1);
                        FireAndForgetAck::QueueFull
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
                _histdb_queue_offline "$request_type" "$json"
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

> **Review:** not a real guarantee (fsync/checkpoint/busy can exceed it; timing tests will flake). Read-your-writes via the daemon is proposed. [F5](REVIEW.md#f5)

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

> **Review:** the hybrid index→SQLite merge is wrong for non-recency orderings and host/dir filters, duplicates SQLite's page cache, and is unreachable from standalone CLI processes. Proposed: drop it. [F6](REVIEW.md#f6)

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
    /// Command → history IDs (for dedup/lookup)
    by_command: HashMap<Box<str>, Vec<HistoryId>>,
    /// Directory → history IDs (for --in/--at queries)
    by_dir: HashMap<Box<Path>, Vec<HistoryId>>,
    /// Last N entries per session (for isearch)
    by_session: HashMap<SessionId, VecDeque<HistoryId>>,
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
        let now = chrono::Utc::now();
        let age_threshold = now - chrono::Duration::seconds(self.bounds.max_age_secs as i64);

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

    fn evict_older_than(&mut self, threshold: DateTime<Utc>) {
        let to_remove: Vec<_> = self.recent.iter()
            .filter(|e| e.start_time < threshold)
            .map(|e| e.history_id)
            .collect();

        for history_id in to_remove {
            self.remove_entry(history_id);
        }
    }

    fn evict_oldest_entry(&mut self) {
        if let Some(oldest) = self.recent.iter().min_by_key(|e| e.start_time) {
            let history_id = oldest.history_id;
            self.remove_entry(history_id);
        }
    }

    fn remove_entry(&mut self, history_id: HistoryId) {
        if let Some(pos) = self.recent.iter().position(|e| e.history_id == history_id) {
            let entry = self.recent.remove(pos);
            self.estimated_memory -= Self::entry_memory_size(&entry);

            // Clean up maps
            if let Some(ids) = self.by_command.get_mut(&entry.argv) {
                ids.retain(|&eid| eid != history_id);
            }
            if let Some(ids) = self.by_dir.get_mut(&entry.dir) {
                ids.retain(|&eid| eid != history_id);
            }
            if let Some(ids) = self.by_session.get_mut(&entry.session_id) {
                ids.retain(|&eid| eid != history_id);
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
    pub history_id: HistoryId,   // UUIDv7 (shell-generated)
    pub session_id: SessionId,   // UUIDv7 (shell-generated)
    pub argv: Box<str>,
    pub dir: Box<Path>,
    pub host: Box<str>,
    pub start_time: DateTime<Utc>,
    pub exit_status: Option<i32>,
    pub duration_ms: Option<u64>,
}

/// Operations that can be applied to the index
pub enum IndexOp {
    Insert(RecentEntry),
    UpdateExitStatus { history_id: HistoryId, exit_status: i32, duration_ms: u64 },
    Evict { before: DateTime<Utc> },  // Evict entries older than timestamp
    EvictIfNeeded,                     // Check bounds and evict as needed
}

impl Absorb<IndexOp> for HistoryIndex {
    fn absorb_first(&mut self, op: &mut IndexOp, _: &Self) {
        match op {
            IndexOp::Insert(entry) => {
                // IMPORTANT: Evict BEFORE insert to ensure bounds are never exceeded
                // This prevents temporary memory spikes and maintains strict bounds
                if self.needs_eviction() {
                    self.evict_if_needed();
                }

                // Track memory
                self.estimated_memory += Self::entry_memory_size(entry);

                self.by_command
                    .entry(entry.argv.clone())
                    .or_default()
                    .push(entry.history_id);
                self.by_dir
                    .entry(entry.dir.clone())
                    .or_default()
                    .push(entry.history_id);

                // Enforce per-session limit (evict before adding)
                let session_entries = self.by_session
                    .entry(entry.session_id)
                    .or_default();
                if session_entries.len() >= self.bounds.max_per_session {
                    // Evict oldest from this session
                    if let Some(old_id) = session_entries.pop_front() {
                        self.remove_entry(old_id);
                    }
                }
                session_entries.push_back(entry.history_id);

                self.recent.push(entry.clone());

                // Post-insert safety check: if memory estimation drifted or entry was
                // larger than expected, evict again to guarantee bounds are satisfied
                if self.needs_eviction() {
                    self.evict_if_needed();
                }
            }
            IndexOp::UpdateExitStatus { history_id, exit_status, duration_ms } => {
                if let Some(entry) = self.recent.iter_mut().find(|e| e.history_id == *history_id) {
                    entry.exit_status = Some(*exit_status);
                    entry.duration_ms = Some(*duration_ms);  // Issue 10: Store duration
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

    fn absorb_second(&mut self, mut op: IndexOp, other: &Self) {
        // Issue 7: Fixed borrow pattern - take ownership and reborrow
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

    pub fn update_exit_status(&mut self, history_id: HistoryId, exit_status: i32, duration_ms: u64) {
        self.write.append(IndexOp::UpdateExitStatus { history_id, exit_status, duration_ms });
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
            .filter_map(|history_id| guard.recent.iter().find(|e| e.history_id == *history_id))
            .take(limit)
            .cloned()
            .collect()
    }

    /// Get oldest timestamp in index (for hybrid query decisions)
    pub fn oldest_timestamp(&self) -> Option<DateTime<Utc>> {
        let guard = self.read();
        guard.recent.first().map(|e| e.start_time)  // Vec uses first(), not front()
    }

    /// General search using QueryBuilder filters
    /// Used by hybrid query to apply same filters to in-memory index
    pub fn search(&self, builder: &QueryBuilder) -> Vec<RecentEntry> {
        let guard = self.read();
        let limit = builder.limit.unwrap_or(100);

        guard.recent
            .iter()
            .rev()  // Most recent first
            .filter(|e| {
                // Apply pattern filter
                if let Some(ref pattern) = builder.pattern {
                    if !e.argv.contains(pattern) {
                        return false;
                    }
                }
                // Apply directory filter
                if let Some(ref dir_filter) = builder.dir_filter {
                    match dir_filter {
                        DirFilter::Exact(d) => if e.dir != *d { return false; }
                        DirFilter::Prefix(d) => if !e.dir.starts_with(d) { return false; }
                    }
                }
                // Apply session filter
                if let Some(ref session) = builder.session_filter {
                    if e.session_id != *session { return false; }
                }
                // Apply time range filter
                if let Some(ref time_range) = builder.time_range {
                    if let Some(start) = time_range.start {
                        if e.start_time < start { return false; }
                    }
                    if let Some(end) = time_range.end {
                        if e.start_time > end { return false; }
                    }
                }
                // Apply exclude_ids filter
                if let Some(ref exclude) = builder.exclude_ids {
                    if exclude.contains(&e.history_id) { return false; }
                }
                true
            })
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
    fn flush_batch(&mut self, batch: &mut Vec<PendingCommand>) -> Result<(), Error> {
        let tx = self.conn.transaction()?;
        let mut index_ops = Vec::with_capacity(batch.len());

        for cmd in batch.drain(..) {
            match cmd.kind {
                CommandKind::Start { history_id, session_id, argv, dir, host, start_time, response_tx } => {
                    self.insert_history_entry(&tx, history_id, session_id, &argv, &dir, &host, start_time)?;

                    // Queue index update
                    index_ops.push(IndexOp::Insert(RecentEntry {
                        history_id,
                        session_id,
                        argv: argv.clone(),
                        dir: dir.clone(),
                        host: host.clone(),
                        start_time,  // Already DateTime<Utc> from CommandKind
                        exit_status: None,
                        duration_ms: None,
                    }));

                    if let Some(tx) = response_tx {
                        let _ = tx.send(history_id);
                    }
                }
                CommandKind::Finish { history_id, exit_status, duration_ms } => {
                    // Handle orphaned FinishCommand (no matching StartCommand)
                    // This can happen if:
                    // - StartCommand was lost (crash, network issue)
                    let updated = self.update_finish(&tx, history_id, exit_status, duration_ms)?;

                    if updated {
                        // StartCommand existed - update index to reflect exit status
                        index_ops.push(IndexOp::UpdateExitStatus {
                            history_id,
                            exit_status,
                            duration_ms,
                        });
                    } else {
                        // Orphaned FinishCommand - create DB placeholder, skip index update
                        // Index has no entry for this history_id (StartCommand never arrived)
                        // When StartCommand is replayed later from the spool, it will populate the index
                        tracing::warn!("Orphaned FinishCommand for {}, creating placeholder", history_id);
                        self.insert_placeholder(&tx, history_id, exit_status, duration_ms)?;
                        // No index update - entry doesn't exist in index
                    }
                }
            }
        }

        // Commit transaction - propagate error instead of panicking
        if let Err(e) = tx.commit() {
            tracing::error!("Failed to commit batch: {}", e);
            // Index ops not applied - data remains consistent
            // Entries will be retried on next batch
            return Err(e.into());
        }

        // Apply index updates and publish to readers
        for op in index_ops {
            self.index_writer.append(op);
        }
        self.index_writer.publish();  // Readers now see new entries
        Ok(())
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
        let exclude_ids: HashSet<_> = recent.iter().map(|e| e.history_id).collect();
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
            if seen.insert(entry.history_id) {
                merged.push(entry.into());
                if merged.len() >= limit {
                    return merged;
                }
            }
        }

        // Add DB results (older entries, already filtered to exclude index IDs)
        for row in db_results {
            if seen.insert(row.history_id) {
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
        self.session_filter.is_some() || self.recent_only
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
│ 4. Cleanup              │  ← WAL checkpoint, remove socket/pidfile
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
    /// Process any pending spool queue entries
    ProcessSpoolQueue,
    /// Flush to disk
    FlushStorage,
    /// Final cleanup
    Terminate,
}

#[derive(Debug)]
pub struct ShutdownConfig {
    /// Total shutdown timeout
    pub total_timeout: Duration,
    /// Time to drain queues
    pub drain_timeout: Duration,
    /// Time to process spool queue
    pub spool_queue_timeout: Duration,
    /// Time to flush storage
    pub flush_timeout: Duration,
}

impl Default for ShutdownConfig {
    fn default() -> Self {
        Self {
            total_timeout: Duration::from_secs(10),
            drain_timeout: Duration::from_millis(500),
            spool_queue_timeout: Duration::from_secs(2),  // Process pending spool entries
            flush_timeout: Duration::from_secs(2),
        }
    }
}

/// Tracks completion of each component
struct ShutdownCompletion {
    socket_listener: AtomicBool,
    writer_thread: AtomicBool,
    offline_processor: AtomicBool,
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

        // Phase 3: Process spool queue
        // Process any queued spool entries before shutdown
        let _ = self.notify_tx.send(ShutdownPhase::ProcessSpoolQueue);
        let spool_result = tokio::time::timeout(
            self.config.spool_queue_timeout,
            Self::process_spool_queue(&spool_processor)
        ).await;

        match spool_result {
            Ok(Ok(processed)) => {
                report.spool_entries_processed = processed;
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

        // Phase 6: Final termination
        let _ = self.notify_tx.send(ShutdownPhase::Terminate);

        report.duration = start.elapsed();
        tracing::info!("Shutdown complete in {:?}", report.duration);
        report
    }

    /// Process pending spool queue entries
    async fn process_spool_queue(
        processor: &SpoolQueueProcessor,
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

        // 4. Rebuild left-right index from recent history using sequence tracking
        // Uses SQLite rowid as sequence number for efficient incremental recovery
        let last_indexed_rowid: i64 = conn.query_row(
            "SELECT COALESCE((SELECT value FROM index_state WHERE key = 'last_indexed_rowid'), 0)",
            [],
            |row| row.get(0)
        )?;

        let recent = conn.prepare("
            SELECT rowid, * FROM history
            WHERE rowid > ?1
              AND start_time > ?2
            ORDER BY rowid ASC
            LIMIT ?3
        ")?.query_map(
            params![last_indexed_rowid, now() - self.config.index.max_age, self.config.index.max_entries],
            |row| {
                let rowid: i64 = row.get(0)?;
                let entry = RecentEntry::from_row_offset(row, 1)?;
                Ok((rowid, entry))
            }
        )?;

        let mut max_rowid = last_indexed_rowid;
        for result in recent {
            let (rowid, entry) = result?;
            self.index_writer.insert(entry);
            max_rowid = max_rowid.max(rowid);
        }
        self.index_writer.publish();

        // Persist high water mark for next recovery
        if max_rowid > last_indexed_rowid {
            conn.execute(
                "INSERT OR REPLACE INTO index_state (key, value) VALUES ('last_indexed_rowid', ?1)",
                params![max_rowid],
            )?;
        }

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
| Corrupt database | All local data | Restore from backup (`histdb export`, file copy, or legacy git sync) |

### Optional Durable Mode

> **Review:** default "fast" mode loses already-acknowledged commands on crash, which the shell spool cannot recover. Durable-by-default is proposed. [F8](REVIEW.md#f8)

For users requiring zero data loss (at cost of latency):

```toml
# ~/.config/histdb/config.toml
[daemon]
# "fast" = ack before persist (default, <1ms, may lose ≤10ms on crash)
# "durable" = ack after persist (~5-10ms, crash-safe)
ack_mode = "fast"
```

---
