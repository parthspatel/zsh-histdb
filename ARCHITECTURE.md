# zsh-histdb-rs: Rust Rewrite Architecture

## Overview

A Rust rewrite of zsh-histdb focusing on **safety**, **performance**, and **correctness**.

## Goals

### Safety
- **No SQL injection**: All queries use parameterized statements
- **Type-safe schema**: Compile-time checked database operations
- **Proper error handling**: All errors propagated via `Result<T, E>`
- **Transaction safety**: ACID guarantees for all multi-step operations
- **Input validation**: All user input validated before use

### Performance
- **Lock-free ingestion**: Wait-free MPSC queue for concurrent shell sessions
- **Zero-copy where possible**: Minimize allocations in hot paths
- **Batched writes**: Coalesce multiple commands into single transactions
- **Prepared statements**: Pre-compiled queries for hot paths
- **Efficient merging**: Optimized 3-way merge algorithm
- **Minimal shell overhead**: Fire-and-forget IPC, sub-millisecond latency

### Correctness
- **Comprehensive tests**: Unit, integration, and property-based testing
- **Migration safety**: Atomic migrations with automatic rollback
- **Concurrent access**: Proper WAL handling for multi-shell scenarios

---

## Project Structure

```
zsh-histdb-rs/
├── Cargo.toml                    # Workspace manifest
├── crates/
│   ├── histdb-core/              # Core library
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── db/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── connection.rs # Connection management
│   │   │   │   ├── schema.rs     # Type-safe schema definitions
│   │   │   │   ├── queries.rs    # Prepared query definitions
│   │   │   │   └── migrations.rs # Schema migrations
│   │   │   ├── ingest/           # Lock-free ingestion pipeline
│   │   │   │   ├── mod.rs
│   │   │   │   ├── queue.rs      # Lock-free MPSC queue
│   │   │   │   ├── writer.rs     # Single-consumer writer thread
│   │   │   │   └── batch.rs      # Batch accumulation logic
│   │   │   ├── models/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── command.rs    # Command model
│   │   │   │   ├── place.rs      # Place (host+dir) model
│   │   │   │   ├── history.rs    # History entry model
│   │   │   │   └── session.rs    # Session model
│   │   │   ├── ops/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── record.rs     # Record new history (via queue)
│   │   │   │   ├── query.rs      # Query history
│   │   │   │   ├── forget.rs     # Delete history
│   │   │   │   ├── merge.rs      # 3-way merge for sync
│   │   │   │   └── stats.rs      # Statistics
│   │   │   ├── config.rs         # Configuration management
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
│   │   │   │   ├── import.rs     # Import from zsh_history
│   │   │   │   └── export.rs     # Export to JSON/CSV
│   │   │   ├── output.rs         # Output formatting
│   │   │   └── args.rs           # Argument parsing (clap)
│   │   └── Cargo.toml
│   │
│   └── histdb-zsh/               # Zsh integration daemon
│       ├── src/
│       │   ├── main.rs           # Daemon entry point
│       │   ├── ipc.rs            # Unix socket IPC
│       │   ├── protocol.rs       # Wire protocol
│       │   └── hooks.rs          # Hook message handlers
│       ├── shell/
│       │   ├── histdb.zsh        # Minimal zsh wrapper
│       │   └── histdb-isearch.zsh
│       └── Cargo.toml
│
├── migrations/                   # SQL migration files
│   ├── 001_initial.sql
│   └── 002_add_indexes.sql
│
└── tests/                        # Integration tests
    ├── migration_tests.rs
    ├── merge_tests.rs
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

When the queue is full (extremely rare with 4096 capacity):

```rust
// Option 1: Drop command (lossy, fast)
if queue.push(cmd).is_err() {
    tracing::warn!("Queue full, dropping command");
}

// Option 2: Inline write (blocking, reliable)
if queue.push(cmd).is_err() {
    tracing::warn!("Queue full, falling back to sync write");
    direct_write_to_db(&cmd);
}

// Option 3: Retry with backoff (balanced)
for attempt in 0..3 {
    if queue.push(cmd.clone()).is_ok() {
        break;
    }
    std::thread::sleep(Duration::from_micros(100 << attempt));
}
```

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
│    - Handle queries (read from SQLite)                   │
├──────────────────────────────────────────────────────────┤
│  Thread 2: Writer Thread                                 │
│    - Drain queue (single consumer)                       │
│    - Batch writes to SQLite                              │
│    - Own the write connection                            │
├──────────────────────────────────────────────────────────┤
│  Shared: Arc<CommandQueue>                               │
│    - Lock-free ArrayQueue                                │
│    - Atomic counters for monitoring                      │
└──────────────────────────────────────────────────────────┘
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
pub struct Config {
    /// Database file path (default: ~/.histdb/zsh-history.db)
    pub database_path: PathBuf,

    /// Hostname override
    pub hostname: String,

    /// Commands to ignore (regex patterns)
    pub ignore_patterns: Vec<Regex>,

    /// Whether to respect histignorespace
    pub ignore_space_prefix: bool,

    /// Query timeout in milliseconds
    pub timeout_ms: u64,

    /// Maximum results for interactive search
    pub max_results: usize,
}

impl Config {
    /// Load from ~/.config/histdb/config.toml
    pub fn load() -> Result<Self, ConfigError>;

    /// Load with environment variable overrides
    pub fn load_with_env() -> Result<Self, ConfigError>;
}
```

**Config file format** (`~/.config/histdb/config.toml`):
```toml
database_path = "~/.histdb/zsh-history.db"
hostname = "my-machine"
timeout_ms = 1000
max_results = 100

[ignore]
patterns = [
    "^ls$",
    "^cd$",
    "^ ",           # commands starting with space
    "^histdb",
    "^top$",
    "^htop$",
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

### 6. Zsh Integration (`histdb-zsh/`)

**Architecture**: Long-running daemon with Unix socket IPC

```
┌─────────────┐     Unix Socket      ┌─────────────────┐
│   zsh       │◄────────────────────►│  histdb-daemon  │
│  (hooks)    │   JSON messages      │  (Rust binary)  │
└─────────────┘                      └────────┬────────┘
                                              │
                                              ▼
                                     ┌─────────────────┐
                                     │    SQLite DB    │
                                     └─────────────────┘
```

#### Protocol (`protocol.rs`)
```rust
#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Request {
    StartCommand {
        argv: String,
        dir: String,
        #[serde(default)]
        session: Option<i64>,
    },
    FinishCommand {
        id: i64,
        exit_status: i32,
    },
    Query {
        pattern: Option<String>,
        limit: usize,
        offset: usize,
        host_filter: bool,
        dir_filter: bool,
    },
    Shutdown,
}

#[derive(Serialize, Deserialize)]
pub enum Response {
    CommandStarted { id: i64 },
    CommandFinished,
    QueryResults { entries: Vec<HistoryEntry> },
    Error { message: String },
}
```

#### Minimal Zsh Wrapper (`shell/histdb.zsh`)
```zsh
# Thin wrapper - all logic in Rust daemon
typeset -g HISTDB_SOCKET="${XDG_RUNTIME_DIR:-/tmp}/histdb-${UID}.sock"
typeset -g HISTDB_LAST_ID=""

_histdb_send() {
    echo "$1" | socat - UNIX-CONNECT:${HISTDB_SOCKET} 2>/dev/null
}

_histdb_addhistory() {
    local response=$(_histdb_send "{\"type\":\"StartCommand\",\"argv\":\"${1[0,-2]}\",\"dir\":\"${PWD}\"}")
    HISTDB_LAST_ID=$(echo "$response" | jq -r '.id // empty')
}

_histdb_precmd() {
    local status=$?
    [[ -n "$HISTDB_LAST_ID" ]] && \
        _histdb_send "{\"type\":\"FinishCommand\",\"id\":${HISTDB_LAST_ID},\"exit_status\":${status}}" &|
    HISTDB_LAST_ID=""
}

add-zsh-hook zshaddhistory _histdb_addhistory
add-zsh-hook precmd _histdb_precmd
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
parking_lot = "0.12"          # Fast mutexes (for non-hot paths only)

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
- ✅ **Concurrency**: Lock-free MPSC queue with single writer (no contention)

### Still Open

1. **Config format**: TOML, JSON, or environment-only?

2. **FTS5**: Add full-text search index for better pattern matching?
   - Pro: Much faster substring search on large histories
   - Con: Increases DB size, complexity

3. **Encryption**: Support SQLCipher for encrypted history?
   - Pro: Security for sensitive commands
   - Con: Build complexity, performance overhead

4. **Shell support**: Zsh-only, or also bash/fish?
   - Recommendation: Start with zsh, design IPC to be shell-agnostic

5. **Backpressure strategy**: When queue is full, should we:
   - Drop commands (fast, lossy)
   - Block briefly with retry (balanced)
   - Fall back to sync write (reliable, slow)

6. **History ID correlation**: For `FinishCommand`, should we:
   - Require shell to track returned ID (more reliable)
   - Use session + "most recent" heuristic (simpler shell code)

---

## Summary

This architecture prioritizes:
- **Safety**: Parameterized queries, type-safe models, proper error handling
- **Performance**: Connection pooling, prepared statements, daemon architecture
- **Correctness**: Transaction safety, comprehensive testing, atomic migrations
- **Maintainability**: Clean separation of concerns, well-defined interfaces
