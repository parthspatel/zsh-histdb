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
- **Connection pooling**: Reuse SQLite connections
- **Prepared statements**: Pre-compiled queries for hot paths
- **Async I/O**: Non-blocking operations where beneficial
- **Efficient merging**: Optimized 3-way merge algorithm
- **Minimal shell overhead**: Fast IPC for zsh integration

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
│   │   │   ├── models/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── command.rs    # Command model
│   │   │   │   ├── place.rs      # Place (host+dir) model
│   │   │   │   ├── history.rs    # History entry model
│   │   │   │   └── session.rs    # Session model
│   │   │   ├── ops/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── record.rs     # Record new history
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
rusqlite = { version = "0.31", features = ["bundled", "backup"] }
r2d2 = "0.8"
r2d2_sqlite = "0.24"
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "1"
chrono = { version = "0.4", features = ["serde"] }
toml = "0.8"
regex = "1"
tokio = { version = "1", features = ["rt", "net", "io-util"] }
tracing = "0.1"
tracing-subscriber = "0.3"
```

---

## Migration Path

1. **Phase 1**: Core library + CLI (can coexist with zsh version)
2. **Phase 2**: Daemon + zsh integration
3. **Phase 3**: Import tool for existing databases
4. **Phase 4**: Deprecate zsh version

---

## Open Questions for Review

1. **Daemon vs embedded**: Should we use a daemon (faster) or spawn per-command (simpler)?

2. **Async runtime**: Use `tokio` for the daemon, or keep it synchronous?

3. **Config format**: TOML, JSON, or environment-only?

4. **FTS5**: Add full-text search index for better pattern matching?

5. **Encryption**: Support SQLCipher for encrypted history?

6. **Shell support**: Zsh-only, or also bash/fish?

---

## Summary

This architecture prioritizes:
- **Safety**: Parameterized queries, type-safe models, proper error handling
- **Performance**: Connection pooling, prepared statements, daemon architecture
- **Correctness**: Transaction safety, comprehensive testing, atomic migrations
- **Maintainability**: Clean separation of concerns, well-defined interfaces
