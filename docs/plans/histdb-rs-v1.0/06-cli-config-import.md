# Configuration, CLI & Import

*Part of the [histdb-rs v1.0 architecture](README.md). Source: original ARCHITECTURE.md L2405–2938, L6074–6121.*

## Configuration (`histdb-core/src/config.rs`)

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

/// Duration configuration that accepts multiple formats
/// Supports: "7d" (days), "24h" (hours), "3600s" (seconds), or bare number (seconds)
#[derive(Debug, Clone, Copy)]
pub struct ConfigDuration(pub u64);  // Stored as seconds internally

/// Maximum allowed duration: 365 days (prevents overflow in as_chrono)
const MAX_DURATION_SECS: u64 = 365 * 24 * 60 * 60;

impl<'de> Deserialize<'de> for ConfigDuration {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        use serde::de::Error;
        let s = String::deserialize(deserializer)?;

        let secs = if let Some(days) = s.strip_suffix('d') {
            let d: u64 = days.parse().map_err(D::Error::custom)?;
            d * 86400
        } else if let Some(hours) = s.strip_suffix('h') {
            let h: u64 = hours.parse().map_err(D::Error::custom)?;
            h * 3600
        } else if let Some(secs) = s.strip_suffix('s') {
            secs.parse().map_err(D::Error::custom)?
        } else {
            // Bare number = seconds (backwards compatible)
            s.parse().map_err(D::Error::custom)?
        };

        if secs > MAX_DURATION_SECS {
            return Err(D::Error::custom(format!(
                "duration {} exceeds maximum of 365 days ({} seconds)",
                s, MAX_DURATION_SECS
            )));
        }

        Ok(ConfigDuration(secs))
    }
}

impl ConfigDuration {
    pub fn as_secs(&self) -> u64 { self.0 }
    pub fn as_chrono(&self) -> chrono::Duration { chrono::Duration::seconds(self.0 as i64) }
}

#[derive(Debug, Deserialize)]
pub struct IndexConfig {
    /// Maximum entries in in-memory index (default: 1000)
    /// Queries beyond this limit fall back to SQLite (hybrid query)
    #[serde(default = "default_max_entries")]
    pub max_entries: usize,
    /// Evict entries older than (e.g., "7d", "168h", "604800s", or 604800)
    #[serde(default = "default_max_age")]
    pub max_age: ConfigDuration,
    /// Memory limit (bytes)
    pub memory_limit: usize,
}

fn default_max_entries() -> usize { 1000 }
fn default_max_age() -> ConfigDuration { ConfigDuration(7 * 24 * 60 * 60) }  // 7 days

impl From<&IndexConfig> for IndexBounds {
    fn from(config: &IndexConfig) -> Self {
        IndexBounds {
            max_entries: config.max_entries,
            max_memory_bytes: config.memory_limit,
            max_age_secs: config.max_age.as_secs(),
            max_per_session: 500,  // Could add to config if needed
        }
    }
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
max_age = "7d"                   # Evict older entries (supports: "7d", "168h", "604800s", or 604800)
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

## CLI (`histdb-cli/`)

Using `clap` for argument parsing:

```rust
#[derive(Parser)]
#[command(name = "histdb")]
pub enum Cli {
    /// Query command history
    Query(QueryArgs),

    /// Show most frequent commands/directories
    Top(TopArgs),

    /// Sync history via git (legacy zsh-histdb behaviour + 3-way merge; no P2P in v1)
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

// Note: Offline replay is handled by the daemon (SpoolQueueProcessor),
// not the CLI - see daemon/offline.rs
```

---

## Import Interface (`histdb-core/src/import/`)

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


## CLI Commands for Forget

```rust
// cli/forget.rs

/// Forget (delete) a command from history.
/// v1: hard delete of the history row and its FTS row in one transaction.
/// (Tombstones and `resurrect` only matter for P2P sync — see deferred/sync.md.)
pub fn handle_forget(uuid: Uuid, service: &HistoryService) -> Result<()> {
    if service.forget_command(uuid)? {
        println!("Entry {} permanently deleted", uuid);
    } else {
        println!("No entry {}", uuid);
    }
    Ok(())
}
```
