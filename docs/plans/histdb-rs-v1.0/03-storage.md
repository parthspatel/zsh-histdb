# Storage: Database, Schema, Migrations & Retention

*Part of the [histdb-rs v1.0 architecture](README.md). Source: original ARCHITECTURE.md L1887–2245, L7257–7580.*

## Database Layer (`histdb-core/src/db/`)

### Connection Management
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

### Schema (Type-Safe)
```rust
// schema.rs
pub struct Command {
    pub id: i64,
    pub argv: String,
}

pub struct Place {
    pub id: i64,           // Internal surrogate key (for joins)
    pub host: String,
    pub dir: PathBuf,
}

/// History entry - UUID is primary identifier (idempotent offline replay)
pub struct HistoryEntry {
    pub history_id: HistoryId,        // Primary key (UUIDv7, shell-generated)
    pub session_id: SessionId,        // UUIDv7 session
    pub argv: String,                 // Command text (denormalized for simplicity)
    pub dir: PathBuf,                 // Working directory
    pub host: String,                 // Hostname
    pub exit_status: Option<i32>,     // NULL while running
    pub start_time: DateTime<Utc>,
    pub duration_ms: Option<u64>,     // NULL while running
}
```

### Migrations

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

## Models Layer (`histdb-core/src/models/`)

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


## Data Retention & Pruning

> **Review:** the default limits (365 days, 100k/host, 1M total) silently delete history; proposed default is unlimited with `prune --dry-run`. [F7](REVIEW.md#f7)

Configurable limits and strategies to manage history growth.

### Retention Limits

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

### Retention Strategies

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

### Smart Strategy Scoring

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

### Usage Tracking

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


    Ok(())
}
```

### Pruning Implementation

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
                // v1: hard delete (history row + FTS row) in one transaction
                self.delete_entry(*uuid)?;
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

### Starred Commands

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
