# Operations, Query & Search

*Part of the [histdb-rs v1.0 architecture](README.md). Source: original ARCHITECTURE.md L2246–2404, L2939–3137.*

## Operations Layer (`histdb-core/src/ops/`)

### Recording History
```rust
// record.rs
pub struct Recorder<'a> {
    conn: &'a Connection,
    session_id: SessionId,  // UUIDv7 - shell-generated
    host: &'a str,
}

impl<'a> Recorder<'a> {
    /// Record a new command execution (called on zshaddhistory)
    /// history_id is shell-generated UUIDv7 for offline queue correlation
    pub fn start_command(
        &self,
        history_id: HistoryId,  // Shell-generated
        argv: &str,
        dir: &Path,
    ) -> Result<HistoryId, Error>;

    /// Update with exit status (called on precmd)
    pub fn finish_command(
        &self,
        history_id: HistoryId,
        exit_status: i32,
        duration_ms: u64,
    ) -> Result<(), Error>;
}
```

**Safety**: Uses a single transaction for the insert sequence:
```rust
/// Start command recording
/// history_id is shell-generated (UUIDv7) for offline queue correlation
pub fn start_command(&self, history_id: HistoryId, argv: &str, dir: &Path) -> Result<HistoryId, Error> {
    let tx = self.conn.transaction()?;

    // history_id comes from shell - enables:
    // 1. Offline queue: shell tracks ID for later FinishCommand
    // 2. Idempotent replay: ON CONFLICT deduplicates by UUID
    // 3. Race handling: upsert fills placeholder if FinishCommand arrived first
    // 4. No round-trip: shell proceeds immediately after spawning command

    // All inserts in one transaction - no race conditions
    tx.execute(
        "INSERT OR IGNORE INTO commands (argv) VALUES (?1)",
        params![argv],
    )?;

    tx.execute(
        "INSERT OR IGNORE INTO places (host, dir) VALUES (?1, ?2)",
        params![self.host, dir.display().to_string()],
    )?;

    // INSERT ... ON CONFLICT provides idempotent upsert - handles both:
    // 1. Normal case: insert new entry
    // 2. Race case: FinishCommand arrived first and created placeholder
    //    -> UPDATE fills in session_id, command_id, place_id, start_time
    tx.execute(
        "INSERT INTO history (uuid, session, command_id, place_id, start_time)
         SELECT ?1, ?2, c.id, p.id, ?3
         FROM commands c, places p
         WHERE c.argv = ?4 AND p.host = ?5 AND p.dir = ?6
         ON CONFLICT(uuid) DO UPDATE SET
             session = COALESCE(excluded.session, session),
             command_id = COALESCE(excluded.command_id, command_id),
             place_id = COALESCE(excluded.place_id, place_id),
             start_time = COALESCE(excluded.start_time, start_time)",
        params![history_id.to_string(), self.session_id.to_string(), Utc::now().timestamp(),
                argv, self.host, dir.display().to_string()],
    )?;

    tx.commit()?;
    Ok(history_id)
}
```

### Querying History
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
    exclude_ids: Option<HashSet<HistoryId>>,  // For hybrid query deduplication
    recent_only: bool,                         // Constrain to recent entries
}

impl QueryBuilder {
    pub fn new() -> Self;
    pub fn host(self, host: &str) -> Self;
    pub fn in_dir(self, dir: &Path) -> Self;      // dir and subdirs
    pub fn at_dir(self, dir: &Path) -> Self;      // exact dir
    pub fn session(self, session: SessionId) -> Self;
    pub fn from(self, from: DateTime<Utc>) -> Self;
    pub fn until(self, until: DateTime<Utc>) -> Self;
    pub fn pattern(self, pattern: &str) -> Self;  // LIKE pattern
    pub fn glob(self, glob: &str) -> Self;        // GLOB pattern (escaped!)
    pub fn limit(self, n: usize) -> Self;
    pub fn descending(self) -> Self;

    /// Exclude specific IDs from results (for hybrid query deduplication)
    /// Used to avoid returning entries already found in the in-memory index
    pub fn exclude_ids(self, ids: HashSet<HistoryId>) -> Self {
        Self { exclude_ids: Some(ids), ..self }
    }

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

### 3-Way Merge
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


## Extensible Search Backend (`histdb-core/src/search/`)

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
