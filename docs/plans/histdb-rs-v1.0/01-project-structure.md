# Project Structure

*Part of the [histdb-rs v1.0 architecture](README.md). Source: original ARCHITECTURE.md L39–146.*

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
│   │   │   │   ├── sync.rs       # histdb sync (legacy git-based sync + 3-way merge; no P2P)
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
│   └── 003_add_fts5.sql          # Optional FTS5 virtual table
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
