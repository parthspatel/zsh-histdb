# Deferred: P2P Synchronization (not in v1)

> **Status: deferred, not scheduled.** Decided 2026-11-03: P2P sync is out of histdb-rs v1
> because its value to end users hasn't been established. v1 keeps zsh-histdb's existing
> git-based `histdb sync` + 3-way merge. This file preserves the original design verbatim
> (ARCHITECTURE.md L4628–7256 and L7581–7613), followed by an appendix of sync-coupled
> fragments cut from v1 sections.

## Before reviving this: answer the user question

What would a user get that git sync doesn't already give them? Candidates worth
validating with real users:
- Seconds-fresh history across machines, e.g. "the command I just ran on my desktop".
- Zero-setup sync for ephemeral machines (Codespaces, devcontainers, K8s pods).
- Not having to run `histdb sync` or manage a git remote.

If none of these matters enough, stop here.

## Known problems in the design below

See [REVIEW.md › F10](../REVIEW.md#f10). In summary:
1. **"Merge as single node" loses writes.** Instances sharing an identity run independent
   HLCs and mint colliding `(origin, wall, logical)` keys, so writes are dropped.
2. **Delta sync skips late ops.** The watermark per origin permanently skips ops that
   arrive late from a second instance with the same identity.
3. **mDNS can't reach the target machines.** Ephemeral machines (Codespaces, K8s) are on
   other networks behind NAT.
4. **Leader election is unnecessary.** History is a grow-only set plus delete markers;
   monotone data needs no coordination (CALM).
5. **The forward-compatibility claim fails.** bincode 1.x isn't self-describing, so
   `#[serde(default)]` doesn't give forward compatibility. The schema comment also says
   CBOR while the code uses bincode.

## Suggested direction if revived

Cheapest option first:
1. **Dumb-store sync, end-to-end encrypted.**
   - Model it on Atuin's record store: each machine has its own random ID and
     **appends only to its own log**. Never share a log between machines; this removes
     bugs 1–2 by construction.
   - Sync = "for each `(host_id)`, send records after the peer's max index".
   - Transport is anything dumb: the existing git remote, S3, a synced folder, or a tiny
     relay.
   - No HLC (a per-host sequence number plus wall time is enough), no leader, no mDNS.
2. **Live P2P** only if (1) proves too slow. Use iroh (dial by public key, relay + hole
   punching) instead of hand-rolled Noise + mDNS. Keep the same per-host append-only
   record model.
3. **Off-the-shelf:** after Turso 1.0, evaluate its CDC + sync engine. Also consider the
   SQLite session extension (changesets) on top of approach (1).
4. **Testing:** whichever option is chosen, use deterministic simulation testing
   (turmoil / madsim) before shipping.

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

// ============================================================================
// Core Type Definitions
// ============================================================================

/// Node identifier - unique per machine/identity
///
/// Derived from identity key: `NodeId = hash(public_key)[0..8]`
/// This provides:
/// - Compact representation (8 bytes vs 32 byte public key)
/// - Deterministic derivation from identity
/// - Efficient comparison and hashing
pub type NodeId = u64;

/// Human-readable node information (for display/config, not sync)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Numeric ID used in protocol
    pub id: NodeId,
    /// Human-readable name (e.g., "laptop", "desktop", "my-codespaces")
    pub name: String,
    /// Full public key (Ed25519)
    pub public_key: [u8; 32],
}

impl NodeInfo {
    /// Derive NodeId from public key
    pub fn id_from_pubkey(pubkey: &[u8; 32]) -> NodeId {
        // Use first 8 bytes of SHA-256 hash of public key
        use sha2::{Sha256, Digest};
        let hash = Sha256::digest(pubkey);
        u64::from_le_bytes(hash[0..8].try_into().unwrap())
    }
}

// ============================================================================
// Hybrid Logical Clock
// ============================================================================

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
///
/// Ordering: HLCs are ordered by (wall_time, logical, node_id)
/// This provides a total order even for concurrent events.
///
/// CORRECTNESS NOTE: The derived Ord implementation compares fields in declaration order:
/// 1. wall_time (primary) - timestamp in milliseconds
/// 2. logical (secondary) - counter for events at same wall_time
/// 3. node_id (tertiary) - tie-breaker for concurrent events
///
/// This ordering is correct because:
/// - Later wall_time means causally later or concurrent
/// - Higher logical at same wall_time means later in causal chain
/// - node_id provides deterministic tie-breaking for truly concurrent events
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct HLC {
    /// Wall clock time (milliseconds since epoch)
    /// IMPORTANT: Field order matters for derived Ord - do not reorder
    pub wall_time: u64,
    /// Logical counter for events at same wall time
    pub logical: u32,
    /// Node ID that created this timestamp (u64 derived from identity key)
    pub node_id: NodeId,
}

impl HLC {
    /// Create an initial HLC (at epoch)
    pub fn new(node_id: NodeId) -> Self {
        Self {
            wall_time: 0,
            logical: 0,
            node_id,
        }
    }

    /// Create HLC at current wall time (for local-only operations)
    /// Use this when daemon is unavailable and shell needs a timestamp
    pub fn local_now(node_id: NodeId) -> Self {
        Self {
            wall_time: Self::current_wall_time(),
            logical: 0,
            node_id,
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
    /// Returns Result to handle logical counter overflow (extremely rare)
    pub fn tick(&mut self, node_id: NodeId) -> Result<HLC, HlcError> {
        let wall = Self::current_wall_time();

        let (new_wall, new_logical) = if wall > self.wall_time {
            (wall, 0)
        } else {
            // Check for logical counter overflow
            let new_logical = self.logical.checked_add(1)
                .ok_or(HlcError::LogicalOverflow)?;
            (self.wall_time, new_logical)
        };

        // Update internal state
        self.wall_time = new_wall;
        self.logical = new_logical;

        Ok(HLC {
            wall_time: new_wall,
            logical: new_logical,
            node_id,
        })
    }

    /// Tick with automatic retry on LogicalOverflow
    /// If logical counter overflows (2^32 ops in same ms), sleep 1ms and retry
    /// This is a rare edge case but handles it gracefully
    pub fn tick_with_retry(&mut self, node_id: NodeId) -> HLC {
        loop {
            match self.tick(node_id) {
                Ok(hlc) => return hlc,
                Err(HlcError::LogicalOverflow) => {
                    // Rare: 2^32 ops in same millisecond
                    // Sleep 1ms to advance wall clock, then retry
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    tracing::warn!("HLC logical counter overflow, waited 1ms");
                }
                Err(e) => {
                    // Other errors (clock skew) - shouldn't happen in tick()
                    panic!("Unexpected HLC error: {:?}", e);
                }
            }
        }
    }

    /// Update clock after receiving a remote timestamp
    /// Fast-forwards local clock to max(local, remote, wall) to ensure causality
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

        // Fast-forward to max(local, remote, wall)
        // This ensures we don't waste logical counter on old messages
        let max_wall = wall.max(self.wall_time).max(remote.wall_time);

        let new_logical = if max_wall > self.wall_time && max_wall > remote.wall_time {
            // Wall clock advanced past both - reset logical to 0
            0
        } else if max_wall == self.wall_time && max_wall > remote.wall_time {
            // Local is ahead - no need to increment for old remote
            // Return current state (already causally ahead)
            self.logical
        } else if max_wall == remote.wall_time && max_wall > self.wall_time {
            // Remote is ahead - advance past remote
            remote.logical.checked_add(1).ok_or(HlcError::LogicalOverflow)?
        } else {
            // Same wall time - advance past max logical
            self.logical.max(remote.logical).checked_add(1)
                .ok_or(HlcError::LogicalOverflow)?
        };

        self.wall_time = max_wall;
        self.logical = new_logical;

        Ok(HLC {
            wall_time: max_wall,
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
    /// Tombstone for deletion (type determines resurrection policy)
    pub deleted: Option<Tombstone>,
}

/// Tombstone types with different resurrection semantics
///
/// Merge rules:
/// - SecurityTombstone ALWAYS wins (privacy protection, irrecoverable)
/// - GenericTombstone can be "resurrected" by a newer Insert/Update
/// - Higher timestamp wins within same tombstone type
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum TombstoneReason {
    /// Normal deletion (e.g., cleanup, user preference)
    /// Can be resurrected by newer operations
    Generic,
    /// Security/privacy deletion (e.g., sensitive command, credential leak)
    /// ALWAYS wins - entry cannot be resurrected
    Security,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tombstone {
    pub timestamp: HLC,
    pub reason: TombstoneReason,
}

impl Tombstone {
    pub fn generic(timestamp: HLC) -> Self {
        Self { timestamp, reason: TombstoneReason::Generic }
    }

    pub fn security(timestamp: HLC) -> Self {
        Self { timestamp, reason: TombstoneReason::Security }
    }

    /// Merge tombstones: Security always wins, otherwise newer timestamp wins
    pub fn merge(&self, other: &Tombstone) -> Tombstone {
        // Security tombstone always takes precedence (privacy protection)
        match (&self.reason, &other.reason) {
            (TombstoneReason::Security, _) => self.clone(),
            (_, TombstoneReason::Security) => other.clone(),
            // Both Generic: higher timestamp wins
            _ => {
                if other.timestamp > self.timestamp {
                    other.clone()
                } else {
                    self.clone()
                }
            }
        }
    }

    /// Check if this tombstone blocks resurrection
    pub fn is_permanent(&self) -> bool {
        self.reason == TombstoneReason::Security
    }
}

/// Core history entry data (local storage)
/// Does NOT contain CRDT metadata - that's stored separately and joined at sync time
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntryData {
    /// Session ID (UUIDv7 - unique per shell session, time-ordered)
    pub session_id: Uuid,
    /// Command line (argv joined)
    pub argv: String,
    /// Working directory
    pub dir: String,
    /// Hostname
    pub host: String,
    /// Start time (Unix timestamp)
    pub start_time: i64,
    /// LWW-Register: last-writer-wins for mutable fields
    pub exit_status: Option<LWWRegister<i32>>,
    pub duration: Option<LWWRegister<u64>>,
}

/// CRDT metadata for sync (separate from EntryData)
/// Stored in same table but populated separately:
/// - uuid, origin_node, hlc_* populated on first insert
/// - deleted_at_*, tombstone_reason populated on delete
/// Joined with EntryData when preparing sync messages
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrdtMetadata {
    /// Unique ID for this entry (UUIDv7 - from shell)
    pub uuid: Uuid,
    /// Node that created this entry
    pub origin_node: NodeId,
    /// HLC timestamp when created
    pub hlc: HLC,
    /// Tombstone (if deleted)
    pub deleted: Option<Tombstone>,
}

impl CrdtMetadata {
    /// Create metadata for a new local entry
    pub fn new_local(uuid: Uuid, node_id: NodeId, hlc: HLC) -> Self {
        Self { uuid, origin_node: node_id, hlc, deleted: None }
    }

    /// Check if entry is deleted (tombstoned)
    pub fn is_deleted(&self) -> bool { self.deleted.is_some() }

    /// Check if permanently deleted (SecurityTombstone - cannot resurrect)
    pub fn is_permanently_deleted(&self) -> bool {
        self.deleted.as_ref().map_or(false, |t| t.is_permanent())
    }
}

/// Full entry for sync (combines EntryData + CrdtMetadata)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyncEntry {
    pub data: EntryData,
    pub crdt: CrdtMetadata,
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
    /// 2. If HLC equal, higher node_id wins (numeric comparison)
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

        // Tie-breaker: numeric node_id comparison (NodeId is u64)
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
            HLC { wall_time: 100, logical: 0, node_id: 1 },
        );
        let reg2 = LWWRegister::new(
            99,
            HLC { wall_time: 200, logical: 0, node_id: 2 },
        );

        reg1.merge(&reg2);
        assert_eq!(reg1.value, 99);
    }

    #[test]
    fn test_lww_equal_time_node_id_tiebreaker() {
        // Same wall_time and logical, different node_id
        let mut reg1 = LWWRegister::new(
            42,
            HLC { wall_time: 100, logical: 5, node_id: 1 },
        );
        let reg2 = LWWRegister::new(
            99,
            HLC { wall_time: 100, logical: 5, node_id: 100 },
        );

        reg1.merge(&reg2);
        // node_id 100 > 1 numerically, so reg2 wins
        assert_eq!(reg1.value, 99);
    }

    #[test]
    fn test_lww_convergence_is_deterministic() {
        // Both nodes start with different values, same timestamp
        let ts_a = HLC { wall_time: 100, logical: 0, node_id: 1 };
        let ts_b = HLC { wall_time: 100, logical: 0, node_id: 2 };

        let mut node_a = LWWRegister::new("value-a", ts_a.clone());
        let mut node_b = LWWRegister::new("value-b", ts_b.clone());

        // Cross-merge in any order
        let node_b_copy = node_b.clone();
        node_a.merge(&node_b_copy);

        let node_a_copy = LWWRegister::new("value-a", ts_a);
        node_b.merge(&node_a_copy);

        // Both nodes converge to same value (node_id 2 > 1, so node_b wins)
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

    /// Delete entry (tombstone) - reason determines resurrection policy
    Delete {
        entry_id: Uuid,
        tombstone: Tombstone,
    },

    /// Resurrect entry (only works against GenericTombstone)
    Resurrect {
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

            Operation::Delete { entry_id, tombstone } => {
                if let Some(entry) = state.entries.get_mut(entry_id) {
                    // Merge tombstones: Security always wins, otherwise newer timestamp
                    match &entry.deleted {
                        Some(existing) => {
                            entry.deleted = Some(existing.merge(tombstone));
                        }
                        None => {
                            entry.deleted = Some(tombstone.clone());
                        }
                    }
                }
            }

            Operation::Resurrect { entry_id, timestamp } => {
                if let Some(entry) = state.entries.get_mut(entry_id) {
                    // Only resurrect if:
                    // 1. Entry has a tombstone
                    // 2. Tombstone is Generic (not Security)
                    // 3. Resurrection timestamp is newer than tombstone
                    if let Some(tombstone) = &entry.deleted {
                        if !tombstone.is_permanent() && *timestamp > tombstone.timestamp {
                            entry.deleted = None;
                        }
                        // Security tombstones cannot be resurrected - silently ignored
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tombstone_tests {
    use super::*;

    #[test]
    fn test_security_tombstone_wins_over_generic() {
        let generic = Tombstone::generic(HLC { wall_time: 200, logical: 0, node_id: "a".into() });
        let security = Tombstone::security(HLC { wall_time: 100, logical: 0, node_id: "b".into() });

        // Security wins even with older timestamp
        let merged = generic.merge(&security);
        assert_eq!(merged.reason, TombstoneReason::Security);
    }

    #[test]
    fn test_generic_tombstone_newer_wins() {
        let older = Tombstone::generic(HLC { wall_time: 100, logical: 0, node_id: "a".into() });
        let newer = Tombstone::generic(HLC { wall_time: 200, logical: 0, node_id: "b".into() });

        let merged = older.merge(&newer);
        assert_eq!(merged.timestamp.wall_time, 200);
    }

    #[test]
    fn test_resurrect_blocked_by_security_tombstone() {
        let mut state = HistoryState::new();
        // ... setup entry with security tombstone
        // Resurrect operation should be silently ignored
    }
}
```

### Log Replication (Raft-style, but without consensus)

```rust
// sync/log.rs

/// Local operation log - append-only, replicated to peers
///
/// Uses HLC for ordering instead of sequence numbers. This enables
/// "Merge as Single Node" where multiple instances share the same origin.
pub struct OperationLog {
    /// SQLite-backed persistent log
    conn: Connection,
    /// Local node ID (derived from identity key)
    node_id: NodeId,
    /// Hybrid logical clock (monotonically increasing)
    hlc: Mutex<HLC>,
    /// Vector clock: tracks latest HLC seen from each peer
    /// Used for efficient delta sync ("give me everything newer than X")
    vector_clock: RwLock<HashMap<NodeId, HLC>>,
}

/// Log entry for replication - uses HLC for ordering (not seq numbers)
///
/// HLC-based ordering is essential for "Merge as Single Node" where multiple
/// instances share the same origin (node_id). HLC is globally unique and
/// provides total ordering without coordination.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogEntry {
    /// Node that created this entry (same as timestamp.node_id)
    pub origin: NodeId,
    /// HLC timestamp - provides unique identity AND ordering
    /// Note: origin == timestamp.node_id always
    pub timestamp: HLC,
    /// The operation
    pub operation: Operation,
    /// CRC32 checksum for integrity validation (defense-in-depth)
    /// Catches memory corruption, storage bitflips, and transmission errors
    pub checksum: u32,
}

impl LogEntry {
    /// Create a new log entry with computed checksum
    pub fn new(origin: NodeId, timestamp: HLC, operation: Operation) -> Self {
        debug_assert_eq!(origin, timestamp.node_id, "origin must match HLC node_id");
        let mut entry = Self {
            origin,
            timestamp,
            operation,
            checksum: 0,
        };
        entry.checksum = entry.compute_checksum();
        entry
    }

    /// Compute CRC32 checksum over entry contents
    /// Performance: < 1μs per operation (minimal overhead)
    pub fn compute_checksum(&self) -> u32 {
        use crc32fast::Hasher;
        let mut hasher = Hasher::new();

        // Hash all fields in deterministic order
        // Note: origin == timestamp.node_id, so only hash once
        hasher.update(&self.origin.to_le_bytes());
        hasher.update(&self.timestamp.wall_time.to_le_bytes());
        hasher.update(&self.timestamp.logical.to_le_bytes());
        hasher.update(&bincode::serialize(&self.operation).unwrap_or_default());

        hasher.finalize()
    }

    /// Verify checksum matches computed value
    pub fn verify_checksum(&self) -> bool {
        self.checksum == self.compute_checksum()
    }
}

/// Error when checksum validation fails
#[derive(Debug, Error)]
pub enum ChecksumError {
    #[error("Checksum mismatch for entry hlc={hlc:?} origin={origin}: expected {expected}, got {actual}")]
    Mismatch {
        hlc: HLC,
        origin: NodeId,
        expected: u32,
        actual: u32,
    },
}

impl OperationLog {
    /// Create a new operation log
    pub fn new(conn: Connection, node_id: NodeId) -> Self {
        Self {
            conn,
            node_id,
            hlc: Mutex::new(HLC::new(node_id)),
            vector_clock: RwLock::new(HashMap::new()),
        }
    }

    /// Append a new operation (local write)
    pub fn append(&self, op: Operation) -> Result<LogEntry, Error> {
        // Get next HLC timestamp (guaranteed > any previous)
        let timestamp = {
            let mut hlc = self.hlc.lock();
            let ts = hlc.tick(self.node_id)?;
            *hlc = ts;
            ts
        };

        // Create entry with checksum
        let entry = LogEntry::new(self.node_id, timestamp, op);

        // Persist to local log
        // PRIMARY KEY is (origin, timestamp_wall, timestamp_logical) for HLC ordering
        self.conn.execute(
            "INSERT INTO operation_log (origin, timestamp_wall, timestamp_logical, operation, checksum)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                entry.origin,
                entry.timestamp.wall_time,
                entry.timestamp.logical,
                bincode::serialize(&entry.operation)?,
                entry.checksum,
            ],
        )?;

        // Update our own vector clock with latest HLC
        self.vector_clock.write().insert(self.node_id, entry.timestamp);

        Ok(entry)
    }

    /// Get entries after a given vector clock (for replication)
    ///
    /// Vector clock maps NodeId -> latest HLC seen from that node.
    /// Returns all entries with HLC > the given HLC for that origin.
    pub fn entries_since(&self, vector_clock: &HashMap<NodeId, HLC>) -> Result<Vec<LogEntry>, Error> {
        let mut entries = Vec::new();

        // Get entries from each known node that are newer than peer's last seen
        for (node_id, our_latest) in self.vector_clock.read().iter() {
            let peer_latest = vector_clock.get(node_id);

            // If peer hasn't seen any entries from this node, or our latest is newer
            let needs_sync = match peer_latest {
                None => true,
                Some(peer_hlc) => our_latest > peer_hlc,
            };

            if needs_sync {
                // Fetch entries newer than peer's last seen HLC
                let (after_wall, after_logical) = match peer_latest {
                    Some(hlc) => (hlc.wall_time, hlc.logical),
                    None => (0, 0),
                };

                let node_entries: Vec<LogEntry> = self.conn
                    .prepare(
                        "SELECT origin, timestamp_wall, timestamp_logical, operation, checksum
                         FROM operation_log
                         WHERE origin = ?1
                           AND (timestamp_wall > ?2 OR (timestamp_wall = ?2 AND timestamp_logical > ?3))
                         ORDER BY timestamp_wall, timestamp_logical"
                    )?
                    .query_map(params![node_id, after_wall, after_logical], |row| {
                        Ok(LogEntry {
                            origin: row.get(0)?,
                            timestamp: HLC {
                                wall_time: row.get(1)?,
                                logical: row.get(2)?,
                                node_id: row.get(0)?,  // origin == timestamp.node_id
                            },
                            operation: bincode::deserialize(&row.get::<_, Vec<u8>>(3)?).unwrap(),
                            checksum: row.get(4)?,
                        })
                    })?
                    .filter_map(|r| r.ok())
                    .collect();

                entries.extend(node_entries);
            }
        }

        // Sort all entries by HLC for consistent ordering
        entries.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        Ok(entries)
    }

    /// Apply entries from a remote peer
    ///
    /// Validates CRC32 checksums before applying - defense-in-depth against:
    /// - Memory corruption during transmission
    /// - Storage bitflips on remote peer
    /// - Network corruption (additional layer beyond TLS)
    pub fn apply_remote(&self, entries: Vec<LogEntry>, state: &mut HistoryState) -> Result<(), Error> {
        let mut hlc_guard = self.hlc.lock();
        let mut corrupted_count = 0;

        for entry in entries {
            // Verify checksum before applying (defense-in-depth)
            if !entry.verify_checksum() {
                let expected = entry.compute_checksum();
                tracing::error!(
                    "Checksum mismatch for entry origin={} hlc=({}, {}): expected {}, got {}",
                    entry.origin, entry.timestamp.wall_time, entry.timestamp.logical,
                    expected, entry.checksum
                );
                corrupted_count += 1;
                continue;
            }

            // Update our HLC to be at least as advanced as remote
            if let Ok(new_hlc) = hlc_guard.receive(&entry.timestamp, self.node_id) {
                *hlc_guard = new_hlc;
            }

            // Check if we already have this entry (idempotent via HLC)
            let dominated = {
                let vc = self.vector_clock.read();
                if let Some(our_latest) = vc.get(&entry.origin) {
                    entry.timestamp <= *our_latest
                } else {
                    false
                }
            };

            if !dominated {
                // Persist to local log
                // INSERT OR IGNORE handles duplicate HLCs (shouldn't happen, but defensive)
                self.conn.execute(
                    "INSERT OR IGNORE INTO operation_log (origin, timestamp_wall, timestamp_logical, operation, checksum)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        entry.origin,
                        entry.timestamp.wall_time,
                        entry.timestamp.logical,
                        bincode::serialize(&entry.operation)?,
                        entry.checksum,
                    ],
                )?;

                // Apply to CRDT state (idempotent)
                entry.operation.apply(state);

                // Update vector clock with latest HLC from this origin
                let mut vc = self.vector_clock.write();
                let current = vc.entry(entry.origin).or_insert(entry.timestamp);
                if entry.timestamp > *current {
                    *current = entry.timestamp;
                }
            }
        }

        if corrupted_count > 0 {
            tracing::warn!("Skipped {} corrupted entries during sync", corrupted_count);
        }

        Ok(())
    }

    /// Verify integrity of local log entries (for diagnostics)
    pub fn verify_log_integrity(&self) -> Result<IntegrityReport, Error> {
        let mut report = IntegrityReport::default();

        let entries: Vec<LogEntry> = self.conn
            .prepare(
                "SELECT origin, timestamp_wall, timestamp_logical, operation, checksum
                 FROM operation_log"
            )?
            .query_map([], |row| {
                let origin: NodeId = row.get(0)?;
                Ok(LogEntry {
                    origin,
                    timestamp: HLC {
                        wall_time: row.get(1)?,
                        logical: row.get(2)?,
                        node_id: origin,  // origin == timestamp.node_id
                    },
                    operation: bincode::deserialize(&row.get::<_, Vec<u8>>(3)?).unwrap(),
                    checksum: row.get(4)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        for entry in entries {
            report.total_entries += 1;
            if !entry.verify_checksum() {
                report.corrupted_entries.push((entry.origin, entry.timestamp));
            }
        }

        Ok(report)
    }

    /// Get the oldest entry's HLC in the log (for compaction/sync decisions)
    pub fn oldest_entry_hlc(&self) -> Result<HLC, Error> {
        self.conn
            .query_row(
                "SELECT origin, timestamp_wall, timestamp_logical
                 FROM operation_log
                 ORDER BY timestamp_wall ASC, timestamp_logical ASC
                 LIMIT 1",
                [],
                |row| {
                    Ok(HLC {
                        wall_time: row.get(1)?,
                        logical: row.get(2)?,
                        node_id: row.get(0)?,
                    })
                },
            )
            .map_err(Error::from)
    }
}

#[derive(Default, Debug)]
pub struct IntegrityReport {
    pub total_entries: usize,
    /// Corrupted entries: (origin, HLC timestamp)
    pub corrupted_entries: Vec<(NodeId, HLC)>,
}

impl IntegrityReport {
    pub fn is_healthy(&self) -> bool {
        self.corrupted_entries.is_empty()
    }
}
```

### Sync Protocol (Raft-style RPCs, CRDT semantics)

```rust
// sync/protocol.rs

/// Sync messages between peers
#[derive(Clone, Serialize, Deserialize)]
pub enum SyncMessage {
    /// Request entries since a vector clock (HLC-based)
    SyncRequest {
        from_node: NodeId,
        vector_clock: HashMap<NodeId, HLC>,
    },

    /// Response with new entries
    SyncResponse {
        entries: Vec<LogEntry>,
        vector_clock: HashMap<NodeId, HLC>,
    },

    /// Push new entries (eager replication)
    PushEntries {
        entries: Vec<LogEntry>,
    },

    /// Acknowledge received entries
    Ack {
        vector_clock: HashMap<NodeId, HLC>,
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
    ///
    /// The shell provides:
    /// - history_id: UUIDv7 generated by shell (for offline correlation)
    /// - session_id: UUIDv7 generated on shell startup
    pub async fn handle_command(&self, cmd: PendingCommand) -> Result<Uuid, Error> {
        // Use shell-provided history_id (for offline queue correlation)
        // Shell generates this to ensure idempotent replay
        let history_id = cmd.history_id;

        // Get HLC timestamp
        let hlc = self.log.hlc.lock()
            .tick(self.node_id)
            .map_err(|_| Error::HlcOverflow)?;

        // 1. Create CRDT entry
        let entry = CrdtEntry {
            id: history_id,
            hlc,
            origin: self.node_id,
            data: EntryData {
                session_id: cmd.session_id,  // UUIDv7 from shell
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
        self.state.write().entries.insert(history_id, entry);

        // 4. Broadcast to peers (async, fire-and-forget)
        let sync = self.sync.clone();
        tokio::spawn(async move {
            sync.broadcast(log_entry).await;
        });

        Ok(history_id)
    }

    /// Update exit status - LWW semantics
    pub async fn finish_command(&self, history_id: Uuid, exit_status: i32, duration: u64) -> Result<(), Error> {
        let timestamp = self.log.hlc.lock().tick(self.node_id)?;

        let op = Operation::UpdateStatus {
            entry_id: history_id,
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

    /// Delete/forget a command - user-initiated deletions use SecurityTombstone
    ///
    /// SecurityTombstone ensures the entry cannot be resurrected by sync from
    /// other peers, protecting privacy even if other machines still have the entry.
    pub async fn forget_command(&self, uuid: Uuid, security: bool) -> Result<(), Error> {
        let timestamp = self.log.hlc.lock().tick(self.node_id)?;

        // User-initiated deletions default to Security (privacy protection)
        // Only use Generic for programmatic/automatic cleanup
        let tombstone = if security {
            Tombstone::security(timestamp)
        } else {
            Tombstone::generic(timestamp)
        };

        let op = Operation::Delete {
            entry_id: uuid,
            tombstone,
        };

        let log_entry = self.log.append(op.clone())?;

        // Apply locally
        op.apply(&mut self.state.write());

        // Broadcast tombstone to peers
        let sync = self.sync.clone();
        tokio::spawn(async move {
            sync.broadcast(log_entry).await;
        });

        Ok(())
    }

    /// Resurrect a previously deleted entry (only works for GenericTombstone)
    ///
    /// Use case: Undo accidental deletion from auto-cleanup, restore from backup
    /// SecurityTombstone entries cannot be resurrected (privacy protection)
    pub async fn resurrect_command(&self, uuid: Uuid) -> Result<bool, Error> {
        // Check if entry exists and has a non-security tombstone
        let can_resurrect = {
            let state = self.state.read();
            if let Some(entry) = state.entries.get(&uuid) {
                match &entry.deleted {
                    Some(tombstone) => !tombstone.is_permanent(),
                    None => false, // Not deleted, nothing to resurrect
                }
            } else {
                false // Entry doesn't exist
            }
        };

        if !can_resurrect {
            return Ok(false);
        }

        let timestamp = self.log.hlc.lock().tick(self.node_id)?;

        let op = Operation::Resurrect {
            entry_id: uuid,
            timestamp,
        };

        let log_entry = self.log.append(op.clone())?;

        // Apply locally
        op.apply(&mut self.state.write());

        // Broadcast
        let sync = self.sync.clone();
        tokio::spawn(async move {
            sync.broadcast(log_entry).await;
        });

        Ok(true)
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
-- Tombstone reason: 0 = Generic (can resurrect), 1 = Security (permanent)
ALTER TABLE history ADD COLUMN tombstone_reason INTEGER DEFAULT NULL;

-- Operation log (append-only, replicated)
-- Uses HLC as primary key for total ordering (supports "Merge as Single Node")
CREATE TABLE IF NOT EXISTS operation_log (
    origin INTEGER NOT NULL,           -- NodeId of the originating node
    timestamp_wall INTEGER NOT NULL,   -- HLC wall time (milliseconds since epoch)
    timestamp_logical INTEGER NOT NULL, -- HLC logical counter
    operation BLOB NOT NULL,           -- CBOR-encoded Operation
    -- CRC32 checksum for integrity validation (defense-in-depth)
    checksum INTEGER NOT NULL,         -- u32 stored as INTEGER
    PRIMARY KEY (origin, timestamp_wall, timestamp_logical)
);

-- Vector clock state: tracks latest HLC seen from each peer
CREATE TABLE IF NOT EXISTS vector_clock (
    node_id INTEGER PRIMARY KEY,
    hlc_wall INTEGER NOT NULL,
    hlc_logical INTEGER NOT NULL
);

-- Known peers
CREATE TABLE IF NOT EXISTS peers (
    node_id INTEGER PRIMARY KEY,
    addr TEXT NOT NULL,
    last_seen INTEGER
);

-- Index recovery state: tracks high water mark for crash recovery
-- Uses SQLite's INTEGER PRIMARY KEY as implicit sequence (rowid alias)
CREATE TABLE IF NOT EXISTS index_state (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
-- INSERT OR REPLACE INTO index_state (key, value) VALUES ('last_indexed_rowid', ?)

-- Indexes
CREATE INDEX IF NOT EXISTS operation_log_hlc ON operation_log(timestamp_wall, timestamp_logical);
CREATE INDEX IF NOT EXISTS operation_log_origin_hlc ON operation_log(origin, timestamp_wall, timestamp_logical);
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

#### Duplicate Ephemeral Identity: Merge as Single Node

When multiple ephemeral machines (e.g., two Codespaces) share the same portable
identity key and run simultaneously, they are treated as a **single logical node**
with their operations merged together.

**Design Decision: Option C - Merge as Single Node**

All instances sharing a portable identity are merged into one unified node:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           Option C: Merge as Single Node                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   HISTDB_IDENTITY (shared secret)                                            │
│           │                                                                  │
│           ▼                                                                  │
│   ┌───────────────────────────────────────────────────────────┐             │
│   │           SINGLE LOGICAL NODE                              │             │
│   │           node_id = hash(identity_key)                     │             │
│   │                                                            │             │
│   │   Operations from all instances merged into one stream:    │             │
│   │   ┌─────────────────────────────────────────────────────┐ │             │
│   │   │ HLC(100,0): Insert(cmd_A) ← from Codespace A        │ │             │
│   │   │ HLC(100,1): Insert(cmd_B) ← from Codespace B        │ │             │
│   │   │ HLC(101,0): Update(cmd_A) ← from Codespace A        │ │             │
│   │   │ HLC(102,0): Insert(cmd_C) ← from Codespace B        │ │             │
│   │   └─────────────────────────────────────────────────────┘ │             │
│   └───────────────────────────────────────────────────────────┘             │
│                     ▲                           ▲                            │
│           ┌─────────┴─────────┐       ┌─────────┴─────────┐                 │
│           │   Codespace A     │       │   Codespace B     │                 │
│           │   session=01910a  │       │   session=01910b  │                 │
│           │   (contributor)   │       │   (contributor)   │                 │
│           └───────────────────┘       └───────────────────┘                 │
│                                                                              │
│   To static peers: appears as ONE peer with unified operation log           │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Merge Semantics:**

| Aspect | Behavior |
|--------|----------|
| **node_id** | Same for all instances: `hash(identity_key)` |
| **Operation log** | Unified - all instances contribute to same sequence |
| **Sequence numbers** | Coordinated via HLC - no conflicts possible |
| **To peers** | Appears as single node with one operation stream |
| **Sessions** | Distinguishable via unique `session_id` (UUIDv7) |
| **Commands** | Each has unique `history_id` (UUIDv7) - no duplicates |

**How Merge Works:**

```rust
// When syncing to a static peer, all operations from all instances
// with the same identity appear as coming from one node:

// Codespace A writes: Insert(cmd_A) at HLC(100, 0, node_X)
// Codespace B writes: Insert(cmd_B) at HLC(100, 1, node_X)  // Same node_id!
// Codespace A writes: Update(cmd_A) at HLC(101, 0, node_X)

// Static peer sees: single node "node_X" with operations:
//   - Insert(cmd_A)
//   - Insert(cmd_B)
//   - Update(cmd_A)
// All merged into one coherent stream, ordered by HLC
```

**Why Merge as Single Node Is Correct:**

1. **User intent**: Portable identity means "all these machines are me"
2. **Simple mental model**: One identity = one node in the sync topology
3. **Clean peer management**: One peer entry covers all instances
4. **No conflicts**: UUIDs ensure uniqueness, HLC ensures ordering
5. **Unified history**: All commands from all instances in one stream

**Sequence Number Coordination:**

Instances don't need explicit coordination because:
- HLC provides globally unique timestamps (wall_time + logical + node_id)
- Operations are ordered by HLC as primary key (no sequence numbers)
- HLC's logical counter handles concurrent operations at same wall time

```rust
// HLC handles concurrent operations automatically:
// Codespace A: LogEntry { timestamp: HLC(100, 0, node_X), ... }
// Codespace B: LogEntry { timestamp: HLC(100, 1, node_X), ... }
//
// After merge, ordered by HLC (wall_time, logical, node_id):
// 1. HLC(100, 0, node_X) - from A
// 2. HLC(100, 1, node_X) - from B (logical counter incremented)
//
// Primary key is (origin, timestamp_wall, timestamp_logical) - no seq column
```

**Alternatives Considered:**

| Option | Description | Why Rejected |
|--------|-------------|--------------|
| **A: Instance-unique node_id** | Each instance gets unique node_id | Creates N peer entries, confusing |
| **B: Shared node_id, separate logs** | Same node_id but independent logs | Inconsistent, complex merge |
| **C: Merge as single node** | ✅ **Chosen** - unified operation log | Clean, matches user intent |

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
| **Delete vs update** | SecurityTombstone always wins; GenericTombstone can be resurrected |
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
    /// Determines if a log entry can be safely compacted (removed)
    ///
    /// Uses HLC comparison: entry can be compacted when all static peers
    /// have acknowledged operations at or after this entry's timestamp
    pub fn can_compact(&self, entry: &LogEntry, peer_acks: &HashMap<NodeId, HLC>) -> bool {
        // Rule 1: All static peers have ACKed at or past this entry's timestamp
        let static_peers_acked = self.static_peers.iter()
            .all(|peer| {
                peer_acks.get(peer)
                    .map(|ack_hlc| ack_hlc >= &entry.timestamp)
                    .unwrap_or(false)
            });

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
    let their_vc: HashMap<NodeId, HLC> = peer.get_vector_clock().await?;
    let our_oldest_hlc = self.log.oldest_entry_hlc()?;

    // Peer needs full sync if any of their tracked HLCs are older than our oldest log entry
    let needs_full_sync = their_vc.values().any(|hlc| hlc < &our_oldest_hlc);

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
pub struct SyncMessageV1 {
    pub entries: Vec<LogEntry>,
    pub vector_clock: HashMap<NodeId, HLC>,

    #[serde(default)]  // v1.1: older peers ignore
    pub compression: Option<CompressionType>,

    #[serde(default)]  // v1.2: older peers ignore
    pub checksum: Option<u32>,  // CRC32 is u32
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


#### Sync Considerations

| Scenario | Behavior |
|----------|----------|
| **Different retention settings** | Deletes sync (most restrictive wins) |
| **Starred on host A, pruned on B** | Star should sync first; if not, conflict resolution needed |
| **Usage stats sync** | LWW-Register for use_count and last_used |
| **Offline host reconnects** | Receives tombstones, prunes locally |

**Conflict resolution for starred:**

```rust
// Starred flag uses pure LWW with node_id tie-breaker (consistent with HLC ordering)
// No value bias - deterministic based on node_id for reproducible merge results
pub fn merge_starred(local: &LWWRegister<bool>, remote: &LWWRegister<bool>) -> bool {
    match local.timestamp.cmp(&remote.timestamp) {
        Ordering::Greater => local.value,
        Ordering::Less => remote.value,
        Ordering::Equal => {
            // Tie-breaker: higher node_id wins (consistent with HLC Ord)
            // This ensures deterministic merge regardless of which node merges first
            if local.timestamp.node_id >= remote.timestamp.node_id {
                local.value
            } else {
                remote.value
            }
        }
    }
}
```

---

---

<a id="appendix-sync-coupled-fragments-removed-from-v1-docs"></a>
## Appendix: sync-coupled fragments removed from v1 docs

These were embedded in v1 sections of the original ARCHITECTURE.md and were cut when
sync was deferred. They are kept verbatim so the sync design can be restored later.

### From `01-project-structure.md`: sync/ module tree

~~~~text
│   │   │   ├── sync/             # Hybrid sync (log replication + CRDTs)
│   │   │   │   ├── mod.rs
│   │   │   │   ├── hlc.rs        # Hybrid Logical Clock
│   │   │   │   ├── crdt.rs       # CRDT types (G-Set, LWW-Register)
│   │   │   │   ├── log.rs        # Operation log (append-only)
│   │   │   │   ├── protocol.rs   # Sync protocol messages
│   │   │   │   └── peer.rs       # Peer connection management
~~~~

### From `02-daemon-and-ingest.md`: async send: hlc

~~~~text
    local hlc_json=$(_histdb_get_hlc)
~~~~

### From `02-daemon-and-ingest.md`: orphan finish comment: sync

~~~~text
                    // - Sync brought FinishCommand before StartCommand
~~~~

### From `02-daemon-and-ingest.md`: shutdown diagram: best-effort sync box

~~~~text
┌─────────────────────────┐
│ 4. Best-effort sync     │  ← Push pending ops to peers
│    (timeout: 1s)        │
└───────────┬─────────────┘
            ▼
~~~~

### From `02-daemon-and-ingest.md`: ShutdownPhase::SyncPeers

~~~~text
    /// Sync with peers (best effort)
    SyncPeers,
~~~~

### From `02-daemon-and-ingest.md`: ShutdownConfig.sync_timeout

~~~~text
    /// Time for peer sync (best effort)
    pub sync_timeout: Duration,
~~~~

### From `02-daemon-and-ingest.md`: ShutdownConfig default sync_timeout

~~~~text
            sync_timeout: Duration::from_secs(1),
~~~~

### From `02-daemon-and-ingest.md`: shutdown(): sync param

~~~~text
        sync: Option<Arc<PeerSync>>,
~~~~

### From `02-daemon-and-ingest.md`: shutdown phase 5: peer sync

~~~~text
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

~~~~

### From `02-daemon-and-ingest.md`: ShutdownReport.peers_synced

~~~~text
    pub peers_synced: bool,
~~~~

### From `02-daemon-and-ingest.md`: crash recovery: rebuild vector clock

~~~~text
        // 5. Rebuild vector clock from operation log (tracks latest HLC per origin)
        let vector_clock: HashMap<NodeId, HLC> = conn.prepare("
            SELECT origin, timestamp_wall, timestamp_logical
            FROM operation_log o1
            WHERE (timestamp_wall, timestamp_logical) = (
                SELECT MAX(timestamp_wall), MAX(timestamp_logical)
                FROM operation_log o2
                WHERE o2.origin = o1.origin
                  AND o2.timestamp_wall = (
                    SELECT MAX(timestamp_wall) FROM operation_log o3 WHERE o3.origin = o1.origin
                  )
            )
            GROUP BY origin
        ")?.query_map([], |row| {
            let origin: NodeId = row.get(0)?;
            let wall: i64 = row.get(1)?;
            let logical: u32 = row.get(2)?;
            Ok((origin, HLC { wall_time: wall, logical, node_id: origin }))
        })?.collect::<Result<_, _>>()?;

        *self.log.vector_clock.write() = vector_clock;

~~~~

### From `03-storage.md`: migration 4: add_sync_tables

~~~~text
            Migration {
                version: 4,
                name: "add_sync_tables",
                up: include_str!("../migrations/004_sync.up.sql"),
                down: include_str!("../migrations/004_sync.down.sql"),
                destructive: false,
            },
~~~~

### From `03-storage.md`: record_command_use: CRDT usage op

~~~~text
    // Also create CRDT operation to sync usage stats
    let op = Operation::UpdateUsage {
        entry_id: uuid,
        use_count: LWWRegister::new(/* ... */),
        last_used: LWWRegister::new(/* ... */),
    };
    self.log.append(op)?;
~~~~

### From `03-storage.md`: pruner: GenericTombstone via op log

~~~~text
                // Create GenericTombstone for automatic cleanup (can be resurrected)
                let timestamp = self.hlc.lock().tick(self.node_id)?;
                let op = Operation::Delete {
                    entry_id: *uuid,
                    tombstone: Tombstone::generic(timestamp),
                };
                self.log.append(op)?;
~~~~

### From `05-shell-integration.md`: Request::GetHlc

~~~~text
    /// Get current HLC (for offline queue)
    GetHlc,

~~~~

### From `05-shell-integration.md`: Response::Hlc

~~~~text
    /// Current HLC
    Hlc {
        wall_time: i64,
        logical: u32,
        node_id: NodeId,
    },

~~~~

### From `05-shell-integration.md`: spool JSON: hlc

~~~~text
    "hlc": {"wall_time": 1704067200000, "logical": 0, "node_id": 12345678},
~~~~

### From `05-shell-integration.md`: key decision 2: HLC in offline queue

~~~~text
2. **HLC stored in offline queue**: Preserves causal ordering when replaying. Without this, replayed commands would get "wrong" timestamps and break CRDT merge.
~~~~

### From `05-shell-integration.md`: _histdb_get_hlc

~~~~text
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

~~~~

### From `05-shell-integration.md`: spool writer: hlc field

~~~~text
        print -r -- "\"hlc\":${hlc_json},"
~~~~

### From `05-shell-integration.md`: send_or_queue: get HLC

~~~~text
    # Get HLC for this operation
    local hlc_json=$(_histdb_get_hlc)

~~~~

### From `05-shell-integration.md`: cli/offline.rs (SQLite offline_queue + HLC)

~~~~text
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

    /// HLC timestamp as JSON: {"wall_time":..., "logical":..., "node_id":...}
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
            hlc.node_id,
        ],
    )?;

    Ok(())
}

#[derive(Deserialize)]
struct HlcJson {
    wall_time: u64,
    logical: u32,
    node_id: u64,  // NodeId is u64 (hash of identity key)
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

~~~~

### From `05-shell-integration.md`: daemon SpoolEntry.hlc

~~~~text
    hlc: HlcJson,        // Preserved HLC timestamp
~~~~

### From `05-shell-integration.md`: spool replay_entry with HLC

~~~~text
    async fn replay_entry(&self, entry: &SpoolEntry) -> Result<(), Error> {
        // Reconstruct HLC from stored values
        let hlc = HLC {
            wall_time: entry.hlc.wall_time,
            logical: entry.hlc.logical,
            node_id: entry.hlc.node_id.clone(),
        };

        // Parse and create PendingCommand with original HLC
        let cmd = parse_offline_request(&entry.r#type, &entry.payload, hlc)?;

        // Enqueue to normal ingest pipeline
        self.ingest_queue.push(cmd)
            .map_err(|_| Error::QueueFull)?;

        Ok(())
    }
~~~~

### From `05-shell-integration.md`: dedupe rationale: sync from peers

~~~~text
- Sync from peers won't duplicate already-synced entries
~~~~

### From `05-shell-integration.md`: histdb-cli HlcNow subcommand

~~~~text
    /// Get current HLC from daemon (for offline queue)
    HlcNow,

~~~~

### From `05-shell-integration.md`: histdb-cli hlc-now handler

~~~~text
/// Get current HLC from daemon
pub fn handle_hlc_now() -> Result<()> {
    let socket_path = get_socket_path();
    let mut stream = UnixStream::connect(&socket_path)?;
    stream.set_read_timeout(Some(Duration::from_secs(1)))?;

    // Request HLC from daemon
    let request = r#"{"type":"GetHlc"}"#;
    writeln!(stream, "{}", request)?;

    let mut response = String::new();
    BufReader::new(stream).read_line(&mut response)?;

    // Output HLC as JSON for shell to parse
    print!("{}", response.trim());
    Ok(())
}

~~~~

### From `05-shell-integration.md`: handle_queue: get cached HLC

~~~~text
    // Get cached HLC or create local one
    let hlc = get_cached_hlc_or_local();
~~~~

### From `05-shell-integration.md`: handle_queue: hlc field

~~~~text
        hlc,
~~~~

### From `05-shell-integration.md`: SpoolEntry.hlc

~~~~text
    hlc: HLC,
~~~~

### From `05-shell-integration.md`: histdb-cli: HLC cache helpers

~~~~text
/// Get cached HLC or generate local one
/// Maintains causal ordering even when offline by:
/// 1. Caching last HLC received from daemon
/// 2. Incrementing locally when daemon unavailable
fn get_cached_hlc_or_local() -> HLC {
    let cache_path = get_hlc_cache_path();

    // Try to read cached HLC
    if let Ok(cached) = read_cached_hlc(&cache_path) {
        // Increment locally for this operation
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        // HLC tick: if wall time advanced, reset logical; else increment logical
        if now_ms > cached.wall_time {
            return HLC {
                wall_time: now_ms,
                logical: 0,
                node_id: cached.node_id,
            };
        } else {
            // Same or earlier wall time - increment logical counter
            let new_hlc = HLC {
                wall_time: cached.wall_time,
                logical: cached.logical.saturating_add(1),
                node_id: cached.node_id,
            };
            // Update cache with new logical value
            let _ = write_cached_hlc(&cache_path, &new_hlc);
            return new_hlc;
        }
    }

    // No cache - create local HLC (wall time only, no node_id yet)
    // This will be corrected when daemon replays with proper HLC
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;

    HLC {
        wall_time: now_ms,
        logical: 0,
        node_id: 0, // Will be set by daemon on replay
    }
}

/// Cache HLC from daemon response (called after successful GetHlc)
fn cache_hlc_from_daemon(hlc: &HLC) -> Result<()> {
    let cache_path = get_hlc_cache_path();
    write_cached_hlc(&cache_path, hlc)
}

fn get_hlc_cache_path() -> PathBuf {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR")
        .unwrap_or_else(|_| "/tmp".to_string());
    let uid = unsafe { libc::getuid() };
    PathBuf::from(format!("{}/histdb-{}-hlc.json", runtime_dir, uid))
}

fn read_cached_hlc(path: &Path) -> Result<HLC> {
    let content = std::fs::read_to_string(path)?;
    serde_json::from_str(&content).map_err(Error::from)
}

fn write_cached_hlc(path: &Path, hlc: &HLC) -> Result<()> {
    let content = serde_json::to_string(hlc)?;
    std::fs::write(path, content)?;
    Ok(())
}
~~~~

### From `06-cli-config-import.md`: forget/resurrect with Security/Generic tombstones

~~~~text
### CLI Commands for Forget/Resurrect

```rust
// cli/forget.rs

/// Forget (delete) a command from history
///
/// By default uses SecurityTombstone to ensure privacy protection.
/// The --generic flag uses GenericTombstone which can be resurrected.
pub fn handle_forget(
    uuid: Uuid,
    generic: bool,  // --generic flag for resurrectable deletion
    service: &HistoryService,
) -> Result<()> {
    let security = !generic;

    tokio::runtime::Runtime::new()?
        .block_on(service.forget_command(uuid, security))?;

    if security {
        println!("Entry {} permanently deleted (SecurityTombstone)", uuid);
        println!("This deletion will sync to all peers and cannot be undone.");
    } else {
        println!("Entry {} deleted (GenericTombstone)", uuid);
        println!("This entry can be resurrected with: histdb resurrect {}", uuid);
    }

    Ok(())
}

/// Resurrect a previously deleted entry
pub fn handle_resurrect(uuid: Uuid, service: &HistoryService) -> Result<()> {
    let result = tokio::runtime::Runtime::new()?
        .block_on(service.resurrect_command(uuid))?;

    if result {
        println!("Entry {} resurrected successfully", uuid);
    } else {
        println!("Cannot resurrect entry {}:", uuid);
        println!("  - Entry may not exist");
        println!("  - Entry may not be deleted");
        println!("  - Entry may have a SecurityTombstone (cannot be resurrected)");
    }

    Ok(())
}
```
~~~~

### From `07-errors-testing-deps.md`: dependencies: sync crates

~~~~text
# Hybrid sync (CRDT + log replication)
bincode = "1.3"               # Efficient serialization for operation log
crc32fast = "1.3"             # CRC32 checksums for operation log integrity

# Sync security & networking
snow = "0.9"                  # Noise Protocol Framework
ed25519-dalek = "2"           # Ed25519 signatures (SSH key compatible)
x25519-dalek = "2"            # X25519 key exchange
mdns-sd = "0.10"              # mDNS for peer discovery
base64 = "0.21"               # Key encoding for secrets

~~~~
