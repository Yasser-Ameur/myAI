"""SQLite schema for the cognitive graph.

One local database, versioned. Tables cover the temporal knowledge graph,
memories, personalities, relationships, goals, sources and agent memory.
"""

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    summary TEXT DEFAULT '',
    confidence REAL DEFAULT 0.5,
    importance REAL DEFAULT 0.3,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    valid_from TEXT DEFAULT '',
    valid_to TEXT DEFAULT '',
    is_deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);

CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_id TEXT,
    value TEXT,
    confidence REAL DEFAULT 0.5,
    importance REAL DEFAULT 0.3,
    created_at TEXT NOT NULL,
    valid_from TEXT DEFAULT '',
    valid_to TEXT DEFAULT '',
    source_episode_id TEXT DEFAULT '',
    source_id TEXT DEFAULT '',
    last_confirmed_at TEXT DEFAULT '',
    embedding_id TEXT DEFAULT '',
    is_deleted INTEGER DEFAULT 0,
    provenance TEXT DEFAULT 'conversation'
);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject_id, predicate);
CREATE INDEX IF NOT EXISTS idx_facts_object ON facts(object_id);

CREATE TABLE IF NOT EXISTS relations (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    properties TEXT DEFAULT '{}',
    confidence REAL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    valid_from TEXT DEFAULT '',
    valid_to TEXT DEFAULT '',
    is_deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject_id);
CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(type);

CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    transcript TEXT DEFAULT '[]',
    participants TEXT DEFAULT '[]',
    user_state_before TEXT DEFAULT '{}',
    user_state_after TEXT DEFAULT '{}',
    assistant_state TEXT DEFAULT '{}',
    topics TEXT DEFAULT '[]',
    entities TEXT DEFAULT '[]',
    actions TEXT DEFAULT '[]',
    outcome TEXT DEFAULT '',
    importance REAL DEFAULT 0.3,
    is_consolidated INTEGER DEFAULT 0,
    summary TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_episodes_started ON episodes(started_at);

CREATE TABLE IF NOT EXISTS turns (
    id TEXT PRIMARY KEY,
    episode_id TEXT,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    source TEXT DEFAULT 'text',
    user_state TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_turns_episode ON turns(episode_id);

CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    payload TEXT DEFAULT '{}',
    episode_id TEXT DEFAULT '',
    timestamp TEXT NOT NULL,
    confidence REAL DEFAULT 0.5
);
CREATE INDEX IF NOT EXISTS idx_observations_kind ON observations(kind);
CREATE INDEX IF NOT EXISTS idx_observations_episode ON observations(episode_id);

CREATE TABLE IF NOT EXISTS beliefs (
    id TEXT PRIMARY KEY,
    target_type TEXT NOT NULL,
    target_name TEXT NOT NULL,
    predicate TEXT DEFAULT 'has',
    value TEXT DEFAULT '{}',
    confidence REAL DEFAULT 0.5,
    evidence TEXT DEFAULT '[]',
    status TEXT DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    importance REAL DEFAULT 0.3
);
CREATE INDEX IF NOT EXISTS idx_beliefs_target ON beliefs(target_type, target_name);

CREATE TABLE IF NOT EXISTS personality_traits (
    name TEXT PRIMARY KEY,
    value REAL DEFAULT 0.5,
    confidence REAL DEFAULT 0,
    stability REAL DEFAULT 0.5,
    evidence_count INTEGER DEFAULT 0,
    stability_class TEXT DEFAULT 'very_stable',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS personality_values (
    name TEXT PRIMARY KEY,
    importance REAL DEFAULT 0.5,
    confidence REAL DEFAULT 0,
    stability REAL DEFAULT 0.9,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS personality_preferences (
    name TEXT PRIMARY KEY,
    value REAL DEFAULT 0.5,
    confidence REAL DEFAULT 0,
    stability REAL DEFAULT 0.4,
    evidence_count INTEGER DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS personality_evidence (
    id TEXT PRIMARY KEY,
    target TEXT NOT NULL,
    direction TEXT NOT NULL,
    strength REAL DEFAULT 0.3,
    confidence REAL DEFAULT 0.3,
    source_episode TEXT DEFAULT '',
    source TEXT DEFAULT 'conversation',
    timestamp TEXT NOT NULL,
    context TEXT DEFAULT '',
    kind TEXT DEFAULT 'statement'
);
CREATE INDEX IF NOT EXISTS idx_pe_target ON personality_evidence(target);

CREATE TABLE IF NOT EXISTS contradictions (
    id TEXT PRIMARY KEY,
    statement_a TEXT NOT NULL,
    statement_b TEXT NOT NULL,
    subject TEXT DEFAULT '',
    predicate TEXT DEFAULT '',
    contexts TEXT DEFAULT '[]',
    timestamps TEXT DEFAULT '[]',
    resolution_status TEXT DEFAULT 'unresolved'
);

CREATE TABLE IF NOT EXISTS states (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    snapshot TEXT DEFAULT '{}',
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_states_kind ON states(kind, timestamp);

CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    status TEXT DEFAULT 'active',
    priority REAL DEFAULT 0.5,
    progress REAL DEFAULT 0,
    confidence REAL DEFAULT 0.5,
    source_episode_id TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);

CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    type TEXT DEFAULT 'person',
    name TEXT DEFAULT '',
    trust REAL DEFAULT 0.5,
    familiarity REAL DEFAULT 0,
    emotional_valence REAL DEFAULT 0,
    interaction_count INTEGER DEFAULT 0,
    last_interaction TEXT DEFAULT '',
    important_events TEXT DEFAULT '[]',
    confidence REAL DEFAULT 0.3,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    notes TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_id);

CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT DEFAULT '',
    uri TEXT DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge (
    id TEXT PRIMARY KEY,
    source_id TEXT DEFAULT '',
    source_type TEXT DEFAULT 'document',
    content TEXT NOT NULL,
    title TEXT DEFAULT '',
    confidence REAL DEFAULT 0.5,
    embedding_id TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_knowledge_source ON knowledge(source_id);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    importance REAL DEFAULT 0.3,
    confidence REAL DEFAULT 0.5,
    status TEXT DEFAULT 'candidate',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    accessed_at TEXT DEFAULT '',
    retrieval_count INTEGER DEFAULT 0,
    source_episode_id TEXT DEFAULT '',
    embedding_id TEXT DEFAULT '',
    locked INTEGER DEFAULT 0,
    meta TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);

CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    vector BLOB NOT NULL,
    owner_type TEXT DEFAULT '',
    owner_id TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_id);

CREATE TABLE IF NOT EXISTS agent_memory (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,          -- belief | observation | action | outcome
    content TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_memory_kind ON agent_memory(kind);

CREATE TABLE IF NOT EXISTS system_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""
