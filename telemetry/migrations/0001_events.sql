-- One row per beacon the collector accepts. Analytics Engine drops data after three months; this
-- table keeps it, and the dashboard reads it. payload is the sanitized beacon as JSON, so a new
-- client field needs no migration: query it with payload ->> '$.field'.
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY,
  received_at TEXT NOT NULL,              -- ISO 8601 UTC, stamped by the Worker
  install_id TEXT NOT NULL DEFAULT '',
  session_id TEXT NOT NULL DEFAULT '',
  event TEXT NOT NULL DEFAULT '',         -- run | crash | session | command | launch
  interface TEXT NOT NULL DEFAULT '',     -- cli | gui
  app_version TEXT NOT NULL DEFAULT '',
  schema_version INTEGER NOT NULL DEFAULT 0,
  payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS events_received_at ON events (received_at);
CREATE INDEX IF NOT EXISTS events_install ON events (install_id, received_at);
