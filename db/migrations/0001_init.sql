-- 0001_init.sql — BeevR for Legal core schema + matter isolation (RLS).
-- Materializes doc 12 §2 (DDL) and doc 16 §2 (RLS defense-in-depth).
-- Forward-only. Runs offline in the customer VPC (doc 20 §5.1).
--
-- Isolation is enforced in TWO layers:
--   (1) the application data-access layer (beevr/store.py) — the primary filter
--   (2) Postgres RLS below — defense in depth behind the app filter (doc 12 §3)
-- Session context the app must set per request:
--   SET app.current_matter = '<matter_id>';
--   SET app.matter_grants  = '<csv of granted matter_ids>';

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;   -- pgvector (doc 12 §1)

-- ---- scoping -------------------------------------------------------------
CREATE TABLE matter (
  id                 TEXT PRIMARY KEY,               -- ULID
  client             TEXT NOT NULL,
  name               TEXT NOT NULL,
  status             TEXT NOT NULL DEFAULT 'active',
  ethical_wall_group TEXT,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE document (
  id            TEXT PRIMARY KEY,
  matter_id     TEXT NOT NULL REFERENCES matter(id),
  source        TEXT NOT NULL,                        -- upload|imanage|netdocs|m365
  mime          TEXT NOT NULL,
  sha256        TEXT NOT NULL,
  object_key    TEXT NOT NULL,
  ocr           BOOLEAN NOT NULL DEFAULT false,
  ingest_status TEXT NOT NULL DEFAULT 'queued',       -- queued|processing|done|failed
  ingested_at   TIMESTAMPTZ,
  UNIQUE (matter_id, sha256)
);

CREATE TABLE chunk (
  id          TEXT PRIMARY KEY,
  document_id TEXT NOT NULL REFERENCES document(id),
  matter_id   TEXT NOT NULL,                          -- denormalized for isolation
  locator     JSONB NOT NULL,
  text        TEXT NOT NULL,
  embedding   VECTOR(1024),                           -- bge-m3 dim (doc 14)
  ts          TSVECTOR
);
CREATE INDEX chunk_hnsw   ON chunk USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunk_ts_gin ON chunk USING gin (ts);
CREATE INDEX chunk_matter ON chunk (matter_id);       -- isolation-critical

CREATE TABLE entity (
  id        TEXT PRIMARY KEY,
  matter_id TEXT NOT NULL,
  type      TEXT NOT NULL,                             -- party|person|date|obligation|...
  value     TEXT NOT NULL
);
CREATE TABLE edge (
  from_entity     TEXT NOT NULL REFERENCES entity(id),
  rel             TEXT NOT NULL,
  to_entity       TEXT NOT NULL REFERENCES entity(id),
  source_chunk_id TEXT NOT NULL REFERENCES chunk(id),  -- mandatory provenance (doc 22 §3)
  matter_id       TEXT NOT NULL
);

CREATE TABLE answer (
  id         TEXT PRIMARY KEY,
  matter_id  TEXT NOT NULL,
  user_id    TEXT NOT NULL,
  query      TEXT NOT NULL,
  confidence TEXT NOT NULL,
  abstained  BOOLEAN NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE citation (
  id          TEXT PRIMARY KEY,
  answer_id   TEXT NOT NULL REFERENCES answer(id),
  matter_id   TEXT NOT NULL,
  document_id TEXT NOT NULL,
  chunk_id    TEXT NOT NULL,
  locator     JSONB NOT NULL,
  verified    BOOLEAN NOT NULL
);

-- ---- audit (append-only, hash-chained — doc 16 §3) -----------------------
CREATE TABLE audit_event (
  id           TEXT PRIMARY KEY,
  seq          BIGSERIAL,
  matter_id    TEXT,
  actor        TEXT NOT NULL,
  actor_kind   TEXT NOT NULL,                          -- user|agent
  type         TEXT NOT NULL,
  target       TEXT,
  payload_hash TEXT NOT NULL,
  prev_hash    TEXT NOT NULL,
  this_hash    TEXT NOT NULL,
  ts           TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---- roles ---------------------------------------------------------------
-- app_role is what the application connects as; it may never UPDATE/DELETE audit.
DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'app_role') THEN
    CREATE ROLE app_role NOLOGIN;
  END IF;
END $$;
REVOKE UPDATE, DELETE ON audit_event FROM app_role;    -- append-only (FR-AU-02)

-- ---- Row-Level Security (defense in depth, doc 16 §2) --------------------
-- Every matter-scoped table only exposes rows for matters the session was granted.
CREATE OR REPLACE FUNCTION current_matter_grants() RETURNS TEXT[]
  LANGUAGE sql STABLE AS
$$ SELECT string_to_array(current_setting('app.matter_grants', true), ',') $$;

DO $$
DECLARE t TEXT;
BEGIN
  FOREACH t IN ARRAY ARRAY['chunk','entity','edge','answer','citation','document']
  LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
    EXECUTE format($f$
      CREATE POLICY %1$s_matter_isolation ON %1$I
        USING (matter_id = ANY (current_matter_grants()))
        WITH CHECK (matter_id = ANY (current_matter_grants()))
    $f$, t);
  END LOOP;
END $$;

COMMIT;
