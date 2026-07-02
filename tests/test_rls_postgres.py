"""Postgres RLS integration test — the SECOND isolation layer (doc 16 §2).

Verifies db/migrations/0001_init.sql actually blocks cross-matter rows at the
database, independent of the app filter. Skips cleanly when psycopg or a test
database is unavailable, so the offline suite stays green; CI with a Postgres
service runs it for real.

    DATABASE_URL=postgresql://postgres:postgres@localhost/beevr_test  \
        python -m pytest tests/test_rls_postgres.py
"""
import os
import pathlib

import pytest

psycopg = pytest.importorskip("psycopg")

DB_URL = os.environ.get("DATABASE_URL")
if not DB_URL:
    pytest.skip("DATABASE_URL not set — RLS integration test skipped",
                allow_module_level=True)

MIGRATION = (pathlib.Path(__file__).parent.parent
             / "db" / "migrations" / "0001_init.sql").read_text(encoding="utf-8")


@pytest.fixture()
def conn():
    c = psycopg.connect(DB_URL, autocommit=True)
    with c.cursor() as cur:
        # isolated schema so the test is repeatable
        cur.execute("DROP SCHEMA IF EXISTS beevr_test CASCADE; "
                    "CREATE SCHEMA beevr_test; SET search_path TO beevr_test;")
        cur.execute(MIGRATION)
        # seed two matters with identical chunk text (grant both so WITH CHECK passes)
        cur.execute("SET app.matter_grants = 'A,B'")
        cur.execute("INSERT INTO matter(id,client,name) VALUES "
                    "('A','X','Facility A'),('B','Y','Facility B')")
        cur.execute("INSERT INTO document(id,matter_id,source,mime,sha256,object_key) "
                    "VALUES ('dA','A','upload','application/pdf','h1','kA'),"
                    "       ('dB','B','upload','application/pdf','h2','kB')")
        cur.execute("INSERT INTO chunk(id,document_id,matter_id,locator,text) VALUES "
                    "('cA','dA','A','{}','leverage ratio below 3.0x'),"
                    "('cB','dB','B','{}','leverage ratio below 3.0x')")
    yield c
    c.close()


def test_rls_scopes_select_to_granted_matter(conn):
    with conn.cursor() as cur:
        cur.execute("SET search_path TO beevr_test")
        cur.execute("SET app.matter_grants = 'A'")
        cur.execute("SELECT id FROM chunk ORDER BY id")
        assert [r[0] for r in cur.fetchall()] == ["cA"]      # cB hidden by RLS

        # even an explicit cross-matter predicate returns nothing (TC-207 at DB layer)
        cur.execute("SELECT count(*) FROM chunk WHERE matter_id = 'B'")
        assert cur.fetchone()[0] == 0

        cur.execute("SET app.matter_grants = 'B'")
        cur.execute("SELECT id FROM chunk ORDER BY id")
        assert [r[0] for r in cur.fetchall()] == ["cB"]


def test_rls_denies_write_outside_grant(conn):
    with conn.cursor() as cur:
        cur.execute("SET search_path TO beevr_test")
        cur.execute("SET app.matter_grants = 'A'")
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute("INSERT INTO chunk(id,document_id,matter_id,locator,text) "
                        "VALUES ('cX','dB','B','{}','sneaky')")   # WITH CHECK blocks B
