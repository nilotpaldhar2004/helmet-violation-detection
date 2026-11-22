"""SQLite storage for runs and violation events."""
import os, sqlite3
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = os.getenv("DB_PATH", str(ROOT / "data" / "events.db"))
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT DEFAULT (datetime('now')),
  source TEXT, filename TEXT, conf REAL, frames INTEGER, fps REAL, processing_ms REAL,
  riders INTEGER, violations INTEGER, compliant INTEGER, compliance_rate REAL, output_file TEXT
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER REFERENCES runs(id) ON DELETE CASCADE,
  created_at TEXT DEFAULT (datetime('now')),
  event_type TEXT, track_id INTEGER, class_name TEXT, confidence REAL,
  frame INTEGER, t_sec REAL, snapshot TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
"""


def conn():
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    return c


def init():
    with closing(conn()) as c:
        c.executescript(SCHEMA)
        c.commit()


def save_run(source, filename, conf, frames, fps, ms, riders, violations, compliant, output_file, events=()):
    rate = round(compliant / riders, 3) if riders else None
    with closing(conn()) as c:
        cur = c.execute(
            "INSERT INTO runs(source,filename,conf,frames,fps,processing_ms,riders,violations,compliant,compliance_rate,output_file)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (source, filename, conf, frames, fps, ms, riders, violations, compliant, rate, output_file))
        run_id = cur.lastrowid
        c.executemany(
            "INSERT INTO events(run_id,event_type,track_id,class_name,confidence,frame,t_sec,snapshot) VALUES(?,?,?,?,?,?,?,?)",
            [(run_id, e.get("event_type", "no_helmet_violation"), e.get("track_id"), e.get("class", "no_helmet"),
              e.get("confidence"), e.get("frame"), e.get("t_sec"), e.get("snapshot")) for e in events])
        c.commit()
    return run_id


def list_runs(limit=50):
    with closing(conn()) as c:
        return [dict(r) for r in c.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))]


def list_events(run_id=None, limit=100):
    q, args = "SELECT * FROM events", []
    if run_id is not None:
        q += " WHERE run_id=?"; args.append(run_id)
    q += " ORDER BY id DESC LIMIT ?"; args.append(limit)
    with closing(conn()) as c:
        return [dict(r) for r in c.execute(q, args)]


def stats():
    with closing(conn()) as c:
        t = c.execute("SELECT COUNT(*) AS runs, COALESCE(SUM(violations),0) AS violations, "
                      "COALESCE(SUM(riders),0) AS riders, COALESCE(SUM(compliant),0) AS compliant FROM runs").fetchone()
        by_source = [dict(r) for r in c.execute(
            "SELECT source, COUNT(*) AS runs, SUM(violations) AS violations FROM runs GROUP BY source")]
    rate = round(t["compliant"] / t["riders"], 3) if t["riders"] else None
    return {"runs": t["runs"], "violations": t["violations"], "riders": t["riders"],
            "compliance_rate": rate, "by_source": by_source}


def clear():
    with closing(conn()) as c:
        files = [r[0] for r in c.execute("SELECT output_file FROM runs WHERE output_file IS NOT NULL")]
        files += [r[0] for r in c.execute("SELECT snapshot FROM events WHERE snapshot IS NOT NULL")]
        c.execute("DELETE FROM events"); c.execute("DELETE FROM runs"); c.commit()
    return files
