"""Step 5: a read-only SQL tool over the orders database (m04/data/orders.db).

    python m04/orders_sql.py A1003                         # named query: one order
    python m04/orders_sql.py --customer "Tom B."           # named query: a customer's orders
    python m04/orders_sql.py --free-sql "Which orders are still processing?"

Two ways to let an agent read a database:

  named queries (what the desk uses)
      A few fixed, parameterised queries: order_status(order_id), order_product(order_id),
      orders_for_customer(name). The model only supplies a value such as "A1003"; SQLite
      binds it with "?", so it can never change the query. Safe and predictable, and all
      a 3B model can be trusted with, but it only answers the questions we planned for.

  model-written SQL (--free-sql, for comparison)
      The model writes a SELECT from the table schema. Flexible, but now model output
      runs against your data, so it passes two guards:
        1. a validator: exactly one statement, a SELECT, no write statements, and only the
           orders and order_items tables. The table rule is enforced by SQLite itself
           (Connection.set_authorizer): while the query is compiled, SQLite reports every
           table and column it would read, and anything else is denied. A regex over the
           SQL text can't do that: a quoted name ("sqlite_master") or a second table in a
           comma join slips past it. The rows are capped structurally (the query runs as
           SELECT * FROM (<query>) LIMIT 20), so a LIMIT in a comment can't lift the cap.
        2. the connection itself is read-only (SQLite "file:...?mode=ro"): even a write
           that got past the validator is refused by the database
      The demo plants a DELETE to show both guards stop it.

Knowing an order ID is not permission to read that order. These named queries answer for
any ID; Module 9 adds the caller's identity and scopes every query to the caller's orders.

Every connection here is read-only; only make_orders_db.py writes the file.
"""
import argparse
import pathlib
from contextlib import closing
import re
import sqlite3

from pydantic import BaseModel, Field

import make_orders_db
from llm_calls import chat, describe

HERE = pathlib.Path(__file__).resolve().parent
DB = make_orders_db.DB
SQL_PROMPT = (HERE / "prompts" / "sql.txt").read_text()
TABLES = {"orders", "order_items"}
# write statements, as keywords outside string literals; replace( the function is a read
WRITES = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex)\b(?!\s*\()", re.I)
ROW_CAP = 20             # every model-written query returns at most this many rows
PLANTED = "DELETE FROM orders WHERE order_id = 'A1001'"


def connect() -> sqlite3.Connection:
    """Open orders.db read-only. Builds the file first if it doesn't exist yet."""
    if not DB.exists():
        make_orders_db.build()
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


# ---- named, parameterised queries (the desk's SQL tool) --------------------------

def order_status(order_id: str) -> dict | None:
    """One order with its product: the facts the desk needs for an order question."""
    with closing(connect()) as con:
        row = con.execute(
            "SELECT o.order_id, o.customer, o.status, o.carrier, o.eta, o.delivered, o.note, "
            "i.product_code AS product, i.item FROM orders o JOIN order_items i ON i.order_id = o.order_id "
            "WHERE o.order_id = ?", (order_id.upper(),)).fetchone()
    return {k: row[k] for k in row.keys() if row[k] is not None} if row else None


def order_product(order_id: str) -> str | None:
    """The product code of an order, e.g. A1002 -> D300 (used to filter the manual search)."""
    with closing(connect()) as con:
        row = con.execute("SELECT product_code FROM order_items WHERE order_id = ?", (order_id.upper(),)).fetchone()
    return row["product_code"] if row else None


def orders_for_customer(name: str) -> list[dict]:
    with closing(connect()) as con:
        rows = con.execute(
            "SELECT o.order_id, o.status, i.product_code FROM orders o JOIN order_items i ON i.order_id = o.order_id "
            "WHERE o.customer = ? ORDER BY o.order_id", (name,)).fetchall()
    return [dict(r) for r in rows]


# ---- model-written SQL, behind a validator and a read-only connection -------------

class SqlQuery(BaseModel):
    sql: str = Field(description="one SQLite SELECT query ending with LIMIT 20")


def _strip_literals(q: str) -> str:
    """The query with string literals and comments blanked, so keywords inside them don't count."""
    return re.sub(r"'(?:[^']|'')*'|--[^\n]*|/\*.*?(?:\*/|$)", " ", q, flags=re.S)


def _authorizer(shadowed: set[str], denied: list[str]):
    """SQLite's own access check for model-written SQL.

    Allowed: the SELECT itself, functions, and reads of the orders and order_items tables.
    When a TEMP view shadows a table (Module 9's caller scope), the base table may only be
    read through that view: a direct read of main.orders is denied, so the scope is the
    boundary rather than a naming convention. Everything else (sqlite_master, PRAGMA, ATTACH,
    any write) is denied."""
    def check(action, arg1, arg2, db, inner):
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ and arg1 in TABLES:
            # db is None when no column is read (count(*), a table only joined): it can only be
            # main then, and under a scope that still means "not through the view": denied
            if db == "temp" or inner in shadowed or (db in ("main", None) and arg1 not in shadowed):
                return sqlite3.SQLITE_OK
            denied.append(f"{db or 'main'}.{arg1} directly (only through your scoped view)")
            return sqlite3.SQLITE_DENY
        denied.append(f"table {arg1}" if action == sqlite3.SQLITE_READ else f"operation {action}")
        return sqlite3.SQLITE_DENY
    return check


def guard(con: sqlite3.Connection) -> list[str]:
    """Install the authorizer on `con`; returns the list it appends denials to."""
    shadowed = {r[0] for r in con.execute("SELECT name FROM sqlite_temp_master WHERE type = 'view'")}
    denied: list[str] = []
    con.set_authorizer(_authorizer(shadowed, denied))
    return denied


def capped(q: str) -> str:
    """The query as it runs: wrapped, so its own LIMIT (or a LIMIT in a comment) can't lift the cap."""
    return f"SELECT * FROM (\n{q}\n) LIMIT {ROW_CAP}"


def validate(sql: str) -> tuple[str, list[str]]:
    """Return (the query to run, problems). An empty problems list means it may run."""
    q = sql.strip().rstrip(";").strip()
    bare = _strip_literals(q)
    problems = []
    if ";" in bare:
        problems.append("more than one statement")
    if not re.match(r"\s*(select|with)\b", bare, re.I):
        problems.append("not a SELECT")
    if WRITES.search(bare):
        problems.append(f"write keyword {WRITES.search(bare).group(1).upper()}")
    if problems:
        return q, problems
    with closing(connect()) as con:      # compile it (no rows read) with the authorizer on
        denied = guard(con)
        try:
            con.execute("EXPLAIN " + capped(q))
        except sqlite3.DatabaseError as e:
            problems.append(f"not allowed: {', '.join(dict.fromkeys(denied))}" if denied else f"SQLite: {e}")
    return capped(q), problems


def run_readonly(sql: str) -> list[dict]:
    """Run a query on a read-only connection, with the authorizer on and at most ROW_CAP rows."""
    with closing(connect()) as con:
        guard(con)
        return [dict(r) for r in con.execute(sql).fetchmany(ROW_CAP)]


def free_sql(question: str, say=print) -> dict:
    """The model writes a SELECT; the validator and the read-only connection guard it."""
    try:
        sql = chat([("system", SQL_PROMPT), ("user", question)], schema=SqlQuery).sql
    except Exception as e:
        say(f"[sql] the model gave no usable query ({type(e).__name__}); nothing runs")
        return {"sql": None, "rows": None}
    say(f"[sql] model wrote: {sql}")
    q, problems = validate(sql)
    if problems:
        say(f"[guard] BLOCKED by the validator: {'; '.join(problems)}")
        return {"sql": sql, "rows": None}
    say(f"[guard] PASS: one SELECT on known tables (at most {ROW_CAP} rows)")
    try:
        rows = run_readonly(q)
    except sqlite3.Error as e:
        say(f"[sql] SQLite error: {e}")
        return {"sql": q, "rows": None}
    for r in rows[:8]:
        say("   " + ", ".join(f"{k}={v}" for k, v in r.items()))
    say(f"[sql] {len(rows)} row{'s' if len(rows) != 1 else ''}")
    return {"sql": q, "rows": rows}


def planted_write(say=print) -> dict:
    """Show both guards refusing a write."""
    say(f"[INFO] planted query: {PLANTED}")
    _, problems = validate(PLANTED)
    say(f"[guard] BLOCKED by the validator: {'; '.join(problems)}")
    say("[INFO] now skip the validator and send it straight to the read-only connection")
    try:
        with closing(connect()) as con:
            con.execute(PLANTED)
        refused = None
    except sqlite3.OperationalError as e:
        refused = str(e)
    say(f"[guard] BLOCKED by SQLite: {refused}" if refused else "[FAIL] the write went through")
    still = order_status("A1001") is not None
    say(f"[INFO] order A1001 is {'still there' if still else 'GONE'}")
    return {"validator": problems, "sqlite": refused, "intact": still}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("order_id", nargs="?", help="an order ID such as A1003")
    ap.add_argument("--customer", help="list a customer's orders, e.g. \"Tom B.\"")
    ap.add_argument("--free-sql", metavar="QUESTION", help="let the model write the SELECT (guarded)")
    a = ap.parse_args()
    if a.free_sql:
        print(f"[INFO] model: {describe()} | database: {DB.relative_to(HERE.parent)} (read-only)")
        print(f"Question: {a.free_sql}")
        free_sql(a.free_sql)
        print()
        planted_write()
    elif a.customer:
        rows = orders_for_customer(a.customer)
        print(f"[sql] orders_for_customer({a.customer!r}): {len(rows)} orders")
        for r in rows:
            print("   " + ", ".join(f"{k}={v}" for k, v in r.items()))
    else:
        oid = a.order_id or "A1003"
        row = order_status(oid)
        print(f"[sql] order_status({oid!r}) -> {len(row) if row else 0} fields")
        items = [f"{k}={v}" for k, v in (row or {}).items()]
        for i in range(0, len(items), 4):
            print("   " + ", ".join(items[i:i + 4]))
        print(f"[sql] order_product({oid!r}) -> {order_product(oid)}")


if __name__ == "__main__":
    main()
