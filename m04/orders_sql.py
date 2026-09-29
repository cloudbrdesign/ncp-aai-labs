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
        1. a validator: exactly one statement, a SELECT, only the orders and order_items
           tables, no write keywords, a LIMIT (LIMIT 20 is added if missing)
        2. the connection itself is read-only (SQLite "file:...?mode=ro"): even a write
           that got past the validator is refused by the database
      The demo plants a DELETE to show both guards stop it.

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
WRITES = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex)\b", re.I)
MAX_LIMIT = 50
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


def validate(sql: str) -> tuple[str, list[str]]:
    """Return (the query to run, problems). An empty problems list means it may run."""
    q = sql.strip().rstrip(";").strip()
    problems = []
    if ";" in q:
        problems.append("more than one statement")
    if not re.match(r"select\b", q, re.I):
        problems.append("not a SELECT")
    if WRITES.search(q):
        problems.append(f"write keyword {WRITES.search(q).group(1).upper()}")
    tables = {t.lower() for t in re.findall(r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)", q, re.I)}
    if tables - TABLES:
        problems.append(f"unknown table {', '.join(sorted(tables - TABLES))}")
    limit = re.search(r"\blimit\s+(\d+)\s*$", q, re.I)
    if not limit and not problems:
        q += " LIMIT 20"
    elif limit and int(limit.group(1)) > MAX_LIMIT:
        problems.append(f"LIMIT over {MAX_LIMIT}")
    return q, problems


def run_readonly(sql: str) -> list[dict]:
    with closing(connect()) as con:
        return [dict(r) for r in con.execute(sql).fetchall()]


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
    say(f"[guard] PASS: one SELECT on known tables{' (added LIMIT 20)' if q != sql.strip().rstrip(';') else ''}")
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
