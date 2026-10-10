"""Regression tests for the free-SQL guard (orders_sql.validate / run_readonly).

    python m04/tests/test_sql_guard.py        # no model, no network: SQLite only

Added after the October 2026 technical review: a regex over the SQL text let quoted
identifiers ("sqlite_master") and comma joins past the table check, a LIMIT in a trailing
comment counted as the cap, harmless reads such as replace() were refused, and Module 9's
caller scope (TEMP views) could be skipped by naming main.orders directly. The guard is now
SQLite's authorizer plus a structural row cap; these cases keep it that way.
"""
import pathlib
import sys
from contextlib import closing

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import orders_sql  # noqa: E402

FAILS = []


def expect(label: str, ok: bool, detail="") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        FAILS.append(label)


def run(sql: str):
    q, problems = orders_sql.validate(sql)
    return problems, (None if problems else orders_sql.run_readonly(q))


# ---- legitimate reads must run ------------------------------------------------------------
ALLOWED = [
    ("plain select", "SELECT order_id, status FROM orders", 8),
    ("join", "SELECT o.order_id, i.product_code FROM orders o JOIN order_items i ON i.order_id = o.order_id", 8),
    ("comma join of the two allowed tables", "SELECT o.order_id FROM orders o, order_items i WHERE i.order_id = o.order_id", 8),
    ("replace() is a function, not a write", "SELECT replace(note, 'a', 'b') AS n FROM orders", 8),
    ("a write word inside a string literal", "SELECT order_id FROM orders WHERE note = 'update, please'", 0),
    ("LIMIT with OFFSET", "SELECT order_id FROM orders ORDER BY order_id LIMIT 3 OFFSET 2", 3),
    ("comma-style LIMIT", "SELECT order_id FROM orders ORDER BY order_id LIMIT 2, 3", 3),
    ("a CTE over an allowed table", "WITH p AS (SELECT * FROM orders WHERE status = 'processing') SELECT order_id FROM p", None),
    ("trailing semicolon", "SELECT count(*) AS n FROM orders;", 1),
]
for label, sql, n in ALLOWED:
    problems, rows = run(sql)
    expect(f"allowed: {label}", not problems and rows is not None and (n is None or len(rows) == n),
           f"problems={problems} rows={None if rows is None else len(rows)}")

# ---- the row cap is structural --------------------------------------------------------------
problems, rows = run("SELECT o.order_id FROM orders o, order_items a, order_items b LIMIT 999")
expect(f"a LIMIT 999 still returns at most {orders_sql.ROW_CAP} rows", not problems and len(rows) == orders_sql.ROW_CAP,
       f"{problems} {rows and len(rows)}")
problems, rows = run(f"SELECT o.order_id FROM orders o, order_items a, order_items b -- LIMIT 5\n")
expect("a LIMIT in a comment is just a comment (cap still applies)", not problems and len(rows) == orders_sql.ROW_CAP,
       f"{problems} {rows and len(rows)}")

# ---- bypass attempts must be refused --------------------------------------------------------
BLOCKED = [
    ("quoted system table", 'SELECT * FROM "sqlite_master" LIMIT 20'),
    ("system table in a comma join", "SELECT * FROM orders, sqlite_master"),
    ("system table in a subquery", "SELECT (SELECT sql FROM sqlite_master LIMIT 1) AS s FROM orders"),
    ("table-valued pragma", "SELECT * FROM pragma_table_info('orders')"),
    ("DELETE", "DELETE FROM orders WHERE order_id = 'A1001'"),
    ("two statements", "SELECT 1; DROP TABLE orders"),
    ("PRAGMA", "PRAGMA table_info(orders)"),
    ("ATTACH", "ATTACH DATABASE 'x.db' AS x"),
]
for label, sql in BLOCKED:
    problems, _ = run(sql)
    expect(f"blocked: {label}", bool(problems), "it ran")

# ---- Module 9's caller scope: base tables only through the TEMP views ------------------------
_orig = orders_sql.connect


def scoped_connect(name="Tom B."):
    """The same TEMP views as m09/guarded_desk.scoped_connect()."""
    con = _orig()
    con.execute(f"CREATE TEMP VIEW orders AS SELECT * FROM main.orders WHERE customer = '{name}'")
    con.execute("CREATE TEMP VIEW order_items AS SELECT * FROM main.order_items WHERE order_id IN "
                f"(SELECT order_id FROM main.orders WHERE customer = '{name}')")
    return con


orders_sql.connect = scoped_connect
try:
    problems, rows = run("SELECT DISTINCT customer FROM orders")
    expect("scoped: the view shows only the caller's orders", not problems and [r["customer"] for r in rows] == ["Tom B."],
           f"{problems} {rows}")
    problems, rows = run("SELECT DISTINCT o.customer FROM order_items i JOIN orders o ON o.order_id = i.order_id")
    expect("scoped: a join of the two views stays in scope", not problems and {r["customer"] for r in rows} == {"Tom B."},
           f"{problems} {rows}")
    for label, sql in [
        ("main.orders by name", "SELECT DISTINCT customer FROM main.orders"),
        ("quoted \"main\".\"orders\"", 'SELECT DISTINCT customer FROM "main"."orders"'),
        ("main.orders in a comma join", "SELECT DISTINCT m.customer FROM orders o, main.orders m"),
        ("main.orders inside a CTE", "WITH x AS (SELECT * FROM main.orders) SELECT DISTINCT customer FROM x"),
        ("main.order_items in a subquery", "SELECT (SELECT count(*) FROM main.order_items) AS n FROM orders"),
    ]:
        problems, rows = run(sql)
        expect(f"scoped: blocked {label}", bool(problems), f"ran: {rows}")
    # run_readonly enforces the same rule on its own (it is the last line, not the validator)
    with closing(scoped_connect()) as con:
        orders_sql.guard(con)
        try:
            con.execute("SELECT customer FROM main.orders").fetchall()
            leaked = True
        except Exception:
            leaked = False
    expect("scoped: the authorizer refuses main.orders even without the validator", not leaked)
finally:
    orders_sql.connect = _orig

print(f"\n{'OK' if not FAILS else 'FAILED'}: {len(FAILS)} failure(s)")
sys.exit(1 if FAILS else 0)
