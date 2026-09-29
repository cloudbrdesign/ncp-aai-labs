"""Build the orders database for Module 4: m04/data/orders.db (SQLite).

    python m04/make_orders_db.py

In Modules 1 to 3 the orders lived in m01/data/orders.json. M4 moves them into a real
database with two tables, so the desk can look up structured facts with SQL:

    orders       one row per order: status, carrier, ETA, delivery
    order_items  what was in the order, with the product code (H200, D300, M270)

A1001 to A1003 are the M1 orders, unchanged; A1004 to A1008 are new. The product code is
the link between the two kinds of knowledge: SQL tells us that order A1002 was a D300
dock, and the D300 code then filters the manual search to the dock's manual.
The file is rebuilt from scratch every time (the data is made up for the course).
"""
import pathlib
import sqlite3

DB = pathlib.Path(__file__).resolve().parent / "data" / "orders.db"

ORDERS = [
    # order_id, customer, status, carrier, eta, delivered, note
    ("A1001", "Priya N.", "shipped", "DHL", "2 business days", None, None),
    ("A1002", "Tom B.", "delivered", None, None, "3 days ago", None),
    ("A1003", "Lena K.", "processing", None, None, None, "awaiting stock"),
    ("A1004", "Marco D.", "delivered", None, None, "12 days ago", None),
    ("A1005", "Tom B.", "shipped", "UPS", "4 business days", None, None),
    ("A1006", "Aisha R.", "delivered", None, None, "40 days ago", None),
    ("A1007", "Lena K.", "processing", None, None, None, "payment check"),
    ("A1008", "Marco D.", "shipped", "DHL", "1 business day", None, None),
]
ITEMS = [
    # order_id, product_code, item, qty
    ("A1001", "H200", "Wireless headset", 1),
    ("A1002", "D300", "USB-C dock", 1),
    ("A1003", "M270", "4K monitor", 1),
    ("A1004", "H200", "Wireless headset", 2),
    ("A1005", "M270", "4K monitor", 1),
    ("A1006", "D300", "USB-C dock", 1),
    ("A1007", "D300", "USB-C dock", 1),
    ("A1008", "M270", "4K monitor", 1),
]

SCHEMA = """
CREATE TABLE orders (
    order_id  TEXT PRIMARY KEY,
    customer  TEXT NOT NULL,
    status    TEXT NOT NULL CHECK (status IN ('processing', 'shipped', 'delivered')),
    carrier   TEXT,
    eta       TEXT,
    delivered TEXT,
    note      TEXT
);
CREATE TABLE order_items (
    order_id     TEXT NOT NULL REFERENCES orders(order_id),
    product_code TEXT NOT NULL,
    item         TEXT NOT NULL,
    qty          INTEGER NOT NULL
);
"""


def build(path: pathlib.Path = DB) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    with con:   # one transaction: committed at the end of the block
        con.executescript(SCHEMA)
        con.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)", ORDERS)
        con.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?)", ITEMS)
    con.close()
    return path


if __name__ == "__main__":
    p = build()
    print(f"[INFO] wrote {p.relative_to(p.parents[2])}: {len(ORDERS)} orders, {len(ITEMS)} order items")
