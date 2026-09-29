"""Long-term memory: what the desk knows about a customer across all threads.

It lives in a LangGraph store (SqliteStore, file m03/state/memory.db), as JSON documents
under a namespace and a key. One customer has three namespaces:

    ("customers", id, "profile")    semantic memory: facts, kept as ONE document we update
    ("customers", id, "episodes")   episodic memory: past resolved requests, a growing collection
    ("customers", id, "lessons")    reflection memory: one-line lessons from failed drafts, at most 3

Every document has a "text" field, a plain sentence. The NAT memory provider in
m03/desk_memory searches that field, so the toolkit agent can recall the same facts.
Similarity here is plain word overlap: no embedding model needed.
"""
import contextlib
import os
import pathlib
import re
import time

from langgraph.store.sqlite import SqliteStore

HERE = pathlib.Path(__file__).resolve().parent
STATE_DIR = pathlib.Path(os.environ.get("M03_STATE_DIR", HERE / "state"))
MEMORY_DB = STATE_DIR / "memory.db"
MAX_LESSONS = 3
STOPWORDS = {"the", "and", "can", "you", "my", "is", "it", "a", "an", "of", "to", "i", "me", "for", "on",
             "in", "please", "what", "where", "when", "will", "your", "our", "with", "from", "order"}


def ns(customer: str, kind: str) -> tuple:
    return ("customers", customer, kind)


@contextlib.contextmanager
def open_store():
    """Open the SQLite store. from_conn_string sets up the connection the store needs
    (a plain sqlite3.connect() fails with 'cannot start a transaction within a transaction')."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with SqliteStore.from_conn_string(str(MEMORY_DB)) as store:
        store.setup()
        yield store


# ---- semantic memory: the profile -------------------------------------------------

def detect_facts(text: str) -> dict:
    """Simple rules that spot a preference in a customer message (no model needed)."""
    t = text.lower()
    facts = {}
    if re.search(r"e-?mail only|only (by |via )?e-?mail|no (phone )?calls|don'?t (phone|call)", t):
        facts["contact"] = "email"
    elif re.search(r"call me|phone me|by phone", t):
        facts["contact"] = "phone"
    lang = re.search(r"\b(?:in|reply in|answer in) (english|german|french|spanish|italian)\b", t)
    if lang:
        facts["language"] = lang.group(1).capitalize()
    return facts


def profile_text(profile: dict) -> str:
    parts = []
    if profile.get("contact") == "email":
        parts.append("Contact preference: email only, no phone calls.")
    elif profile.get("contact"):
        parts.append(f"Contact preference: {profile['contact']}.")
    if profile.get("language"):
        parts.append(f"Language: {profile['language']}.")
    return " ".join(parts)


def get_profile(store, customer: str) -> dict:
    item = store.get(ns(customer, "profile"), "main")
    return {k: v for k, v in item.value.items() if k != "text"} if item else {}


def save_profile(store, customer: str, facts: dict) -> dict:
    profile = {**get_profile(store, customer), **facts}
    store.put(ns(customer, "profile"), "main", {**profile, "text": profile_text(profile)})
    return profile


# ---- episodic memory: past requests -----------------------------------------------

def words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9-]+", text.lower()) if w not in STOPWORDS and len(w) > 2}


def similarity(a: str, b: str) -> float:
    """Word overlap (Jaccard): shared words / all words. 0 = nothing in common, 1 = same words."""
    wa, wb = words(a), words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def add_episode(store, customer: str, request: str, steps: list[dict], reply: str) -> str:
    key = f"ep-{time.time_ns()}"
    done = ", ".join(f"{s['action']} {s.get('order_id', '')}".strip() for s in steps)
    store.put(ns(customer, "episodes"), key, {
        "request": request, "steps": steps, "reply": reply,
        "text": f"Past request: {request} Steps: {done}. Reply sent: {reply}"})
    return key


def episodes(store, customer: str) -> list[dict]:
    return [i.value for i in store.search(ns(customer, "episodes"), limit=100)]


def similar_episode(store, customer: str, request: str) -> dict | None:
    """The past episode whose request shares the most words with this one."""
    best, best_score = None, 0.0
    for episode in episodes(store, customer):
        score = similarity(request, episode["request"])
        if score > best_score:
            best, best_score = episode, score
    return {**best, "score": round(best_score, 2)} if best else None


# ---- reflection memory: lessons (bounded, like Reflexion) --------------------------

def lessons(store, customer: str) -> list[dict]:
    items = [i.value | {"key": i.key} for i in store.search(ns(customer, "lessons"), limit=100)]
    return sorted(items, key=lambda v: v["at"])


def add_lesson(store, customer: str, text: str) -> list[dict]:
    """Add a one-line lesson; keep only the newest MAX_LESSONS (the oldest drops off)."""
    for old in lessons(store, customer):
        if old["text"] == text:            # same lesson again: refresh it instead of duplicating
            store.delete(ns(customer, "lessons"), old["key"])
    store.put(ns(customer, "lessons"), f"lesson-{time.time_ns()}", {"text": text, "at": time.time()})
    current = lessons(store, customer)
    for old in current[:-MAX_LESSONS]:
        store.delete(ns(customer, "lessons"), old["key"])
    return current[-MAX_LESSONS:]
