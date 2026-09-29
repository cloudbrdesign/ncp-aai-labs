"""A NeMo Agent Toolkit memory provider over the desk's own memory file (m03/state/memory.db).

NAT's memory tools (add_memory, get_memory) talk to a MemoryEditor: add_items(),
search() and remove_items(). NAT 1.9.0 ships no local provider (Mem0, Redis, Zep and
MemMachine are external services), so this one reads and writes the same LangGraph
SqliteStore as m03/desk_graph.py. A preference the LangGraph desk saved is found by the
NAT agent, with no embedding model: search is plain word overlap on each item's "text".

After `pip install -e m03/desk_memory` a config can say:

    memory:
      desk:
        _type: desk_memory
"""
import os
import pathlib
import re
import uuid

from langgraph.store.sqlite import SqliteStore
from nat.plugin_api import Builder, MemoryBaseConfig, MemoryEditor, MemoryItem, register_memory

M03 = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_DB = pathlib.Path(os.environ.get("M03_STATE_DIR", M03 / "state")) / "memory.db"   # m03/state/memory.db


class DeskMemoryConfig(MemoryBaseConfig, name="desk_memory"):
    """Local memory for the course's support desk: SQLite file, keyword search, no embedder."""
    db_path: str = str(DEFAULT_DB)


STOPWORDS = {"the", "and", "how", "what", "should", "can", "you", "for", "with", "this", "that", "about"}


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9-]+", text.lower()) if len(w) > 2 and w not in STOPWORDS}


class DeskMemoryEditor(MemoryEditor):
    """Items live under ("customers", user_id, <kind>). NAT's add_memory writes kind "notes";
    search looks through every kind the desk wrote (profile, episodes, lessons, notes)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def _store(self):
        return SqliteStore.from_conn_string(self.db_path)   # opened per call: short, safe with other processes

    async def add_items(self, items: list[MemoryItem]) -> None:
        with self._store() as store:
            store.setup()
            for item in items:
                text = item.memory or " ".join(m.get("content", "") for m in item.conversation or [])
                store.put(("customers", item.user_id, "notes"), f"note-{uuid.uuid4().hex[:8]}",
                          {"text": text, "tags": item.tags, "metadata": item.metadata})

    async def search(self, query: str, top_k: int = 5, **kwargs) -> list[MemoryItem]:
        user_id = kwargs["user_id"]
        with self._store() as store:
            store.setup()
            found = store.search(("customers", user_id), limit=200)   # prefix search: all kinds
        wanted = _words(query)
        scored = []
        for f in found:
            text = f.value.get("text", "")
            overlap = len(wanted & _words(text)) / (len(wanted) or 1)
            if overlap > 0 or f.namespace[-1] == "profile":   # the profile is always worth returning
                scored.append((overlap, f, text))
        # the profile (semantic memory) first, then the rest by word overlap
        scored.sort(key=lambda s: (s[1].namespace[-1] == "profile", s[0]), reverse=True)
        return [MemoryItem(user_id=user_id, memory=text, tags=[f.namespace[-1]],
                           metadata={"key": f.key}, similarity_score=round(score, 2))
                for score, f, text in scored[:top_k]]

    async def remove_items(self, **kwargs) -> None:
        """Delete this user's notes (the desk's profile, episodes and lessons stay)."""
        user_id = kwargs["user_id"]
        with self._store() as store:
            store.setup()
            for f in store.search(("customers", user_id, "notes"), limit=1000):
                store.delete(f.namespace, f.key)


@register_memory(config_type=DeskMemoryConfig)
async def desk_memory(config: DeskMemoryConfig, builder: Builder):
    yield DeskMemoryEditor(config.db_path)
