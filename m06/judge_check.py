"""Step 5: how far can the judge be trusted? Its verdicts against hand labels, repeated and swapped.

    python m06/judge_check.py                              # qwen3:4b and llama3.2:3b (the desk's model)
    python m06/judge_check.py --judges qwen3:4b,nemotron-mini --reps 3
    python m06/judge_check.py --temperature 0.8            # how stable are the verdicts when it samples?

data/judge_labels.jsonl holds 24 replies to support questions, each with the passages the
reply may use and a human verdict (12 pass, 12 fail), and 8 pairs of replies with the one a
human prefers. Some are planted:

    right_but_terse        short, but right and cited (a human passes it)
    wrong_but_long_polite  friendly and long, with a made-up cause (a human fails it)
    wrong_number           one number changed (90 W instead of 65 W)
    citation_wrong_manual  right facts, but it cites the dock manual for a monitor question
    verbosity (pair)       a long, repetitive, wrong reply against a short right one

For every judge model:
  single replies  each reply is judged --reps times (default 3) with a pass/fail JSON verdict.
                  The majority verdict is compared with the human one: agreement, false passes
                  (the judge passes what a human fails: too lenient) and false fails. Self-
                  consistency: items where all reps gave the same verdict.
  pairs           each pair is judged twice, A/B and then B/A. Position-consistent: the judge
                  picks the same reply both times. A reply "wins" only if both orders agree
                  (the mitigation for position bias). First-slot picks: how often it chose
                  whichever reply came first.

The desk's own model as a judge is the self-judge. The replies here are hand-written, not
produced by the desk, so this does not measure self-enhancement bias; it shows how a 3B
general model and a separate 4B judge grade the same labels. Nothing is asserted about how
good the numbers are: 24 labels are a small sample, for this desk only.

Writes m06/state/judge_check.json.
"""
import argparse
import collections
import json
import pathlib
import re
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_calls  # noqa: E402  (first)
import testset  # noqa: E402

LABELS = HERE / "data" / "judge_labels.jsonl"
OUT = HERE / "state" / "judge_check.json"

SINGLE = """Task: grade one reply.
You check a reply from an electronics shop's support desk. The reply passes only if it answers the
question, every fact in it is supported by the passages, and every ID it cites in square brackets is
one of the passage IDs. Otherwise it fails.

QUESTION: {question}

PASSAGES:
{passages}

REPLY: {reply}

Answer with a JSON object only: {{"verdict": "pass" or "fail", "reason": "one short sentence"}}"""

PAIR = """Task: compare two replies.
Two replies from an electronics shop's support desk answer the same question. Decide which one
answers it better, judged by the passages.

QUESTION: {question}

PASSAGES:
{passages}

REPLY A: {a}

REPLY B: {b}

Answer with a JSON object only: {{"better": "A" or "B", "reason": "one short sentence"}}"""


def load() -> tuple[list[dict], list[dict]]:
    rows = [json.loads(line) for line in LABELS.read_text().splitlines() if line.strip()]
    return [r for r in rows if r["kind"] == "single"], [r for r in rows if r["kind"] == "pair"]


def passages_text(passages: list[dict]) -> str:
    return "\n".join(f"[{p['id']}] {p['text']}" for p in passages)


def parse(text: str, key: str, allowed: set[str]) -> str | None:
    """The value of `key` from the first JSON object in the reply, or None if there is none."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        value = str(json.loads(m.group(0)).get(key, "")).strip()
    except (ValueError, AttributeError):
        return None
    for v in allowed:
        if value.lower() == v.lower():
            return v
    return None


def ask(client, model: str, prompt: str, temperature: float, counter: collections.Counter) -> str:
    counter["calls"] += 1
    try:
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                           response_format={"type": "json_object"}, temperature=temperature,
                                           max_tokens=200, **llm_calls.judge_args(model))
    except Exception as e:
        counter["errors"] += 1
        return f"ERROR {type(e).__name__}: {e}"
    if r.usage:
        counter["tokens"] += r.usage.total_tokens or 0
    return r.choices[0].message.content or ""


def judge_singles(client, model, singles, reps, temperature, counter) -> dict:
    items = []
    for s in singles:
        prompt = SINGLE.format(question=s["question"], passages=passages_text(s["passages"]), reply=s["reply"])
        verdicts = [parse(ask(client, model, prompt, temperature, counter), "verdict", {"pass", "fail"})
                    for _ in range(reps)]
        got = [v for v in verdicts if v]
        counts = collections.Counter(got)
        majority = counts.most_common(1)[0][0] if got else None
        if counts["pass"] and counts["pass"] == counts["fail"]:
            majority = "fail"                      # a tie counts as fail: not convincingly supported
        items.append({"id": s["id"], "human": s["human"], "verdicts": verdicts, "majority": majority,
                      "planted": s.get("planted", "")})
    parsed = [i for i in items if i["majority"]]
    return {"items": items, "n": len(items), "parseable": len(parsed),
            "parse_rate_calls": round(sum(v is not None for i in items for v in i["verdicts"]) / max(len(items) * reps, 1), 3),
            "agree": sum(i["majority"] == i["human"] for i in parsed),
            "false_pass": sum(i["majority"] == "pass" and i["human"] == "fail" for i in parsed),
            "false_fail": sum(i["majority"] == "fail" and i["human"] == "pass" for i in parsed),
            "consistent": sum(len(set(i["verdicts"])) == 1 and i["verdicts"][0] is not None for i in items)}


def judge_pairs(client, model, pairs, temperature, counter) -> dict:
    items = []
    for p in pairs:
        text = passages_text(p["passages"])
        first = parse(ask(client, model, PAIR.format(question=p["question"], passages=text, a=p["reply_a"],
                                                     b=p["reply_b"]), temperature, counter), "better", {"A", "B"})
        second = parse(ask(client, model, PAIR.format(question=p["question"], passages=text, a=p["reply_b"],
                                                      b=p["reply_a"]), temperature, counter), "better", {"A", "B"})
        # which underlying reply was chosen: in the swapped order, slot A holds reply b
        pick1 = {"A": "a", "B": "b"}.get(first)
        pick2 = {"A": "b", "B": "a"}.get(second)
        consistent = pick1 is not None and pick1 == pick2
        items.append({"id": p["id"], "human": p["better"], "order_ab": first, "order_ba": second,
                      "pick_ab": pick1, "pick_ba": pick2, "consistent": consistent,
                      "win": pick1 if consistent else None, "planted": p.get("planted", "")})
    both = [i for i in items if i["order_ab"] and i["order_ba"]]
    return {"items": items, "n": len(items), "parseable": len(both),
            "position_consistent": sum(i["consistent"] for i in both),
            "wins_match_human": sum(i["win"] == i["human"] for i in items if i["win"]),
            "first_slot_picks": sum((i["order_ab"] == "A") + (i["order_ba"] == "A") for i in both),
            "slots": 2 * len(both)}


def run(judges: list[str], reps: int = 3, temperature: float = 0.0, say=print) -> dict:
    singles, pairs = load()
    client = llm_calls.judge_client()
    results = {}
    for model in judges:
        counter = collections.Counter()
        t = time.perf_counter()
        say(f"[INFO] judge {model}: {len(singles)} replies x {reps} reps, {len(pairs)} pairs x 2 orders ...", )
        s = judge_singles(client, model, singles, reps, temperature, counter)
        p = judge_pairs(client, model, pairs, temperature, counter)
        results[model] = {"single": s, "pairs": p, "calls": counter["calls"], "errors": counter["errors"],
                          "tokens": counter["tokens"], "seconds": round(time.perf_counter() - t, 1),
                          "self_judge": model == llm_calls.SELF_JUDGE}
    out = {"reps": reps, "temperature": temperature, "judges": results, "base_url": llm_calls.judge_base_url()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    show(out, singles, pairs, say)
    return out


def show(out: dict, singles, pairs, say=print) -> None:
    say(f"\nSingle replies ({len(singles)} labelled: {sum(s['human'] == 'pass' for s in singles)} pass, "
        f"{sum(s['human'] == 'fail' for s in singles)} fail; {out['reps']} reps each, temperature {out['temperature']:g})")
    say(f"{'judge':<22}{'parseable':>10}{'agree':>8}{'false pass':>12}{'false fail':>12}{'same verdict x' + str(out['reps']):>18}")
    for model, r in out["judges"].items():
        s = r["single"]
        name = model + (" (self)" if r["self_judge"] else "")
        say(f"{name:<22}{s['parseable']:>5}/{s['n']:<4}{s['agree']:>5}/{s['parseable']:<3}{s['false_pass']:>9}"
            f"{s['false_fail']:>12}{s['consistent']:>14}/{s['n']}")
    say(f"\nPairs ({len(pairs)}, each judged A/B and B/A)")
    say(f"{'judge':<22}{'parseable':>10}{'position-consistent':>21}{'wins (both orders)':>20}{'first-slot picks':>18}")
    for model, r in out["judges"].items():
        p = r["pairs"]
        name = model + (" (self)" if r["self_judge"] else "")
        say(f"{name:<22}{p['parseable']:>5}/{p['n']:<4}{p['position_consistent']:>16}/{p['parseable']:<4}"
            f"{p['wins_match_human']:>15}/{p['n']:<4}{p['first_slot_picks']:>13}/{p['slots']}")
    say("\nPlanted cases (human -> the judge's majority verdict; for the pair, the reply it picked in A/B, B/A order)")
    for model, r in out["judges"].items():
        planted = [f"{i['planted']} {i['human']}->{i['majority'] or '?'}" for i in r["single"]["items"] if i["planted"]]
        planted += [f"{i['planted']} {i['human']}->{i['pick_ab'] or '?'},{i['pick_ba'] or '?'}"
                    for i in r["pairs"]["items"] if i["planted"]]
        say(f"  {model:<20} " + "; ".join(planted))
    for model, r in out["judges"].items():
        say(f"[INFO] {model}: {r['calls']} calls, {r['errors']} errors, {r['tokens']} tokens, {r['seconds']} s")
    say("[INFO] no threshold is applied: 24 labels for one desk are a small sample. "
        f"Details in {testset.rel(OUT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judges", default=f"{llm_calls.judge_model()},{llm_calls.SELF_JUDGE}",
                    help="comma-separated judge models (default: the judge and the desk's model)")
    ap.add_argument("--reps", type=int, default=3, help="verdicts per labelled reply (default 3)")
    ap.add_argument("--temperature", type=float, default=0.0, help="the judges' temperature (default 0, as in steps 3-4)")
    a = ap.parse_args()
    run([j.strip() for j in a.judges.split(",") if j.strip()], a.reps, a.temperature)


if __name__ == "__main__":
    main()
