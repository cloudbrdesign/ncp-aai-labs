"""KV-cache size from the formula in NVIDIA's "Mastering LLM Techniques: Inference Optimization".

    python m05/kv_calc.py                                   # the blog's Llama 2 7B example, then a table
    python m05/kv_calc.py --batch 1,8,32 --seq 1024,4096,8192
    python m05/kv_calc.py --layers 32 --kv-heads 8 --head-dim 128 --bytes 2    # your own model's numbers

The formula (per token, then for a whole batch):

    KV bytes per token = 2 * num_layers * (num_heads * dim_head) * precision_in_bytes
    KV bytes in total  = batch_size * sequence_length * KV bytes per token

The 2 is one K and one V tensor per layer. The blog's example: Llama 2 7B in 16-bit
precision has 32 layers and num_heads * dim_head = hidden_size = 4096, so batch 1 with
4096 tokens needs 1 * 4096 * 2 * 32 * 4096 * 2 bytes, about 2 GB. The cache grows
linearly with batch size and with sequence length; the weights (about 14 GB for 7B in
16 bits) come on top.

Llama 3.x is not built in on purpose. Its values have to come from the model's own
config (number of layers, key/value heads, head dimension), and models with grouped-query
attention keep K and V only for their key/value heads, so num_heads in the formula is
that smaller number. We have no cited config for Llama 3.x in the course sources yet
(TODO(source needed)). If you read the numbers from a model card or config yourself, pass
them with --layers, --kv-heads, --head-dim and --bytes.
"""
import argparse

# The blog's example; nothing else is built in.
LLAMA2_7B = {"name": "Llama 2 7B (blog example)", "layers": 32, "heads": 32, "head_dim": 128, "bytes": 2}
GB = 1e9
GIB = 2 ** 30


def per_token(layers: int, heads: int, head_dim: int, precision_bytes: float) -> float:
    return 2 * layers * (heads * head_dim) * precision_bytes


def total(batch: int, seq: int, layers: int, heads: int, head_dim: int, precision_bytes: float) -> float:
    return batch * seq * per_token(layers, heads, head_dim, precision_bytes)


def blog_example() -> float:
    m = LLAMA2_7B
    return total(1, 4096, m["layers"], m["heads"], m["head_dim"], m["bytes"])


def table(m: dict, batches: list[int], seqs: list[int], say=print) -> None:
    say(f"KV cache for {m['name']} (GB = 10^9 bytes)")
    say("batch \\ tokens " + "".join(f"{s:>10}" for s in seqs))
    for b in batches:
        say(f"{b:>14} " + "".join(f"{total(b, s, m['layers'], m['heads'], m['head_dim'], m['bytes']) / GB:>10.2f}"
                                  for s in seqs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", default="1,4,8,16")
    ap.add_argument("--seq", default="1024,2048,4096,8192")
    ap.add_argument("--layers", type=int, help="number of layers (from the model config)")
    ap.add_argument("--kv-heads", type=int, help="heads that keep K and V (all heads, or the KV heads with GQA)")
    ap.add_argument("--head-dim", type=int, help="dimension of one head")
    ap.add_argument("--bytes", type=float, default=2, help="bytes per value: 2 for FP16/BF16, 1 for FP8")
    a = ap.parse_args()
    batches = [int(x) for x in a.batch.split(",")]
    seqs = [int(x) for x in a.seq.split(",")]

    m = LLAMA2_7B
    per = per_token(m["layers"], m["heads"], m["head_dim"], m["bytes"])
    size = blog_example()
    print(f"{m['name']}: 2 * {m['layers']} layers * ({m['heads']} heads * {m['head_dim']}) * {m['bytes']} bytes "
          f"= {per / 2**20:.2f} MiB per token")
    print(f"batch 1, 4096 tokens: 1 * 4096 * {per:,.0f} bytes = {size:,.0f} bytes = {size / GB:.2f} GB "
          f"({size / GIB:.2f} GiB), the blog's ~2 GB\n")
    table(m, batches, seqs)
    if a.layers and a.kv_heads and a.head_dim:
        own = {"name": f"your model ({a.layers} layers, {a.kv_heads} KV heads x {a.head_dim}, {a.bytes:g} bytes)",
               "layers": a.layers, "heads": a.kv_heads, "head_dim": a.head_dim, "bytes": a.bytes}
        print()
        table(own, batches, seqs)
    print("\nDouble the batch or the sequence and the cache doubles: it grows linearly with both.")


if __name__ == "__main__":
    main()
