# huffman_pdf_streams.py
# Token-aware Huffman coding for PDF-like text/glyph streams.
# - Greedy Huffman with deterministic tiebreaks
# - Canonical codes (compact header via code lengths)
# - Decoder + round-trip check
# - Plots: compression ratio, avglen vs entropy, runtime ~ n log n

from collections import Counter
import heapq, math, random, time
import matplotlib.pyplot as plt


def entropy(freqs):
    N = sum(freqs.values())
    if N == 0:
        return 0.0
    H = 0.0
    for f in freqs.values():
        if f == 0:
            continue
        p = f / N
        H -= p * math.log2(p)
    return H

def avg_codelen(lengths, freqs):
    N = sum(freqs.values())
    if N == 0:
        return 0.0
    return sum((freqs[s] / N) * lengths[s] for s in lengths)

def fixed8_bits(stream):
    return 8 * len(stream)

#Building the Huffman through greedy
class Node:
    __slots__ = ("freq", "sym", "left", "right")
    def __init__(self, freq, sym=None, left=None, right=None):
        self.freq, self.sym, self.left, self.right = freq, sym, left, right

def build_huffman_code(freqs):
    """
    Build canonical Huffman code from symbol->frequency dict.
    Returns (code_map: sym->bitstring, length_map: sym->int)
    Greedy with deterministic tiebreaks via (freq, counter, node).
    """
    if not freqs:
        return {}, {}
    heap = []
    counter = 0
    for s, f in freqs.items():
        heapq.heappush(heap, (f, counter, Node(f, s)))
        counter += 1
    if len(heap) == 1:
        _, _, only = heap[0]
        return {only.sym: "0"}, {only.sym: 1}

    while len(heap) > 1:
        f1, _, x = heapq.heappop(heap)
        f2, _, y = heapq.heappop(heap)
        z = Node(f1 + f2, None, x, y)
        heapq.heappush(heap, (z.freq, counter, z))
        counter += 1

    _, _, root = heap[0]
    lengths = {}
    def dfs(node, d):
        if node.sym is not None:
            lengths[node.sym] = max(1, d)
            return
        dfs(node.left, d+1)
        dfs(node.right, d+1)
    dfs(root, 0)

    code = canonical_from_lengths(lengths)
    return code, lengths

def canonical_from_lengths(lengths):
    items = sorted(lengths.items(), key=lambda kv: (kv[1], str(kv[0])))
    code = {}
    current_code = 0
    current_len = items[0][1]
    for sym, L in items:
        while current_len < L:
            current_code <<= 1
            current_len += 1
        code[sym] = format(current_code, '0{}b'.format(L))
        current_code += 1
    return code

#Header codebook cost
def rle_encode_lengths(lengths_sorted):
    out = []
    if not lengths_sorted:
        return out
    cur = lengths_sorted[0]
    run = 1
    for L in lengths_sorted[1:]:
        if L == cur:
            run += 1
        else:
            out.append((cur, run))
            cur, run = L, 1
    out.append((cur, run))
    return out

def varint_bits(x):
    if x <= 0:
        return 8  # one byte minimum
    bits = 0
    while x > 0:
        x >>= 7
        bits += 8
    return bits

def header_bits_canonical(lengths):
    if not lengths:
        return 0
    lengths_sorted = [L for _, L in sorted(lengths.items(), key=lambda kv: (kv[1], str(kv[0])))]
    rle = rle_encode_lengths(lengths_sorted)
    bits = varint_bits(len(lengths_sorted))
    for L, run in rle:
        bits += varint_bits(L) + varint_bits(run)
    return bits

#Encoding/Decoding 
def encode_stream(stream, code):
    return ''.join(code[s] for s in stream)

def invert_code_to_trie(code):
    trie = {}
    for s, bits in code.items():
        node = trie
        for b in bits:
            node = node.setdefault(b, {})
        node['$'] = s
    return trie

def decode_bitstring(bitstr, trie):
    out = []
    node = trie
    for b in bitstr:
        node = node[b]
        if '$' in node:
            out.append(node['$'])
            node = trie
    return ''.join(out)


#Datasets
def dataset_text_english():
    return (
        "In PDF/PostScript, text and glyph streams exhibit skewed statistics. "
        "Spaces, vowels, and common punctuation appear frequently, while other "
        "symbols are rare. Huffman coding assigns shorter codes to frequent "
        "symbols to reduce storage cost without loss."
    )

def dataset_code_snippet():
    return (
        "for (int i=0; i<n; ++i) { sum += a[i]; }\n"
        "printf(\"total=%d\\n\", sum);\n"
        "if (sum > THRESH) { alert(); }\n"
    )

def dataset_glyph_ids_from_text(text):
    unique = sorted(set(text))
    mapping = {ch: i for i, ch in enumerate(unique)}
    ids = [mapping[ch] for ch in text]
    return ''.join(chr(i % 256) for i in ids)


#experiments
def bitsize_of_stream(stream, code):
    return sum(len(code[s]) for s in stream)

def run_experiment():
    textA = dataset_text_english()
    textB = dataset_code_snippet()
    glyph_sim = dataset_glyph_ids_from_text(textA + textB)
    streams = {"EnglishText": textA, "CodeSnippet": textB, "GlyphIDSim": glyph_sim}

    rows = []
    for name, s in streams.items():
        freqs = Counter(s)
        H = entropy(freqs)

        t0 = time.perf_counter()
        huff_code, huff_lengths = build_huffman_code(freqs)
        t1 = time.perf_counter()
        huff_build_ms = (t1 - t0) * 1000.0

        bitstr = encode_stream(s, huff_code)
        payload_bits = len(bitstr)
        header_bits = header_bits_canonical(huff_lengths)
        total_bits = payload_bits + header_bits

        # round-trip check
        trie = invert_code_to_trie(huff_code)
        assert decode_bitstring(bitstr, trie) == s

        fixed_bits = fixed8_bits(s)

        rows.append({
            "dataset": name,
            "N": len(s),
            "n_symbols": len(freqs),
            "entropy": H,
            "fixed_bits": fixed_bits,
            "huff_payload_bits": payload_bits,
            "huff_header_bits": header_bits,
            "huff_total_bits": total_bits,
            "huff_avg_len": avg_codelen(huff_lengths, freqs),
            "huff_build_ms": huff_build_ms,
        })

    # Compression ratio bars (payload vs total)
    labels = [r["dataset"] for r in rows]
    fixed = [r["fixed_bits"] for r in rows]
    huff_payload = [r["huff_payload_bits"] for r in rows]
    huff_total = [r["huff_total_bits"] for r in rows]

    def safe_ratio(num, den):
        return num / den if den else 1.0

    payload_cr = [safe_ratio(h, f) for h, f in zip(huff_payload, fixed)]
    total_cr = [safe_ratio(h, f) for h, f in zip(huff_total, fixed)]

    plt.figure()
    x = range(len(labels)); width = 0.32
    plt.bar([i - width/2 for i in x], payload_cr, width, label="Token-Huff (payload)")
    plt.bar([i + width/2 for i in x], total_cr, width, label="Token-Huff (payload+header)")
    plt.xticks(list(x), labels, rotation=15)
    plt.ylabel("Compression ratio (compressed / uncompressed) ↓ better")
    plt.title("Compression Ratio by Dataset")
    # legend above the plot area so it never overlaps bars
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show(); plt.close()

    # Avg code length vs entropy (H and H+1 guide lines)
    Hs = [r["entropy"] for r in rows]
    avgs = [r["huff_avg_len"] for r in rows]
    plt.figure()
    plt.scatter(Hs, avgs)
    for r in rows:
        plt.annotate(r["dataset"], (r["entropy"], r["huff_avg_len"]))
    xmin, xmax = min(Hs) - 0.2, max(Hs) + 0.2
    xs = [xmin, xmax]
    plt.plot(xs, xs, linestyle="--", label="y = H")
    plt.plot(xs, [x + 1 for x in xs], linestyle="--", label="y = H + 1")
    plt.xlabel("Entropy H (bits/symbol)")
    plt.ylabel("Average code length (Huffman)")
    plt.title("Huffman Average Code Length vs Entropy (expect H ≤ l̄ < H+1)")
    plt.legend(); plt.tight_layout()
    plt.show(); plt.close()

    # Runtime vs n log n
    ks = [8, 16, 32, 64, 128, 256, 512, 1024]
    times = []; nlogn = []
    for k in ks:
        freqs = {i: random.randint(1, 1000) for i in range(k)}
        t0 = time.perf_counter(); build_huffman_code(freqs); t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0); nlogn.append(k * math.log2(k))
    c = times[-1] / nlogn[-1]; model = [c * v for v in nlogn]
    plt.figure()
    plt.plot(ks, times, marker="o", label="Measured build (ms)")
    plt.plot(ks, model, linestyle="--", label="c · n log2 n (shape)")
    plt.xscale("log", base=2)
    plt.xlabel("Alphabet size n (log scale)")
    plt.ylabel("Build time (ms)")
    plt.title("Huffman Build Time vs n log n")
    plt.legend(); plt.tight_layout()
    plt.show(); plt.close()

    # Header amortization vs N
    base = streams["EnglishText"]; lengths = []; overhead_pct = []
    for rep in [1, 2, 4, 8, 16, 32, 64]:
        s = base * rep
        freqs = Counter(s)
        code, Ls = build_huffman_code(freqs)
        payload = bitsize_of_stream(s, code)
        header = header_bits_canonical(Ls)
        total = payload + header
        lengths.append(len(s))
        overhead_pct.append(100.0 * header / total if total else 0.0)
    plt.figure()
    plt.plot(lengths, overhead_pct, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("Stream length N (log scale)")
    plt.ylabel("Header overhead (%)")
    plt.title("Header Amortization vs Stream Length")
    plt.tight_layout()
    plt.show(); plt.close()

    # Console summary
    print("\nDATASET SUMMARY (token-level Huffman)")
    for r in rows:
        cr_payload = r["huff_payload_bits"] / r["fixed_bits"]
        cr_total = r["huff_total_bits"] / r["fixed_bits"]
        pct_red_payload = 100.0 * (1 - cr_payload)
        pct_red_total = 100.0 * (1 - cr_total)
        print(
            f"{r['dataset']}: N={r['N']}, n={r['n_symbols']}, "
            f"H={r['entropy']:.3f}, lbar={r['huff_avg_len']:.3f}, "
            f"build={r['huff_build_ms']:.2f} ms, "
            f"CR(payload)={cr_payload:.3f} ({pct_red_payload:.1f}%↓), "
            f"CR(total)={cr_total:.3f} ({pct_red_total:.1f}%↓)"
        )

if __name__ == "__main__":
    run_experiment()
