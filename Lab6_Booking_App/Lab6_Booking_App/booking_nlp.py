"""
booking_nlp.py — NLP engine for the Flight Booking Assistant (Lab 6).

Everything from Chapter 6, written from scratch so students can read it:
  1. PCFG + probabilistic CKY (Viterbi, with k-best parses)
  2. Constituency -> dependency conversion (head rules) + arc-eager transition parser
  3. Slot extraction (action, object, origin, destination, date) from dependency arcs
  4. Phrase-based SMT decoder: French -> English, noisy channel + distortion + beam search
  5. BLEU (clipped n-gram precision, brevity penalty, multiple references)

No Streamlit code lives here, so the notebook can import and test it directly.
"""
import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache


# =====================================================================
# 0. Tokenisation
# =====================================================================
def tokenize(text):
    return re.findall(r"[a-zà-ÿ0-9']+", text.lower())


def strip_accents(w):
    return "".join(c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) != "Mn")


# =====================================================================
# 1. PCFG for booking commands
# =====================================================================
# Base lexical weights; each tag's distribution is normalised to sum to 1.
BASE_LEXICON = {
    "V":     {"book": 5, "reserve": 2, "find": 1.5, "show": 1, "flights": 0.5},
    "N":     {"flight": 4, "flights": 4, "ticket": 2, "tickets": 1.5, "book": 1,
              "morning": 1, "evening": 1, "monday": 1, "friday": 1, "tomorrow": 1},
    "Det":   {"a": 4, "the": 4, "an": 1, "one": 1, "two": 0.5, "three": 0.5},
    "Adj":   {"cheap": 2, "cheapest": 2, "direct": 2, "nonstop": 1, "early": 1, "late": 1},
    "P":     {"to": 4, "from": 3, "on": 2, "for": 2, "in": 1},
    "PropN": {"paris": 1, "beirut": 1, "london": 1, "dubai": 1, "rome": 1, "cairo": 1},
}
UNKNOWN_TAGS = {"PropN": 0.02, "N": 0.005}   # unknown words: probably a city name, maybe a noun

TAG_NAMES = {"V": "Verb", "N": "Noun", "Det": "Determiner", "Adj": "Adjective",
             "P": "Preposition", "PropN": "Proper noun"}


def make_grammar(p_v_book=0.5, p_n_book=0.1, pp_verb_pref=0.5):
    """Build the PCFG.

    p_v_book     : P(V -> book)
    p_n_book     : P(N -> book)
    pp_verb_pref : 0..1, how much a PP prefers to attach to the verb (obl) vs the noun (nmod)
    """
    lex = {}
    for tag, weights in BASE_LEXICON.items():
        w = dict(weights)
        fixed = {"V": p_v_book, "N": p_n_book}.get(tag)
        if fixed is not None:
            others = {k: v for k, v in w.items() if k != "book"}
            tot = sum(others.values())
            dist = {k: (1 - fixed) * v / tot for k, v in others.items()}
            dist["book"] = fixed
        else:
            tot = sum(w.values())
            dist = {k: v / tot for k, v in w.items()}
        lex[tag] = dist

    vp_pp = 0.05 + 0.6 * pp_verb_pref          # VP -> VP PP
    np_pp = 0.05 + 0.4 * (1 - pp_verb_pref)    # NP -> NP PP
    rest = 1 - np_pp
    binary = [  # (A, B, C, P)
        ("VP", "V", "NP", 1 - vp_pp),
        ("VP", "VP", "PP", vp_pp),
        ("NP", "Det", "Nom", rest * 0.50),
        ("NP", "NP", "PP", np_pp),
        ("Nom", "Adj", "Nom", 0.20),
        ("Nom", "N", "Nom", 0.15),
        ("PP", "P", "NP", 1.0),
    ]
    unary = [   # (A, B, P)
        ("S", "VP", 0.75),
        ("S", "NP", 0.25),
        ("NP", "Nom", rest * 0.25),
        ("NP", "PropN", rest * 0.25),
        ("Nom", "N", 0.65),
    ]
    return {"lexicon": lex, "binary": binary, "unary": unary, "start": "S"}


def grammar_rules_table(g):
    """Rows (rule, P) for display."""
    rows = [(f"{a} → {b} {c}", p) for a, b, c, p in g["binary"]]
    rows += [(f"{a} → {b}", p) for a, b, p in g["unary"]]
    return sorted(rows)


def word_tags(g, word):
    tags = {t: d[word] for t, d in g["lexicon"].items() if word in d}
    if not tags:
        tags = dict(UNKNOWN_TAGS)
    return tags


# ---------------------------------------------------------------------
# Trees: preterminal = (tag, word, index) ; internal = (label, child, [child])
# ---------------------------------------------------------------------
def is_pre(t):
    return isinstance(t[1], str)


def tree_words(t):
    return [t[1]] if is_pre(t) else [w for c in t[1:] for w in tree_words(c)]


def bracket(t):
    if is_pre(t):
        return f"[{t[0]} {t[1]}]"
    return f"[{t[0]} " + " ".join(bracket(c) for c in t[1:]) + "]"


def pcky(words, g, k_best=8):
    """Probabilistic CKY with k-best lists.

    Returns
      chart : {(i, j): {symbol: best_prob}}  — the Viterbi chart from the slides
      best  : function(sym, i, j) -> list[(prob, tree)] sorted, at most k_best
    """
    n = len(words)
    unary_by_child, binary_by_children = {}, {}
    for a, b, p in g["unary"]:
        unary_by_child.setdefault(b, []).append((a, p))
    for a, b, c, p in g["binary"]:
        binary_by_children.setdefault((b, c), []).append((a, p))
    symbols = {a for a, *_ in g["binary"]} | {a for a, *_ in g["unary"]} | set(g["lexicon"])

    @lru_cache(maxsize=None)
    def items(sym, i, j):
        out = []
        if j == i + 1:                                         # lexical rules
            p = word_tags(g, words[i]).get(sym)
            if p:
                out.append((p, (sym, words[i], i)))
        for a, b, p in g["unary"]:                             # unary rules A -> B
            if a == sym:
                out += [(p * q, (sym, t)) for q, t in items(b, i, j)]
        for a, b, c, p in g["binary"]:                         # binary rules A -> B C
            if a == sym:
                for k in range(i + 1, j):
                    left, right = items(b, i, k), items(c, k, j)
                    out += [(p * ql * qr, (sym, tl, tr)) for ql, tl in left for qr, tr in right]
        out.sort(key=lambda x: -x[0])
        return tuple(out[:k_best])

    chart = {}
    for span in range(1, n + 1):
        for i in range(0, n - span + 1):
            j = i + span
            cell = {s: items(s, i, j)[0][0] for s in symbols if items(s, i, j)}
            chart[(i, j)] = cell
    return chart, (lambda sym, i, j: list(items(sym, i, j)))


def parse(sentence, g, k_best=8):
    words = tokenize(sentence)
    if not words:
        return {"words": [], "parses": [], "chart": {}}
    chart, best = pcky(words, g, k_best)
    parses = best(g["start"], 0, len(words))
    unknown = [w for w in words if not any(w in d for d in g["lexicon"].values())]
    return {"words": words, "parses": parses, "chart": chart, "unknown": unknown}


def tree_to_dot(t, highlight=False):
    """Graphviz DOT for a constituency tree."""
    lines = ['digraph T { node [fontname="Helvetica", fontsize=12, style="filled,rounded", shape=box, '
             'penwidth=0]; edge [color="#9aa5b1", arrowhead=none];']
    counter = [0]

    def walk(node):
        my = f"n{counter[0]}"; counter[0] += 1
        if is_pre(node):
            lines.append(f'{my} [label="{node[0]}", fillcolor="#1f4e79", fontcolor="white"];')
            w = f"n{counter[0]}"; counter[0] += 1
            lines.append(f'{w} [label="{node[1]}", fillcolor="#fbe5d6", fontcolor="#c55a11", shape=plaintext];')
            lines.append(f"{my} -> {w};")
            return my
        color = "#2e75b6" if node[0] in ("S", "VP", "NP", "PP", "Nom") else "#1f4e79"
        lines.append(f'{my} [label="{node[0]}", fillcolor="{color}", fontcolor="white"];')
        for c in node[1:]:
            lines.append(f"{my} -> {walk(c)};")
        return my

    walk(t)
    lines.append("}")
    return "\n".join(lines)


# =====================================================================
# 2. Dependency parsing
# =====================================================================
# Head rules: (parent, children...) -> (index of head child, labels for the other children)
HEAD_RULES = {
    ("S", "VP"): (0, {}),
    ("S", "NP"): (0, {}),
    ("VP", "V", "NP"): (0, {1: "obj"}),
    ("VP", "VP", "PP"): (0, {1: "obl"}),
    ("NP", "Det", "Nom"): (1, {0: "det"}),
    ("NP", "NP", "PP"): (0, {1: "nmod"}),
    ("NP", "Nom"): (0, {}),
    ("NP", "PropN"): (0, {}),
    ("Nom", "Adj", "Nom"): (1, {0: "amod"}),
    ("Nom", "N", "Nom"): (1, {0: "compound"}),
    ("Nom", "N"): (0, {}),
    ("PP", "P", "NP"): (1, {0: "case"}),
}


def tree_to_dependencies(tree):
    """Convert a constituency tree to dependency arcs using head rules.
    Word positions are 1..n; 0 = ROOT. Returns heads{dep: head}, labels{dep: label}."""
    heads, labels = {}, {}

    def head_of(node):
        if is_pre(node):
            return node[2] + 1
        key = (node[0],) + tuple(c[0] for c in node[1:])
        h_idx, dep_labels = HEAD_RULES[key]
        child_heads = [head_of(c) for c in node[1:]]
        h = child_heads[h_idx]
        for ci, ch in enumerate(child_heads):
            if ci != h_idx:
                heads[ch], labels[ch] = h, dep_labels.get(ci, "dep")
        return h

    r = head_of(tree)
    heads[r], labels[r] = 0, "root"
    return heads, labels


def arc_eager(words, heads, labels):
    """Run the arc-eager transition system with a static oracle (stands in for the classifier).
    Returns a list of states: dict(action, stack, buffer, arcs, new_arc)."""
    n = len(words)
    stack, buffer, arcs = [0], list(range(1, n + 1)), []
    has_head = set()
    states = [dict(action="INIT", stack=list(stack), buffer=list(buffer), arcs=[], new_arc=None)]

    def log(action, new=None):
        states.append(dict(action=action, stack=list(stack), buffer=list(buffer),
                           arcs=list(arcs), new_arc=new))

    guard = 0
    while (buffer or len(stack) > 1) and guard < 10 * n + 10:
        guard += 1
        s = stack[-1]
        if not buffer:                                   # buffer empty: pop what's left
            stack.pop(); log("REDUCE"); continue
        b = buffer[0]
        if s != 0 and heads[s] == b:                     # LEFT-ARC: b is the head of s
            arc = (b, labels[s], s)
            arcs.append(arc); has_head.add(s); stack.pop()
            log(f"LEFT-ARC({labels[s]})", arc)
        elif heads[b] == s:                              # RIGHT-ARC: s is the head of b
            arc = (s, labels[b], b)
            arcs.append(arc); has_head.add(b); stack.append(buffer.pop(0))
            log(f"RIGHT-ARC({labels[b]})", arc)
        elif s in has_head and any(heads[b] == k or heads.get(k) == b for k in stack[:-1]):
            stack.pop(); log("REDUCE")                   # s is finished
        else:
            stack.append(buffer.pop(0)); log("SHIFT")
    return states


def validate_graph(n, arcs):
    heads = Counter(d for _, _, d in arcs)
    head_of = {d: h for h, _, d in arcs}
    single = all(heads[i] == 1 for i in range(1, n + 1))
    connected, acyclic = True, True
    for i in range(1, n + 1):
        seen, cur = set(), i
        while cur != 0:
            if cur in seen:
                acyclic = False; break
            if cur not in head_of:
                connected = False; break
            seen.add(cur); cur = head_of[cur]
    one_root = sum(1 for h, _, _ in arcs if h == 0) == 1
    return {"Single head": single, "Acyclic": acyclic, "Connected": connected, "Exactly one ROOT": one_root}


# =====================================================================
# 3. Slot extraction — "direct extraction of meaning" (slide 31)
# =====================================================================
TIME_WORDS = {"monday", "friday", "tomorrow", "morning", "evening"}


def extract_slots(words, tags, arcs):
    """words/tags are 0-indexed lists; arcs use 1..n positions (0 = ROOT)."""
    W = ["ROOT"] + words
    T = ["ROOT"] + tags
    deps = {}
    for h, l, d in arcs:
        deps.setdefault(h, []).append((l, d))
    root = next(d for h, l, d in arcs if h == 0)
    slots = {"Action": None, "Object": None, "Preferences": [], "Origin": None,
             "Destination": None, "Date / time": None, "Is a command": T[root] == "V"}
    if T[root] == "V":
        slots["Action"] = W[root]
    obj = next((d for l, d in deps.get(root, []) if l == "obj"), root if T[root] != "V" else None)
    if obj:
        mods = sorted(deps.get(obj, []), key=lambda x: x[1])
        compounds = [W[d] for l, d in mods if l == "compound"]
        slots["Object"] = " ".join(compounds + [W[obj]])
        slots["Preferences"] = [W[d] for l, d in mods if l == "amod"]
    # Every word that has a 'case' dependent is the object of a preposition
    for h, l, d in arcs:
        if l != "case":
            continue
        prep, noun = W[d], W[h]
        if prep == "from":
            slots["Origin"] = noun
        elif prep == "to" or (prep == "for" and T[h] == "PropN"):
            slots["Destination"] = noun
        elif prep in ("on", "in", "for") and noun in TIME_WORDS:
            slots["Date / time"] = noun
    return slots


# =====================================================================
# 4. Phrase-based SMT: French -> English
# =====================================================================
FR_TABLE = {   # french: (POS, [(english, P(f|e))])
    "réserver": ("V", [("book", 0.6), ("reserve", 0.4)]),
    "trouver":  ("V", [("find", 0.8), ("look", 0.2)]),
    "un":       ("Det", [("a", 0.8), ("one", 0.2)]),
    "une":      ("Det", [("a", 0.8), ("one", 0.2)]),
    "le":       ("Det", [("the", 1.0)]),
    "la":       ("Det", [("the", 1.0)]),
    "les":      ("Det", [("the", 1.0)]),
    "vol":      ("N", [("flight", 0.85), ("theft", 0.15)]),
    "vols":     ("N", [("flights", 0.85), ("thefts", 0.15)]),
    "billet":   ("N", [("ticket", 1.0)]),
    "billets":  ("N", [("tickets", 1.0)]),
    "direct":   ("Adj", [("direct", 1.0)]),
    "directs":  ("Adj", [("direct", 1.0)]),
    "économique": ("Adj", [("cheap", 0.6), ("economy", 0.4)]),
    "matinal":  ("Adj", [("early", 0.5), ("morning", 0.5)]),
    "pour":     ("P", [("to", 0.5), ("for", 0.5)]),
    "vers":     ("P", [("to", 1.0)]),
    "de":       ("P", [("from", 0.6), ("of", 0.4)]),
    "à":        ("P", [("to", 0.6), ("at", 0.4)]),
    "lundi":    ("N", [("monday", 1.0)]),
    "vendredi": ("N", [("friday", 1.0)]),
    "demain":   ("N", [("tomorrow", 1.0)]),
    "paris":    ("PropN", [("paris", 1.0)]),
    "beyrouth": ("PropN", [("beirut", 1.0)]),
    "londres":  ("PropN", [("london", 1.0)]),
    "dubaï":    ("PropN", [("dubai", 1.0)]),
    "rome":     ("PropN", [("rome", 1.0)]),
}
_FR_PLAIN = {strip_accents(k): k for k in FR_TABLE}


def fr_lookup(w):
    key = w if w in FR_TABLE else _FR_PLAIN.get(strip_accents(w))
    if key:
        return FR_TABLE[key]
    return ("X", [(w, 1.0)])                       # unknown: copy it through


EN_CORPUS = """
book a flight to paris
book a direct flight to london
book a cheap flight from beirut to dubai
reserve a flight to rome on monday
book the cheapest flight on friday
find a direct flight from paris to beirut
book a ticket to cairo
i want a flight to paris tomorrow
book an early flight to london
show me flights from london to paris
book two tickets for monday
find flights to dubai
the flight to paris is direct
book a morning flight to beirut
the theft of the car
""".strip().splitlines()


class BigramLM:
    def __init__(self, sentences, k=0.1):
        self.k = k
        toks = [["<s>"] + s.split() + ["</s>"] for s in sentences]
        self.uni = Counter(w for s in toks for w in s)
        self.bi = Counter((a, b) for s in toks for a, b in zip(s, s[1:]))
        self.V = len(self.uni) + 1

    def prob(self, prev, w):
        return (self.bi[(prev, w)] + self.k) / (self.uni[prev] + self.k * self.V)

    def sentence_logprob(self, words):
        seq = ["<s>"] + list(words) + ["</s>"]
        return sum(math.log(self.prob(a, b)) for a, b in zip(seq, seq[1:]))


LM = BigramLM(EN_CORPUS)


def smt_decode(french, p_swap=0.9, lm_weight=1.0, beam=5):
    """Monotone phrase-based decoding with an N+Adj swap option and beam search.

    score = log P(F|E) + log d + λ · log P(E)
    Returns dict(candidates, trace, table)."""
    f = tokenize(french)
    info = [fr_lookup(w) for w in f]
    pos = [i[0] for i in info]
    n = len(f)
    p_mono = 1 - p_swap

    # hypothesis: (score, i, out_words, log_tm, log_d, log_lm)
    hyps = [(0.0, 0, (), 0.0, 0.0, 0.0)]
    finished, trace = [], []

    def extend(h, emitted, tm_p, d_p, new_i):
        score, i, out, ltm, ld, llm = h
        prev = out[-1] if out else "<s>"
        add_lm = 0.0
        for w in emitted:
            add_lm += math.log(LM.prob(prev, w)); prev = w
        ltm2, ld2, llm2 = ltm + math.log(tm_p), ld + math.log(d_p), llm + add_lm
        if new_i == n:
            llm2 += math.log(LM.prob(prev, "</s>"))
        return (ltm2 + ld2 + lm_weight * llm2, new_i, out + tuple(emitted), ltm2, ld2, llm2)

    step = 0
    while hyps:
        new = []
        for h in hyps:
            i = h[1]
            pair = i + 1 < n and pos[i] == "N" and pos[i + 1] == "Adj"
            # (a) translate word i in place
            for e, p in info[i][1]:
                new.append(extend(h, (e,), p, p_mono if pair else 1.0, i + 1))
            # (b) swap N Adj -> Adj N
            if pair:
                for ea, pa in info[i + 1][1]:
                    for en, pn in info[i][1]:
                        new.append(extend(h, (ea, en), pa * pn, p_swap, i + 2))
        # keep top-K per coverage length (simple beam)
        new.sort(key=lambda x: -x[0])
        kept = new[:beam]
        step += 1
        trace.append([(" ".join(h[2]), h[0], h[1]) for h in kept])
        finished += [h for h in kept if h[1] == n]
        hyps = [h for h in kept if h[1] < n]

    finished.sort(key=lambda x: -x[0])
    seen, cands = set(), []
    for h in finished:
        s = " ".join(h[2])
        if s in seen:
            continue
        seen.add(s)
        cands.append({"english": s, "score": h[0], "log P(F|E)": h[3], "log d": h[4], "log P(E)": h[5]})
    table = [{"French": w, "POS": info[k][0],
              "Options  e : P(f|e)": ",  ".join(f"{e}: {p:g}" for e, p in info[k][1])}
             for k, w in enumerate(f)]
    return {"candidates": cands, "trace": trace, "table": table, "french": f}


# =====================================================================
# 5. BLEU
# =====================================================================
def ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(candidate, references, N=4, smooth=False):
    """BLEU with clipping against the max count over references, closest-length BP."""
    cand = tokenize(candidate)
    refs = [tokenize(r) for r in references if r.strip()]
    if not cand or not refs:
        return None
    details, ps = [], []
    for n in range(1, N + 1):
        cc = ngram_counts(cand, n)
        max_ref = Counter()
        for r in refs:
            for g, c in ngram_counts(r, n).items():
                max_ref[g] = max(max_ref[g], c)
        rows = [{"n-gram": " ".join(g), "Count cand": c, "Max count ref": max_ref[g],
                 "Clipped = min": min(c, max_ref[g])} for g, c in cc.items()]
        num = sum(r["Clipped = min"] for r in rows)
        den = max(sum(cc.values()), 1)
        p = num / den
        if smooth and num == 0:
            p = 0.1 / den                                   # epsilon smoothing for zero counts
        ps.append(p)
        details.append({"n": n, "rows": rows, "num": num, "den": den, "p": p})
    c = len(cand)
    r = min((len(x) for x in refs), key=lambda L: (abs(L - c), L))
    bp = 1.0 if c > r else math.exp(1 - r / c)
    score = 0.0 if min(ps) == 0 else bp * math.exp(sum(math.log(p) / N for p in ps))
    return {"bleu": score, "bp": bp, "c": c, "r": r, "p": ps, "details": details}
