"""
✈️ Flight Booking Assistant — NLP Lab 6 (Syntax Parsing & Classical MT)
Run with:   streamlit run app.py
"""
import math

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import booking_nlp as nlp

st.set_page_config(page_title="Flight Booking NLP", page_icon="✈️", layout="wide")

NAVY, BLUE, ORANGE, GREEN, RED, GRAY = "#1f4e79", "#2e75b6", "#c55a11", "#2e7d32", "#c62828", "#7f7f7f"
LABEL_COLORS = {"root": GRAY, "obj": ORANGE, "obl": BLUE, "nmod": "#8e44ad", "det": "#5d6d7e",
                "amod": GREEN, "compound": "#16a085", "case": "#b7950b"}

st.markdown("""
<style>
.chip {display:inline-block;padding:4px 10px;margin:3px;border-radius:8px;font-family:monospace;font-size:15px}
.stack {background:#1f4e79;color:white}
.buffer {background:#fbe5d6;color:#c55a11}
.root {background:#7f7f7f;color:white}
.card {border:1px solid #d0d7de;border-radius:12px;padding:16px 20px;background:rgba(46,117,182,0.06)}
.small {color:#7f7f7f;font-size:13px}
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("✈️ Booking NLP")
    st.caption("NLP Lab 6 · Faculty of Engineering · BAU")
    st.markdown("""
**Pipeline**
1. **CKY** finds the most probable grammar tree
2. **Dependency parser** turns it into head → dependent arcs
3. Arcs give the **booking details**
4. **SMT** translates French requests
5. **BLEU** scores the translation
""")
    st.divider()
    st.subheader("⚙️ Tune the PCFG")
    p_v_book = st.slider("P(V → book)", 0.05, 0.95, 0.70, 0.05,
                         help="How likely the verb tag is to produce the word 'book'. Slide 11 uses 0.7.")
    p_n_book = st.slider("P(N → book)", 0.0, 0.9, 0.10, 0.05,
                         help="How likely the noun tag is to produce 'book' (as in 'a book'). Slide 11 uses 0.1.")
    pp_pref = st.slider("PP attaches to verb ← → noun", 0.0, 1.0, 0.55, 0.05,
                        help="High = 'to paris' modifies the verb (obl). Low = it modifies the flight (nmod).")
    st.caption("💡 Hint: drag P(V → book) to the minimum and P(N → book) to the maximum, and watch 'book flights' turn into a noun phrase.")

g = nlp.make_grammar(p_v_book=p_v_book, p_n_book=p_n_book, pp_verb_pref=1 - pp_pref)

st.title("✈️ Flight Booking Assistant")
st.caption("Syntax parsing & classical machine translation, applied to booking requests — everything computed live from the lecture formulas.")

tab1, tab2, tab3, tab4 = st.tabs(["① CKY Parsing", "② Dependency & Booking", "③ Translate (SMT)", "④ BLEU Evaluation"])

EXAMPLES = [
    "book flights",
    "book a direct flight to paris",
    "book the cheapest flight from beirut to dubai on monday",
    "book a morning flight to london",
    "reserve two tickets to rome",
    "find a cheap flight from paris to cairo on friday",
]

# =====================================================================
# TAB 1 — CKY
# =====================================================================
def tab_cky():
    c1, c2 = st.columns([2, 1])
    with c1:
        ex = st.selectbox("Pick an example", EXAMPLES, index=1)
        sentence = st.text_input("…or type your own booking request", value=ex,
                                 help="Use simple English commands. Unknown words are treated as city names or nouns.")
    with c2:
        st.markdown("**Grammar G = (V, Σ, R, S)**")
        st.markdown(f"<span class='small'>V: S, VP, NP, Nom, PP, V, N, Det, Adj, P, PropN<br>"
                    f"|Σ| = {sum(len(d) for d in g['lexicon'].values())} words · "
                    f"|R| = {len(g['binary']) + len(g['unary'])} structural rules · S = start</span>",
                    unsafe_allow_html=True)

    res = nlp.parse(sentence, g)
    words = res["words"]

    if words:
        st.markdown("**Step 1 — base level: possible tags for each word** (lexical rules, span length = 1)")
        cols = st.columns(len(words))
        for c, w in zip(cols, words):
            tags = nlp.word_tags(g, w)
            c.markdown(f"**{w}**" + (" ❓" if w in res["unknown"] else ""))
            for t, p in sorted(tags.items(), key=lambda x: -x[1]):
                c.markdown(f"<span class='small'>{t}: {p:.3f}</span>", unsafe_allow_html=True)
        if res["unknown"]:
            st.info(f"Unknown word(s) {res['unknown']} → treated as a proper noun (p=0.02) or noun (p=0.005).")

    if not res["parses"]:
        st.error("❌ No valid parse — this sentence is not in the language of the grammar.")
        st.caption("Try a command like *book a flight to paris*. Vocabulary: " +
                   ", ".join(sorted({w for d in g['lexicon'].values() for w in d})))
        return None, words

    parses = res["parses"]
    best_p, best_tree = parses[0]

    m1, m2, m3 = st.columns(3)
    m1.metric("P(best tree)", f"{best_p:.3e}")
    m2.metric("Parses found", len(parses))
    second = parses[1][0] if len(parses) > 1 else 0
    m3.metric("Best ÷ 2nd best", f"{best_p / second:.2f}×" if second else "unique")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### 🌳 Best parse tree (Viterbi)")
        st.graphviz_chart(nlp.tree_to_dot(best_tree))
        st.code(nlp.bracket(best_tree), language=None)
    with right:
        st.markdown("#### 🔢 CKY chart")
        st.caption("Row = start index *i*, column = end index *j*. Each cell keeps the best probability for every symbol.")
        n = len(words)
        show_syms = st.multiselect("Symbols to show", ["S", "VP", "NP", "PP", "Nom", "V", "N"],
                                   default=["S", "VP", "NP", "V", "N"])
        grid = []
        for i in range(n):
            row = []
            for j in range(1, n + 1):
                if j <= i:
                    row.append("")
                    continue
                cell = res["chart"].get((i, j), {})
                txt = ",  ".join(f"{s}:{cell[s]:.2g}" for s in show_syms if s in cell)
                row.append(txt or "∅")
            grid.append(row)
        df_chart = pd.DataFrame(grid, index=[f"i={i} ({words[i]})" for i in range(n)],
                                columns=[f"j={j}" for j in range(1, n + 1)])
        st.dataframe(df_chart)
        st.caption("💡 Hint: the answer lives in the top-right cell (0, n) under **S**.")

    st.markdown("#### ⚖️ Competing parses — resolving ambiguity")
    comp = pd.DataFrame({"Parse": [f"#{k+1}" for k in range(len(parses))],
                         "P": [p for p, _ in parses],
                         "Share": [p / sum(q for q, _ in parses) for p, _ in parses],
                         "Bracketing": [nlp.bracket(t) for _, t in parses]})
    chart = alt.Chart(comp).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("Share:Q", axis=alt.Axis(format="%"), title="share of total probability"),
        y=alt.Y("Parse:N", sort=None),
        color=alt.condition(alt.datum.Parse == "#1", alt.value(GREEN), alt.value("#9aa5b1")),
        tooltip=["Parse", alt.Tooltip("P:Q", format=".3e"), alt.Tooltip("Share:Q", format=".1%"), "Bracketing"],
    ).properties(height=40 + 28 * len(parses))
    st.altair_chart(chart)
    with st.expander("See the other trees"):
        for k, (p, t) in enumerate(parses[1:4], start=2):
            st.markdown(f"**Parse #{k}** — P = {p:.3e}")
            st.graphviz_chart(nlp.tree_to_dot(t))
    with st.expander("📜 All grammar rules R with their probabilities"):
        st.dataframe(pd.DataFrame(nlp.grammar_rules_table(g), columns=["Rule", "P"]).style.format({"P": "{:.3f}"}),
                     hide_index=True)
        st.caption("Rules are in CNF (binary) plus unary rules handled inside each cell, as on slide 13.")
    return parses, words


with tab1:
    parses, words = tab_cky()

# =====================================================================
# TAB 2 — Dependency parsing + booking slots
# =====================================================================
def draw_arcs(words, arcs, new_arc=None):
    toks = ["ROOT"] + words
    fig, ax = plt.subplots(figsize=(max(6, 1.25 * len(toks)), 2.8))
    for i, w in enumerate(toks):
        ax.text(i, 0, w, ha="center", va="top", fontsize=12, fontweight="bold" if i == 0 else "normal",
                bbox=dict(boxstyle="round,pad=0.3", fc="#dddddd" if i == 0 else "#eef3f8", ec="none"))
    for h, lab, d in arcs:
        span = abs(d - h)
        height = 0.25 + 0.22 * span
        t = np.linspace(0, 1, 60)
        xs, ys = h + (d - h) * t, 0.12 + 4 * height * t * (1 - t)
        col = LABEL_COLORS.get(lab, "black")
        lw = 3.2 if new_arc == (h, lab, d) else 1.8
        ax.plot(xs, ys, color=col, lw=lw)
        ax.annotate("", xy=(d, 0.1), xytext=(xs[-4], ys[-4]), arrowprops=dict(arrowstyle="-|>", color=col, lw=lw))
        ax.text((h + d) / 2, 0.14 + height, lab, ha="center", fontsize=9.5, color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
    ax.set_xlim(-0.6, len(toks) - 0.4)
    ax.set_ylim(-0.35, 0.4 + 0.22 * len(toks))
    ax.axis("off")
    fig.tight_layout()
    return fig


def chips(indices, words, cls):
    toks = ["ROOT"] + words
    if not indices:
        return "<span class='small'>empty</span>"
    return "".join(f"<span class='chip {'root' if i == 0 else cls}'>{toks[i]}</span>" for i in indices)


def tab_dep():
    st.markdown("The best tree from tab ① is converted to **head → dependent** arcs with head rules, then rebuilt "
                "step by step by an **arc-eager transition parser**. An *oracle* plays the role of the classifier.")

    choice = st.selectbox("Which parse should the parser follow?",
                          [f"#{k+1}  (P = {p:.2e})" for k, (p, _) in enumerate(parses[:4])], index=0,
                          help="Different trees → different PP attachments → different arcs (obl vs nmod).")
    tree = parses[int(choice.split()[0][1:]) - 1][1]

    tags = []
    def _collect(t):
        if nlp.is_pre(t): tags.append(t[0])
        else: [_collect(c) for c in t[1:]]
    _collect(tree)

    heads, labels = nlp.tree_to_dependencies(tree)
    states = nlp.arc_eager(words, heads, labels)
    final_arcs = states[-1]["arcs"]

    k = st.slider("Transition step", 0, len(states) - 1, len(states) - 1)
    s = states[k]
    a1, a2 = st.columns([1, 2])
    with a1:
        st.markdown(f"### Step {k}: `{s['action']}`")
        if s["new_arc"]:
            h, l, d = s["new_arc"]
            toks = ["ROOT"] + words
            st.success(f"New arc: **{toks[h]}** —{l}→ **{toks[d]}**")
        st.markdown("**Stack σ** (top on the right)")
        st.markdown(chips(s["stack"], words, "stack"), unsafe_allow_html=True)
        st.markdown("**Buffer β**")
        st.markdown(chips(s["buffer"], words, "buffer"), unsafe_allow_html=True)
        st.markdown(f"**Arc set A**: {len(s['arcs'])} / {len(words)} arcs")
    with a2:
        st.pyplot(draw_arcs(words, s["arcs"], s["new_arc"]))

    with st.expander("📋 Full transition sequence", expanded=False):
        toks = ["ROOT"] + words
        st.dataframe(pd.DataFrame([{
            "Step": i, "Action": x["action"],
            "Stack": "[" + ", ".join(toks[j] for j in x["stack"]) + "]",
            "Buffer": "[" + ", ".join(toks[j] for j in x["buffer"]) + "]",
            "New arc": f"{toks[x['new_arc'][0]]} →{x['new_arc'][1]}→ {toks[x['new_arc'][2]]}" if x["new_arc"] else ""}
            for i, x in enumerate(states)]), hide_index=True)
        st.caption("Actions: SHIFT · LEFT-ARC (buffer front = head) · RIGHT-ARC (stack top = head, dependent pushed) · "
                   "REDUCE (pop a finished word). Here ROOT is linked by a RIGHT-ARC when the main verb arrives — "
                   "the standard arc-eager order; the slides anchor ROOT at the end instead. Same final graph.")

    v1, v2 = st.columns([1, 2])
    with v1:
        st.markdown("#### ✅ Graph validity (slide 22)")
        for name, ok in nlp.validate_graph(len(words), final_arcs).items():
            st.markdown(f"{'✅' if ok else '❌'} {name}")
        st.markdown(f"#### ⚡ Speed")
        n = len(words)
        st.markdown(f"Transitions: **{len(states)-1}** (≈ 2n = {2*n})  \nCKY work ≈ n³·|R| = **{n**3 * (len(g['binary'])):,}**")
    with v2:
        st.markdown("#### 🎫 Booking extracted from the arcs")
        slots = nlp.extract_slots(words, tags, final_arcs)
        if not slots["Is a command"]:
            st.warning("The parser read this as a **noun phrase**, not a command — no action found. "
                       "Raise P(V → book) in the sidebar.")
        c = st.columns(3)
        c[0].metric("Action", slots["Action"] or "—")
        c[1].metric("Object", slots["Object"] or "—")
        c[2].metric("Preferences", ", ".join(slots["Preferences"]) or "—")
        c = st.columns(3)
        c[0].metric("From", (slots["Origin"] or "—").title())
        c[1].metric("To", (slots["Destination"] or "—").title())
        c[2].metric("When", (slots["Date / time"] or "—").title())
        st.caption("💡 Hint: *Destination* = the noun whose **case** dependent is “to”. "
                   "*Object* = the **obj** dependent of the root verb. No phrase tree needed — just follow arrows.")

with tab2:
    if parses:
        tab_dep()
    else:
        st.info("Enter a sentence the grammar can parse in tab ① first.")

# =====================================================================
# TAB 3 — SMT
# =====================================================================
FR_EXAMPLES = {
    "Réserver un vol direct pour Paris": "book a direct flight to paris",
    "Réserver un vol économique de Beyrouth à Dubaï lundi": "book a cheap flight from beirut to dubai on monday",
    "Trouver les vols directs vers Londres": "find the direct flights to london",
    "Réserver un billet pour Rome": "book a ticket to rome",
}

with tab3:
    st.markdown(r"Noisy channel: $\hat{E} = \arg\max_E \; P(F\mid E)\cdot d \cdot P(E)^{\lambda}$  "
                r"— translation model × distortion × language model.")
    t1, t2 = st.columns([2, 1])
    with t1:
        fr_ex = st.selectbox("French booking request", list(FR_EXAMPLES), index=0)
        french = st.text_input("…or type your own (French)", value=fr_ex)
    with t2:
        p_swap = st.slider("P(Adj–Noun reorder)", 0.0, 1.0, 0.9, 0.05,
                           help="Slide 40: 0.9 for swapping, so 0.1 for keeping French order.")
        lam = st.slider("Language model weight λ", 0.0, 2.0, 1.0, 0.1,
                        help="0 = ignore fluency entirely, only fidelity P(F|E) counts.")
        beam = st.slider("Beam width K", 1, 20, 5,
                         help="How many partial translations are kept at each step (slide 37).")

    out = nlp.smt_decode(french, p_swap=max(p_swap, 1e-6), lm_weight=lam, beam=beam)
    cands = out["candidates"]
    if not cands:
        st.error("Nothing to translate.")
    else:
        best = cands[0]["english"]
        st.session_state["smt_best"] = best
        st.session_state["smt_ref"] = FR_EXAMPLES.get(french, "")
        st.markdown(f"### 🇬🇧 {best.capitalize()}")

        b1, b2 = st.columns([1, 1.3])
        with b1:
            st.markdown("**Phrase table lookups**")
            st.dataframe(pd.DataFrame(out["table"]), hide_index=True)
            st.caption("💡 Hint: *vol* can mean *flight* **or** *theft*. The translation model alone prefers flight; "
                       "*pour* is 50/50 between *to* and *for* — only the language model can break that tie.")
        with b2:
            st.markdown("**Final candidates in the beam**")
            dfc = pd.DataFrame(cands[:10])
            mx = dfc["score"].max()
            dfc["relative"] = np.exp(dfc["score"] - mx)
            dfc["relative"] /= dfc["relative"].sum()
            ch = alt.Chart(dfc).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("relative:Q", axis=alt.Axis(format="%"), title="relative probability among candidates"),
                y=alt.Y("english:N", sort="-x", title=None),
                color=alt.condition(alt.datum.english == best, alt.value(GREEN), alt.value("#9aa5b1")),
                tooltip=["english", alt.Tooltip("score:Q", format=".2f"), alt.Tooltip("log P(F|E):Q", format=".2f"),
                         alt.Tooltip("log d:Q", format=".2f"), alt.Tooltip("log P(E):Q", format=".2f")],
            ).properties(height=40 + 26 * len(dfc))
            st.altair_chart(ch)

        st.markdown("**Score breakdown (log space — higher is better)**")
        st.dataframe(pd.DataFrame(cands[:10]).style.format(
            {c: "{:.2f}" for c in ["score", "log P(F|E)", "log d", "log P(E)"]})
            .background_gradient(subset=["score"], cmap="Greens"), hide_index=True)

        with st.expander("🔦 Beam search trace"):
            for i, kept in enumerate(out["trace"], start=1):
                st.markdown(f"**Step {i}** — kept {len(kept)}: " +
                            " · ".join(f"`{h or '∅'}` ({sc:.1f})" for h, sc, _ in kept))
        st.info("Try it: set **λ = 0** → fluency is ignored, and *for paris* ties with *to paris*. "
                "Set **reorder = 0.1** → the literal *flight direct* wins, just like ‘The cat black’ on slide 42.")

# =====================================================================
# TAB 4 — BLEU
# =====================================================================
with tab4:
    mode = st.radio("Evaluate", ["Translation from tab ③", "Slide 53 example", "Custom"], horizontal=True)
    if mode == "Translation from tab ③":
        cand_default = st.session_state.get("smt_best", "")
        ref_default = st.session_state.get("smt_ref") or "book a direct flight to paris"
    elif mode == "Slide 53 example":
        cand_default, ref_default = "The cat sat on mat", "The cat sat on the mat"
    else:
        cand_default, ref_default = "book a flight to paris", "book a direct flight to paris\nreserve a direct flight to paris"

    e1, e2 = st.columns(2)
    with e1:
        candidate = st.text_input("Machine output (candidate)", value=cand_default, key=f"cand_{mode}")
    with e2:
        refs = st.text_area("Human reference(s) — one per line", value=ref_default, key=f"ref_{mode}", height=90)
    o1, o2 = st.columns(2)
    N = o1.slider("Max n-gram order N", 1, 4, 2, help="BLEU-2 on the slides, BLEU-4 is the standard in research.")
    smooth = o2.checkbox("Smoothing for zero counts", value=False,
                         help="Short sentences often have no matching 4-grams → BLEU = 0. Smoothing replaces 0 with a tiny value.")

    b = nlp.bleu(candidate, refs.splitlines(), N=N, smooth=smooth)
    if b is None:
        st.warning("Enter a candidate and at least one reference.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"BLEU-{N}", f"{b['bleu']*100:.1f}%")
        k2.metric("Brevity penalty", f"{b['bp']:.4f}")
        k3.metric("Candidate length c", b["c"])
        k4.metric("Reference length r", b["r"])

        terms = " \\times ".join(f"{p:.3f}^{{1/{N}}}" for p in b["p"])
        st.latex(rf"\text{{BLEU}} = BP \times \prod_n p_n^{{w_n}} = {b['bp']:.4f} \times {terms} = {b['bleu']:.4f}")
        if b["c"] <= b["r"]:
            st.latex(rf"BP = e^{{1 - r/c}} = e^{{1 - {b['r']}/{b['c']}}} = {b['bp']:.4f}")
        else:
            st.caption("c > r → no brevity penalty (BP = 1).")

        g1, g2 = st.columns([1, 1.4])
        with g1:
            dfp = pd.DataFrame({"component": [f"p{n}" for n in range(1, N + 1)] + ["BP", f"BLEU-{N}"],
                                "value": b["p"] + [b["bp"], b["bleu"]]})
            st.altair_chart(alt.Chart(dfp).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("component:N", sort=None, title=None), y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("component:N", legend=None,
                                scale=alt.Scale(range=[BLUE] * N + [ORANGE, GREEN])),
                tooltip=["component", alt.Tooltip("value:Q", format=".3f")]).properties(height=280))
        with g2:
            for d in b["details"]:
                with st.expander(f"p{d['n']} = {d['num']}/{d['den']} = {d['p']:.3f}", expanded=d["n"] <= 2):
                    dd = pd.DataFrame(d["rows"])
                    st.dataframe(dd.style.apply(lambda r: ["background-color: #fde2e2" if r["Clipped = min"] < r["Count cand"]
                                                           else "" for _ in r], axis=1),
                                 hide_index=True)
        st.caption("💡 Hint: red rows are n-grams that were **clipped** — the candidate used them more often "
                   "than any reference does (or they never appear). Try candidate *the the the the*.")

st.divider()
st.caption("Built for NLP Lab 6 · all algorithms implemented from scratch in `booking_nlp.py`")
