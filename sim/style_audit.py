"""Quantitative audit of AI-style writing signals in the manuscript prose."""
import re, sys, statistics as st
from collections import Counter

files = ["abstract", "sec_intro", "sec_related", "sec_methods", "sec_results1", "sec_results2", "sec_discussion"]
raw = "\n".join(open(f"paper/{f}.tex").read() for f in files)
num = {m.group(1): m.group(2) for m in re.finditer(r"\\newcommand\{\\n(\w+)\}\{([^}]*)\}", open("paper/numbers.tex").read())}
txt = re.sub(r"\\n([A-Za-z]+)\{?\}?", lambda m: num.get(m.group(1), "X"), raw)
txt = re.sub(r"\\begin\{(table|figure)\}.*?\\end\{\1\}", " ", txt, flags=re.S)
txt = re.sub(r"%.*", "", txt)
txt = re.sub(r"\$[^$]*\$", "X", txt)
txt = re.sub(r"\\(cite|citep|ref|label|eqref|input|includegraphics)\{[^}]*\}", "", txt)
txt = re.sub(r"\\(section|subsection|paragraph|emph|textbf|item)\*?(\[[^\]]*\])?", "", txt)
txt = re.sub(r"\\[a-zA-Z]+", "", txt)
txt = re.sub(r"[{}~]", " ", txt)
txt = re.sub(r"\s+", " ", txt)
words = re.findall(r"[A-Za-z][A-Za-z'-]*", txt)
n = len(words)
sents = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", txt) if len(s.split()) > 3]
L = [len(s.split()) for s in sents]
print(f"words {n} | sentences {len(sents)} | mean len {st.mean(L):.1f} | sd {st.pstdev(L):.1f} | >35 words {sum(l > 35 for l in L)} | <10 words {sum(l < 10 for l in L)}")
print("em-dash (---):", raw.count("---"), "| unicode em-dash:", raw.count("\u2014"))
flags = ["Furthermore", "Moreover", "Additionally", "Consequently", "Notably", "Importantly", "Overall",
         "In conclusion", "delve", "leverage", "crucial", "pivotal", "robust", "comprehensive", "underscore",
         "facilitate", "ensure", "showcase", "seamless", "intricate", "realm", "landscape", "tapestry",
         "It is worth noting", "It should be noted", "plays a key role", "paramount", "novel", "state-of-the-art",
         "demonstrate", "clinically plausible", "realistic", "utilize", "in order to", "whilst"]
hits = {f: len(re.findall(f, txt, flags=re.I)) for f in flags}
print("flagged terms:", {k: v for k, v in hits.items() if v})
passive = len(re.findall(r"\b(is|are|was|were|been|being|be)\s+(\w+ly\s+)?\w+ed\b", txt))
print(f"passive constructions: {passive} ({passive / len(sents):.2f} per sentence)")
we = len(re.findall(r"\bwe\b|\bWe\b|\bour\b|\bOur\b", txt))
print(f"first-person plural: {we}")
op = Counter(s.split()[0].strip("(") for s in sents)
print("top sentence openers:", op.most_common(8))
for s in sents:
    if len(s.split()) > 38:
        print("LONG:", len(s.split()), s[:140], "...")
