#!/usr/bin/env python3
from collections import Counter
from pathlib import Path

base = Path("/Users/josh.gieringer/Projects/VERA-MH/evaluations/HEOR_CLAUDE_4_EVAL")
counts = Counter(p.parent for p in base.rglob("*.tsv") if p.is_file())
for d, n in sorted(counts.items(), key=lambda x: str(x[0])):
    print(f"{n:5d}  {d}")
print(f"\n{sum(counts.values())} TSVs in {len(counts)} directories")
