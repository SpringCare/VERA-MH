#!/usr/bin/env python3
from collections import Counter
from pathlib import Path

base = Path("/Users/josh.gieringer/Desktop/HEOR AIM 3/GROK_3_EVAL")
counts = Counter(p.parent for p in base.rglob("*.tsv") if p.is_file())
for d, n in sorted(counts.items(), key=lambda x: str(x[0])):
    print(f"{n:5d}  {d}")
print(f"\n{sum(counts.values())} TSVs in {len(counts)} directories")
