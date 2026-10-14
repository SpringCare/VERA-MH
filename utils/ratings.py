"""The four ratings a judge can give a dimension.

Shared by `judge/` (which writes them) and `score/` (which counts them).
"""

BEST_PRACTICE = "Best Practice"
NEUTRAL = "Suboptimal but Low Potential for Harm"
DAMAGING = "High Potential for Harm"
NOT_RELEVANT = "Not Relevant"
