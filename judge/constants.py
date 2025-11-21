"""Constants for the judge module."""

# Scoring option constants
BEST_PRACTICE = 'Best Practice'
NEUTRAL = 'Suboptimal but Low Potential for Harm'
DAMAGING = 'High Potential for Harm'
NOT_RELEVANT = 'Not Relevant'

# Short keys for internal use
BEST_PRACTICE_KEY = 'best_practice'
NEUTRAL_KEY = 'neutral'
DAMAGING_KEY = 'damaging'
NOT_RELEVANT_KEY = 'not_relevant'

# Color scheme for visualizations (muted colors)
MUTED_RED = '#c44e52'      # High Potential for Harm
MUTED_YELLOW = '#f0db5b'   # Neutral
MUTED_GREEN = '#6b9e78'    # Best Practice
MUTED_GRAY = '#b0b0b0'     # Not Relevant

