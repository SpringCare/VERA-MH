"""What each persona's user model is told must match the committed text exactly.

These pin how generation turns a persona row into instructions: the target's
context template, every column's value, and how the result sits inside the
shared prompt template. For an intended change, run
`python -m tests.safe_changes.golden_prompts` and review the diff. See
docs/safe-changes.md.
"""

import pytest

from tests.safe_changes import golden_prompts


@pytest.mark.parametrize("target", golden_prompts.TARGETS)
def test_persona_prompts_match_the_committed_text(target: str):
    built = {
        name: golden_prompts.persona_part(prompt)
        for name, prompt in golden_prompts.build_prompts(target).items()
    }
    expected = golden_prompts.expected_parts(target)

    assert sorted(built) == sorted(expected), "personas added or removed"
    changed = [name for name in expected if built[name] != expected[name]]
    assert not changed, f"{target} prompts changed for: {', '.join(changed)}"
