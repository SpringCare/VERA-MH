"""Shared helpers for generation-run folder layout (nested vs legacy flat)."""

import os
from typing import Optional

from .naming import is_generation_run_folder_basename


def resolve_conversation_input(folder: str) -> tuple[str, Optional[str], str]:
    """
    Resolve where transcript ``.txt`` files live under a generation or
    conversation folder.

    Returns:
        (transcripts_dir, gen_run_root_or_none, conversation_run_basename)

    - If ``folder/conversations/`` contains ``.txt`` files, ``folder`` is treated as
      a generation run root; transcripts are read from ``folder/conversations``.
    - If ``folder`` is named ``conversations`` (with or without ``.txt`` files yet)
      and its parent's basename passes ``is_generation_run_folder_basename``
      (canonical ``p_*__a_*__t*__r*__*`` layout from ``utils.naming``), treat
      ``folder`` as that run's nested transcript directory (same as passing the
      parent). This keeps judge run folder names and score metadata aligned with
      the run id and avoids resolving to a bogus ``.../conversations/conversations``
      path before the first transcript is written.
    - Otherwise, if ``folder`` itself contains ``.txt`` files, use the legacy flat
      layout (``gen_run_root`` is None).
    - If ``folder/conversations/`` exists but has no ``.txt`` files yet, use that
      path (default nested layout for new runs).
    - If there is no ``conversations/`` subdirectory and no transcripts at the root,
      default to ``folder/conversations`` as the transcript directory so generation
      and judging agree on the nested layout.
    """
    folder = os.path.normpath(os.path.abspath(folder))
    nested = os.path.join(folder, "conversations")
    basename = os.path.basename(folder)

    def dir_has_txt_files(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        return any(name.endswith(".txt") for name in os.listdir(path))

    if dir_has_txt_files(nested):
        return nested, folder, basename
    if dir_has_txt_files(folder):
        if basename == "conversations":
            parent = os.path.dirname(folder)
            parent_base = os.path.basename(parent)
            if (
                parent
                and parent_base
                and is_generation_run_folder_basename(parent_base)
            ):
                return folder, parent, parent_base
        return folder, None, basename
    # Path already ends in .../conversations/ but no .txt yet (or only non-.txt files):
    # do not fall through to folder/conversations (double "conversations").
    if basename == "conversations" and os.path.isdir(folder):
        parent = os.path.dirname(folder)
        parent_base = os.path.basename(parent) if parent else ""
        if parent and parent_base and is_generation_run_folder_basename(parent_base):
            return folder, parent, parent_base
        return folder, None, basename
    if os.path.isdir(nested):
        return nested, folder, basename
    return nested, folder, basename
