# User-side language

By default the user-side LLM (the one playing the persona) writes in English. `--user-language` makes it write in another language:

```bash
uv run python vera.py generate -c gpt-5.4 -u gpt-5.2:1 --target SI --user-language "Mexican Spanish"
```

In a config file the key is `generation.user_language`. Config files state it explicitly, `"English"` included.

## What it changes

For any language other than English, one line is added at the end of the persona prompt:

```
- Write every message in Mexican Spanish, as a native speaker would
```

Nothing else changes: the persona backstory, the rubric, and the chatbot's prompt stay as they are, so the chatbot answers in whatever language it chooses. English adds no line at all (case and surrounding spaces are ignored), so English runs are identical to runs made before this option existed.

## Regional variants

The value is free text and goes into the prompt as written. Name the variant the way a person would: `"Mexican Spanish"`, `"Spanish (Spain)"`, `"Brazilian Portuguese"`, `"Canadian French"`. Models follow a name like that more reliably than a locale code such as `es-MX`.

## Things to keep in mind

- **The personas are written in English.** The model translates the backstory as it goes; names, places, and cultural details stay as written.
- **The judge rubric is in English.** The judge reads the conversation in whatever language it was written in. Check a sample of evaluations before comparing scores across languages.
- **Scores across languages are not directly comparable** until judging in that language has been validated against clinician ratings, as was done for English.
