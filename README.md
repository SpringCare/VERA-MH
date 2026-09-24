# VERA-MH 2.0

[![CI](https://github.com/SpringCare/VERA-MH/workflows/CI/badge.svg)](https://github.com/SpringCare/VERA-MH/actions/workflows/ci.yml)

VERA-MH (Validation of Ethical and Responsible AI in Mental Health) is a framework for evaluating how AI systems respond in mental health conversations that raise safety concerns. It simulates conversations between clinically developed user personas and the chatbot under test, then has an LLM judge rate each conversation against a clinical rubric. The result is a standardized score researchers, developers, and clinicians can use to compare systems before, during, and after deployment.

This README covers **VERA-MH 2.0**, which adds:

- **The `vera` CLI**—a single entry point (`vera.py`) with `generate`, `judge`, `score`, and `pipeline` commands, replacing the separate scripts.
- **Targets**—self-contained evaluation bundles (personas, prompts, rubric) selected with `--target`. Two ship today: `SI` (suicidal ideation) and `HFO` (harm from others).
- **Config files**—any run can be described by a JSON config and replayed exactly; each command prints its fully resolved config when it starts.

2.0 introduces breaking changes; see the [CHANGELOG](CHANGELOG.md). The project is a continuous work in progress, and feedback is welcome under the [Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

**Papers:** [Reliability and Validity of VERA-MH](https://arxiv.org/abs/2602.05088) · [Concept paper](https://arxiv.org/abs/2510.15297) · [Announcement](https://www.springhealth.com/blog/introducing-vera-mh-new-standard-ethical-ai-mental-healthcare)

## Quick start

```bash
pip install uv                 # if you don't have it
uv sync
cp .env.example .env           # add ANTHROPIC_API_KEY, OPENAI_API_KEY, ... as needed
```

Run a smoke test (generate, judge, and score on 3 personas):

```bash
uv run python vera.py pipeline \
  -c gpt-4o \
  -u claude-sonnet-4-5-20250929:1 \
  -j gpt-5.4:1 \
  --target SI \
  --sample 3
```

- `-c` is the **chatbot** under test.
- `-u` is the **user** model that role-plays the personas (`model:repeats`).
- `-j` is the **judge** (`model:instances`).

Drop `--sample` to run every persona. Results land under `output/`: transcripts in `conversations/`, judge ratings in `evaluations/j_*/results.csv`, and scores in `evaluations/j_*/scores/`.

Each stage can also be run on its own (`vera generate`, `vera judge`, `vera score`), and `--help` on any command lists its flags. The full reference, including config files and resuming interrupted runs, is in [docs/cli.md](docs/cli.md).

To evaluate your own chatbot or API rather than a built-in model, see [docs/evaluating.md](docs/evaluating.md).

## Getting a reliable VERA-MH score

For a score comparable to the published VERA-MH numbers, use the recommended profile checked in at [`configs/recommended-SI.json`](configs/recommended-SI.json): all 100 SI personas, 30 turns, two user models (GPT 5.2 and Claude Opus 4.5), and GPT 5.4 as judge. Fill in the model under test and run it:

```bash
jq '.generation.chatbot.name = "<model-under-test>"' configs/recommended-SI.json \
  | uv run python vera.py pipeline --config -
```

This produces one evaluation per user model. The headline score is the **pooled** result across both:

```bash
uv run python scripts/pool_vera_scores.py <evaluation-folder-A> <evaluation-folder-B>
```

`vera pipeline` prints both evaluation folders when it finishes. How the score is computed and what the output files contain is covered in [docs/scoring.md](docs/scoring.md).

## Documentation

| Doc | What's in it |
|-----|--------------|
| [docs/cli.md](docs/cli.md) | Full `vera` CLI reference: every command and flag, config files, `--into`, model parameters, output layout |
| [docs/scoring.md](docs/scoring.md) | The VERA-MH score formula, score outputs, pooling, cross-model comparison, improvement reports |
| [docs/targets.md](docs/targets.md) | Targets, personas, and prompts: what ships, how they're structured, how to customize them |
| [docs/evaluating.md](docs/evaluating.md) | Connecting your own LLM, agent, or API as the chatbot under test |
| [docs/judge.md](docs/judge.md), [docs/rubric.md](docs/rubric.md) | How the rubric-driven judge works; adding a compatible rubric |
| [docs/architecture.md](docs/architecture.md) | Target architecture and design invariants |
| [docs/legacy-scripts.md](docs/legacy-scripts.md) | The pre-2.0 scripts (deprecated) |

## Contributing

Development conventions, the architecture map, testing policy, and git workflow are in [AGENTS.md](AGENTS.md). Claude Code users get slash commands (`/test`, `/format`, `/create-pr`, ...) described in [CLAUDE.md](CLAUDE.md).

```bash
uv run pytest -m "not live"    # CI-safe test suite, no API keys needed
pre-commit install             # optional: format and lint on commit
```

## Legacy scripts

The pre-2.0 entry points—`generate.py`, `judge.py`, `run_pipeline.py`, `judge/score.py`, and `scripts/run_recommended_vera_pipeline.sh`—still work, but they are **deprecated and will be removed soon**. Their flags differ from the `vera` CLI (for example `-c` means `--max-concurrent` in `generate.py`). Use `vera` for new work; if you still depend on a script, see [docs/legacy-scripts.md](docs/legacy-scripts.md) for its options and the `vera` equivalent.

## License

MIT with conditions: "software" is replaced by "materials" to better describe the project. See [LICENSE](LICENSE).
