# `vera_cli` — how the unified CLI fits together

This package is the argument-and-config layer behind `vera.py`. It resolves user
input into canonical values and then calls domain functions. It contains no
generation, judging, or scoring logic.

Currently implemented: `generate`. `judge`, `score`, `pool`, `pipeline`, and
`resume` are specified in [../docs/architecture.md](../docs/architecture.md) but
not built yet.

## Layout

| Module | Responsibility |
|---|---|
| `../vera.py` | Root parser and dispatcher. Registers each command, routes to its handler, turns `ConfigError` into a standard CLI error. |
| `generate.py` | The `generate` command: flags, input resolution, and the call into the generation domain. The reference implementation for new commands. |
| `config.py` | Shared input handling: loading config JSON, the config-or-flags rule, path resolution, resolved-run rendering. |
| `targets.py` | Target discovery and manifest validation — turning a target name into concrete, verified file paths. |
| `../utils/config_schema.py` | *Not a CLI module.* Shared canonical types (`RunConfig` and friends) in the leaf `utils/` layer, so domain packages may consume them too. Validation and serialization only; no parsing, no defaults. |

## Adding a command

**Registering** a command means binding its name in the root parser and attaching
the function to call for it. A command's `register` function does both.

`subparsers.add_parser("<name>")` binds the name: it maps the literal word the
user types to a new, independent `ArgumentParser` that owns that command's flags.
Flags belong to that sub-parser rather than the root one, which is why `-c` is
free to mean something different under each command.
`parser.set_defaults(handler=run)` attaches the handler by putting a `handler`
attribute on the parsed namespace, so the namespace carries its own destination
and `vera.py` dispatches with `args.handler(args)` instead of comparing command
names. Those two dispatch-added attributes, `command` and `handler`, are what
`config.DISPATCH_ATTRIBUTES` exists to filter out — see the flag-presence rule
below, which infers user input from what is present on the namespace.

**An operation does not exist until it is registered.** A command module can be
complete and tested, and `vera <name>` will still fail with `invalid choice`
until `build_parser` calls its `register` — the module is unreachable code. Two
steps:

1. In your command module, define `register(subparsers)`. It adds a subparser,
   declares that command's flags, and ends with
   `parser.set_defaults(handler=run)` so the dispatcher knows what to call.
2. In `vera.py`, import that `register` and call it inside `build_parser`.

Your handler receives the parsed `argparse.Namespace` and returns an exit code.

## The rules a command must follow

**One input form per run.** A run is defined either by CLI flags or by a config
file — never a mixture. Invocation controls (`--sample`, `--debug`, `--print`)
are exempt and may accompany either form. `config.resolve_input` enforces this;
each command supplies its own run-defining flags and valid config sections,
because the rule is shared but the fields are per-command.

**Flag presence is the signal, so run-defining flags use
`argparse.SUPPRESS`.** A suppressed flag is absent from the namespace unless the
user passed it, which is how the rule above is enforceable. `None` cannot serve
as that sentinel because `None` is a legitimate value for several flags. The
consequence: parser defaults are unavailable, so CLI defaults live in a
`DEFAULTS` dict beside the flags and are applied during resolution.

It is also what lets `resolve_input` derive the run-defining set by subtracting
the command's invocation-only flags, rather than each command maintaining a
second list. A flag added to a command's parser is covered by the rule
automatically.

**Config-driven runs state every behavior field.** Defaults are a CLI
convenience; a stored config is meant to be a complete, reproducible
description, so a missing field is an error rather than something the code
fills in.

**Unknown config fields are rejected, not ignored.** A typo, or a section
belonging to a command that does not exist yet, fails loudly.

**Resolve fully, then execute.** Names become paths, relative becomes absolute,
and every referenced file is checked before any model is called. CLI paths
resolve against the working directory; config paths resolve against the
repository root, since config files are checked in and shared.

**Hand domains resolved values.** Domain entry points receive canonical types —
never an `argparse.Namespace`, a config file, or a target manifest — and are
called as Python functions, never as subprocesses or through another parser.

## Transitional state

`generate` calls into the root `generate.py` module, which is a legacy script
whose reusable functions have not yet moved into a permanent `generate/`
package. Two stopgaps live there and are labeled as such:
`run_for_user_models`, which expands a run's user models into individual
generations, and `_legacy_model_config`, which flattens a `ModelSpec` into the
dict shape the old signature expects. Both disappear when the generation domain
accepts `ModelSpec` directly. See
[../docs/architecture.md](../docs/architecture.md) for the boundary contract.
