"""The job loop: runs generation and judging jobs with a concurrency cap per kind.

Import the submodule you need (`workers.loop`, `workers.events`,
`workers.reporters`). Nothing here imports a domain package (`generate`,
`judge`, `score`, `vera_cli`): the domain hands its jobs to the loop, never the
reverse. Design: docs/design/job-loop.md and docs/design/terminal-interface.md.
"""
