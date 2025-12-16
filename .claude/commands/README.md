# VERA-MH Custom Commands

This directory contains custom slash commands for the VERA-MH project.

## Available Commands

### Development & Setup
- `/setup-dev` - Set up complete development environment (includes test infrastructure)

### Code Quality
- `/format` - Run code formatting and linting

### Running VERA-MH
- `/run-generator` - Run conversation generator with prompts
- `/run-judge` - Run conversation evaluator with prompts

### Testing
- `/test` - Run test suite (with coverage by default, use --no-cov to skip)
- `/fix-tests` - Fix failing tests iteratively until all pass, show branch-focused coverage
- `/create-tests [module_path] [--layer=unit|integration|e2e]` - Create tests (focused: single module, or coverage analysis: find and fix gaps)

### Git Workflow
- `/create-commits` - Analyze changes and create logical, well-organized commits (with optional branch creation)
- `/create-pr` - Create GitHub pull request with auto-generated summary

## Creating New Commands

To create a new command:
1. Create a `.md` file in this directory (e.g., `my-command.md`)
2. Write the prompt/instructions in the file
3. Use it with `/my-command` in Claude Code

## Command Best Practices

- Keep commands focused on a single task
- Use clear, imperative language
- Include parameter prompts for user input
- Document expected outputs

## Maintenance Guidelines

**When adding new testing commands:**
1. Update `.claude/agents/test-engineer.md` "Reference Documentation" section if the command contains testing patterns or conventions
2. Update this README and the main `README.md`

**When updating existing testing commands:**
1. The `test-engineer` agent reads command files directly, so updates automatically propagate
2. Verify that the agent's "Reference Documentation" section still accurately describes what each command provides
3. If command purpose changes significantly, update agent documentation

**Agent-Command Relationship:**
- Slash commands are living documentation that agents read via the Read tool
- Changes to commands automatically benefit agents (they read the files directly)
- Only update the agent when you add/remove commands or change what information they contain
