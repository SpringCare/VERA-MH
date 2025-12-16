Analyze the current working tree and create logical, well-organized commits:

## Workflow

1. **Check Current Branch**
   - Run `git branch --show-current` to see current branch
   - If on `main` or `master`:
     - Read CLAUDE.md for any branch naming conventions
     - Analyze the changes to propose an appropriate branch name
     - Suggest branch name (e.g., `feature/add-authentication`, `fix/validation-bug`, `refactor/cleanup-utils`)
     - Ask user for approval (allow editing the name)
     - Create branch: `git checkout -b [branch-name]`

2. **Analyze Current State**
   - Run `git status` to see all changes (modified, untracked, deleted, staged)
   - Run `git diff` for unstaged changes
   - Run `git diff --cached` for staged changes
   - Note any existing staged files

3. **Check Guidelines**
   - Read CLAUDE.md to understand:
     - File organization rules
     - Commit message conventions
     - Code style requirements
   - Identify any changes that violate guidelines
   - If violations found, alert user and ask how to proceed

4. **Handle Untracked Files**
   - List all untracked files
   - For each untracked file, determine if it's:
     - New code that should be committed
     - Generated/build artifact (suggest .gitignore)
     - Temporary file (suggest ignoring)
   - Ask user how to handle unclear cases

5. **Propose Logical Commit Groupings**
   - Analyze all changes and group by:
     - **Type**: feat, fix, refactor, test, docs, chore, config
     - **Scope**: Related functionality or files
     - **Separation of concerns**: Don't mix unrelated changes
   - Follow conventional commit format if used in repo
   - Each commit should:
     - Have a single, clear purpose
     - Be atomic (could be merged/reverted independently)
     - Include related test changes with code changes

6. **Present Proposal**
   - Show each proposed commit with:
     ```
     Commit [N]: [type]: [clear description]
     Files:
       - path/to/file1.py
       - path/to/file2.py
     Rationale: [Why these files are grouped together]
     ```
   - Ask user for approval or modifications
   - Allow user to:
     - Approve all
     - Modify commit messages
     - Regroup files
     - Skip certain files

7. **Execute Commits**
   - For each approved commit group:
     - Stage the specific files: `git add [files]`
     - Create commit with message:
       ```
       [type]: [description]

       [Optional detailed explanation if needed]

       🤖 Generated with [Claude Code](https://claude.com/claude-code)

       Co-Authored-By: Claude <noreply@anthropic.com>
       ```
     - Show commit hash and message
   - After all commits, run `git log --oneline -n [count]` to show results

## Important Notes

- **NEVER run `git add .`** - Always add specific files per commit
- **Check for secrets** - Warn if `.env`, `credentials.json`, etc. are being committed
- **Preserve context** - If some files are already staged, incorporate them appropriately
- **Be interactive** - Get user approval before creating branches or commits
- **Handle errors** - If git commands fail, explain the error and suggest fixes
- **No empty commits** - Skip if there are no changes

## Example Commit Types

- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code restructuring without behavior change
- `test`: Adding or updating tests
- `docs`: Documentation changes
- `chore`: Maintenance tasks (deps, config, etc.)
- `style`: Code style/formatting only
