Create a GitHub pull request with auto-generated summary from branch commits:

## Workflow

1. **Validate Current State**
   - Check current branch: `git branch --show-current`
   - Error if on `main` or `master` (can't create PR from main)
   - Check if branch has commits ahead of main: `git log main..HEAD --oneline`
   - If no commits, error with message "No commits to create PR from"

2. **Check Remote Status**
   - Check if remote branch exists: `git ls-remote --heads origin <branch-name>`
   - If branch not on remote:
     - Ask user: "Branch not pushed to remote. Push now? (yes/no)"
     - If yes: `git push -u origin <branch-name>`
     - If no: Exit with "Please push branch manually before creating PR"
   - Check if branch is up-to-date with remote

3. **Analyze Commits**
   - Get all commits in this branch: `git log main..HEAD --oneline`
   - Get full diff: `git diff main...HEAD --stat`
   - Identify commit types (feat, fix, docs, etc.) to understand scope
   - Note: Handle case where base branch might be 'master' instead of 'main'

4. **Scan for TODOs**
   - Search for TODO comments in the changeset: `git diff main...HEAD | grep -i "TODO"`
   - Parse TODO comments and their context
   - Extract file paths and line numbers if available
   - Group TODOs by category or file if helpful
   - If no TODOs found, omit the TODOs section entirely

5. **Generate PR Content**
   - **Title**:
     - If single commit: Use that commit message
     - If multiple commits of same type: `<type>: <summary of changes>`
     - If mixed types: Use most prominent type or "chore: multiple updates"
     - Follow conventional commit format

   - **Body**:
     ```markdown
     ## Summary
     - <Bullet point 1: key change>
     - <Bullet point 2: key change>
     - <Bullet point 3: key change>

     ## TODOs
     <Only include if TODOs found in changeset>
     - [ ] <TODO item 1 with file reference>
     - [ ] <TODO item 2 with file reference>
     ```

6. **Present for Review**
   - Show generated PR title and body
   - Ask user: "Create PR with this content? (yes/edit/cancel)"
   - If "yes": Proceed to step 7
   - If "edit":
     - Ask for new title (or keep current)
     - Ask for new description (or keep current)
     - Show updated content
     - Ask again: "Create PR with this content? (yes/cancel)"
   - If "cancel": Exit without creating PR

7. **Create Pull Request**
   - Use GitHub CLI: `gh pr create --title "..." --body "..."`
   - Use heredoc for multiline body:
     ```bash
     gh pr create --title "PR Title" --body "$(cat <<'EOF'
     PR body content here
     with multiple lines
     EOF
     )"
     ```
   - Capture PR URL from output
   - Display success message with PR URL

8. **Error Handling**
   - Check if `gh` CLI is installed: `gh --version`
   - If not installed: "GitHub CLI (gh) is not installed. Install from: https://cli.github.com/"
   - Check if authenticated: `gh auth status`
   - If not authenticated: "Run `gh auth login` to authenticate with GitHub"
   - Check if repo has remote: `git remote -v`
   - If no remote: "No git remote configured. Cannot create PR."
   - Handle git command failures gracefully
   - Provide clear error messages and next steps

## Important Notes

- **NEVER force push** - Only push with `-u` flag for new branches
- **Check authentication** - Ensure `gh` is authenticated before creating PR
- **Handle edge cases** - Empty branches, no remote, merge conflicts
- **Use interactive approval** - Get user confirmation before creating PR
- **Follow conventions** - Use conventional commit format for PR title
- **Be descriptive** - Generate meaningful summaries, not just commit lists

## Examples

**Single feature commit:**
```
Title: feat: add GPT-4 support for evaluation

Body:
## Summary
- Added GPT-4 model configuration
- Updated judge to support GPT-4 evaluation
- Added documentation for GPT-4 usage
```

**Multiple commits:**
```
Title: feat: improve testing infrastructure

Body:
## Summary
- Added comprehensive unit test suite
- Set up pytest configuration and fixtures
- Created test-engineer agent for automated testing
```

**With TODOs in changeset:**
```
Title: feat: add user authentication system

Body:
## Summary
- Implemented JWT-based authentication
- Added login and signup endpoints
- Created middleware for protected routes

## TODOs
- [ ] Add rate limiting to login endpoint (auth/routes.py:45)
- [ ] Implement refresh token rotation (auth/jwt.py:78)
- [ ] Add password reset functionality (auth/routes.py:120)
```

## Tips

- Use `/create-commits` first to organize your changes before creating PR
- Review the generated content carefully - edit if needed
- Keep PR title concise (under 72 characters)
- Check that branch is ready (tests pass, code formatted, etc.)
- TODOs section is auto-generated from code comments in your changeset
