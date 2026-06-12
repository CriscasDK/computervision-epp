---
name: changelog-updater
description: Automatically updates the CHANGELOG.md file whenever a new commit is made. Use this skill immediately before or during any process where you are creating a commit for the user. It guarantees that the changelog is kept in sync with the commit history.
---

# Changelog Updater

This skill ensures that every codebase change is properly documented in the project's `CHANGELOG.md` file before a commit is finalized.

## When to Use This Skill

- Any time the user asks you to "make a commit", "commit my changes", or similar.
- When you are completing a task and are about to commit the changes to the repository.
- After creating a conventional commit message, but *before* you execute the `git commit` command.

## How to use this skill

Whenever you are asked to commit changes, follow this exact workflow:

### Step 1: Generate the Commit Message
If not already provided or generated, create a suitable, conventional commit message for the changes currently staged or to be staged.

### Step 2: Update the CHANGELOG.md file
Read the current `CHANGELOG.md` file (if it exists). 
If it does not exist, create a new `CHANGELOG.md` file in the root of the project with a `# Changelog` header.

Prepend the new commit's information to the top of the changelog (or directly under an unreleased/current version header, if applicable).
Format the entry clearly. A good default format is:
```markdown
## [Date: YYYY-MM-DD]
- **<Type of commit>** (e.g. feat, fix, chore): <Commit message summary>
  - <Optional bullet points with more details if the commit is large>
```

### Step 3: Stage the CHANGELOG.md
Run `git add CHANGELOG.md` to stage the updated changelog along with the other files you are committing.

### Step 4: Execute the Commit
Run the `git commit -m "<Commit message>"` command with the generated commit message and include both the code changes and the changelog update in this single commit.
