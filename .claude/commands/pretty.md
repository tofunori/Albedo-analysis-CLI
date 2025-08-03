---
description: "Make code more readable and beautiful while preserving exact functionality"
---

I'll improve code readability while preserving exact functionality.

## Strategic Thinking Process

<think>
Before making any changes, I need to consider:

1. **Code Analysis**
   - What makes this code "ugly" or hard to read?
   - Are there complex nested structures that need simplification?
   - Is the naming unclear or inconsistent?
   - Are there repeated patterns that could be abstracted?

2. **Safety Considerations**
   - Which changes are purely cosmetic (safe)?
   - Which might affect behavior (risky)?
   - Are there implicit dependencies on current structure?
   - Could renaming break external references?

3. **Improvement Strategy**
   - Priority 1: Clear naming (variables, functions, files)
   - Priority 2: Reduce complexity (extract functions, simplify logic)
   - Priority 3: Remove redundancy (DRY principle)
   - Priority 4: Improve type safety (if applicable)

4. **Validation Approach**
   - How can I ensure functionality remains identical?
   - What tests exist to verify behavior?
   - Should I add temporary logging to verify flow?
</think>

Based on this analysis, let me create a safety commit and backup:

```bash
# Create a commit before changes
git add -A
git commit -m "Backup before prettifying code" || echo "No changes to commit"

# Create backup folder
BACKUP_DIR="$HOME/.claude/.prettify_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo "Creating backup at: $BACKUP_DIR"
```

I'll identify files to beautify based on:
- Files you specify, or if none specified, analyze the entire application
- Recently modified code
- Our conversation context

```bash
# Copy files to backup before modifications
if [ -n "$ARGUMENTS" ]; then
    cp -r "$ARGUMENTS" "$BACKUP_DIR/" 2>/dev/null || true
else
    # Backup all source files based on project structure
    echo "Backing up source files..."
fi
```

I'll improve:
- Variable and function names for clarity
- Code organization and structure
- Remove unused code and clutter
- Simplify complex expressions
- Group related functionality
- Fix loose or generic type declarations
- Add missing type annotations where supported
- Make types more specific based on usage

My approach:
1. Analyze current code patterns and type usage
2. Apply consistent naming conventions
3. Improve type safety where applicable
4. Reorganize for better readability
5. Remove redundancy without changing logic

I'll ensure:
- All functionality remains identical
- Tests continue to pass (if available)
- No behavior changes occur
- Backups are available for rollback

After beautifying, I'll:
- Show a summary of improvements
- Verify everything still works
- Create another commit with the improvements

```bash
# After prettifying, commit the changes
git add -A
git commit -m "Prettify code: improve readability and organization" || echo "No changes made"
```

**Important**: I will NEVER:
- Add "Co-authored-by" or any Claude signatures
- Include "Generated with Claude Code" or similar messages
- Modify git config or user credentials
- Add any AI/assistant attribution to the commit

This helps transform working code into maintainable code without risk.