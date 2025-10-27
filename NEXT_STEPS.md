# Next Steps for Brio-Esperanto Manual Testing

## What We Accomplished Today

### 1. Removed `<out>` Tags from LLM Prompts
- **Why**: LLMs are unreliable at formatting. Let them generate content, brio_ext handles fencing.
- **What changed**: Updated all adapters (Qwen, Llama, Mistral, Gemma, Phi) to remove `<out>` from prompts and stop tokens
- **Result**: Cleaner separation - LLM generates content, brio_ext wraps it in `<out>...</out>` for BrioDocs

### 2. Created Manual Testing Environment
- Download scripts for 7 models from registry.briodocs.ai
- Server startup script with numbered selection (1-7)
- Test runner with full pipeline visibility
- Models directory added to .gitignore

### 3. Started Component-Based Test System
- **New structure**: Separate files for system prompts, user prompts, content, insights
- **Goal**: Test exactly how BrioDocs assembles messages
- **Benefit**: Mix and match components for realistic test battery

## What's Next (Not Complete Yet)

### A. Finish Component-Based Test System

**Current State**:
- ✅ Created directory structure: `fixtures/prompts/{system,user,content,insights}/`
- ✅ Created sample components (pirate, patent_analyst, etc.)
- ✅ Created test_cases.yaml config
- ❌ Need to update test_with_llm.py to:
  1. Load components from files
  2. Assemble messages like BrioDocs does (system + content + insights + user)
  3. Use numbered model selection (1-7, openai) like server script
  4. Show all 4 pipeline steps clearly

**How BrioDocs Assembles Messages**:
```python
# System message gets: role description + source context
system_content = f"""
{system_prompt}

# SOURCE CONTEXT

## SOURCE CONTENT
{content}

## SOURCE INSIGHTS
{insight_1}

{insight_2}

## CONTEXT METADATA
- Source count: 1
- Insight count: 2
"""

messages = [
    {"role": "system", "content": system_content},
    {"role": "user", "content": user_prompt}
]
```

### B. Update test_with_llm.py

**Changes needed**:
1. Load test cases from test_cases.yaml
2. Load component files (system, user, content, insights)
3. Assemble messages in BrioDocs format
4. Update model selection to use numbers (1-7, openai)
5. Show 4-step pipeline clearly:
   - Step 1: Test → esperanto (show assembled messages)
   - Step 2: esperanto → LLM (show rendered prompt with ChatML tags)
   - Step 3: LLM → esperanto (show raw response)
   - Step 4: esperanto → Test (show fenced response)

### C. Create More Test Components

Add realistic BrioDocs prompts:
- `system/legal_clerk.txt` - Legal document review
- `user/make_concise.txt` - Conciseness task
- `user/extract_key_points.txt` - Extraction task
- `content/engagement_letter.txt` - Legal document
- `content/contract_sample.txt` - Contract
- `insights/contract_key_terms.txt` - Extracted terms

### D. Documentation Updates

Update these docs with the new approach:
- TESTING_QUICKSTART.md - How to use numbered model selection
- scripts/README.md - Explain component-based tests
- test_cases.yaml - Add more example test cases

## Ready to Commit

### Files Ready to Commit Now:
- All adapter changes (removed `<out>` tags)
- .gitignore (models directory)
- scripts/start_server.sh (numbered selection)
- scripts/download_models.sh (registry URLs)
- fixtures/prompts/* (component files)
- fixtures/test_cases.yaml (config)

### Commit Message:
```
feat: remove LLM fencing, add component-based test system

Remove <out> tags from LLM prompts - let LLMs generate content,
brio_ext handles all fencing. More reliable separation of concerns.

Add component-based test fixture system that assembles messages
like BrioDocs does (system + content + insights + user).

Changes:
- Remove <out> from all adapter prompts and stop tokens
- Update DEFAULT_STOP to empty list in renderer
- Simplify server script to use numbered selection (1-7)
- Add component-based test fixtures (prompts/system, user, content, insights)
- Add test_cases.yaml to configure test scenarios

INCOMPLETE: test_with_llm.py still needs update to:
1. Load and assemble components
2. Use numbered model selection
3. Show 4-step pipeline clearly

Next: Complete test runner update to match new component system.
```

### Files Not Ready (Need Work):
- test_with_llm.py - Still uses old hardcoded JSON, needs complete rewrite

## Testing Plan

Once test_with_llm.py is updated:

1. **Test Qwen 2.5 7B** (the main bug target)
   ```bash
   ./scripts/start_server.sh 1
   python scripts/test_with_llm.py inventor 1
   ```

2. **Compare with baseline**
   ```bash
   python scripts/test_with_llm.py inventor openai
   ```

3. **Test other models** (Llama, Mistral, Phi)

4. **Create custom test cases** by mixing components

## Questions to Resolve

1. How exactly does BrioDocs format the system message with multiple insights?
2. Are there other message assembly patterns besides system+content+insights+user?
3. What other realistic test scenarios should we include?
4. Should we add conversation history (multi-turn) support?

## Architecture Summary

```
BrioDocs Application
  ↓ (assembles from components)
  ↓ system + content + insights + user
  ↓
brio_ext (BrioAIFactory)
  ↓ (renders to LLM format - ChatML, [INST], etc.)
  ↓ (NO <out> tags in prompt!)
  ↓
llama.cpp server (or cloud API)
  ↓ (generates natural content)
  ↓
brio_ext
  ↓ (wraps in <out>...</out>)
  ↓
BrioDocs Application
  ↓ (strips tags, uses content)
```

Key insight: **LLMs don't know about `<out>` tags. They just generate content. brio_ext handles all formatting.**
