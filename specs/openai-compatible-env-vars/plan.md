# Tasks: Provider-Specific Environment Variables for OpenAI-Compatible Providers

**Branch**: `openai-compatible-env-vars`
**Specs**: [spec.md](./spec.md)
**Architecture**: [architecture.md](./architecture.md)
**Status**: Not Started ⏳

---

## Phase 1: US-001 - Environment Variable Implementation (P1) ⏳ 🎯 MVP (~2h)

**Goal**: Implement provider-specific environment variable lookup with fallback for all four OpenAI-compatible providers

**Independent Test**: Each provider can be configured with its own endpoint using provider-specific env vars, and falls back to generic ones when specific vars aren't set

### Tasks

- [ ] T001 [P] [US1] Update LLM provider env var lookup in src/esperanto/providers/llm/openai_compatible.py
  - Add `OPENAI_COMPATIBLE_BASE_URL_LLM` and `OPENAI_COMPATIBLE_API_KEY_LLM` with fallback to generic vars
  - Update error message to mention both specific and generic env vars
  - Lines to modify: ~33-42, ~46-48

- [ ] T002 [P] [US1] Update Embedding provider env var lookup in src/esperanto/providers/embedding/openai_compatible.py
  - Add `OPENAI_COMPATIBLE_BASE_URL_EMBEDDING` and `OPENAI_COMPATIBLE_API_KEY_EMBEDDING` with fallback to generic vars
  - Update error message to mention both specific and generic env vars
  - Lines to modify: ~57-64, ~68-72

- [ ] T003 [P] [US1] Update STT provider env var lookup in src/esperanto/providers/stt/openai_compatible.py
  - Add `OPENAI_COMPATIBLE_BASE_URL_STT` and `OPENAI_COMPATIBLE_API_KEY_STT` with fallback to generic vars
  - Update error message to mention both specific and generic env vars
  - Lines to modify: ~63-72, ~76-80

- [ ] T004 [P] [US1] Update TTS provider env var lookup in src/esperanto/providers/tts/openai_compatible.py
  - Add `OPENAI_COMPATIBLE_BASE_URL_TTS` and `OPENAI_COMPATIBLE_API_KEY_TTS` with fallback to generic vars
  - Update error message to mention both specific and generic env vars
  - Lines to modify: ~57-66, ~70-74

**Checkpoint**: ✋ All four providers support provider-specific env vars with fallback

**Notes:**
- [Space for learnings and comments during implementation]

---

## Phase 2: US-002 - Test Suite Updates (P2) ⏳ (~2h)

**Goal**: Add comprehensive test coverage for provider-specific environment variables and fallback behavior

**Independent Test**: All tests pass, validating both provider-specific env vars and backward compatibility

### Tasks

- [ ] T005 [P] [US2] Add LLM provider-specific env var tests in tests/providers/llm/test_openai_compatible_provider.py
  - Test provider-specific env var takes precedence over generic
  - Test fallback to generic env var when specific not set
  - Test config parameter overrides env vars
  - Test direct parameter overrides everything
  - Verify existing tests still pass (backward compatibility)

- [ ] T006 [P] [US2] Add Embedding provider-specific env var tests in tests/providers/embedding/test_openai_compatible.py
  - Test provider-specific env var takes precedence over generic
  - Test fallback to generic env var when specific not set
  - Test config parameter overrides env vars
  - Test direct parameter overrides everything
  - Verify existing tests still pass (backward compatibility)

- [ ] T007 [P] [US2] Add STT provider-specific env var tests in tests/providers/stt/test_openai_compatible.py
  - Test provider-specific env var takes precedence over generic
  - Test fallback to generic env var when specific not set
  - Test config parameter overrides env vars
  - Test direct parameter overrides everything
  - Verify existing tests still pass (backward compatibility)

- [ ] T008 [P] [US2] Add TTS provider-specific env var tests in tests/providers/tts/test_openai_compatible.py
  - Test provider-specific env var takes precedence over generic
  - Test fallback to generic env var when specific not set
  - Test config parameter overrides env vars
  - Test direct parameter overrides everything
  - Verify existing tests still pass (backward compatibility)

- [ ] T009 [US2] Run full test suite to verify no breaking changes
  - Execute: `uv run pytest tests/providers/llm/test_openai_compatible_provider.py -v`
  - Execute: `uv run pytest tests/providers/embedding/test_openai_compatible.py -v`
  - Execute: `uv run pytest tests/providers/stt/test_openai_compatible.py -v`
  - Execute: `uv run pytest tests/providers/tts/test_openai_compatible.py -v`
  - Verify all tests pass

**Checkpoint**: ✋ Complete test coverage validates provider-specific env vars and backward compatibility

**Notes:**
- [Space for learnings and comments during implementation]

---

## Phase 3: US-003 - Documentation Updates (P2) ⏳ (~1h)

**Goal**: Update documentation to clearly explain new environment variable options and precedence

**Independent Test**: Documentation accurately describes all new env vars, provides examples, and explains precedence

### Tasks

- [ ] T010 [P] [US3] Update README.md with provider-specific env var documentation
  - Add section explaining provider-specific environment variables
  - Document all 8 new env vars (4 BASE_URL + 4 API_KEY)
  - Explain precedence: Direct params > Config > Provider-specific env > Generic env > Default
  - Add examples showing how to configure different endpoints for different provider types
  - Update "OpenAI-Compatible Endpoints" section (around line 351-381)

- [ ] T011 [P] [US3] Update .env.example with provider-specific env var examples
  - Add commented examples for all 8 new environment variables
  - Group with existing OPENAI_COMPATIBLE variables (if any)
  - Include helpful comments explaining when to use provider-specific vs generic vars

**Checkpoint**: ✋ Documentation complete and clear for end users

**Notes:**
- [Space for learnings and comments during implementation]

---

## Phase 4: US-004 - Backward Compatibility Validation (P1) ⏳ (~30min)

**Goal**: Verify that all existing configurations continue to work without changes

**Independent Test**: Full test suite passes, no breaking changes detected

### Tasks

- [ ] T012 [US4] Run complete test suite across all providers
  - Execute: `uv run pytest -v`
  - Verify all existing tests pass
  - Confirm no regressions in any provider functionality

- [ ] T013 [US4] Validate error messages are clear and informative
  - Test each provider without any env vars set
  - Verify error messages mention both provider-specific and generic options
  - Confirm error messages are user-friendly

**Checkpoint**: ✋ Backward compatibility confirmed, feature complete

**Notes:**
- [Space for learnings and comments during implementation]

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1: US-001 (Environment Variable Implementation) - Can start immediately
    ↓
Phase 2: US-002 (Test Suite Updates) - Requires Phase 1 complete
    ↓
Phase 3: US-003 (Documentation Updates) - Can run in parallel with Phase 2
    ↓
Phase 4: US-004 (Backward Compatibility Validation) - Requires Phases 1+2 complete
```

### User Story Independence

- **US-001 (P1)**: Core implementation - No dependencies ✅ MVP
- **US-002 (P2)**: Depends on US-001 being complete
- **US-003 (P2)**: Can run in parallel with US-002 after US-001 completes
- **US-004 (P1)**: Final validation - Depends on US-001 and US-002

### Parallel Opportunities

**Within Phase 1:**
- Tasks T001-T004 can run in parallel (different provider files)
- Recommended: Complete T001 first as a reference, then parallelize T002-T004

**Within Phase 2:**
- Tasks T005-T008 can run in parallel (different test files)
- Task T009 must run after T005-T008 complete

**Within Phase 3:**
- Tasks T010-T011 can run in parallel (different files)

**Across phases:**
- Phase 3 can start as soon as Phase 1 completes (parallel with Phase 2)

---

## Implementation Strategy

### 🎯 MVP First (Recommended)

1. Complete Phase 1: US-001 ✅
   - Implement provider-specific env vars for all 4 providers
   - **VALIDATE**: Manually test one provider with specific env var
2. Complete Phase 2: US-002 ✅
   - Add comprehensive tests
   - **VALIDATE**: All tests pass
3. Complete Phase 3: US-003 ✅
   - Update documentation
   - **VALIDATE**: Documentation is clear and accurate
4. Complete Phase 4: US-004 ✅
   - Final validation
   - **VALIDATE**: No breaking changes, feature complete

### 📈 Incremental Delivery

Each phase builds on the previous:
- After Phase 1: Core functionality works ✅
- After Phase 2: Functionality is tested ✅
- After Phase 3: Feature is documented ✅
- After Phase 4: Feature is production-ready ✅

### 🔧 Development Workflow

**For Phase 1 (Implementation):**
1. Start with T001 (LLM provider) as reference implementation
2. Test manually with environment variables
3. Once working, use same pattern for T002-T004
4. Commit after each provider is updated

**For Phase 2 (Tests):**
1. Write tests for one provider first (T005)
2. Run tests to ensure they work
3. Replicate pattern for other providers (T006-T008)
4. Run full test suite (T009)

**For Phase 3 (Documentation):**
1. Document while implementation is fresh in mind
2. Include real examples from manual testing
3. Review for clarity before moving to Phase 4

---

## Progress Tracking

**Emoji Legend:**
- ⏳ Not Started
- ⏰ In Progress
- ✅ Completed

**Update this file as you work:**
1. Change phase status emoji (⏳ → ⏰ → ✅)
2. Check off tasks as completed: `- [x]`
3. Add notes/learnings in the Notes sections
4. Update time estimates if needed

---

## Task Execution Tips

**Before starting a task:**
- Read the task description and line numbers carefully
- Review the current code in the file to understand the pattern
- Check if any [P] sibling tasks can run in parallel

**While executing:**
- Follow the exact pattern shown in architecture.md
- Maintain consistency across all four providers
- Test each change incrementally

**After completing a task:**
- Mark it complete: `- [x]`
- Run affected tests to verify no breakage
- Add any learnings to Notes section

**At checkpoints:**
- Validate the entire phase works as expected
- Test manually if needed
- Get user approval before continuing

---

## Reference: Implementation Pattern

Each provider should use this exact pattern (from architecture.md):

```python
# For base_url
self.base_url = (
    base_url or                                          # Direct parameter (highest)
    config.get("base_url") or                           # Config dict
    os.getenv("OPENAI_COMPATIBLE_BASE_URL_<TYPE>") or  # Provider-specific env var
    os.getenv("OPENAI_COMPATIBLE_BASE_URL")            # Generic env var (fallback)
)

# For api_key
self.api_key = (
    api_key or                                          # Direct parameter (highest)
    config.get("api_key") or                           # Config dict
    os.getenv("OPENAI_COMPATIBLE_API_KEY_<TYPE>") or  # Provider-specific env var
    os.getenv("OPENAI_COMPATIBLE_API_KEY")            # Generic env var (fallback)
)

# Updated error message
if not self.base_url:
    raise ValueError(
        "OpenAI-compatible base URL is required. "
        f"Set OPENAI_COMPATIBLE_BASE_URL_<TYPE> or OPENAI_COMPATIBLE_BASE_URL "
        "environment variable or provide base_url in config."
    )
```

**Where `<TYPE>` is:**
- `LLM` for language models
- `EMBEDDING` for embedding models
- `STT` for speech-to-text models
- `TTS` for text-to-speech models

---

## Reference: Test Cases Per Provider

Each provider test file should include these test cases:

1. **test_provider_specific_env_var_precedence**
   - Set both `OPENAI_COMPATIBLE_BASE_URL_<TYPE>` and `OPENAI_COMPATIBLE_BASE_URL`
   - Verify provider uses the specific one

2. **test_fallback_to_generic_env_var**
   - Set only `OPENAI_COMPATIBLE_BASE_URL`
   - Verify provider uses the generic one

3. **test_config_overrides_env_vars**
   - Set env var and pass config dict
   - Verify config takes precedence

4. **test_direct_parameter_overrides_all**
   - Set env var, config, and direct parameter
   - Verify direct parameter wins

5. **Existing tests should all pass** (backward compatibility)

---

## Notes

**Key Conventions:**
- [P] = Can run in parallel (different files, no dependencies)
- [US1] = Belongs to User Story 1
- File paths are absolute from project root
- Commit after each logical grouping of tasks
- Stop at checkpoints to validate

**Remember:**
- Maintain identical pattern across all four providers
- Test backward compatibility thoroughly
- Clear, informative error messages
- Document with examples

**Environment Variables Being Added:**
- `OPENAI_COMPATIBLE_BASE_URL_LLM`
- `OPENAI_COMPATIBLE_API_KEY_LLM`
- `OPENAI_COMPATIBLE_BASE_URL_EMBEDDING`
- `OPENAI_COMPATIBLE_API_KEY_EMBEDDING`
- `OPENAI_COMPATIBLE_BASE_URL_STT`
- `OPENAI_COMPATIBLE_API_KEY_STT`
- `OPENAI_COMPATIBLE_BASE_URL_TTS`
- `OPENAI_COMPATIBLE_API_KEY_TTS`
