# AgriGPT Backend Agent — Claude Instructions

## Project Overview

FastAPI + LangGraph agent serving an agricultural chatbot.

- **Entry point**: `app.py`
- **Deploy**: EC2 via SSH (`.github/workflows/deploy.yml`)
- **LLM**: Google Gemini 2.5 Flash via LangChain
- **Tools**: Discovered dynamically from MCP servers (Alumnx, Vignan)
- **Memory**: MongoDB (`agrigpt.chats` collection)

## Architecture

```
Local development
  ├── /qa  (Claude Code skill)  → generates/updates tests, runs them, fixes failures
  └── pre-push hook             → blocks git push if tests fail

PR opened
  ├── claude-review.yml         → Claude reviews code, posts comment, approves/requests changes
  └── claude-test-gen.yml       → Claude generates tests from rules below, then runs them

Merge to main
  ├── run-tests.yml             → Runs pytest automatically (merge gate)
  └── deploy.yml                → SSH into EC2, git pull, pip install, restart service
```

## CI/CD Workflows

| Workflow              | Trigger               | What happens                                                        |
| --------------------- | --------------------- | ------------------------------------------------------------------- |
| `claude-review.yml`   | PR opened/updated     | Claude reviews, comments, auto-enables merge if approved            |
| `claude-test-gen.yml` | PR with `.py` changes | Claude generates tests per rules below → runs them → fixes failures |
| `run-tests.yml`       | Every push / every PR | Runs `pytest tests/` — must pass before merge                       |
| `deploy.yml`          | Push to `main`        | SSH deploy to EC2                                                   |

## Required GitHub Secrets

| Secret         | Purpose                         |
| -------------- | ------------------------------- |
| `GITHUB_TOKEN` | Auto-provided by GitHub Actions |
| `EC2_HOST`     | EC2 instance IP                 |
| `EC2_USER`     | SSH username (ubuntu)           |
| `EC2_SSH_KEY`  | Contents of `.pem` file         |

> Claude auth is handled by the installed GitHub App — no `ANTHROPIC_API_KEY` needed.

---

## Code Review Rules

When Claude reviews a PR, enforce all of these:

1. **No secrets in code** — all config via `os.getenv()` + `.env`
2. **`global_tool_results` must be cleared** at the start of every `/test/chat` call (thread safety)
3. **MCP client failures must be caught** — server down must not crash the app
4. **`/webhook` must not block** — long work goes in `BackgroundTasks`
5. **MongoDB operations** must handle connection errors
6. **All request/response bodies** use Pydantic models
7. **No hardcoded URLs** — MCP URLs from env vars only
8. **No bare `except`** — always catch specific exception types

---

## Test Strategy

> This section is the single source of truth for how Claude generates tests.
> The `claude-test-gen.yml` workflow tells Claude to read and follow these rules exactly.
> Do not add test logic to the workflow yaml — add it here.

### What to generate

Always generate or update `tests/test_app.py`. Always create `tests/__init__.py` (empty).

Analyse the current state of `app.py` to determine what functions and endpoints exist.
For every function and endpoint, generate tests covering:

- **Happy path** — expected inputs produce expected outputs
- **Edge cases** — empty strings, `None`, missing optional fields, zero-length lists
- **Error path** — dependencies fail (MongoDB down, MCP server unreachable, LLM error)

### Mocking rules

Never call real external services in tests. Always mock:

| Dependency                            | How to mock                                         |
| ------------------------------------- | --------------------------------------------------- |
| MongoDB (`chat_sessions`)             | `unittest.mock.patch("app.chat_sessions")`          |
| MCP HTTP calls (`httpx.Client`)       | `unittest.mock.patch("app.httpx.Client")`           |
| Gemini LLM (`ChatGoogleGenerativeAI`) | `unittest.mock.patch("app.ChatGoogleGenerativeAI")` |
| `app_agent.invoke` (LangGraph)        | `unittest.mock.patch("app.app_agent")`              |

Use `pytest-mock` (`mocker` fixture) or `unittest.mock.patch` as decorator/context manager.
Never import or use real env vars — patch `os.getenv` or set `monkeypatch.setenv`.

### Test framework

```
pytest
pytest-asyncio       # for async endpoints
httpx                # AsyncClient for FastAPI testing
pytest-mock          # mocker fixture
```

Use `httpx.AsyncClient(app=app, base_url="http://test")` for endpoint tests.
Mark async tests with `@pytest.mark.asyncio`.

### Fixtures to always define

```python
@pytest.fixture
def chat_id():
    return "test-chat-123"

@pytest.fixture
def phone():
    return "919999999999"

@pytest.fixture
def mock_ai_message():
    from langchain_core.messages import AIMessage
    return AIMessage(content="Test agricultural answer")
```

### Specific test cases required for this project

**`load_history`**

- Returns `[]` when `chat_id` not found in MongoDB
- Reconstructs `HumanMessage`, `AIMessage`, `SystemMessage` from stored dicts
- Ignores unknown roles silently

**`save_history` — sliding window**

- Stores all messages when count ≤ `MAX_MESSAGES` (20)
- Trims to last N human/AI pairs when count > `MAX_MESSAGES`
- Upserts (creates doc on first save, updates on subsequent)
- Sets `phone_number` only when provided

**`extract_sources_from_tool_results`**

- Extracts `filename` from `sources[].filename` dict format
- Extracts plain strings from `sources[]` list format
- Extracts `source` field from `results[].source` format
- Extracts `source` / `document` / `filename` from list-of-dicts format (VignanUniversity style)
- Returns `[]` when tool_results is empty
- Falls back to tool name when list result has no source field

**`has_meaningful_tool_results`**

- Returns `True` for non-empty list results
- Returns `True` when `sources` list is non-empty
- Returns `True` when `information` string is > 50 chars
- Returns `False` for error status results
- Returns `False` for empty list

**`/webhook GET`** — verify token handshake

- Returns `hub.challenge` when mode=subscribe and token matches
- Returns 403 when token mismatches

**`/webhook POST`** — message parsing

- Returns `{"status": "ok"}` for non-text message types
- Returns `{"status": "ok"}` when messages array is empty
- Parses phone number and body correctly for valid text message

**`/test/chat POST`**

- Returns `ChatResponse` with sources when MCP tools return results
- Uses Gemini fallback when `has_meaningful_tool_results` is False
- Clears `global_tool_results` at start of each request
- Returns 500 on unhandled exception

**`/hello GET`**

- Returns `{"message": "Hello Claude!!"}`

### Commit after generating

After writing `tests/test_app.py` and `tests/__init__.py`, commit with:

```
git add tests/
git commit -m "test: auto-generate tests via Claude [skip ci]"
git push
```

---

## Development Conventions

- Match existing code style exactly
- `global_tool_results` is a module-level list, cleared per request (single uvicorn worker)
- Tool names prefixed with server name on collision (e.g., `vignan_search`)
- System prompt injected fresh each request, stripped from history before MongoDB save
- Commit prefix convention: `feat:`, `fix:`, `test:`, `chore:`

## Onboarding (run once after cloning)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-asyncio httpx pytest-mock
sh setup-hooks.sh        # activates pre-push hook — tests run before every git push
```

## QA Skill — /qa

Type `/qa` in Claude Code at any time to:

- Generate or update `tests/test_app.py` for the current state of the code
- Run the tests automatically
- Fix any failures

**When to use it:** after writing new functions or endpoints, before opening a PR.
The pre-push hook will also block your push if tests are failing.

## Running Tests Manually

```bash
pytest tests/ -v
```
