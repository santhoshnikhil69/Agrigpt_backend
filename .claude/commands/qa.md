Analyze all Python files in this project to understand the current code.

Then do these two things in order:

---

## Part 1 — Update CLAUDE.md

Read `app.py` and read the current `CLAUDE.md`.

Compare the "Specific test cases required" section in CLAUDE.md against every function and endpoint that exists in `app.py` right now.

- If a function or endpoint exists in `app.py` but has no entry in CLAUDE.md → add it with the correct test cases to cover
- If a function or endpoint was removed from `app.py` but still has an entry in CLAUDE.md → remove that entry
- If a function or endpoint exists in both but the CLAUDE.md entry is outdated (e.g. parameters changed) → update it
- Do not change anything else in CLAUDE.md — only the "Specific test cases required" section

---

## Part 2 — Generate and run tests

Follow the updated **Test Strategy** section in `CLAUDE.md` exactly to generate or update `tests/test_app.py`.

Steps:

1. Read the updated `CLAUDE.md` — use the Test Strategy section as the single source of truth
2. Read `app.py` — identify every function and endpoint that exists right now
3. Compare with `tests/test_app.py` if it exists — find anything missing or outdated
4. Write or update `tests/test_app.py` so every function and endpoint is covered
5. Also create `tests/__init__.py` if it does not exist (empty file)
6. Run the tests: `pytest tests/ -v --tb=short`
7. If any tests fail, read the failure output, fix `tests/test_app.py`, and run again
8. Report: what was added/updated in CLAUDE.md, how many tests were added/updated, how many pass

Rules:

- Never remove existing passing tests
- Never call real external services — mock everything as described in CLAUDE.md
- Never ask the user what to test — decide based on what exists in the code
