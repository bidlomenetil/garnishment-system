# Garnishment decision service

Reads a court garnishment letter with an LLM and routes it to `auto`, `needs_human` or
`reject`, recording an audit trail that explains the route and can be replayed.

This file is operational. The reasoning — where confidence comes from, why every threshold
is a hypothesis rather than a measurement, what is deliberately out of scope — is in
[`DESIGN.md`](DESIGN.md).

---

## Prerequisites

**Python 3.12, installed by [`uv`](https://docs.astral.sh/uv/).** Nothing else. No Docker, no
Postgres, no API key.

If you do not have `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

`make install` checks for it and prints that line if it is missing. The Python version is
pinned in `.python-version`, and `uv` installs it for you — the system Python is not used and
does not need to be 3.12.

---

## Green tests in two commands

**No API key is needed, and this is the default path.** `LLM_PROVIDER` defaults to `fake`,
which replays extraction payloads recorded offline through the same schema and the same
mapping code a live response would take. Only the transport is fake.

```bash
make install
make test
```

Expected: **658 passed, 3 deselected**, in a few seconds. The three deselected are the live
tests, which need a real provider key; see below.

Other targets:

```bash
make lint        # ruff
make typecheck   # mypy (strict over domain and application)
make check       # lint + typecheck + test
make eval        # the eval report — see below
make live        # the live provider tests — skipped without a key
```

---

## Running the service

```bash
make run    # uvicorn on http://localhost:8000
```

This uses `LLM_PROVIDER=fake` unless you say otherwise. To use a real reader, copy
`.env.example` to `.env` and set one of the three providers below.

| `LLM_PROVIDER` | Reader | Needs a key |
|---|---|---|
| `fake` (default) | Replays payloads recorded offline, looked up by the SHA-256 of the letter | no |
| `anthropic` | `AnthropicExtractor` — one forced tool call to the Messages API | yes |
| `openai` | `OpenAIExtractor` — one strict-JSON-schema reply from the Responses API | yes |

```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-...
ANTHROPIC_MODEL=claude-sonnet-5
```

```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5-nano-2025-08-07
OPENAI_REASONING_EFFORT=minimal   # leave empty to send no reasoning parameter at all
```

The two real adapters are two implementations of one port and behave identically: the same
prompt, the same schema, the same mapper, the same one-retry-then-`H01` semantics. `openai`
defaults to `gpt-5-nano`, the cheapest model on OpenAI's pricing page, which reads one letter
for roughly **$0.0003** — the whole 8-letter × 5-run `make eval` costs about a cent.

Both refuse an aliased model id at startup, because every decision record names the model that
produced it: `claude-sonnet-5-latest` and the undated `gpt-5-nano` are both rejected. Anthropic
and OpenAI pin differently — a dated snapshot for OpenAI, the bare generation id for Anthropic,
since Anthropic publishes no dated snapshots and `-latest` is *its* moving alias — see
`src/garnishment/adapters/llm/model_pinning.py`. An unknown `LLM_PROVIDER` also fails at
startup, a typo rather than one letter at a time.

`LOG_LEVEL` (default `INFO`) controls the decision logger; set it in `.env` alongside the
provider settings.

Interactive API docs are at `http://localhost:8000/docs`.

### Hitting it

```bash
ID=$(curl -s -X POST localhost:8000/decisions \
  -H 'content-type: application/json' \
  --data-binary "$(jq -Rs '{letter_text: .}' evals/samples/sample_03.txt)" \
  | jq -r .id)

curl -s "localhost:8000/decisions/$ID" | jq '{route, decisive_rule_id, reasons}'
```

For `sample_03.txt` — a restriction order citing EUR 4,250.00 and reducing it to
EUR 1,267.00 — that returns:

```json
{
  "route": "auto",
  "decisive_rule_id": "A00_AUTO_DEFAULT",
  "reasons": [
    "All required fields present, grounded, validated and consistent; lowest final confidence across the gated fields (debtor_name, amount, case_reference, plus account_iban when present) is 0.97 ≥ 0.85.",
    "Effective amount EUR 1267.00 taken from the restricted figure (RESTRICTION_SUPERSEDES, doctrine RESTRICTION_DOCTRINE_V1).",
    "IBAN checksum valid."
  ]
}
```

Reasons are listed in rule-table order, so the decisive rule appears among them rather than
first. `decisive_rule_id` is what names the rule that decided the route.

`GET` returns the full audit tail: the letter text and its SHA-256, the verbatim tool-call
payload, the model id, the prompt version, the routing version, the doctrine version, every
fired rule, and the four-part confidence breakdown (`llm` / `grounding` / `validator` /
`final`) for each field.

### Endpoints

| | |
|---|---|
| `POST /decisions` | `{"letter_text": "..."}` as `application/json`, **or the raw letter as `text/plain`** → `201` with the decision and a `Location` header |
| `GET /decisions/{id}` | `200` with the decision plus its audit tail |

`text/plain` exists because a court letter is multi-line and JSON forbids a literal newline
inside a string, so pasting a real letter into a JSON body fails on its first line break.
Posting it raw takes it exactly as written, which is the easier path from `/docs` or by hand:

```bash
curl -s -X POST localhost:8000/decisions \
  -H 'content-type: text/plain' \
  --data-binary @evals/samples/sample_02.txt | jq '{route, decisive_rule_id, reasons}'
```

Both content types converge on the same string and the same rules — blank is `422`,
over-length is `413` — and the stored decision is identical either way.

Errors are RFC 9457 problem details: `422` for a blank letter, `413` over 100 000 characters
(checked twice — a pre-parse `Content-Length` guard rejects an oversized body before it's
read, and the router itself checks the decoded character count), `404` for an unknown id,
`503` when the reader never answered (and **nothing is persisted** — see `DESIGN.md` §8),
`500` for anything unanticipated (logged with the real detail; the response body never
echoes it).

---

## The eval harness

```bash
make eval
```

Reads 8 labelled letters — the 3 from the assignment plus 5 adversarial ones — five times
each and prints a report: per letter, the expected route against every observed route, route
and decisive-rule stability, the resolved operative amount and the observed role confidences;
across the set, the `auto` rate and the wrong-`auto` count.

**The report is the deliverable.** Instability is never a build failure. The exit code is
non-zero on exactly one condition: a **wrong `auto`**, meaning a letter the label says a human
should have seen was cleared automatically.

```bash
uv run python evals/run_eval.py --runs 3 --out report.txt
```

Without an API key this runs against the fake reader, so stability is 5/5 by construction and
is not a signal about any model. The report says so at the top. `DESIGN.md` §9 explains what
it does and does not establish.

---

## Testing against a real provider

`tests/live/` is excluded from `make test` and skips unless `LLM_PROVIDER` names a real
provider *and* that provider's key is set. The tests themselves name no provider: they read
the three assignment letters with whichever reader is configured and assert the labelled
route, so either adapter can be checked with the same command.

```bash
export LLM_PROVIDER=openai OPENAI_API_KEY=sk-...   # or anthropic + ANTHROPIC_API_KEY
make live
make eval
```

This has been run against `gpt-5-nano`: amount-role labelling checked out, and a separate
ambiguity around `document_is_seizure_order` was found and fixed as a result. `DESIGN.md`
§11.5 and §11.7 carry the full account, including a residual confidence-stability finding
left deliberately untouched.

---

## Layout

```
src/garnishment/
  domain/        pure decision logic — stdlib imports only, enforced by a test
  application/   ports and use cases
  adapters/      llm/ (anthropic, openai, fake, schema, prompt, mapping), persistence/, system.py
  api/           FastAPI app, DI, DTOs, two routes
tests/           unit, property (hypothesis), integration, live
evals/           labelled letters, expectations.yaml, run_eval.py
docs/            system-design.excalidraw (open at excalidraw.com)
```

`docs/system-design.excalidraw` is a one-page whiteboard of the whole system: requirements,
load assumptions, the request flow with the probabilistic/deterministic boundary marked, the
extraction schema, confidence derivation, the rule table, the decision record, and what would
change in production. Open it by dragging the file onto https://excalidraw.com.

`route()` in `domain/routing.py` is the whole decision: one pure function of one argument,
no clock, no I/O. `domain/rules.py` is the ordered rule table it walks.

## Deployment boundaries

`make run` binds the API to all network interfaces on port 8000. The service has
no authentication or authorization layer: `GET /decisions/{id}` returns the
original letter and extraction audit data. Deploy behind an authenticated gateway
and restrict access to the API and database. The default SQLite URL writes to
`garnishment.db` in the current directory; persist and back up that file, or supply
an appropriate async SQLAlchemy `DATABASE_URL`. Schema creation happens during
application startup; a production schema migration workflow is not included.

The request middleware rejects oversized declared body lengths without converting
untrusted headers to Python integers. Requests without `Content-Length` still rely
on the decoded character limit; configure a request-size ceiling at the ingress
proxy for streamed uploads. Real provider tests incur external calls and remain
separate from the default offline test suite.

## Repository history and license

The first application source commit is `618054b` (2026-09-23 in its recorded
local timezone). Later commits under `experiments/` use historical author dates
to study Git timestamps; they do not establish application development in those
years. Existing history is preserved.

No license grant is currently included in this repository. Public visibility alone
does not grant redistribution rights; the owner must choose licensing terms before
this project can be presented as licensed open-source software.
