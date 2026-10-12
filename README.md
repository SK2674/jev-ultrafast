<img src="docs/banner.svg" alt="Jev Ultrafast · Browser Use × TypeSafe" width="100%" />

# Jev Ultrafast ⚡

> [!IMPORTANT]
> **The Browser Use Cloud waitlist is open.** Get early access to ultrafast browser agents in the cloud.
> **[Join the waitlist →](https://browser-use.com/ultrafast?utm_source=github&utm_medium=readme&utm_campaign=jev-ultrafast)**

**A browser agent with a dynamic, indexed action space.**

Give it one goal. [TypeSafe's Jev](https://docs.typesafe.ai/introduction) picks an operation and an element. A small LLM writes text only when the operation is `TYPE_TEXT`.

**Zürich → London on Google Flights in 7.1 seconds.** One natural-language goal, actual text generation, and loading waits included.

<a href="docs/demo.mp4"><img src="docs/demo.gif" alt="A real Google Flights search at 1× speed, with generated city names and dynamic operation/target decisions" width="100%" /></a>

[Watch the MP4](docs/demo.mp4) · [Measurements](docs/performance.md) · [Read the loop](jev_ultrafast/agent.py)

## The action space

Every observation produces a new element table:

```text
[1] button    Change ticket type · Round trip
[2] combobox  Where from?        · San Francisco
[3] combobox  Where to?          · empty
[4] textbox   Departure          · empty
...
```

The operations are `CLICK`, `TYPE_TEXT`, `SELECT`, `SCROLL_UP`, `SCROLL_DOWN`, `WAIT`, `DONE`, and `BLOCKED`. Only supported operations and targets are offered.

```text
                      one TypeSafe request
                     ┌───────────────────────────┐
page → element table → operation                 │
                     │ click_target              │
                     │ type_text_target          │
                     │ select_target, if present │
                     └─────────────┬─────────────┘
                         use the matching target
                                   │
                    CLICK [7] ─────┤──→ browser
                TYPE_TEXT [3] ─────┘
                          ↓
                   small LLM → text → browser
```

Target questions are speculative. If the operation is `CLICK`, only `click_target` can execute. Two decisions, **one network round trip**. Each target head contains only compatible elements. Native dropdown choices carry an observed element/option index.

There are no site-specific action scripts or prepared field strings in the policy. The Flights example supplies a goal and independently verifies the outcome. The screenshot renderer adds labels afterward; it does not drive the browser.

## Try it

```bash
git clone https://github.com/browser-use/jev-ultrafast.git
cd jev-ultrafast
uv sync
cp .env.example .env
# Add TYPESAFE_API_KEY and TEXT_MODEL_API_KEY.
uv run jev
```

Open **http://127.0.0.1:8766** and click **Start demo → Run automatically**. The inspector shows numbered elements, operation probabilities, target probabilities, and executed actions. **Choose next** pauses before execution.

Chrome connects through [Browser Harness](https://github.com/browser-use/browser-harness), installed by `uv sync`. Run `uv run browser-harness --doctor` if it needs connecting. Allow remote debugging in Chrome when prompted.

`TEXT_MODEL_API_KEY` is an OpenRouter key in the example configuration. The current demo uses `inception/mercury-2.5` with reasoning disabled. Gemini, GLM, and DeepSeek can also use the OpenAI-compatible text helper; configure the appropriate model, endpoint, and reasoning setting.

## Use the library

```python
from jev_ultrafast import Agent

with Agent(
    "https://www.google.com/travel/flights?hl=en",
    "Find one-way flights from Zurich to London on September 20, 2026, "
    "for one adult in economy. Stop when matching flight options are visible.",
) as agent:
    for state in agent.run():
        print(state["elapsed_ms"], state["status"])
```

Run with `uv run --env-file .env python your_script.py`. The same policy can run a different task:

```bash
uv run --env-file .env python examples/run.py \
  --url https://en.wikipedia.org/wiki/Main_Page \
  --goal 'Find and open the Wikipedia article about Gödel’s incompleteness theorems.'
```

`uv run --env-file .env python examples/flights.py --keep-open` performs the flight search, checks the actual route/date/results, and saves its trace. It does not select or book a flight.

## Self-hosted decision models

Two local servers implement the same `POST /v1/systemone` wire protocol as the hosted TypeSafe API, so `TYPESAFE_BASE_URL` is the only switch needed. Only the decision model changes; `TEXT_MODEL_API_KEY` is still an OpenAI-compatible helper for `TYPE_TEXT` values unless you point it at your own server.

| Backend | Start it | `.env` |
| --- | --- | --- |
| [laya](https://github.com/NandhaKishorM/laya) — recommended | `pip install "laya[serve]"` then `LAYA_DEVICE=cuda LAYA_MODELS=typed-decisions laya-serve` | `TYPESAFE_BASE_URL=http://127.0.0.1:8000`<br>`TYPESAFE_PROFILE=laya` |
| [jeff](https://github.com/logan-markewich/jeff) | `uv sync --extra dev`, `uv run hf download knowledgator/gliformer-large-v1 --local-dir models/gliformer-large-v1`, then `JEFF_API_KEYS=devkey JEFF_WARMUP=1 uv run jeff` | `TYPESAFE_BASE_URL=http://127.0.0.1:8000`<br>`TYPESAFE_PROFILE=jeff` |

```bash
TYPESAFE_BASE_URL=http://127.0.0.1:8000
TYPESAFE_PROFILE=laya
TYPESAFE_API_KEY=devkey
```

The `hosted` profile stays the default and is unchanged upstream behaviour: nothing ranked, nothing dropped, full policy text, `TYPESAFE_MODEL=jev-latest`.

### Read this before you expect it to work

Both backends answer correctly and confidently when the answer is **present in the page text**. Neither one reliably follows this agent's policy, and that is a model-capability limit, not a configuration problem. On six realistic Google Flights states, scored against the move a careful person would make:

| Backend | Correct | Operator confidence |
| --- | --- | --- |
| Hosted TypeSafe | runs the task | calibrated |
| laya `typed-decisions` | 1 / 6 | 0.05 – 0.08 |
| jeff `gliformer-large-v1` | 0 / 6 | 0.08 |

jeff's six answers were the *same distribution* on six different pages (`SELECT` .21 / `CLICK` .21 / `BLOCKED` .21), which is what a model with no signal looks like. Probing jeff directly shows why: asked which city is France's capital it answers `paris` at 0.94, and still 0.95 when the options are reordered, but asked how much change Marie got it answers `14` — the ticket price, the most salient number — rather than `8`. It reads documents; it does not reason over one. This agent asks the harder question: the right control for "Where from?" when the goal says "Zurich" and the field label shares no word with it.

**To run this agent on a local model you need to fine-tune it on this policy.** laya is the better starting point: it is 421M instead of 2.3 GB, 33 ms instead of 3 s per decision, and it already documents a fine-tuning path that took its own decision benchmark from 0.362 to 0.766. Until then treat a local backend as a working transport and a policy you still have to teach.

### Why the profiles exist

A self-hosted System 1 encoder does not tolerate what the hosted API does, and the failure is silent. Measured on a real Google Flights page (140 result links, a 140-option dropdown, three fields), the unbudgeted payload is a 53 KB request with a 145-option click head and a 32,579-character state. Against jeff that is refused twice over: **HTTP 422** above 64 options per head, and again above 20,000 characters. On a 4 GB card it can instead answer **HTTP 500** from a CUDA out-of-memory, because eager attention is quadratic in the prompt — measured, 2,385 tokens is fine, a 13,689-character state tries to allocate 572 MB and dies, and the server stays broken afterwards.

laya fails differently and just as quietly: it trims **each question to 256 tokens**, and it trims the *instructions* first. This agent inlines its policy into every head, so on laya the policy is silently deleted and the model is left with bare options. Removing the rules from three heads by hand changed the token count by 53 and the answer not at all — proof they were never being read.

So a profile bounds and ranks what is offered:

| Setting | `hosted` | `jeff` | `laya` | What it does |
| --- | --- | --- | --- | --- |
| `ELEMENT_BUDGET` | `0` | `16` | `8` | Actions offered across all heads, ranked by goal relevance |
| `OPTION_BUDGET` | `0` | `4` | `2` | Values kept per `<select>` |
| `STATE_CHARS` | `0` | `6000` | `800` | Page text is halved until the serialized state fits |
| `RULES` | full | `compact` | `terse` | Which policy text to inline (`full`/`compact`/`terse`) |
| `MODEL` | `jev-latest` | `jev-latest` | `typed-decisions` | Default model name; laya needs a real checkpoint name |

Ranking never removes the only route to a value: `fill` and `select` nodes are offered ahead of better-matching links, and a field keeps both its `TYPE_TEXT` and its "Open …" `CLICK`. Every setting is an independent `TYPESAFE_*` override, and `0` restores the unbounded hosted behaviour.

On the jeff side, `JEFF_MAX_BATCH=1` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` keep a small GPU from fragmenting, and its `JEFF_MAX_LABELS` / `JEFF_MAX_STATE_CHARS` can be raised if you also raise the matching budget above.

## Why it moves

- **One request per decision cycle.** Operation and target heads share the same observed state.
- **No screenshots in the default agent loop.** Jev consumes structured state. The inspector opts into screenshots; the video uses a separate continuous screencast.
- **One browser call per snapshot.** Read visible controls, their names, values, and text atomically. Keep references to the actual DOM nodes.
- **Validate the selected target.** Clicks check the document, form values, target, and nearby context. Animation alone does not force another prediction. Resolve current geometry and reject covered controls before input.
- **Wait for useful state.** After typing into a combobox, wait for visible suggestions, capped at 200 ms. Other interactions get at most two animation frames or 50 ms. These reads happen after execution is logged.
- **Keep hidden tabs rendering.** Focus emulation prevents background animation throttling without switching Chrome's visible tab.
- **Send visible text.** Offscreen article bodies and footers do not fill the model context.
- **Reuse an interrupted text request.** A generated value survives a stale-page retry only if the entire text-helper input is unchanged.

Every executed target is resolved from an observed node. The executor rechecks page freshness and click occlusion. Model output never becomes selectors, coordinates, shell commands, or executable JavaScript. Text-helper output must parse as a small JSON object before typing.

## Small enough to read

| File | Job |
| --- | --- |
| [agent.py](jev_ultrafast/agent.py) | The complete loop and text-helper handoff |
| [snapshot.js](jev_ultrafast/snapshot.js) | Atomic DOM snapshot, indexed controls, freshness guards |
| [browser.py](jev_ultrafast/browser.py) | Browser connection, current geometry, execution |
| [model.py](jev_ultrafast/model.py) | Dynamic operation/target heads and text generation |
| [questions.py](jev_ultrafast/questions.py) | Model instructions |
| [demo.py](jev_ultrafast/demo.py) | Local inspector |

## Evidence and limits

The current video is a **7,073 ms** Google Flights run. Timing starts after initial page observation and includes model calls, generated text, browser work, stale decisions, and loading waits. A fresh independent check verifies the one-way setting, Zürich, London, September 20, 2026, and visible flight options. The video plays at 1×, with no opening hold and a 0.5-second final hold.

In six alternating runs with identical models and settings, both versions passed **3/3**. Median task time went from **9.450 s → 7.092 s**, a **25% reduction**; median browser protocol calls went from **1,092 → 101**. This is three repeats of one task on one browser profile, not a general reliability benchmark.

The same policy opened the requested Wikipedia article in **2.798 s** and passed a local hotel search/filter task in **1.896 s**. Runs, failures, source hashes, and measurement boundaries are in [performance.md](docs/performance.md).

A `DONE` choice still requires independent outcome verification. The DOM reader handles common HTML and ARIA controls, not the full accessible-name specification. Shadow roots, frames, canvas, uploads, pop-up tabs, nested scrolling, and arbitrary keyboard widgets remain outside this MVP. Owned tabs share the existing Chrome profile.

## Development

```bash
uv run ruff check .
uv run pytest
node --check jev_ultrafast/static/app.js
node --check jev_ultrafast/snapshot.js
uv build
```

Tests are offline. `tests/test_agent.py` asserts that the `jeff` and `laya` profiles each fit their backend's limits and that the `hosted` profile is unchanged, and ends with one opt-in end-to-end check that runs only against a server named by `JEV_TEST_JEFF_URL` (default `http://127.0.0.1:8000`) and skips when it is unreachable or is not a jeff, so it never reaches the paid API. `uv run python scripts/check_guards.py` checks real controls in a local browser without model calls. Live examples and recording scripts make paid API calls. `scripts/record_flights.py <new-folder>` captures original browser timestamps; `scripts/render_demo.py <recording-folder>` renders that verified run at 1× and crops out the Google account strip. Credentials and raw traces stay ignored.

---

[Browser Use](https://github.com/browser-use/browser-use) · [Browser Harness](https://github.com/browser-use/browser-harness) · [TypeSafe speculative fan-out](https://docs.typesafe.ai/patterns/fan-out)
