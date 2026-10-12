"""TypeSafe makes choices; an optional small OpenAI-compatible model writes field values."""

import json
import math
import os
import re
import time

import httpx

from .questions import (
    NEXT_ACTION,
    NEXT_ACTION_COMPACT,
    NEXT_ACTION_TERSE,
    TARGET,
    TARGET_COMPACT,
    TARGET_TERSE,
    TEXT_VALUE,
)

CLIENT = httpx.Client(http2=True, timeout=25)
HOSTED_URL = "https://api.typesafe.ai"
SYSTEM_ONE = "/v1/systemone"
OPERATIONS = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}
# Entering a value is only possible through these, so they are never ranked away.
ESSENTIAL = {"fill", "select"}

# A hosted model reads a real page's whole element table. A self-hosted System 1 encoder does
# not: jeff refuses oversized requests and runs out of VRAM, and laya trims each question to its
# own token budget. A profile therefore bounds, ranks, and shortens what is offered, instead of
# letting a silent truncation drop the tail of the table or the rules themselves.
DEFAULTS = {"ELEMENT_BUDGET": False, "OPTION_BUDGET": False, "STATE_CHARS": False,
            "RULES": "", "MODEL": ""}
RULE_SETS = {
    "": (NEXT_ACTION, TARGET),
    "compact": (NEXT_ACTION_COMPACT, TARGET_COMPACT),
    "terse": (NEXT_ACTION_TERSE, TARGET_TERSE),
}
PROFILES = {
    # Unchanged upstream behaviour: nothing ranked, nothing dropped, full policy text.
    "hosted": {},
    # jeff caps a head at 64 options and a state at 20,000 characters, and its eager attention
    # is O(n^2) in the prompt, so a 4 GB card runs out of memory somewhere past 3,400 tokens.
    "jeff": {"ELEMENT_BUDGET": 16, "OPTION_BUDGET": 4, "STATE_CHARS": 6000, "RULES": "compact"},
    # laya caps each question at 256 tokens and the whole state at 1,024, and it trims the
    # instructions first, so the rules must be terse to leave the options any room. Its
    # `jev-latest` is not a checkpoint name, so the decision checkpoint has to be named.
    "laya": {"ELEMENT_BUDGET": 8, "OPTION_BUDGET": 2, "STATE_CHARS": 800, "RULES": "terse",
             "MODEL": "typed-decisions"},
}
WORD = re.compile(r"[^\W_]+")
# Words that appear in almost any goal and say nothing about which control advances it.
STOP = frozenset(
    """a an and any are as at be been but by can do each for from has have how if in into is it its of
    on or set that the their them then there these this to was what when where which who will with you
    your""".split()
)
TRUTHY = {"1", "true", "on", "yes"}
FALSY = {"0", "false", "off", "no"}


def settings():
    """The active backend profile, with TYPESAFE_* env overrides applied on top."""
    profile = PROFILES.get(os.environ.get("TYPESAFE_PROFILE", "hosted").lower(), {})
    out = {**DEFAULTS, **profile}
    for key, value in out.items():
        if isinstance(value, str):
            # A name, not a number: the environment replaces it verbatim.
            override = os.environ.get(f"TYPESAFE_{key}", "").strip()
            if override:
                out[key] = override
            continue
        raw = os.environ.get(f"TYPESAFE_{key}", "").strip().lower()
        if raw in TRUTHY:
            out[key] = True
        elif raw in FALSY:
            out[key] = False
        elif raw:
            out[key] = int(raw)
    return out


def rules_of(config):
    return RULE_SETS.get((config or {}).get("RULES", ""), RULE_SETS[""])


def systemone_url():
    return os.environ.get("TYPESAFE_BASE_URL", HOSTED_URL).rstrip("/") + SYSTEM_ONE


def post_json(url, key, body):
    for attempt in range(3):
        try:
            response = CLIENT.post(url, json=body, headers={"Authorization": f"Bearer {key}"})
        except httpx.HTTPError:
            raise RuntimeError("Model connection failed; no action executed.") from None
        if response.status_code in {429, 529, 503} and attempt < 2:
            time.sleep(0.5 * 2**attempt)
            continue
        if response.is_error:
            raise RuntimeError(f"{_problem(response)}; no action executed.")
        return response.json()
    raise RuntimeError("Model unavailable")


def _problem(response):
    """One-line reason, preferring a self-hosted backend's own limit over its status code."""
    reason = f"Model provider returned HTTP {response.status_code}"
    if not response.headers.get("content-type", "").startswith("application/json"):
        return reason
    detail = response.json()
    detail = detail.get("detail") or detail.get("error")
    first = detail[0] if isinstance(detail, list) and detail else detail
    if isinstance(first, dict):
        return f"{reason}: {first.get('msg') or first.get('message') or first}"
    return f"{reason}: {first}" if first else reason


def validate_choice(answer, ids):
    try:
        probabilities = answer["probabilities"]
        numbers = [*probabilities.values(), answer["confidence"]]
        valid = (
            answer["choice"] in ids
            and set(probabilities) == set(ids)
            and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in numbers)
            and abs(sum(probabilities.values()) - 1) < 0.02
            and probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Invalid TypeSafe response; no action executed.")
    return answer


def terms(text):
    """Meaningful lowercase words of a label, value, or goal."""
    return {w for w in (t.lower() for t in WORD.findall(text or "")) if len(w) > 1 and w not in STOP}


def relevance(goal, actions):
    """How many goal words an element's own label and value explain."""
    best = 0
    for action in actions:
        named = terms(action.get("label")) | terms(action.get("value")) | terms(action.get("current_value"))
        best = max(best, len(goal & named))
    return best


def retained(actions, goal, element_budget, option_budget):
    """Element actions to offer, most goal-relevant first, DOM order breaking ties.

    Every action of a kept node survives, so a field keeps both its fill and its
    "Open ..." click. A native dropdown keeps only its most relevant values, and an
    unbounded backend keeps every observed control in DOM order.
    """
    groups, controls = {}, {}
    for action in actions:
        if action["kind"] in OPERATIONS:
            groups.setdefault(action["node"], []).append(action)
        else:
            controls[action["id"].upper()] = action
    for node, siblings in groups.items():
        # Only a <select> produces this many siblings; a field's pair must not be split.
        if option_budget and len(siblings) > option_budget and all(a["kind"] == "select" for a in siblings):
            ranked = sorted(siblings, key=lambda a: -relevance(goal, [a]))
            groups[node] = ranked[:option_budget]
    if not element_budget:
        # Nothing is dropped, so the offered order stays the observed DOM order.
        return [a for siblings in groups.values() for a in siblings], controls
    order = {node: i for i, node in enumerate(groups)}
    ranked = sorted(groups, key=lambda n: (-relevance(goal, groups[n]), order[n]))
    # A field labelled "Where from?" shares no word with the goal, so relevance alone would
    # drop it behind every link. Typing and selecting are the only routes to a value, so those
    # nodes are offered first; the sort above leaves each tier in relevance order.
    ranked.sort(key=lambda n: 0 if any(a["kind"] in ESSENTIAL for a in groups[n]) else 1)
    kept = []
    for node in ranked:
        if len(kept) + len(groups[node]) > element_budget:
            continue
        kept.extend(groups[node])
    return kept, controls


def action_space(actions, goal="", config=None):
    """One index per observed element; each operation has its own valid target choices."""
    cfg = config if config is not None else settings()
    kept, controls = retained(actions, terms(goal), cfg["ELEMENT_BUDGET"], cfg["OPTION_BUDGET"])
    elements, indices, targets = [], {}, {}
    for action in kept:
        kind = action["kind"]
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = {k: action[k] for k in ("role", "value", "checked", "selected", "expanded") if k in action}
            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])
            if kind == "select":
                element["value"] = action.get("current_value", "")
                element["options"] = []
            elements.append(element)
        index = indices[node]
        operation = OPERATIONS[kind]
        group = targets.setdefault(operation, {})
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        target = index
        if kind == "select":
            target = f"{index}:{len(element['options']) + 1}"
            element["options"].append({"index": target, "label": action["label"], "value": action["value"]})
        group[target] = action
    return elements, targets, controls


def state_size(state):
    """Serialized state size, measured conservatively against any backend's own rendering."""
    return len(json.dumps(state, ensure_ascii=False, separators=(",", ":")))


def fit_state(state, limit):
    """Halve page text until the state fits. The element table is already budgeted and ranked."""
    text = state["page"]["text"]
    while text and limit and state_size(state) > limit:
        text = text[: len(text) // 2]
        state["page"]["text"] = text
    return state


def choose(state, goal, history):
    cfg = settings()
    elements, targets, controls = action_space(state["actions"], goal, cfg)
    rules = rules_of(cfg)
    labels = {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
        "SELECT": "Select an observed dropdown value.",
    }
    operations = {key: labels[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    operations.update(DONE="Every requirement is visibly satisfied.", BLOCKED="No supported operation can progress.")
    questions = {
        "operation": {"type": "choice", "criteria": operations, "instructions": {"goal": goal, "rules": rules[0]}}
    }
    for operation, candidates in targets.items():
        questions[operation.lower() + "_target"] = {
            "type": "choice",
            "criteria": {
                index: {
                    "element": f"[{index}] {a['label']}",
                    "current_value": a.get("current_value", a.get("value", "")),
                    **{k: a[k] for k in ("role", "checked", "selected", "expanded") if k in a},
                }
                for index, a in candidates.items()
            },
            "instructions": {"goal": goal, "operation": operation, "rules": list(rules)},
        }
    body = {
        "model": os.environ.get("TYPESAFE_MODEL") or cfg["MODEL"] or "jev-latest",
        "state": {
            "page": {k: state[k] for k in ("url", "title", "text")},
            "elements": elements,
            "recent_actions": [
                {k: h.get(k) for k in ("action", "kind", "text", "page_changed")} for h in history[-10:]
            ],
        },
        "questions": questions,
    }
    fit_state(body["state"], cfg["STATE_CHARS"])
    started = time.perf_counter()
    result = post_json(systemone_url(), os.environ["TYPESAFE_API_KEY"], body)
    operation_answer = validate_choice(result["answers"].get("operation", {}), operations)
    operation = operation_answer["choice"]
    target = None
    target_answer = None
    probabilities = {}
    if operation in targets:
        # Unused target heads cannot cause an action. Validate the head selected by the operation.
        target_answer = validate_choice(result["answers"].get(operation.lower() + "_target", {}), targets[operation])
        target = target_answer["choice"]
        choice = targets[operation][target]["id"]
        probabilities = {a["id"]: target_answer["probabilities"][index] for index, a in targets[operation].items()}
    else:
        choice = controls[operation]["id"] if operation in controls else operation
        probabilities[choice] = operation_answer["probabilities"][operation]
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": operation_answer["confidence"],
        "probabilities": probabilities,
        "operation_probabilities": operation_answer["probabilities"],
        "target_probabilities": target_answer["probabilities"] if target_answer else {},
        "target_confidence": target_answer["confidence"] if target_answer else None,
        "raw_answers": result["answers"],
        "model": result["model"],
        "usage": result.get("usage", {}),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "request": body,
    }


def field_context(goal, action, page, history):
    return {
        "goal": goal,
        "field": {k: action.get(k) for k in ("label", "role", "value")},
        "page": {"title": page["title"], "text": page["text"][:6000]},
        "recent_actions": [{k: h.get(k) for k in ("action", "text")} for h in history[-6:]],
    }


def field_text(context):
    key = os.environ.get("TEXT_MODEL_API_KEY")
    if not key:
        raise ValueError("TYPE_TEXT needs TEXT_MODEL_API_KEY; no text is hardcoded or guessed by the executor.")
    base = os.environ.get("TEXT_MODEL_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = os.environ.get("TEXT_MODEL", "deepseek-chat")
    reasoning = {"thinking": {"type": "disabled"}} if "api.deepseek.com/" in base else {"reasoning": {"effort": "low"}}
    if os.environ.get("TEXT_MODEL_REASONING") == "none":
        reasoning = {"reasoning": {"enabled": False}}
    started = time.perf_counter()
    result = post_json(
        base + "/chat/completions",
        key,
        {
            "model": model,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
            **reasoning,
            "messages": [
                {"role": "system", "content": TEXT_VALUE},
                {
                    "role": "user",
                    "content": json.dumps(context),
                },
            ],
        },
    )
    try:
        output = json.loads(result["choices"][0]["message"]["content"])
        value = output["text"]
        if set(output) != {"text"} or not isinstance(value, str) or not value.strip() or len(value) > 2000:
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise ValueError("Text helper returned no valid field value; nothing typed.") from None
    return value, {
        "model": model,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": result.get("usage", {}),
    }
