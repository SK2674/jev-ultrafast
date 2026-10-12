"""Offline contracts for a dynamic operation/target policy. No paid APIs."""

import contextlib
import json
import os
import time
from copy import deepcopy
from unittest.mock import Mock

import httpx
import pytest

from jev_ultrafast import agent as loop
from jev_ultrafast import model
from jev_ultrafast.browser import StalePage, browser_operation, fingerprint


def page():
    state = {
        "url": "https://example.test/",
        "title": "Search",
        "text": "Search",
        "scroll": {"y": 0},
        "actions": [
            {"id": "e1", "kind": "fill", "label": "Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e2", "kind": "click", "label": "Open Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e3", "kind": "click", "label": "Go", "role": "button", "value": "", "node": 20},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ],
    }
    state["fingerprint"] = fingerprint(state)
    return state


def choice(ids, selected):
    return {"choice": selected, "confidence": 1.0, "probabilities": {i: float(i == selected) for i in ids}}


# The limits a self-hosted jeff server enforces before it will answer (JEFF_MAX_*).
JEFF_MAX_LABELS = 64
JEFF_MAX_STATE_CHARS = 20000
JEFF_MAX_QUESTIONS = 64


def crowded_page(clicks=140, options=140):
    """A page like a real results list: two fields, a long dropdown, and many links."""
    actions = []
    for i, label in enumerate(["Where from?", "Where to?", "Departure"]):
        actions.append({"id": f"f{i}", "kind": "fill", "label": label, "role": "textbox", "value": "", "node": 10 + i})
        actions.append({"id": f"o{i}", "kind": "click", "label": f"Open {label}", "role": "textbox",
                        "value": "", "node": 10 + i})
    for o in range(options):
        actions.append({"id": f"s{o}", "kind": "select", "label": f"Cabin class → Zurich {o}",
                        "role": "combobox", "value": f"z{o}", "current_value": "Economy", "node": 50})
    for c in range(clicks):
        actions.append({"id": f"c{c}", "kind": "click", "label": f"Flight result {c} nonstop London",
                        "role": "button", "value": "", "node": 200 + c})
    actions.append({"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560})
    actions.append({"id": "wait", "kind": "wait", "label": "Wait for the page to update"})
    return {
        "url": "https://example.test/flights", "title": "Flights",
        "text": " ".join(f"copy line {i} about cheap flights" for i in range(120)),
        "actions": actions, "fingerprint": "x",
    }


def goals():
    return "Find one-way flights from Zurich to London on September 20, 2026, for one adult in economy."


def jeff(monkeypatch):
    monkeypatch.setenv("TYPESAFE_PROFILE", "jeff")
    monkeypatch.setenv("TYPESAFE_API_KEY", "devkey")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "http://127.0.0.1:8000")


def sent_body(monkeypatch, page, goal=None):
    """Run choose() against a stub backend and return the request it would have sent."""
    bodies = []

    def post(_url, _key, body):
        bodies.append(body)
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "TYPE_TEXT"),
                "type_text_target": choice(body["questions"]["type_text_target"]["criteria"], "1"),
            },
        }

    monkeypatch.setattr(model, "post_json", post)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    with contextlib.suppress(ValueError):
        model.choose(page, goal or goals(), [])
    return bodies[0]


def decision(action="e1"):
    return {
        "choice": action,
        "operation": "TYPE_TEXT",
        "target": "1",
        "confidence": 1.0,
        "probabilities": {action: 1.0},
        "latency_ms": 10,
        "usage": {},
    }


@pytest.mark.parametrize("mutation", ["unknown", "nan", "missing", "negative", "non_max", "confidence"])
def test_invalid_choice_is_rejected(mutation):
    a = choice(["a", "b"], "a")
    if mutation == "unknown":
        a["choice"] = "invented"
    elif mutation == "nan":
        a["probabilities"]["a"] = float("nan")
    elif mutation == "missing":
        del a["probabilities"]["b"]
    elif mutation == "negative":
        a["probabilities"]["b"] = -1
    elif mutation == "non_max":
        a["choice"] = "b"
    else:
        a["confidence"] = 5
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.validate_choice(a, {"a", "b"})


def test_one_index_per_node_with_operation_specific_targets():
    elements, targets, controls = model.action_space(page()["actions"])
    assert len(elements) == 2
    assert elements[0]["operations"] == ["TYPE_TEXT", "CLICK"]
    assert targets["TYPE_TEXT"]["1"]["id"] == "e1"
    assert targets["CLICK"]["1"]["id"] == "e2"
    assert targets["CLICK"]["2"]["id"] == "e3"
    assert "WAIT" in controls


def test_all_heads_are_one_request_and_only_matching_head_executes(monkeypatch):
    calls = []

    def post(_url, _key, body):
        calls.append(body)
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "TYPE_TEXT"),
                "type_text_target": choice(["1"], "1"),
                "click_target": {"choice": "invented"},
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(page(), "Find a book", [])
    assert len(calls) == 1
    assert d["operation"] == "TYPE_TEXT" and d["target"] == "1" and d["choice"] == "e1"
    assert set(calls[0]["questions"]) == {"operation", "click_target", "type_text_target"}


def test_click_cannot_consume_a_text_target(monkeypatch):
    def post(_url, _key, body):
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "CLICK"),
                "type_text_target": choice(["1"], "1"),
                "click_target": choice(["1", "2", "999"], "999"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.choose(page(), "Find a book", [])


def test_target_head_receives_control_state_and_full_next_step_rules(monkeypatch):
    p = page()
    p["actions"].insert(0, {
        "id": "toggle", "kind": "click", "label": "Free cancellation", "node": 30,
        "role": "checkbox", "checked": "true", "selected": False,
    })

    def post(_url, _key, body):
        questions = body["questions"]
        target = questions["click_target"]
        assert target["criteria"]["1"]["checked"] == "true"
        assert target["criteria"]["1"]["selected"] is False
        assert questions["operation"]["instructions"]["rules"] in target["instructions"]["rules"]
        return {
            "model": "test",
            "answers": {
                "operation": choice(questions["operation"]["criteria"], "CLICK"),
                "click_target": choice(target["criteria"], "3"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(p, "Search with free cancellation", [])
    assert d["choice"] == "e3"


def test_quoted_task_text_still_uses_the_llm(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    post = Mock(return_value={"choices": [{"message": {"content": '{"text":"Zurich"}'}}]})
    monkeypatch.setattr(model, "post_json", post)
    context = model.field_context('Fly from "Zurich" to London', page()["actions"][0], page(), [])
    assert model.field_text(context)[0] == "Zurich"
    assert post.call_count == 1
    sent = json.loads(post.call_args.args[2]["messages"][1]["content"])
    assert sent["goal"] == 'Fly from "Zurich" to London'


def test_missing_text_credential_stops_before_guessing(monkeypatch):
    monkeypatch.delenv("TEXT_MODEL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TEXT_MODEL_API_KEY"):
        model.field_text({"goal": 'Enter "Zurich"'})


@pytest.fixture
def runner():
    a = loop.Agent.__new__(loop.Agent)
    a.screenshots = False
    a.pending_text = None
    p = page()
    a.state = {
        "browser": Mock(fresh=Mock(return_value=True), observe=Mock(return_value=p)),
        "page": p,
        "decision": decision(),
        "goal": "Find a book",
        "history": [],
        "decisions": [],
        "status": "predicted",
        "started_at": time.perf_counter(),
        "record": False,
        "text_calls": [],
    }
    return a


def test_stale_decision_is_consumed_before_any_mutation(runner):
    runner.state["browser"].fresh.return_value = False
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["browser"].act.assert_not_called()
    assert runner.state["decision"] is None


def test_generated_text_reused_only_for_identical_retry_context(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 1
    assert runner.state["browser"].act.call_count == 2  # The first call rejects before any browser input.
    assert runner.pending_text is None


def test_changed_field_context_does_not_reuse_generated_text(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["page"]["text"] = "Different page context"
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 2


def test_loading_waits_do_not_trigger_no_progress_stop(runner):
    for _ in range(5):
        runner.state["decision"] = decision("wait")
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert len(runner.state["history"]) == 5 and runner.state["status"] == "ready"


def test_stale_observation_preserves_executed_action(runner):
    runner.state["decision"] = decision("e3")
    runner.state["browser"].observe.side_effect = StalePage("changed")
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert runner.state["history"][-1]["action"] == "Go"
    runner.state["browser"].act.assert_called_once()


def test_observation_is_one_atomic_browser_read(monkeypatch):
    import jev_ultrafast.browser as browser

    p = page()
    cdp = Mock(return_value={"result": {"value": p}})
    monkeypatch.setattr(browser, "cdp", cdp)
    actual = browser_operation({"operation": "observe", "session": "test", "screenshot": False})
    assert actual["actions"] == p["actions"]
    assert cdp.call_count == 1
    assert cdp.call_args.args[0] == "Runtime.evaluate"


def test_executor_rejects_a_stale_page_before_browser_input(monkeypatch):
    import jev_ultrafast.browser as browser

    b = browser.Browser.__new__(browser.Browser)
    b.fresh = Mock(return_value=False)
    operation = Mock()
    monkeypatch.setattr(browser, "browser_operation", operation)
    with pytest.raises(StalePage):
        b.act(page()["actions"][0], page(), "book")
    operation.assert_not_called()


@pytest.mark.parametrize("response", [{"exceptionDetails": {}}, {"result": {}}])
def test_interrupted_dropdown_mutation_cannot_be_retried_as_stale(monkeypatch, response):
    import jev_ultrafast.browser as browser

    # A navigation can destroy the evaluation result after the change event already fired.
    if "exceptionDetails" in response:
        response["exceptionDetails"] = {"text": "Execution context destroyed"}
    cdp = Mock(return_value=response)
    monkeypatch.setattr(browser, "cdp", cdp)
    with pytest.raises(RuntimeError, match="Dropdown execution"):
        browser_operation({"operation": "act", "session": "test", "action": {
            "id": "e1", "kind": "select", "node": 1, "value": "Design",
        }})
    assert cdp.call_count == 1


def test_fingerprint_tracks_values_and_identity_not_screenshots():
    p = page()
    other = deepcopy(p)
    other["screenshot"] = "changed"
    assert fingerprint(p) == fingerprint(other)
    other["actions"][0]["node"] = 99
    assert fingerprint(p) != fingerprint(other)


@pytest.mark.parametrize("changed", ["Departure", "Where from?", "Where to?", "year"])
def test_flight_verification_rejects_wrong_trip(changed):
    from examples.flights import verify

    actual = {
        "url": "https://www.google.com/travel/flights/search?tfs=example",
        "text": "Track prices from Zürich to London departing 2026-09-20",
        "actions": [
            {"label": k, "value": v}
            for k, v in [
                ("Change ticket type. One way", "One way"),
                ("Where from?", "Zürich"),
                ("Where to?", "London"),
                ("Departure", "Sun, Sep 20"),
                ("Nonstop flight on Sunday, September 20. Select flight", ""),
            ]
        ],
    }
    assert verify(actual)["passed"]
    if changed == "year":
        actual["text"] = actual["text"].replace("2026", "2027")
    else:
        next(a for a in actual["actions"] if a["label"] == changed)["value"] = "wrong"
    assert not verify(actual)["passed"]


@pytest.mark.parametrize(
    "content", ["Thinking: Zurich", '{"text":null}', '{"text":"Zurich","extra":true}', '{"text":123}']
)
def test_text_helper_rejects_invalid_values(monkeypatch, content):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", Mock(return_value={"choices": [{"message": {"content": content}}]}))
    with pytest.raises(ValueError, match="nothing typed"):
        model.field_text({"goal": "Find a flight"})


def test_navigation_during_prediction_reobserves_without_action(runner):
    runner.state["browser"].fresh.side_effect = StalePage("Document navigating")
    runner.command("tick")
    assert runner.state["status"] == "ready"
    assert runner.state["decision"] is None
    runner.state["browser"].act.assert_not_called()


# --- Running the agent on a self-hosted jeff (TYPESAFE_PROFILE=jeff) ----------------


def test_unbudgeted_request_is_exactly_what_a_self_hosted_server_refuses(monkeypatch):
    """The premise: upstream's full table overruns jeff's documented limits, so it 422s."""
    body = sent_body(monkeypatch, crowded_page())
    assert max(len(q["criteria"]) for q in body["questions"].values()) > JEFF_MAX_LABELS
    assert model.state_size(body["state"]) > JEFF_MAX_STATE_CHARS


def test_jeff_profile_request_fits_every_limit_a_self_hosted_server_enforces(monkeypatch):
    jeff(monkeypatch)
    body = sent_body(monkeypatch, crowded_page())
    for name, question in body["questions"].items():
        assert len(question["criteria"]) <= JEFF_MAX_LABELS, name
    assert len(body["questions"]) <= JEFF_MAX_QUESTIONS
    assert model.state_size(body["state"]) <= JEFF_MAX_STATE_CHARS


def test_fields_are_offered_before_better_matching_result_links(monkeypatch):
    """'Where from?' shares no non-stopword with the goal, but typing is the only route to a value."""
    jeff(monkeypatch)
    body = sent_body(monkeypatch, crowded_page())
    fields = body["questions"]["type_text_target"]["criteria"]
    links = body["questions"]["click_target"]["criteria"]
    assert len(fields) == 3
    assert "London" in json.dumps(links)  # the 140 result links match the goal; the fields match nothing
    assert not any("London" in c["element"] for c in fields.values())
    # The fields lead the click head too, so 'Open ...' is still offered next to its own field.
    assert list(links)[:3] == list(fields)


def test_bounded_action_space_keeps_a_field_whole(monkeypatch):
    """A field's fill and its 'Open ...' click are one choice; splitting them breaks both routes."""
    jeff(monkeypatch)
    body = sent_body(monkeypatch, crowded_page())
    elements = body["state"]["elements"]
    assert [e["operations"] for e in elements[1:4]] == [["TYPE_TEXT", "CLICK"]] * 3
    offered = sum(len(e.get("options") or e["operations"]) for e in elements)
    assert offered == model.settings()["ELEMENT_BUDGET"]


def test_dropdown_values_are_capped_and_most_relevant_first(monkeypatch):
    """A 140-option <select> overruns JEFF_MAX_LABELS on its own; goal words pick the survivors."""
    jeff(monkeypatch)
    body = sent_body(monkeypatch, crowded_page())
    options = next(e for e in body["state"]["elements"] if e.get("options"))["options"]
    assert len(options) == model.settings()["OPTION_BUDGET"]
    goal = model.terms(goals())
    assert options[0] is max(options, key=lambda o: len(goal & model.terms(o["label"])))


def test_hosted_profile_still_offers_every_control_in_dom_order(monkeypatch):
    """The default profile must stay upstream behaviour: nothing ranked away, nothing dropped."""
    body = sent_body(monkeypatch, crowded_page())
    assert [e["label"] for e in body["state"]["elements"]][:4] == [
        "Where from?", "Where to?", "Departure", "Cabin class",
    ]
    assert len(body["state"]["elements"]) == 144  # 3 fields + 1 dropdown + 140 result links
    heads = {name: len(q["criteria"]) for name, q in body["questions"].items()}
    assert (heads["click_target"], heads["select_target"], heads["type_text_target"]) == (143, 140, 3)
    assert {"DONE", "BLOCKED", "SCROLL_DOWN", "WAIT"} <= set(body["questions"]["operation"]["criteria"])


def test_profile_env_overrides_beat_the_profile_default(monkeypatch):
    monkeypatch.setenv("TYPESAFE_PROFILE", "jeff")
    monkeypatch.setenv("TYPESAFE_ELEMENT_BUDGET", "3")
    assert model.settings() == {"ELEMENT_BUDGET": 3, "OPTION_BUDGET": 4, "STATE_CHARS": 6000,
                                "RULES": "compact", "MODEL": ""}
    monkeypatch.setenv("TYPESAFE_RULES", "terse")
    assert model.settings()["RULES"] == "terse"
    monkeypatch.setenv("TYPESAFE_PROFILE", "hosted")
    monkeypatch.setenv("TYPESAFE_ELEMENT_BUDGET", "0")
    assert model.settings()["ELEMENT_BUDGET"] is False  # 0 restores the unbounded hosted behaviour


def test_laya_profile_names_its_checkpoint_and_keeps_rules_short(monkeypatch):
    """laya ignores an unknown model name and trims instructions before options, so both matter."""
    monkeypatch.setenv("TYPESAFE_PROFILE", "laya")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.delenv("TYPESAFE_MODEL", raising=False)
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    body = sent_body(monkeypatch, crowded_page())
    assert body["model"] == "typed-decisions"
    assert body["questions"]["operation"]["instructions"]["rules"] is model.NEXT_ACTION_TERSE
    # laya's cap is 256 tokens per question, shared with that question's options.
    head = body["questions"]["click_target"]
    per_head = len(" ".join(head["instructions"]["rules"]))
    widest = max(len(q["criteria"]) for q in body["questions"].values())
    assert widest <= model.settings()["ELEMENT_BUDGET"]
    assert per_head * 3 < len(model.NEXT_ACTION)


def test_explicit_model_env_beats_the_profile_default(monkeypatch):
    monkeypatch.setenv("TYPESAFE_PROFILE", "laya")
    monkeypatch.setenv("TYPESAFE_MODEL", "multilingual")
    assert sent_body(monkeypatch, page())["model"] == "multilingual"


def test_base_url_points_the_agent_at_a_local_server(monkeypatch):
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    assert model.systemone_url() == "https://api.typesafe.ai/v1/systemone"
    monkeypatch.setenv("TYPESAFE_BASE_URL", "http://127.0.0.1:8000/")
    assert model.systemone_url() == "http://127.0.0.1:8000/v1/systemone"


def test_page_text_is_halved_until_the_state_fits(monkeypatch):
    monkeypatch.setenv("TYPESAFE_PROFILE", "jeff")
    p = crowded_page()
    p["text"] = "chatty page copy. " * 2000
    body = sent_body(monkeypatch, p)
    sent = body["state"]["page"]["text"]
    assert 0 < len(sent) < len(p["text"])
    assert p["text"].startswith(sent)  # only ever truncated, never rewritten
    assert model.state_size(body["state"]) <= model.settings()["STATE_CHARS"]


def test_compact_rules_keep_every_hard_constraint(monkeypatch):
    jeff(monkeypatch)
    body = sent_body(monkeypatch, crowded_page())
    rules = body["questions"]["operation"]["instructions"]["rules"]
    assert rules is model.NEXT_ACTION_COMPACT
    assert rules in body["questions"]["click_target"]["instructions"]["rules"]
    for constraint in ["untrusted", "Do not repeat satisfied", "Fill required fields", "Never toggle",
                       "WAIT only if", "DONE needs visible", "BLOCKED"]:
        assert constraint in rules, constraint
    assert len(model.NEXT_ACTION_COMPACT) * 3 < len(model.NEXT_ACTION) * 2


def test_terse_rules_keep_the_constraints_a_backend_cannot_trim_away(monkeypatch):
    for constraint in ["untrusted", "Do not repeat satisfied", "Fill required fields",
                       "DONE needs visible", "BLOCKED"]:
        assert constraint in model.NEXT_ACTION_TERSE, constraint
    assert "already" in model.TARGET_TERSE and "offered index" in model.TARGET_TERSE


def test_hosted_profile_keeps_the_full_policy_text(monkeypatch):
    monkeypatch.delenv("TYPESAFE_PROFILE", raising=False)
    body = sent_body(monkeypatch, page())
    assert body["questions"]["operation"]["instructions"]["rules"] is model.NEXT_ACTION


def test_self_hosted_limit_message_reaches_the_caller(monkeypatch):
    """jeff names the exact limit it refused; a bare status code leaves the user guessing."""
    detail = {"detail": [{"msg": "At most 64 options/levels"}]}

    class Refused:
        status_code = 422
        headers = {"content-type": "application/json"}
        is_error = True

        def json(self):
            return detail

    monkeypatch.setattr(model.CLIENT, "post", Mock(return_value=Refused()))
    with pytest.raises(RuntimeError, match="At most 64 options/levels"):
        model.post_json("http://127.0.0.1:8000/v1/systemone", "devkey", {})


def test_inspector_numbers_match_what_the_model_saw(monkeypatch):
    """snapshot() and choose() must rank identically, or the user is shown the wrong elements."""
    jeff(monkeypatch)
    p = crowded_page()
    agent = loop.Agent.__new__(loop.Agent)
    agent.state = {"page": p, "goal": goals(), "history": [], "decisions": [], "status": "ready"}
    shown = agent.snapshot()["elements"]
    sent = sent_body(monkeypatch, p)["state"]["elements"]
    assert [e["label"] for e in shown] == [e["label"] for e in sent]
    assert [e["index"] for e in shown] == [e["index"] for e in sent]


def test_local_jeff_answers_a_crowded_page_end_to_end(monkeypatch):
    """Opt-in check against a self-hosted jeff. Never the paid API (AGENTS.md)."""
    url = os.environ.get("JEV_TEST_JEFF_URL", "http://127.0.0.1:8000")
    try:
        health = httpx.get(url + "/healthz", timeout=5)
        served = health.json() if health.status_code == 200 else {}
    except (httpx.HTTPError, ValueError):
        served = {}
    if not served.get("model"):
        pytest.skip(f"no jeff server at {url}; set JEV_TEST_JEFF_URL to run this")
    jeff(monkeypatch)
    body = sent_body(monkeypatch, crowded_page())
    body["model"] = "jev-latest"
    started = time.perf_counter()
    response = httpx.post(
        url + "/v1/systemone", json=body, timeout=180,
        headers={"Authorization": "Bearer " + os.environ.get("JEV_TEST_JEFF_KEY", "devkey")},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["answers"]["operation"]["choice"] in body["questions"]["operation"]["criteria"]
    print(f"\njeff {served['model']}: {result['usage']} in "
          f"{round((time.perf_counter() - started) * 1000)}ms")

