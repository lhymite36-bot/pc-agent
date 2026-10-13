from pc_agent.core.planner import Plan, PlanStep, Planner, _extract_json


def test_extract_json():
    raw = '```json\n{"steps":[{"action":"x","args":{}}]}\n```'
    assert '"steps"' in _extract_json(raw)


def test_execute_with_mock_handlers():
    calls = []

    class FakeBrain:
        client = None
        model = "x"

    handlers = {
        "run_shell": lambda args: calls.append(args) or {"ok": True, "stdout": "hi"},
    }
    planner = Planner(FakeBrain(), handlers, take_screenshot=None)
    plan = Plan(
        goal="echo",
        steps=[PlanStep(id=1, action="run_shell", args={"command": "echo hi"})],
    )
    result = planner.execute(plan)
    assert result.result == "ok"
    assert plan.steps[0].status == "ok"
    assert calls
