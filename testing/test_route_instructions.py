from llm.route_instructions import parse_text_to_steps


def _assert_parse(text, expected_steps):
    steps, clarification = parse_text_to_steps(text)
    assert clarification is None, f"Unexpected clarification: {clarification}"
    assert steps == expected_steps, f"Expected {expected_steps}, got {steps}"


def _assert_clarification(text):
    steps, clarification = parse_text_to_steps(text)
    assert steps is None, f"Expected no steps, got {steps}"
    assert clarification, "Expected a clarification message."


def main():
    _assert_parse("start go forward then right then left", [1, 0, 2])
    _assert_parse("begin u-turn then straight", [3, 1])
    _assert_clarification("turn left then right")  # missing start marker
    _assert_clarification("start turn then right")  # ambiguous turn
    _assert_clarification("start")  # no directions
    steps, clarification = parse_text_to_steps("start forward then left")
    assert clarification is None
    # The binary writer reverses order; this validates the parsed order remains forward.
    assert steps == [1, 2]
    print("test_route_instructions: ok")


if __name__ == "__main__":
    main()
