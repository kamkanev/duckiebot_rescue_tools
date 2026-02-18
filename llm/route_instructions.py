import argparse
import os
import re
from typing import List, Optional, Tuple


DIRECTION_TO_CODE = {
    "right": 0,
    "forward": 1,
    "left": 2,
    "u_turn": 3,
}


_DIRECTION_PATTERN = re.compile(
    r"\b(u[- ]?turn|uturn|left|right|forward|straight)\b", re.IGNORECASE
)
_AMBIGUOUS_TURN_PATTERN = re.compile(r"\bturn\b", re.IGNORECASE)
_TURN_WITH_DIR_PATTERN = re.compile(
    r"\bturn\s+(left|right|straight|forward|u[- ]?turn|uturn)\b", re.IGNORECASE
)
_TURN_INTO_DIR_PATTERN = re.compile(
    r"\bturn\s+into\s+(a\s+)?(left|right|straight|forward|u[- ]?turn|uturn)\b",
    re.IGNORECASE,
)
_DIR_TURN_PATTERN = re.compile(
    r"\b(left|right|straight|forward|u[- ]?turn|uturn)\s+turn\b", re.IGNORECASE
)




def _normalize_token(token: str) -> Optional[str]:
    t = token.lower()
    if t in {"left", "right", "forward"}:
        return t
    if t in {"straight"}:
        return "forward"
    if t in {"u-turn", "uturn", "u turn"}:
        return "u_turn"
    return None


def parse_text_to_steps(text: str) -> Tuple[Optional[List[int]], Optional[str]]:
    """
    Returns (steps, clarification_message).
    If clarification_message is not None, steps will be None.
    """
    if not text or not text.strip():
        return None, "I need a route description with directions."

    # Ignore "u-turn" when checking for ambiguous "turn"
    text_without_uturn = re.sub(r"\bu[- ]?turn\b", "", text, flags=re.IGNORECASE)
    ambiguous_turn = bool(_AMBIGUOUS_TURN_PATTERN.search(text_without_uturn))
    has_turn_with_dir = bool(_TURN_WITH_DIR_PATTERN.search(text)) or bool(
        _TURN_INTO_DIR_PATTERN.search(text)
    ) or bool(_DIR_TURN_PATTERN.search(text))

    tokens = re.findall(r"\b[a-z]+(?:[- ]?turn)?\b", text.lower())
    transition_words = {
        "then",
        "and",
        "next",
        "after",
        "afterwards",
        "into",
        "followed",
        "by",
        "until",
    }
    directions = {"left", "right", "forward", "straight", "u-turn", "uturn"}
    for i, tok in enumerate(tokens):
        if tok == "turn":
            prev_tok = tokens[i - 1] if i - 1 >= 0 else ""
            next_tok = tokens[i + 1] if i + 1 < len(tokens) else ""
            # Ignore transition phrases after "turn"
            if next_tok in transition_words:
                continue
            prev_is_dir = prev_tok in directions
            next_is_dir = next_tok in directions
            if prev_is_dir and next_is_dir:
                return None, "I saw 'turn' with directions on both sides. Please clarify."

    if ambiguous_turn and not has_turn_with_dir:
        return None, "I saw 'turn' without a direction. Please specify left/right/forward/u-turn."

    steps: List[int] = []
    def _extract_steps(segment: str) -> List[int]:
        extracted = []
        for match in _DIRECTION_PATTERN.finditer(segment):
            normalized = _normalize_token(match.group(1))
            if normalized is None:
                continue
            extracted.append(DIRECTION_TO_CODE[normalized])
        return extracted

    # Handle "before" by moving the clause after "before" to the end.
    # Example: "before left and right turn perform a uturn" => uturn, left, right.
    before_match = re.search(r"\bbefore\b", text.lower())
    if before_match:
        after_before = text[before_match.end():].strip()
        # Split at the first action verb to separate the "before" clause
        verb_match = re.search(r"\b(perform|do|take|execute|go)\b", after_before, re.IGNORECASE)
        if verb_match:
            before_clause = after_before[:verb_match.start()].strip()
            after_clause = after_before[verb_match.start():].strip()
        else:
            before_clause = after_before
            after_clause = ""
        after_steps = _extract_steps(after_clause)
        before_steps = _extract_steps(before_clause)
        steps = after_steps + before_steps
    else:
        steps = _extract_steps(text)

    if not steps:
        return None, "I couldn't find any directions. Use left/right/forward/u-turn."

    return steps, None


def write_turns_bin(steps: List[int], output_path: str) -> None:
    if any(step not in (0, 1, 2, 3) for step in steps):
        raise ValueError("All steps must be in {0,1,2,3}.")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(bytes(reversed(steps)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert natural language into turns.bin format."
    )
    parser.add_argument("text", help="Route description containing a start marker.")
    parser.add_argument(
        "--out",
        default="turns.bin",
        help="Output path for the binary turn file (default: turns.bin).",
    )
    args = parser.parse_args()

    steps, clarification = parse_text_to_steps(args.text)
    if clarification:
        print(clarification)
        return

    write_turns_bin(steps, args.out)
    print(f"Wrote {len(steps)} steps to {args.out}: {steps}")


if __name__ == "__main__":
    main()
