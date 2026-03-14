"""Action name mapping between model-facing names and robot legacy names."""
from __future__ import annotations

NEW_TO_LEGACY_ACTIONS: dict[str, str] = {
    "neutral.surprise.quick.low": "sneeze",
    "neutral.deny.default.mid": "standard_no",
    "shy.deny.inquiring.mid": "shy_no",
    "shy.affirm.inquiring.mid": "shy_yes",
    "neutral.affirm.default.mid": "standard_yes",
    "neutral.recover.default.mid": "perk_up",
    "neutral.affirm.expressive.high": "hell_yes",
    "neutral.attention.default.mid": "attention",
    "neutral.deny.quick.low": "neutral_no",
    "neutral.alarm.expressive.high": "scream_scare",
    "shy.affirm.inquiring.low": "shy_trill",
    "neutral.dialogue.default.mid": "dialogue_1",
    "neutral.think.murmur.low": "babble_1",
    "neutral.think.muted.low": "babble_2",
    "neutral.think.animated.mid": "babble_3",
    "neutral.question.default.low": "question_1",
    "neutral.question.default.mid": "question_2",
    "neutral.question.default.high": "question_3",
    "neutral.affirm.playful.low": "yes_1",
    "neutral.affirm.default.high": "yes_2",
    "neutral.apology.default.low": "uhoh",
    "angry.deny.default.high": "angry_no",
    "angry.affirm.default.high": "angry_yes",
    "angry.greet.default.mid": "angry_greeting",
    "angry.deny.quick.mid": "angry_no_short",
    "sad.sigh.default.low": "sad_sigh",
    "sad.deny.default.low": "sad_no",
    "sad.affirm.default.low": "sad_yes",
}


def map_action_to_legacy(action: str) -> str:
    """Map model-facing action name to robot legacy action name."""
    return NEW_TO_LEGACY_ACTIONS.get(action, action)


__all__ = ["NEW_TO_LEGACY_ACTIONS", "map_action_to_legacy"]
