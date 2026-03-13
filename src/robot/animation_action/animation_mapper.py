"""Animation Mapping Module

Converts animation names and function commands to robot command format.
"""

import json
from typing import Literal, Optional

# Lazy import to avoid circular dependency
_sender: Optional["AnimationSender"] = None
_debug_print: bool = False  # Global debug print switch

# ==================== Animation List Constants ====================

IDLE_ANIMATIONS = [
    "idle_neutral",
    "idle_shy",
    "idle_sad",
    "idle_angry",
    "idle_excited",
]

HERO_ANIMATIONS = [
    "hurry_up",
    "tantrum",
    "jump",
    "jump_scare",
    "happy_dance",
    "dance_twist",
    "laugh_big",
    "laugh_giggle",
]

# Special mask animation lists
FACE_ANIMATIONS = [
    "excited_009",
    "notify_008",
    "question_002",
    "dialogue_1",
    "babble_1",
    "babble_2",
    "babble_3",
    "babble_4",
    "question_1",
    "question_2",
    "question_3",
]

EYES_ANIMATIONS = [
    "happy_001",
    "happy_002",
    "happy_003",
    "notify_011",
    "sad_003",
    "sad_004",
    "frustrated_001",
    "frustrated_004",
    "frustrated_005",
    "question_003",
    "question_009",
    "neutral_004",
    "angry_babble_1",
    "angry_babble_2",
    "angry_question_1",
    "sad_babble_1",
    "sad_question_1",
]

# Foreground animations (categorized by emotion, for reference only)
FG_ANIMATIONS = {
    "neutral": [
        "bo_beep",
        "quick_eager_yes",
        "attention",
        "sneeze",
        "perk_up",
        "uhoh",
        "relax",
        "maybe",
        "shy_trill",
        "neutral_no",
        "standard_no",
        "standard_yes",
        "yes_1",
        "yes_2",
        "hell_yes",
    ],
    "shy": [
        "shy_yes",
        "shy_no",
    ],
    "sad": [
        "sad_yes",
        "sad_no",
        "sad_sigh",
    ],
    "angry": [
        "angry_greeting",
        "angry_no_short",
        "angry_no",
        "angry_yes",
        "scream_scare",
    ],
}

# Collect all foreground animations
ALL_FG_ANIMATIONS = set()
for anims in FG_ANIMATIONS.values():
    ALL_FG_ANIMATIONS.update(anims)

# ==================== Core Mapping Functions ====================


def map_to_command(animation: str, blend_duration: float = 0.0) -> str:
    """Map animation name to robot command.

    Args:
        animation: Animation name
        blend_duration: Blend time in seconds, only for idle animations. 0 means use default value.

    Returns:
        Command JSON string
    """
    # Idle animation
    if animation.startswith("idle_"):
        if animation not in IDLE_ANIMATIONS:
            return _error(f"Unknown idle animation: {animation}", valid_list=IDLE_ANIMATIONS)

        # Use default blend time (special handling for idle_sad)
        if blend_duration == 0.0:
            blend_duration = 1.25 if animation == "idle_sad" else 0.3

        return json.dumps({
            "messages": [
                {"animation_addr": "/anim/bg/mask", "animation_args": ["full"]},
                {"animation_addr": "/anim/bg/loop", "animation_args": [animation, blend_duration]},
            ]
        })

    # Hero animation
    if animation in HERO_ANIMATIONS:
        return json.dumps({
            "animation_addr": "/hero",
            "animation_args": [animation],
        })

    # Foreground animation
    if animation in ALL_FG_ANIMATIONS or animation in FACE_ANIMATIONS or animation in EYES_ANIMATIONS:
        # Determine mask
        if animation in FACE_ANIMATIONS:
            mask = "face"
        elif animation in EYES_ANIMATIONS:
            mask = "eyes"
        else:
            mask = "body"

        return json.dumps({
            "animation_addr": "/anim/fg/play",
            "animation_args": [animation, mask],
        })

    # Unknown animation
    return _error(f"Unknown animation: {animation}", valid_idles=IDLE_ANIMATIONS, valid_heroes=HERO_ANIMATIONS)


# ==================== Function Command Dedicated Functions ====================


def map_light_turn(on: bool = True) -> str:
    """Turn light on/off operation."""
    clip = "projector_on" if on else "projector_off"
    value = 1.0 if on else 0.0

    return json.dumps({
        "messages": [
            {"animation_addr": "/audio/clip/play", "animation_args": [clip, 1.0]},
            {"animation_addr": "/func/delta", "animation_args": ["HEAD_LIGHT", value, 5.0]},
        ]
    })


def map_standing(emotion: Literal["neutral", "shy", "sad", "angry", "excited"] = "neutral") -> str:
    """Enter Standing state."""
    idle_anim = f"idle_{emotion}"
    blend = 1.25 if emotion == "sad" else 0.3

    return json.dumps({
        "messages": [
            {"animation_addr": "/estimator/reset", "animation_args": []},
            {"animation_addr": "/idle", "animation_args": []},
            {"animation_addr": "/anim/bg/mask", "animation_args": ["full"]},
            {"animation_addr": "/anim/bg/loop", "animation_args": [idle_anim, blend]},
        ]
    })


def map_emotion(
    emotion: Literal["neutral", "shy", "sad", "angry", "excited"],
    blend_duration: float | None = None
) -> str:
    """Switch emotion state."""
    idle_anim = f"idle_{emotion}"
    if blend_duration is None:
        blend_duration = 1.25 if emotion == "sad" else 0.3

    return json.dumps({
        "messages": [
            {"animation_addr": "/anim/bg/mask", "animation_args": ["full"]},
            {"animation_addr": "/anim/bg/loop", "animation_args": [idle_anim, blend_duration]},
        ]
    })


def map_stop() -> str:
    """Stop all actions."""
    return json.dumps({
        "animation_addr": "/stop",
        "animation_args": []
    })


def map_homing() -> str:
    """Homing operation."""
    return json.dumps({
        "animation_addr": "/pose",
        "animation_args": ["home"]
    })


def map_walking(start: bool = True, foot_id: int = 0) -> str:
    """Start/stop walking."""
    if start:
        return json.dumps({
            "animation_addr": "/gait/start",
            "animation_args": [foot_id]
        })
    else:
        return json.dumps({
            "animation_addr": "/gait/stop",
            "animation_args": []
        })


# ==================== Helper Functions ====================


def is_valid_animation(animation: str) -> bool:
    """Validate if animation name is valid."""
    return (
        animation in IDLE_ANIMATIONS or
        animation in HERO_ANIMATIONS or
        animation in ALL_FG_ANIMATIONS or
        animation in FACE_ANIMATIONS or
        animation in EYES_ANIMATIONS
    )


def get_all_animations() -> list[str]:
    """Get list of all available animations."""
    all_anims = list(IDLE_ANIMATIONS)
    all_anims.extend(HERO_ANIMATIONS)
    all_anims.extend(ALL_FG_ANIMATIONS)
    all_anims.extend(FACE_ANIMATIONS)
    all_anims.extend(EYES_ANIMATIONS)
    return sorted(set(all_anims))


def _error(message: str, **kwargs) -> str:
    """Generate error response JSON."""
    result = {"error": message}
    result.update(kwargs)
    return json.dumps(result, ensure_ascii=False)


# ==================== Sending Functions ====================


def init_sender(host: str, port: int, timeout: float = 1.0) -> None:
    """Initialize global sender.

    Must call this function before using sending functions.

    Args:
        host: Target IP address
        port: Target port
        timeout: socket timeout in seconds

    Example:
        >>> init_sender("192.168.1.102", 5205)
        >>> send_animation("angry_no")  # Send directly
    """
    global _sender
    from .animation_sender import AnimationSender
    _sender = AnimationSender(host, port, timeout)


def close_sender() -> None:
    """Close global sender."""
    global _sender
    if _sender is not None:
        _sender.close()
        _sender = None


def set_debug_print(enabled: bool) -> None:
    """Set global debug print switch.

    Args:
        enabled: Whether to enable debug printing
    """
    global _debug_print
    _debug_print = enabled


def send_animation(animation: str, blend_duration: float = 0.0, print_json: bool | None = None) -> bool:
    """Map and send animation.

    Args:
        animation: Animation name
        blend_duration: Blend time in seconds, only for idle animations
        print_json: Whether to print command content (None means use global setting)

    Returns:
        Whether send succeeded
    """
    global _debug_print
    if print_json is None:
        print_json = _debug_print

    cmd_json = map_to_command(animation, blend_duration)

    if print_json:
        print(f"    {cmd_json}")

    # DRY-RUN mode: print only, no send
    if _sender is None:
        return True

    result = _sender.send_raw(cmd_json)
    return result.success


def send_emotion(
    emotion: Literal["neutral", "shy", "sad", "angry", "excited"],
    blend_duration: float | None = None,
    print_json: bool | None = None
) -> bool:
    """Switch emotion state and send."""
    global _debug_print
    if print_json is None:
        print_json = _debug_print

    cmd_json = map_emotion(emotion, blend_duration)

    if print_json:
        print(f"    {cmd_json}")

    # DRY-RUN mode: print only, no send
    if _sender is None:
        return True

    result = _sender.send_raw(cmd_json)
    return result.success


def send_light_turn(on: bool = True, print_json: bool | None = None) -> bool:
    """Turn light on/off and send."""
    global _debug_print
    if print_json is None:
        print_json = _debug_print

    cmd_json = map_light_turn(on)

    if print_json:
        print(f"    {cmd_json}")

    # DRY-RUN mode: print only, no send
    if _sender is None:
        return True

    result = _sender.send_raw(cmd_json)
    return result.success


def send_standing(
    emotion: Literal["neutral", "shy", "sad", "angry", "excited"] = "neutral",
    print_json: bool | None = None
) -> bool:
    """Enter Standing state and send."""
    global _debug_print
    if print_json is None:
        print_json = _debug_print

    cmd_json = map_standing(emotion)

    if print_json:
        print(f"    {cmd_json}")

    # DRY-RUN mode: print only, no send
    if _sender is None:
        return True

    result = _sender.send_raw(cmd_json)
    return result.success


def send_stop(print_json: bool | None = None) -> bool:
    """Stop all actions and send."""
    global _debug_print
    if print_json is None:
        print_json = _debug_print

    cmd_json = map_stop()

    if print_json:
        print(f"    {cmd_json}")

    # DRY-RUN mode: print only, no send
    if _sender is None:
        return True

    result = _sender.send_raw(cmd_json)
    return result.success


def send_homing(print_json: bool | None = None) -> bool:
    """Homing and send."""
    global _debug_print
    if print_json is None:
        print_json = _debug_print

    cmd_json = map_homing()

    if print_json:
        print(f"    {cmd_json}")

    # DRY-RUN mode: print only, no send
    if _sender is None:
        return True

    result = _sender.send_raw(cmd_json)
    return result.success


def send_walking(start: bool = True, foot_id: int = 0, print_json: bool | None = None) -> bool:
    """Start/stop walking and send."""
    global _debug_print
    if print_json is None:
        print_json = _debug_print

    cmd_json = map_walking(start, foot_id)

    if print_json:
        print(f"    {cmd_json}")

    # DRY-RUN mode: print only, no send
    if _sender is None:
        return True

    result = _sender.send_raw(cmd_json)
    return result.success
