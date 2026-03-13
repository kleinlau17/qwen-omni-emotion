"""Animation Module

Provides robot animation control and sending functionality.

Primary API (send_* functions):
    Most users only need these. They handle mapping + sending in one call.

Advanced API (map_* functions):
    For testing/previewing commands without sending, or custom processing.
"""

from .animation_mapper import (
    # === Primary API: Send functions ===
    init_sender,
    close_sender,
    set_debug_print,
    send_animation,
    send_emotion,
    send_light_turn,
    send_standing,
    send_stop,
    send_homing,
    send_walking,
    # === Advanced API: Map functions (for testing/custom use) ===
    map_to_command,
    map_light_turn,
    map_standing,
    map_emotion,
    map_stop,
    map_homing,
    map_walking,
    # === Helper functions ===
    is_valid_animation,
    get_all_animations,
    # === Constants ===
    IDLE_ANIMATIONS,
    HERO_ANIMATIONS,
    FG_ANIMATIONS,
    FACE_ANIMATIONS,
    EYES_ANIMATIONS,
)
from .animation_sender import AnimationSender, SendResult

__all__ = [
    # === Primary API: Send functions ===
    "init_sender",
    "close_sender",
    "set_debug_print",
    "send_animation",
    "send_emotion",
    "send_light_turn",
    "send_standing",
    "send_stop",
    "send_homing",
    "send_walking",
    # === Advanced API: Map functions ===
    "map_to_command",
    "map_light_turn",
    "map_standing",
    "map_emotion",
    "map_stop",
    "map_homing",
    "map_walking",
    # === Helper functions ===
    "is_valid_animation",
    "get_all_animations",
    # === Constants ===
    "IDLE_ANIMATIONS",
    "HERO_ANIMATIONS",
    "FG_ANIMATIONS",
    "FACE_ANIMATIONS",
    "EYES_ANIMATIONS",
    # === Classes ===
    "AnimationSender",
    "SendResult",
]
