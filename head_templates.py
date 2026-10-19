"""Exact, unmodified head requests that do not need another LLM call.

These are complete phrase aliases, not substring rules or a language detector.
Requests with any extra instruction stay on the existing generative planner.
Only joint 0 is produced; motion_gen still applies its normal validation.
"""

import re
import unicodedata


# Keep each alias an entire, unmodified command. In languages where a generic
# "move/shake your head" can also mean nodding, require explicit sideways words.
_RIGHT = (
    "look right", "look to the right", "look head to the right", "turn your head right", "turn your head to the right",
    "向右看", "往右看", "向右转头", "头向右转",
    "दाईं ओर देखो", "दाहिने देखो", "सिर दाईं ओर घुमाओ",
    "mira a la derecha", "gira la cabeza a la derecha",
    "انظر يمينا", "انظر يمينًا", "انظر يميناً", "انظر إلى اليمين", "أدر رأسك إلى اليمين",
    "regarde à droite", "tourne la tête à droite",
    "olhe para direita", "olhe para a direita", "vire a cabeça para a direita",
    "tengok kanan", "tengok ke kanan", "tengok kepala ke kanan", "tengok kepala kanan",
    "lihat ke kanan", "menoleh kanan", "menoleh ke kanan", "menolehkan kepala ke kanan",
    "右を見て", "右を向いて", "頭を右に向けて",
    "오른쪽을 봐", "오른쪽을 보세요", "고개를 오른쪽으로 돌려",
)

_LEFT = (
    "look left", "look to the left", "look head to the left", "turn your head left", "turn your head to the left",
    "向左看", "往左看", "向左转头", "头向左转",
    "बाईं ओर देखो", "बाएं देखो", "सिर बाईं ओर घुमाओ",
    "mira a la izquierda", "gira la cabeza a la izquierda",
    "انظر يسارا", "انظر يسارًا", "انظر يساراً", "انظر إلى اليسار", "أدر رأسك إلى اليسار",
    "regarde à gauche", "tourne la tête à gauche",
    "olhe para esquerda", "olhe para a esquerda", "vire a cabeça para a esquerda",
    "tengok kiri", "tengok ke kiri", "tengok kepala ke kiri", "tengok kepala kiri",
    "lihat ke kiri", "menoleh kiri", "menoleh ke kiri", "menolehkan kepala ke kiri",
    "左を見て", "左を向いて", "頭を左に向けて",
    "왼쪽을 봐", "왼쪽을 보세요", "고개를 왼쪽으로 돌려",
)

_SHAKE = (
    "shake your head", "shake head",
    "摇头",
    "सिर को दाएं-बाएं हिलाओ", "सिर दाएँ-बाएँ हिलाओ",
    "mueve la cabeza de lado a lado",
    "هز رأسك يمينا ويسارا", "هز رأسك من جانب إلى آخر",
    "secoue la tête de gauche à droite",
    "balance a cabeça de um lado para o outro",
    "geleng kepala", "gelengkan kepala", "menggelengkan kepala",
    "首を横に振って", "頭を左右に振って",
    "고개를 좌우로 흔들어", "머리를 좌우로 흔들어",
)

_KINDS = {phrase: kind for kind, phrases in (
    ("right", _RIGHT), ("left", _LEFT), ("shake", _SHAKE),
) for phrase in phrases}


def _whole_phrase(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).casefold()).strip()
    # Ignore one ordinary sentence terminator only. Questions, ellipses,
    # internal punctuation and quoted/compound instructions remain unmatched.
    if text.endswith((".", "!", "。", "！")):
        text = text[:-1].rstrip()
    return text


def plan_for_common_request(request_text):
    """Return a fresh head-only plan, or None to keep the original LLM path."""
    kind = _KINDS.get(_whole_phrase(request_text))
    if kind is None:
        return None
    if kind == "right":
        angles, hold = (30,), 1.0
    elif kind == "left":
        angles, hold = (-30,), 1.0
    else:
        angles, hold = (-40, 40, -40, 40), 0.4
    return {"steps": [{"joints": {0: angle}, "hold": hold} for angle in angles]}
