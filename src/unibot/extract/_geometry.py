from __future__ import annotations

from typing import Any

from unibot.extract.documents import BoundingBox


def column_name(column_number: int) -> str:
    label = ""
    number = column_number
    while number > 0:
        number, remainder = divmod(number - 1, 26)
        label = chr(65 + remainder) + label
    return label or "A"


def as_bounding_box(raw_box: Any) -> BoundingBox | None:
    if raw_box is None:
        return None

    left = getattr(raw_box, "left", getattr(raw_box, "l", None))
    top = getattr(raw_box, "top", getattr(raw_box, "t", None))
    right = getattr(raw_box, "right", getattr(raw_box, "r", None))
    bottom = getattr(raw_box, "bottom", getattr(raw_box, "b", None))
    if left is None or top is None or right is None or bottom is None:
        return None
    return BoundingBox(
        left=float(left),
        top=float(top),
        right=float(right),
        bottom=float(bottom),
    )
