"""Smart Tagging loader adapted from the official starter (MIT).

Copyright (c) 2026-2027 Metrica Sports. See LICENSE.
Intervals are rounded to frames and half-open on the right, in file order.
"""

import xml.etree.ElementTree as ET

from possession.data.match import FPS


def load_tagging(path, *, fps=FPS):
    rows = []
    try:
        for instance in ET.parse(path).iter("instance"):
            start_s = float(instance.findtext("start"))
            end_s = float(instance.findtext("end"))
            rows.append({
                "code": instance.findtext("code"),
                "start_frame": int(round(start_s * fps)),
                "end_frame": int(round(end_s * fps)),
                "tags": {label.findtext("group"): label.findtext("text")
                         for label in instance.findall("label")},
            })
    except (ET.ParseError, TypeError, ValueError, OverflowError):
        raise ValueError("Invalid Smart Tagging XML or interval time.") from None
    return rows
