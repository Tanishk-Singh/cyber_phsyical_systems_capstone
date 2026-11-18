"""SWaT attack types, derived from the official attack list (Phase A, Step 8).

SWaT attacks are usually grouped by how many points they touch and how many
plant stages those points belong to. The attack list does not label each
attack with its type, so it is derived here from the "Attack Point" column:
a tag's stage is the first digit of its number (LIT-301 -> stage 3).

    SSSP  single stage, single point      (e.g. attack 3: LIT-101)
    SSMP  single stage, multiple points   (e.g. attack 24: P-203, P-205)
    MSMP  multiple stages, multiple points (e.g. attack 26: P-101, LIT-301)

(A single point always sits in a single stage, so "MSSP" cannot be derived
from the attack points alone.)

The CSV ends with junk rows (a legend and a footnote), so only rows whose
"Attack #" is a number are kept.
"""
import re
from pathlib import Path

import pandas as pd

ATTACK_LIST_FILE = "List_of_attacks_Final(Sheet1).csv"


def attack_points(text: str) -> list[str]:
    """Tags in an "Attack Point" cell, e.g. "P-101; P-102" -> ["P-101", "P-102"]."""
    return re.findall(r"[A-Z]+-?\d{3}", str(text).upper())


def attack_type(points: list[str]) -> str:
    stages = {re.search(r"\d", p).group() for p in points}
    if len(points) == 1:
        return "SSSP"
    return "SSMP" if len(stages) == 1 else "MSMP"


def load_attack_types(swat_dir: str | Path) -> pd.DataFrame:
    """One row per documented attack: number, points, stages, type."""
    raw = pd.read_csv(Path(swat_dir) / ATTACK_LIST_FILE, encoding="latin1")
    number = pd.to_numeric(raw["Attack #"], errors="coerce")
    raw = raw[number.notna()].assign(attack=number[number.notna()].astype(int))
    rows = []
    for attack, cell in zip(raw["attack"], raw["Attack Point"]):
        points = attack_points(cell)
        if not points:                       # "No Physical Impact Attack" rows
            continue
        rows.append({"attack": attack, "points": ", ".join(points),
                     "stages": "".join(sorted({re.search(r"\d", p).group() for p in points})),
                     "type": attack_type(points)})
    return pd.DataFrame(rows).set_index("attack")
