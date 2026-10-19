"""Fast .xlsx reading for the SWaT workbooks.

openpyxl takes 10+ minutes on these 100-130MB files; streaming the raw
worksheet XML out of the zip with regex does the same job in under a minute.
Validated against openpyxl's output in the exploratory notebook
(codebase/data_analysis.ipynb) before being promoted here.
"""
import re
import zipfile

import pandas as pd

_CELL_PATTERN = re.compile(
    r'<c[^>]*r="([A-Z]+)\d+"[^>]*?(?: t="(\w+)")?>(?:<v>(.*?)</v>|<is><t[^>]*>(.*?)</t></is>)?</c>'
)


def fast_xlsx_rows(path: str, sheet_index: int = 0) -> list[dict]:
    """Stream raw worksheet XML out of the xlsx zip; return list of {col_letter: value} per row."""
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            ss_xml = z.read("xl/sharedStrings.xml").decode("utf-8", errors="replace")
            shared = re.findall(r"<t[^>]*>(.*?)</t>", ss_xml, re.S)
        xml = z.read(f"xl/worksheets/sheet{sheet_index + 1}.xml").decode("utf-8", errors="replace")

    rows = []
    for row_xml in re.findall(r"<row[^>]*>(.*?)</row>", xml, re.S):
        row = {}
        for col_letter, ctype, val, inline_str in _CELL_PATTERN.findall(row_xml):
            if ctype == "s" and val != "":
                row[col_letter] = shared[int(val)] if val.isdigit() and int(val) < len(shared) else val
            else:
                row[col_letter] = inline_str if inline_str else val
        rows.append(row)
    return rows


def rows_to_df(rows: list[dict], header_row_idx: int) -> pd.DataFrame:
    """Turn raw {col_letter: value} rows into a DataFrame, using `header_row_idx`
    as the header. SWaT's two files need DIFFERENT values here -- see swat.py."""
    all_letters = sorted({c for r in rows for c in r.keys()}, key=lambda s: (len(s), s))
    df = pd.DataFrame(rows, columns=all_letters)
    header = df.iloc[header_row_idx].tolist()
    df = df.iloc[header_row_idx + 1:].reset_index(drop=True)
    df.columns = header
    return df
