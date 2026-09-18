"""Distinguish sparse observed exactness from consistency in repeated groups."""

import pandas as pd

import fieldwork as fw


def investigate():
    frame = pd.DataFrame({"X": [1, 1, 2, 2], "Y": ["a", None, "b", None]})
    result = fw.discover_dependencies(frame, max_key_size=1)
    record = result.to_frame("dependencies").iloc[0]
    assert record["exact"]
    assert record["target_coverage"] == 0.5
    assert record["repeated_rows"] == 0
    assert result["candidates"][0]["determines"] == ["Y"]
    assert result["candidates"][0]["determines_with_repeated_support"] == []
    return frame, result


if __name__ == "__main__":
    _, result = investigate()
    print(fw.render_plaintext(result))
