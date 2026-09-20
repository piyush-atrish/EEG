path = "src/eegpipe/config.py"
with open(path, "r", encoding="utf-8") as f:
    c = f.read()

# 1. Restore the import fixes we accidentally reverted
c = c.replace("import collections.abc", "import copy")
c = c.replace("deep_update(", "_deep_merge(")

# 2. Restore the get_paths function
if "def get_paths" not in c:
    c += "\n\ndef get_paths(cfg: dict):\n    from pathlib import Path\n    return {key: Path(val) for key, val in cfg[\"paths\"].items()}\n"

with open(path, "w", encoding="utf-8") as f:
    f.write(c)
