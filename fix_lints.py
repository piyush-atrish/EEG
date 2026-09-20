# 1. Chop off the dead Member A stubs from the bottom of Member B's scripts
for script, stub_marker in [
    ('scripts/03_make_windows.py', '"""Build window tables, contract C4'),
    ('scripts/04_extract_features.py', '"""Extract features (contract C5)')
]:
    with open(script, 'r', encoding='utf-8') as f:
        content = f.read()
    if stub_marker in content:
        # Keep everything before the old stub
        clean_content = content.split(stub_marker)[0].strip() + '\n'
        with open(script, 'w', encoding='utf-8') as f:
            f.write(clean_content)

# 2. Update pyproject.toml to standard 120 line-length and ignore test limits
with open('pyproject.toml', 'r', encoding='utf-8') as f:
    toml = f.read()

if 'line-length' not in toml:
    toml = toml.replace('[tool.ruff.lint]', 'line-length = 120\n\n[tool.ruff.lint]')
if 'per-file-ignores' not in toml:
    toml += '\n[tool.ruff.lint.per-file-ignores]\n"tests/*" = ["E501"]\n'

with open('pyproject.toml', 'w', encoding='utf-8') as f:
    f.write(toml)
