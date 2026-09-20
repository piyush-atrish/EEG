path = "tests/test_loader_cohort.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Remove the awkwardly placed CH variable from the top
content = content.replace("CH = [\"FP1-F7\", \"F7-T7\", \"T8-P8\"]\n", "")

# Insert it safely below all imports, right before the first function definition
content = content.replace("def ", "CH = [\"FP1-F7\", \"F7-T7\", \"T8-P8\"]\n\ndef ", 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
