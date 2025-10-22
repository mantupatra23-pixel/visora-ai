import re

source = "visora_backend.py"
backup = "visora_backend_backup.py"
fixed = "visora_backend_fixed.py"

# Backup before editing
import shutil
shutil.copy(source, backup)

with open(source) as f:
    lines = f.readlines()

new_lines = []
skip_next = False
for i, line in enumerate(lines):
    if skip_next:
        skip_next = False
        continue

    # Remove duplicate except or pass blocks
    if re.match(r'^\s*except\s+Exception', line) and i + 1 < len(lines):
        # If next line also starts with 'except', skip this duplicate
        if re.match(r'^\s*except\s+Exception', lines[i + 1]):
            continue

    # Fix empty try: blocks
    if re.match(r'^\s*try:\s*$', line):
        # Next line check
        if i + 1 >= len(lines) or re.match(r'^\s*(except|finally|#|$)', lines[i + 1]):
            new_lines.append(line)
            new_lines.append("    pass  # auto-fixed indentation\n")
            continue

    # Normalize tabs → spaces
    fixed_line = line.replace("\t", "    ")

    # Remove trailing spaces
    fixed_line = fixed_line.rstrip() + "\n"

    new_lines.append(fixed_line)

with open(fixed, "w") as f:
    f.writelines(new_lines)

shutil.move(fixed, source)

print("✅ AutoFix Complete: visora_backend.py cleaned and indentation fixed!")
