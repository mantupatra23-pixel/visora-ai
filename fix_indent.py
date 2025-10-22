import re

with open("visora_backend.py") as f:
    lines = f.readlines()

fixed = []
for i, line in enumerate(lines):
    fixed.append(line)
    if re.match(r'^\s*try:\s*$', line):
        # Next line check
        if i + 1 >= len(lines) or not re.match(r'^\s+', lines[i + 1]):
            fixed.append("    pass  # auto-fix for empty try\n")

# Normalize tabs to 4 spaces
fixed = [l.replace("\t", "    ") for l in fixed]

with open("visora_backend_fixed.py", "w") as f:
    f.writelines(fixed)

print("✅ Indentation fix applied. New file: visora_backend_fixed.py")
