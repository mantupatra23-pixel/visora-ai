import re

with open("visora_backend.py") as f:
    lines = f.readlines()

fixed = []
for i, line in enumerate(lines):
    fixed.append(line)
    if re.match(r'^\s*try:\s*$', line):
        if i + 1 >= len(lines) or not re.match(r'^\s+', lines[i + 1]):
            fixed.append("    pass  # global auto-fix for empty try\n")

# Normalize tabs and trim
fixed = [l.expandtabs(4).rstrip() + "\n" for l in fixed]

with open("visora_backend_fixed.py", "w") as f:
    f.writelines(fixed)

print("✅ Global indentation fix applied to all try blocks.")
