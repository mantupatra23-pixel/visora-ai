#!/usr/bin/env python3
# fix_full.py — auto repair common unterminated/quote/triple-quote issues iteratively

import ast, re, sys, shutil

PATH = "visora_backend.py"
BACKUP = PATH + ".bak"

def backup():
    shutil.copy2(PATH, BACKUP)
    print(f"Backup saved to {BACKUP}")

def try_parse(code):
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, e

def remove_non_python_header(code):
    # Remove lines at top that look like non-Python title lines (very loose)
    lines = code.splitlines()
    # If first few lines contain "Visora Backend" or long banner without python syntax,
    # strip leading non-code lines until we hit an import or shebang or triple quote.
    i = 0
    while i < min(30, len(lines)):
        l = lines[i].strip()
        if l.startswith("#!") or l.startswith("import ") or l.startswith("from ") or l.startswith('"""') or l.startswith("'''"):
            break
        # if line contains too many non-code chars (no spaces + letters) consider removing
        if re.match(r'^[A-Za-z0-9\W]{10,}$', l) and not re.search(r'[a-z_]+\s*=|def\s+|class\s+|import\s+|from\s+', l, re.I):
            lines[i] = ""  # blank it
        i += 1
    return "\n".join(lines)

def balance_triple_quotes(code):
    # If odd number of triple-quotes, add closing triple at end
    tq = code.count('"""') + code.count("'''")
    if tq % 2 != 0:
        print("🔧 Uneven triple quotes found — appending closing triple quote at EOF")
        code = code + "\n\"\"\"\n"
    return code

def fix_trailing_triple_after_normal_quotes(code):
    # Fix patterns like: "motivational"""  -> "motivational"
    code = re.sub(r'(".*?")"{2,}', r'\1', code)
    code = re.sub(r"('.*?')'{2,}", r'\1', code)
    # Fix == "motivational"""  -> == "motivational"
    code = re.sub(r'==\s*"([^"]+)"{2,}', r'== "\1"', code)
    code = re.sub(r"==\s*'([^']+)'{2,}", r"== '\1'", code)
    # Fix stray triple quote at end of line -> keep single quote
    code = re.sub(r'("""\s*\n)', '"\n', code)
    code = re.sub(r"('''\s*\n)", "'\n", code)
    return code

def fix_unclosed_quotes_near(line_text):
    # Try to close an unclosed double or single quote in a line
    ln = line_text
    # if odd count of double quotes in line, append a closing quote
    if ln.count('"') % 2 != 0:
        print("  → closing unaired double quote in line")
        return ln + '"'
    if ln.count("'") % 2 != 0:
        print("  → closing unaired single quote in line")
        return ln + "'"
    return ln

def iterative_repair(code):
    # Try iterative approach: parse, on SyntaxError, inspect nearby lines and attempt fixes
    attempts = 0
    max_attempts = 50
    while attempts < max_attempts:
        ok, err = try_parse(code)
        if ok:
            return True, code, None
        if not err:
            return False, code, "Unknown parse failure"
        lineno = getattr(err, "lineno", None)
        msg = getattr(err, "msg", "")
        print(f"Parse error at line {lineno}: {msg}")
        lines = code.splitlines()
        if lineno and 1 <= lineno <= len(lines):
            # Try to repair the specific line and a small window around it
            start = max(1, lineno-3) - 1
            end = min(len(lines), lineno+3)
            repaired = False
            for i in range(start, end):
                old = lines[i]
                new = old
                # quick fixes
                new = new.replace('"""', '"""')  # placeholder (no-op)
                new = fix_unclosed_quotes_near(new)
                # If line ends with stray triple quotes after a normal quote, fix:
                new = re.sub(r'(".*?")"{2,}$', r'\1', new)
                new = re.sub(r"('.*?')'{2,}$", r'\1', new)
                if new != old:
                    print(f"  Fixed line {i+1}: {old!r}  ->  {new!r}")
                    lines[i] = new
                    repaired = True
            if repaired:
                code = "\n".join(lines)
                attempts += 1
                continue
        # fallback global fixes
        code_old = code
        code = fix_trailing_triple_after_normal_quotes(code)
        code = balance_triple_quotes(code)
        if code == code_old:
            return False, code, f"Could not auto-fix parse error at line {lineno}: {msg}"
        attempts += 1
    return False, code, "Max attempts reached"

def main():
    backup()
    code = open(PATH, "r", encoding="utf-8", errors="ignore").read()
    print("Step 1: Remove obvious non-Python header lines...")
    code = remove_non_python_header(code)
    print("Step 2: Apply trailing/triple-quote quick fixes...")
    code = fix_trailing_triple_after_normal_quotes(code)
    code = balance_triple_quotes(code)

    print("Step 3: Iterative parse-and-fix loop...")
    ok, fixed_code, info = iterative_repair(code)
    if not ok:
        print("❌ Auto-fix failed:", info)
        # write back whatever we changed so far but keep backup
        with open(PATH, "w", encoding="utf-8") as f:
            f.write(fixed_code)
        print(f"Partial fixes written to {PATH}. Check backup {BACKUP} or inspect around reported line.")
        sys.exit(2)

    # final sanity parse
    ok2, err2 = try_parse(fixed_code)
    if not ok2:
        print("❌ Still parse error after fixes:", err2)
        with open(PATH, "w", encoding="utf-8") as f:
            f.write(fixed_code)
        sys.exit(3)

    # write final
    with open(PATH, "w", encoding="utf-8") as f:
        f.write(fixed_code)
    print("✅ File repaired and syntax-valid (ast.parse successful).")

if __name__ == "__main__":
    main()
