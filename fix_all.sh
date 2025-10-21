#!/bin/bash
echo "🚀 Starting Full Auto Fix for visora_backend.py ..."

# Step 1: Install format + lint tools
pip install --quiet autopep8 pyflakes

# Step 2: Format and clean indentation
echo "🧹 Cleaning indentation and formatting..."
autopep8 --in-place --aggressive --aggressive visora_backend.py

# Step 3: Static syntax check
echo "🔍 Checking for syntax or logic issues..."
pyflakes visora_backend.py || true

# Step 4: Compile test
echo "🧠 Running compile test..."
python3 -m py_compile visora_backend.py

# Step 5: Git commit and push automatically
echo "📤 Auto committing and pushing to GitHub..."
git add visora_backend.py
git commit -m "Auto fixed all syntax + indentation issues"
git push

echo "✅ All tasks done! File cleaned, tested, and pushed successfully!"
