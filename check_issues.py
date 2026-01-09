#!/usr/bin/env python3
"""Script to check for issues in the ML Summer School project"""

import os
import json
import glob
import sys

issues = []
warnings = []

print("=" * 60)
print("Checking for issues in ML Summer School Project")
print("=" * 60)

# Check 1: Validate notebook JSON structure
print("\n1. Checking notebook JSON validity...")
notebooks = glob.glob("**/*.ipynb", recursive=True)
notebooks = [n for n in notebooks if "environments" not in n]  # Skip venv
print(f"   Found {len(notebooks)} notebooks")

invalid_notebooks = []
for nb in notebooks[:20]:  # Check first 20
    try:
        with open(nb, 'r', encoding='utf-8') as f:
            json.load(f)
    except json.JSONDecodeError as e:
        invalid_notebooks.append((nb, str(e)))
    except Exception as e:
        invalid_notebooks.append((nb, str(e)))

if invalid_notebooks:
    issues.append("Invalid JSON in notebooks:")
    for nb, error in invalid_notebooks:
        issues.append(f"  - {nb}: {error}")
else:
    print("   [OK] All checked notebooks have valid JSON")

# Check 2: Check for required datasets
print("\n2. Checking for required datasets...")
dataset_dir = "Feature_Engineering/Datasets"
if os.path.exists(dataset_dir):
    required_datasets = ["titanic.csv", "houseprice.csv"]
    for ds in required_datasets:
        ds_path = os.path.join(dataset_dir, ds)
        if os.path.exists(ds_path):
            print(f"   [OK] {ds} exists")
        else:
            warnings.append(f"   [WARN] {ds} not found")
else:
    warnings.append(f"   ⚠ Dataset directory not found: {dataset_dir}")

# Check 3: Check Python dependencies
print("\n3. Checking Python dependencies...")
required_packages = {
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "jupyter": "jupyter"
}

missing_packages = []
for module, package in required_packages.items():
    try:
        __import__(module)
        print(f"   [OK] {package} installed")
    except ImportError:
        missing_packages.append(package)
        warnings.append(f"   [WARN] {package} not installed")

# Check 4: Check for common file issues
print("\n4. Checking for common file issues...")
if not os.path.exists("README.md"):
    warnings.append("   [WARN] README.md not found")
else:
    print("   [OK] README.md exists")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if issues:
    print("\n[ERROR] ISSUES FOUND:")
    for issue in issues:
        print(issue)
else:
    print("\n[OK] No critical issues found")

if warnings:
    print("\n[WARN] WARNINGS:")
    for warning in warnings:
        print(warning)
else:
    print("\n[OK] No warnings")

print("\n" + "=" * 60)

sys.exit(1 if issues else 0)

