"""
package_submission.py — Packages the final competition submission ZIP file.

Creates TechForge_submission.zip containing:
  output/matching_results.tsv
  output/candidate_pairs.tsv
  code/business_entity_resolution/src/...
  code/business_entity_resolution/requirements.txt
  code/business_entity_resolution/README.md
  Documentation_template.md
  SOLUTION_DOCUMENTATION.md
"""

import os
import zipfile
import time

def make_submission_zip(zip_name="TechForge_submission.zip"):
    t0 = time.time()
    print(f"Creating submission zip: {zip_name}...", flush=True)

    # Remove existing zip if any
    if os.path.exists(zip_name):
        os.remove(zip_name)
        print(f"  Removed old {zip_name}", flush=True)

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Output files
        for f in ["matching_results.tsv", "candidate_pairs.tsv"]:
            p = os.path.join("output", f)
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Missing required output file: {p}")
            print(f"  Adding {p} ({os.path.getsize(p) / 1024 / 1024:.1f} MB)...", flush=True)
            zf.write(p, arcname=f"output/{f}")

        # 2. Code files
        code_root = os.path.join("code", "business_entity_resolution")
        for root, dirs, files in os.walk(code_root):
            # Skip __pycache__
            if "__pycache__" in root:
                continue
            for f in files:
                if f.endswith((".pyc", ".pyo")):
                    continue
                full_path = os.path.join(root, f)
                arc_name = os.path.relpath(full_path, ".")
                print(f"  Adding {arc_name}...", flush=True)
                zf.write(full_path, arcname=arc_name.replace("\\", "/"))

        # 3. Documentation
        for doc in ["Documentation_template.md", "SOLUTION_DOCUMENTATION.md"]:
            if os.path.isfile(doc):
                print(f"  Adding {doc}...", flush=True)
                zf.write(doc, arcname=doc)

    sz_mb = os.path.getsize(zip_name) / 1024 / 1024
    dt = time.time() - t0
    print(f"\n[SUCCESS] Created {zip_name} ({sz_mb:.1f} MB) in {dt:.1f}s", flush=True)

if __name__ == "__main__":
    make_submission_zip()
