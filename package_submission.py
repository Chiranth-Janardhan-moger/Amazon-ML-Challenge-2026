"""
Package Submission Solution into official Amazon ML Challenge 2026 zip structure.
"""

import os
import sys
import zipfile
import argparse

def package_submission(target_dir: str = "results/3rd_try", zip_name: str = "submission.zip"):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    target_path = os.path.join(base_dir, target_dir)
    zip_path = os.path.join(target_path, zip_name)

    print(f"Creating submission zip: {zip_path}...")
    os.makedirs(target_path, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add matching and candidate outputs
        matching_tsv = os.path.join(target_path, "matching_results.tsv")
        candidate_tsv = os.path.join(target_path, "candidate_pairs.tsv")
        if os.path.exists(matching_tsv):
            print(f"  Adding output/matching_results.tsv ({os.path.getsize(matching_tsv)/(1024*1024):.1f} MB)...")
            zipf.write(matching_tsv, "output/matching_results.tsv")
        if os.path.exists(candidate_tsv):
            print(f"  Adding output/candidate_pairs.tsv ({os.path.getsize(candidate_tsv)/(1024*1024):.1f} MB)...")
            zipf.write(candidate_tsv, "output/candidate_pairs.tsv")

        # Add documentation
        doc_path = os.path.join(base_dir, "Documentation_template.md")
        if os.path.exists(doc_path):
            print("  Adding Documentation_template.md...")
            zipf.write(doc_path, "Documentation_template.md")

        # Add source code
        code_dir = "code/business_entity_resolution"
        readme_path = os.path.join(base_dir, "README.md")
        req_path = os.path.join(base_dir, "requirements.txt")
        if os.path.exists(readme_path):
            zipf.write(readme_path, f"{code_dir}/README.md")
        if os.path.exists(req_path):
            zipf.write(req_path, f"{code_dir}/requirements.txt")

        src_dir = os.path.join(base_dir, "src")
        for root, _, files in os.walk(src_dir):
            for f in files:
                if f.endswith((".py", ".json")):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, base_dir)
                    zipf.write(full_path, f"{code_dir}/{rel_path.replace(os.sep, '/')}")

        models_dir = os.path.join(base_dir, "models")
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith((".txt", ".json")):
                    full_path = os.path.join(models_dir, f)
                    rel_path = os.path.relpath(full_path, base_dir)
                    zipf.write(full_path, f"{code_dir}/{rel_path.replace(os.sep, '/')}")

    print(f"[+] Successfully created {zip_path} ({os.path.getsize(zip_path)/(1024*1024):.1f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package submission zip")
    parser.add_argument("--target-dir", default="results/3rd_try", help="Directory containing TSV files")
    parser.add_argument("--zip-name", default="submission_v3.zip", help="Output zip name")
    args = parser.parse_args()

    package_submission(args.target_dir, args.zip_name)
