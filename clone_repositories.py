# clone_repositories.py → FINAL VERSION (the one I will use forever)
import subprocess
from pathlib import Path

REPO_LIST_FILE = Path("repo_list.txt")
RAW_DATA_DIR = Path("data/01_raw_data")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

if not REPO_LIST_FILE.exists():
    print("repo_list.txt not found!")
    exit(1)

repos = [line.strip() for line in REPO_LIST_FILE.read_text().splitlines() if line.strip() and not line.startswith("#")]
print(f"Found {len(repos)} repositories in repo_list.txt\n")

cloned = 0
skipped = 0

for repo in repos:
    repo_name = repo.split("/")[-1]
    target_dir = RAW_DATA_DIR / repo_name
    url = f"https://github.com/{repo}.git"

    # Check if it's a PROPER bare mirror (must have objects/ and refs/)
    is_valid_mirror = target_dir.exists() and (target_dir / "objects").exists() and (target_dir / "refs").exists()

    if is_valid_mirror:
        print(f"Valid mirror already exists → skipping: {repo}")
        skipped += 1
        continue

    # If folder exists but is broken → delete it
    if target_dir.exists():
        print(f"Broken or incomplete mirror found → removing: {target_dir}")
        import shutil
        shutil.rmtree(target_dir, onerror=lambda f, p, e: print(f"   Could not delete {p}"))

    print(f"Cloning fresh bare mirror: {repo} → {target_dir}/")
    result = subprocess.run([
        "git", "clone", "--mirror", url, str(target_dir)
    ], check=True)
    
    print(f"Successfully cloned: {repo}\n")
    cloned += 1

print(f"DONE!")
print(f"   Cloned: {cloned} | Skipped (already good): {skipped}")
print(f"\nYou are now 100% ready to run:")
print("   python src/1_monthly_miner.py")