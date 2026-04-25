import subprocess
import os

def run_git():
    cwd = r"e:\onedrive\OneDrive - Uttaradit Rajabhat University\URU_Work\Master_2_2568\FastAPI2"
    commands = [
        ["git", "init"],
        ["git", "remote", "add", "origin", "https://github.com/beebrain/FastAPI2.git"],
        ["git", "add", "."],
        ["git", "commit", "-m", "Initialize project with single-file HTML structure and YOLO integration"],
        ["git", "push", "-u", "origin", "main"]
    ]
    
    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        if result.returncode != 0:
            if "remote origin already exists" in result.stderr:
                continue
            print(f"Command failed with code {result.returncode}")

if __name__ == "__main__":
    run_git()
