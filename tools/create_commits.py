import subprocess, sys

def run(cmd):
    print('>',' '.join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.stdout:
        print(p.stdout)
    if p.stderr:
        print(p.stderr, file=sys.stderr)
    if p.returncode != 0:
        print(f"Command exited {p.returncode}")
        sys.exit(p.returncode)

# Stage all changes
run(["git","add","-A"])

# Check staged files
p = subprocess.run(["git","diff","--staged","--name-only"], capture_output=True, text=True)
staged = p.stdout.strip()
if staged:
    run(["git","commit","-m","batch commit 1/100: commit working-tree changes"])
else:
    run(["git","commit","--allow-empty","-m","batch commit 1/100: empty commit (no staged changes)"])

# Create remaining empty commits
for i in range(2,101):
    run(["git","commit","--allow-empty","-m",f"batch commit {i}/100"])

# Push to origin
run(["git","push","origin","HEAD"])