#!/usr/bin/env python3
"""Switch the colormap in config.toml and restart the evolution daemon."""
import re, subprocess, sys

def switch(colormap: str, config_path: str = "config.toml"):
    text = open(config_path).read()
    new_text = re.sub(
        r'^(colormap\s*=\s*")[^"]*(")',
        lambda m: m.group(1) + colormap + m.group(2),
        text, flags=re.MULTILINE
    )
    if new_text == text:
        print(f"WARNING: colormap line not found in {config_path}")
        sys.exit(1)
    open(config_path, "w").write(new_text)
    print(f"colormap → {colormap}")
    result = subprocess.run(["bash", "scripts/evo_daemon.sh", "restart"],
                            capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip())
        sys.exit(result.returncode)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <colormap>")
        sys.exit(1)
    switch(sys.argv[1])
