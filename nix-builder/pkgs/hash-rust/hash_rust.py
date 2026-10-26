import argparse
import json
import subprocess
import tempfile
import tomllib
from pathlib import Path
from string import hexdigits
from urllib.parse import urlsplit


def git_hashes(lock_file: Path) -> dict[str, str]:
    with lock_file.open("rb") as file:
        lock = tomllib.load(file)
    hashes = {}
    fetched = {}
    for package in lock.get("package", []):
        source = package.get("source", "")
        if not source.startswith("git+"):
            continue
        parts = urlsplit(source.removeprefix("git+"))
        revision = parts.fragment
        if len(revision) != 40 or any(char not in hexdigits for char in revision):
            raise ValueError(f"Unsupported Cargo Git source: {source}")
        url = parts._replace(query="", fragment="").geturl()
        key = (url, revision)
        if key not in fetched:
            result = subprocess.run(
                [
                    "nix-prefetch-git",
                    "--url",
                    url,
                    "--rev",
                    revision,
                    "--fetch-submodules",
                ],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            fetched[key] = json.loads(result.stdout)["hash"]
        name = f"{package['name']}-{package['version']}"
        if name in hashes and hashes[name] != fetched[key]:
            raise ValueError(f"Conflicting Git hashes for {name}")
        hashes[name] = fetched[key]
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate rust-git-hashes.json from ./Cargo.lock."
    )
    parser.parse_args()
    try:
        hashes = git_hashes(Path("Cargo.lock"))
        # Keep the temporary file on the same filesystem for atomic replacement.
        with tempfile.TemporaryDirectory(prefix=".rust-git-hashes-", dir=".") as tmp:
            output = Path(tmp) / "hashes.json"
            output.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
            output.replace("rust-git-hashes.json")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"hash-rust: {error}\n")
    print("Wrote rust-git-hashes.json")


if __name__ == "__main__":
    main()
