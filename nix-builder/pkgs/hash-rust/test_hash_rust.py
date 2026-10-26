import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hash_rust import git_hashes, main


class HashRustTests(unittest.TestCase):
    def setUp(self):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        directory = stack.enter_context(tempfile.TemporaryDirectory())
        stack.enter_context(contextlib.chdir(directory))
        self.lock = Path("Cargo.lock")
        self.lock.write_text(
            'version = 4\n[[package]]\nname = "local"\nversion = "1"\n'
        )

    def test_local_and_registry_dependencies_need_no_fetch(self):
        self.lock.write_text(
            self.lock.read_text()
            + '[[package]]\nname = "registry"\nversion = "1"\n'
            + 'source = "registry+https://example.org"\n'
        )
        with patch("hash_rust.subprocess.run") as fetch:
            self.assertEqual(git_hashes(self.lock), {})
            fetch.assert_not_called()

    def test_workspace_dependencies_share_fetch_and_use_resolved_commit(self):
        revision = "a" * 40
        self.lock.write_text(
            "\n".join(
                f'[[package]]\nname = "{name}"\nversion = "1"\n'
                f'source = "git+https://example.org/repo{query}#{revision}"'
                for name, query in [
                    ("one", "?tag=v1"),
                    ("two", "?branch=main"),
                    ("three", ""),
                ]
            )
        )
        sri = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
        with patch("hash_rust.subprocess.run") as fetch:
            fetch.return_value.stdout = json.dumps({"hash": sri})
            self.assertEqual(
                git_hashes(self.lock),
                {f"{name}-1": sri for name in ["one", "two", "three"]},
            )
            fetch.assert_called_once_with(
                [
                    "nix-prefetch-git",
                    "--url",
                    "https://example.org/repo",
                    "--rev",
                    revision,
                    "--fetch-submodules",
                ],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )

    def test_unpinned_source_is_rejected(self):
        for revision in ["", "main", "a" * 39, "g" * 40]:
            with self.subTest(revision=revision):
                self.lock.write_text(
                    '[[package]]\nname = "crate"\nversion = "1"\n'
                    f'source = "git+https://example.org/repo?branch=main#{revision}"\n'
                )
                with patch("hash_rust.subprocess.run") as fetch:
                    with self.assertRaises(ValueError):
                        git_hashes(self.lock)
                    fetch.assert_not_called()

    def test_url_parsing_preserves_repository_and_ignores_selectors(self):
        revision = "a" * 40
        for url in [
            "https://example.org/repo.git",
            "ssh://git@example.org:2222/repo.git",
            "file:///tmp/repo",
        ]:
            with self.subTest(url=url):
                self.lock.write_text(
                    '[[package]]\nname = "crate"\nversion = "1"\n'
                    f'source = "git+{url}?branch=feature%2Ffix%23issue#{revision}"\n'
                )
                with patch("hash_rust.subprocess.run") as fetch:
                    fetch.return_value.stdout = '{"hash": "sha256-test"}'
                    self.assertEqual(git_hashes(self.lock), {"crate-1": "sha256-test"})
                    self.assertEqual(
                        fetch.call_args.args[0],
                        [
                            "nix-prefetch-git",
                            "--url",
                            url,
                            "--rev",
                            revision,
                            "--fetch-submodules",
                        ],
                    )

    def test_failure_preserves_output_and_success_removes_stale_hashes(self):
        output = Path("rust-git-hashes.json")
        original = '{"stale-1": "old-hash"}\n'
        output.write_text(original)
        with patch("sys.argv", ["hash-rust"]):
            with patch(
                "hash_rust.git_hashes",
                side_effect=subprocess.CalledProcessError(1, "fetch"),
            ):
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    main()
                self.assertEqual(error.exception.code, 1)
            self.assertEqual(output.read_text(), original)
            main()
        self.assertEqual(json.loads(output.read_text()), {})
        self.assertEqual(
            sorted(p.name for p in Path(".").iterdir()),
            ["Cargo.lock", "rust-git-hashes.json"],
        )


if __name__ == "__main__":
    unittest.main()
