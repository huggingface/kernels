.PHONY: style kernel-builder-cli-docs quality bump-dev bump-dev-dry-run bump-release bump-release-dry-run pin-actions


export check_dirs := kernels/src kernels/tests kernels-data/bindings/python

all: src/kernels/python_depends.json

kernels/src/kernels/python_depends.json: kernels-data/src/python_dependencies.json
	cp $< $@

style:
	ruff format ${check_dirs}
	ruff check ${check_dirs} --fix

kernel-builder-cli-docs:
	cargo build -p hf-kernel-builder
	./target/debug/kernel-builder generate-docs \
	  | sed 's/hf-kernel-builder/kernel-builder/g' \
	  | sed '1s/^# Command-Line Help for `kernel-builder`/# CLI reference for kernel-builder/' \
	  | sed '/`--backends/,/^\*/{/^  Default value:/d;}' \
	  > docs/source/builder-cli.md
	@echo "Generated docs/source/builder-cli.md"

pin-actions:
	pinact run

quality:
	ruff format --check ${check_dirs}
	ruff check ${check_dirs}

# Bump every version site to the next patch dev release
# (e.g. 0.10.1 -> 0.10.2.dev0, or 0.10.1.dev0 -> 0.10.1.dev1).
# Refreshes Cargo.lock and kernels/uv.lock so all sites stay consistent.
bump-dev:
	python scripts/bump_version.py --dev
	cargo check --workspace
	cd kernels && uv lock

bump-dev-dry-run:
	python scripts/bump_version.py --dev --dry-run

# Bump every version site to the next minor dev release
# (e.g. 0.10.1 -> 0.11.0.dev0, or 0.10.1.dev0 -> 0.11.0.dev0).
# Refreshes Cargo.lock and kernels/uv.lock so all sites stay consistent.
bump-dev-major:
	python scripts/bump_version.py --dev --major
	cargo check --workspace
	cd kernels && uv lock

bump-dev-major-dry-run:
	python scripts/bump_version.py --dev --major --dry-run

# Strip the dev suffix from every version site in prep for a patch release
# (e.g. 0.10.1.dev0 -> 0.10.1, or 0.10.1 -> 0.10.2).
# Refreshes Cargo.lock and kernels/uv.lock so all sites stay consistent.
bump-release:
	python scripts/bump_version.py
	cargo check --workspace
	cd kernels && uv lock

bump-release-dry-run:
	python scripts/bump_version.py --dry-run

# Bump every version site to the next minor release
# (e.g. 0.10.1.dev0 -> 0.11.0, or 0.10.1 -> 0.11.0).
# Refreshes Cargo.lock and kernels/uv.lock so all sites stay consistent.
bump-major:
	python scripts/bump_version.py --major
	cargo check --workspace
	cd kernels && uv lock

bump-major-dry-run:
	python scripts/bump_version.py --major --dry-run
