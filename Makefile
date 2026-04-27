.PHONY: style kernel-builder-cli-docs quality bump-dev bump-dev-dry-run pre-release pre-release-dry-run pin-actions


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

# Bump every version site to the next dev release based on the currently
# installed `kernels` package version (e.g. installed 0.13.0 -> 0.14.0.dev0).
# Refreshes Cargo.lock and kernels/uv.lock so all sites stay consistent.
bump-dev:
	python scripts/bump_to_dev.py
	cargo check --workspace
	cd kernels && uv lock

bump-dev-dry-run:
	python scripts/bump_to_dev.py --dry-run

# Strip the `.dev0` / `-dev0` suffix from every version site in prep for a
# release (e.g. codebase 0.14.0.dev0 -> 0.14.0). Refreshes Cargo.lock and
# kernels/uv.lock so all sites stay consistent.
pre-release:
	python scripts/pre_release.py
	cargo check --workspace
	cd kernels && uv lock

pre-release-dry-run:
	python scripts/pre_release.py --dry-run
