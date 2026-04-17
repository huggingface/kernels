.PHONY: style kernel-builder-cli-docs quality


export check_dirs := kernels/src kernels/tests

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

quality:
	ruff format --check ${check_dirs}
	ruff check ${check_dirs}
