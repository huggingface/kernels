.PHONY: style quality

export check_dirs := kernels/src kernels/tests

all: src/kernels/python_depends.json

kernels/src/kernels/python_depends.json: kernel-builder/src/python_dependencies.json
	cp $< $@

style:
	black ${check_dirs}
	isort ${check_dirs}
	ruff check ${check_dirs} --fix

quality:
	black --check ${check_dirs}
	isort --check-only --diff ${check_dirs}
	ruff check ${check_dirs}
