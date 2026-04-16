.PHONY: style quality

export check_dirs := kernels/src kernels/tests

all: src/kernels/python_depends.json

kernels/src/kernels/python_depends.json: kernels-data/src/python_dependencies.json
	cp $< $@

style:
	ruff format ${check_dirs}
	ruff check ${check_dirs} --fix

quality:
	ruff format --check ${check_dirs}
	ruff check ${check_dirs}
