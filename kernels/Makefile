.PHONY: style

export check_dirs := src examples tests

all: src/kernels/python_depends.json

src/kernels/python_depends.json: ../build2cmake/src/python_dependencies.json
	cp $< $@

style:
	black ${check_dirs}
	isort ${check_dirs}
	ruff check ${check_dirs} --fix
