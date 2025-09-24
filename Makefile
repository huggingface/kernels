.PHONY: style

export check_dirs := src examples tests

style:
	black ${check_dirs}
	isort ${check_dirs}
	ruff check ${check_dirs} --fix