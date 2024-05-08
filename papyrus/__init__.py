[build-system]
requires = ["setuptools", "setuptools-scm"]
build-backend = "setuptools.build_meta"

[project]
name = "my_package"
authors = [
    {name = "Example Author", email = "author@example.com"}
]
description = "package description"
version = "0.0.1"
readme = "README.md"
requires-python = ">=3.10"
dependencies = {file = ["requirements.txt"]}