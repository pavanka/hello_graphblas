set shell := ["bash", "-c"]

setup:
    brew install python@3.10
    python3.10 -m venv .venv/dev
    source .venv/dev/bin/activate && \
    pip install --upgrade pip poetry && \
    poetry install --no-root

docker-build:
    docker build -t graphblas-python .

docker-run:
    docker run -v $(pwd):/app graphblas-python

docker-shell:
    docker run --rm -it -v $(pwd):/app graphblas-python bash

lint:
    source .venv/dev/bin/activate && \
    python -m black *.py

run target:
    source .venv/dev/bin/activate && \
    python {{target}}
