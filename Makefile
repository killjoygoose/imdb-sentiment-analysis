SHELL := /bin/bash

.PHONY: format_black
format_black:
	@poetry run black .


.PHONY: format_prettier
format_prettier:
	@npx prettier --write "**/*.json"


.PHONY: download_models
download_models:
	git lfs install
	mkdir -p models
	cd models && wget https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q5_K_M.gguf
	cd models && wget https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q5_K_M.gguf
	mkdir -p data
	cd data && git clone --depth=1 https://huggingface.co/datasets/ajaykarthick/imdb-movie-reviews

.PHONY: install
install:
	if ! command -v pyenv >/dev/null 2>&1; then \
		apt update; \
		apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
		libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils \
		tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev; \
		curl https://pyenv.run | bash; \
	else echo "pyenv already installed!"; \
	fi && \
	\
	if ! command -v poetry >/dev/null 2>&1; then \
		pip install poetry; \
	fi && \
	if ! pyenv versions --bare | grep -qx "3.11.3"; then \
		pyenv install 3.11.3; \
	fi && \
	\
	poetry config virtualenvs.in-project true && \
	poetry install --no-root && \
	ls -l; \
	source ./venv/bin/activate && \
	CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python


