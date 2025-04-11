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
