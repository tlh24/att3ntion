
all:
	pip install -e .

clean:
	pip uninstall hyper_attn_extensions
	rm -rf build/ dist/ *.egg-info/ __pycache__/
