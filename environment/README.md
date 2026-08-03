# Tested reproduction environment

The final compatibility checks used Ubuntu 20.04.4 LTS on x86-64, Python 3.10.18, CUDA toolkit 11.6, PyTorch 1.13.1+cu116, Transformers 4.37.2, PEFT 0.8.2, tokenizers 0.15.2, safetensors 0.7.0, NumPy 1.26.4, and SciPy 1.15.3.

No GPU device was visible during final validation. The real checkpoint forward was therefore run on CPU. The CUDA build information is recorded for compatibility and is not a claim that a full GPU benchmark was rerun.

`requirements-tested.txt` is an exact record of the tested Python packages. `environment.yml` is a reproducibility recipe; availability of the exact CUDA wheel depends on the configured package channels. The supported ranges in `pyproject.toml` remain broader for installation portability.
