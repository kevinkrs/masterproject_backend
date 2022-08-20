# Attributing
Project structure oriented by the project of mihail911. Thank you for such a beatiful repository.
(https://github.com/mihail911/fake-news)


# Setup
- Select poetry as interpreter
- Run ``poetry build`` and then ``poetry install``
- 
## Running model on M1 Mac
Currently, the support of torch on M1 Metall is a nightly build. Not all features are available.
Following settings are required
- Set _accelerator_ to ``mps`` in the config file
- Check that following `env` variables are set
  - `PYTORCH_ENABLE_MPS_FALLBACK=1`
  - `TOKENIZER_PARALLELISM=true`

# Usefull sources
- https://dvc.org/doc/start/data-management
