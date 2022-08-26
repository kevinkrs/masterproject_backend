# Attributing
Project structure oriented by the project of mihail911. Thank you for such a beatiful repository.
(https://github.com/mihail911/fake-news)


# Setup
- Navigate into the project folder and run `make setup`
  - Creates virtualenv
  - Upgrades pip and installs all relevant requirements

## Running model on M1 Mac
Currently, the support of torch on M1 Metall is a nightly build. Not all features are available.
Following settings are required
- Check that following `env` variables are set
  - `PYTORCH_ENABLE_MPS_FALLBACK=1`
  - `TOKENIZER_PARALLELISM=true`
- Check that the prediction function is passing the inputs to the correct device

# Usefull sources
- https://dvc.org/doc/start/data-management
