import os
import json

from utils.hyperparam_tuning import HyperParamTuning

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(base_dir, "config/config.json")) as f:
    config = json.load(f)



datamodule = TransformerDataModule()

tuner = HyperParamTuning(config, datamodule)
analysis = tuner.run()

print(analysis)
