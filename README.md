# Masterproject FakeChecker Backend
Welcome to our backend. This is a neat little repository to train or utilise any transformer model for inference that is available
on [huggingfacve.co](huggingface.co). Just follow the setup and enjoy playing around with transformers.
For our case, we created a FakeChecker with corresponding UI [Frontend Repository](https://github.com/kevinkrs/fakechecker_mp_frontend/tree/master).
In any case, feel free to reach out if any questions, feedback or suggestions arise.


# Setup
To ease our lives the project is orchestrated with ```make```. So just follow these steps
1. Open project and run ````make setup````. This will create a virtual environment, activate it and install all required libraries.
2. The whole model is managed within the ````config.json```` found in the top level folders. Just change your model type based on the
available transformer models from [huggingface.co](huggingface.co) and your model parameters. Everything else should be fine as it is.
3. Run ````make start_training```` to start the model training. If no data is available, please add data to the defined data path (```config.json```).
Since we build our database university internally, the dataloader will not work for externals.
4. To run hyperparam tuning run ````make run-hyperparam-tuning```` or just combine tuning and training with `make start-training-with-hyperparam-tuning`
5. To re-train the model from a checkpoint, just change ```from_ckp=true``` in the ``config.json``


# Backend API Documentation
- [train.py](https://htmlpreview.github.io/?https://github.com/kevinkrs/masterproject_backend/blob/check-docs/docs/train.html)
- [transformer.py](https://htmlpreview.github.io/?https://github.com/kevinkrs/masterproject_backend/blob/check-docs/docs/transformer.html)
- [dataloader.py](https://htmlpreview.github.io/?https://github.com/kevinkrs/masterproject_backend/blob/check-docs/docs/dataloader.html)
- [dataset_module.py](https://htmlpreview.github.io/?https://github.com/kevinkrs/masterproject_backend/blob/check-docs/docs/dataset_module.html)
- [hyperparam_tuning.py](https://htmlpreview.github.io/?https://github.com/kevinkrs/masterproject_backend/blob/check-docs/docs/hyperparam_tuning.html)
- [inference.py](https://htmlpreview.github.io/?https://github.com/kevinkrs/masterproject_backend/blob/check-docs/docs/inference.html)
- [dataset_module.py](https://htmlpreview.github.io/?https://github.com/kevinkrs/masterproject_backend/blob/check-docs/docs/dataset_module.html)
- [transformer_tokenizer.py](https://htmlpreview.github.io/?https://github.com/kevinkrs/masterproject_backend/blob/check-docs/docs/transformer_tokenizer.html)
