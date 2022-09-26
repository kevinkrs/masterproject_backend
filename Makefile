setup: python-env create-required-files

python-env:
	python3 venv venv && \
	source venv/bin/activate && \
	pip install --upgrade -r requirements.txt && \
	pre-commit install

create-required-files:
	cd config && \
	touch config_secrets.py

run-server:
	cd checker && \
	uvicorn main:app --reload


run-hyperparam-tuning:
	cd checker && \
	python3 hyperparam_main.py


start-training:
	cd checker && \
	python3 train.py

start-training-with-hyperparam-tuning:
	cd checker && \
	python3 hyperparam_tuning.py && \
	python3 train.py
