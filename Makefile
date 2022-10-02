setup: python-env create-required-files

python-env:
	virtualenv -p python3 venv && \
	source venv/bin/activate && \
	pip install --upgrade -r requirements.txt && \
	pre-commit install

create-required-files:
	cd checker/config && \
	touch config_secrets.py

run-server:
	cd checker && \
	uvicorn main:app --port 8082 --reload

run-hyperparam-tuning:
	python3 checker/hyperparam_main.py

start-training:
	python3 checker/train.py

start-training-with-hyperparam-tuning:
	cd checker && \
	python3 hyperparam_tuning.py && \
	python3 train.py
