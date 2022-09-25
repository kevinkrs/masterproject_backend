setup: python-env create-required-files

python-env:
	virtualenv -p python3 venv
	source venv/bin/activate
	venv/bin/pip install --upgrade -r requirements.txt
	pre-commit install

create-required-files:
	cd checker/modules && \
	touch config_secrets.py

run-server:
	cd checker/model && \
	uvicorn main:app --reload


run-hyperparam-tuning:
	cd checker/model && \
	python3 hyperparam_tuning.py


start-training:
	cd checker/model && \
	python3 train.py

start-training-with-hyperparam-tuning:
	cd checker/model && \
	python3 hyperparam_tuning.py && \
	python3 train.py