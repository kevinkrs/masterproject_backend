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
