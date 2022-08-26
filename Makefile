setup: python-env

python-env:
	virtualenv -p python3 venv
	venv/bin/pip install --upgrade -r requirements.txt