setup: python-env

python-env:
	virtualenv -p python3 venv
	source venv/bin/activate
	venv/bin/pip install --upgrade -r requirements.txt