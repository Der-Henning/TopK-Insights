install:
	pip install -r requirements-dev.txt

test:
	python -m pytest --cov tki/

lint:
	pre-commit run -a
