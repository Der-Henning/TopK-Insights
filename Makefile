install:
	poetry install

test:
	poetry run pytest -v --cov tki/

lint:
	poetry run pre-commit run -a
