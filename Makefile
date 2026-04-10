.PHONY: run migrate makemigrations install shell

run:
	python manage.py runserver

migrate:
	python manage.py migrate

makemigrations:
	python manage.py makemigrations

install:
	pip install -r requirements.txt

shell:
	python manage.py shell
