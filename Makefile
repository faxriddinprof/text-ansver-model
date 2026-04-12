.PHONY: run migrate makemigrations install shell test

install:
	pip3 install -r requirements.txt

test:
	python3 test.py
run:
	python3 manage.py runserver

superuser:
	python3 manage.py createsuperuser
	