.PHONY: run migrate makemigrations install shell test

install:
	pip3 install -r requirements.txt

test:
	python3 test.py
