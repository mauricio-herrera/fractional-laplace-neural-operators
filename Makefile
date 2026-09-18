.PHONY: verify fetch-potin
verify:
	python scripts/verify_repository.py

fetch-potin:
	python scripts/fetch_potin_catalog.py
