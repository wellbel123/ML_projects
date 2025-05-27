# Makefile for Alice classification project

# Path to your virtual environment
VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# Default target
.PHONY: help
help:
	@echo "Usage:"
	@echo "  make install       - install dependencies"
	@echo "  make train         - run training script"
	@echo "  make predict       - run prediction script"
	@echo "  make venv          - create virtual environment"
	@echo "  make tree          - show project structure"

# Create virtual environment
venv:
	python3 -m venv $(VENV)
	@echo "Virtual environment created in ./venv"

# Install dependencies
install: $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed"

# Run training
train:
	$(PYTHON) scripts/train.py

# Run prediction
predict:
	$(PYTHON) scripts/predict.py

# Show project structure (optional)
tree:
	@echo "📁 Project structure:"
	@find . -type d -not -path "*/\.*"
