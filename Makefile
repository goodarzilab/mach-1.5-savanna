.PHONY: quality style

check_dirs := savanna

IS_CONDA_ENV=$(if $(CONDA_PREFIX),true,false)
IS_VENV=$(if $(VIRTUAL_ENV),true,false)
export CUDA_PATH=/usr/local/cuda
export CPATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/include:$CPATH"
export PYTHON_MIN_VERSION := 3.12

check-python-env:
ifeq ($(CI), true)
	@echo "Running in CI $(shell python --version)"
else ifeq ($(CONDA_DEFAULT_ENV), base)
	@echo "Not doing anything in the base conda environment, did you forget to activate a new environment?"
	exit 1
else ifeq ($(IS_CONDA_ENV), true)
	@echo "Using conda environment: $(CONDA_PREFIX) $(shell python --version)"
else ifeq ($(IS_VENV), true)
	@echo "Using venv: $(VIRTUAL_ENV) $(shell python --version)"
else
	@echo "No environment detected, did you forget to activate a new environment?"
	exit 1
endif
ifeq ($(PYTHON_OK), no)
	@echo "Python version is less than $(PYTHON_MIN_VERSION)"
	exit 1
endif

check:
	ruff check $(check_dirs)
	ruff format --check $(check_dirs)

style:
	ruff check --fix $(check_dirs)
	ruff format $(check_dirs)

setup-env: check-python-env
	python -m pip install -r ./requirements/requirements.txt 
	python -m pip install -r ./requirements/requirements-torch.txt
	python -m pip install -r ./requirements/requirements-dev.txt
	pip install transformer_engine[pytorch]
	cd savanna/data && make

