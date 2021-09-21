.PHONY: clean create_environment lint install test_predictor test_projections train_predictor train_projections

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = cmmrt 
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies and cmmrt package
install: 
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	# $(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) setup.py install

## Uninstall cmmrt package
uninstall:
	$(PYTHON_INTERPRETER) -m pip uninstall cmmrt -y

## Delete all results/* and compiled Python files
clean:
	rm -rf results/rt/* !(".gitkeep")
	rm -rf results/projection/* !(".gitkeep")
	rm -rf results/optuna/* !(".gitkeep")
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 cmmrt --ignore=E501

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
## Train all RT predictors using hyperparameter tuning 
train_predictor:
	$(PYTHON_INTERPRETER) cmmrt/rt/train_model.py \
		--storage sqlite:///results/optuna/train.db --save_to saved_models \
		--smoke_test # FIXME: remove this line for complete training

## Test the performance of all RT predictors using nested cross-validation 
test_predictor: 
	$(PYTHON_INTERPRETER) cmmrt/rt/validate_model.py \
		--storage sqlite:///results/optuna/cv.db --csv_output results/rt/rt_cv.csv \
		--smoke_test # FIXME remove this line for complete training

## Meta-train a GP for projections using all data from PredRet database
train_projections: 
	$(PYTHON_INTERPRETER) cmmrt/projection/metalearning_train.py -s saved_models \
		-e 10 # FIXME: remove this line for complete training (or set epochs to 0 to train until convergence)

## Test the performance of meta-training for projections using 4 reference CMs
test_projections:
	$(PYTHON_INTERPRETER) cmmrt/projection/metalearning_test.py -s results/projection \
		-e 10 # FIXME: remove this line for complete training (or set epochs to 0 to train until convergence)
#

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
