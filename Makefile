# Reference: https://dida.do/blog/managing-layered-requirements-with-pip-tools

REQUIREMENTS_TXT := $(addsuffix .txt, $(basename $(wildcard requirements/*.in)))
PIP_COMPILE := pip-compile --quiet --no-header --allow-unsafe --resolver=backtracking

.DEFAULT_GOAL := help
.PHONY: reqs clean-reqs help

requirements/constraints.txt: requirements/*.in
	CONSTRAINTS=/dev/null $(PIP_COMPILE) --strip-extras --output-file $@ $^ --extra-index-url https://download.pytorch.org/whl/cpu

requirements/%.txt: requirements/%.in requirements/constraints.txt
	CONSTRAINTS=constraints.txt $(PIP_COMPILE) --no-annotate --output-file $@ $<
	@# Remove --extra-index-url, blank lines, and torch dependency from non-core groups
	@[ $* = core ] || sed '/^--/d; /^$$/d; /^torch==/d' -i $@

reqs: $(REQUIREMENTS_TXT)  ## Generate the requirements files

torch-%: requirements/core.txt  ## Set PyTorch platform to use, e.g. cpu, cu117, rocm5.2
	@echo Generating requirements/core.$*.txt
	@sed 's|cpu|$*|' $< >requirements/core.$*.txt

clean-reqs:  ## Delete the requirements files
	rm -f requirements/constraints.txt requirements/core.*.txt $(REQUIREMENTS_TXT)

help:  ## Display this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
