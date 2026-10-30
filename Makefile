# Author: Sandeep Belgavi
PY ?= python3

.PHONY: setup test test-java test-python lint probe l01-cpu

setup:
	$(PY) -m pip install -e "python[dev,l01]"

test: test-java test-python

test-java:
	mvn -B -ntp -f java/pom.xml test

test-python:
	pytest -q python

lint:
	ruff check python

probe:
	mvn -B -ntp -q -f java/pom.xml compile exec:java -Dexec.mainClass=dev.belgavi.llm.HardwareProbe
	$(PY) -m belgavi_llm.hardware

# Episode 1 (tokens): the whole CPU path, Python then Java. Needs network once, for `fetch`.
l01-cpu:
	PYTHONPATH=python/src $(PY) -m belgavi_llm.l01_tokens.fetch
	PYTHONPATH=python/src $(PY) -m belgavi_llm.l01_tokens.compare
	mvn -B -ntp -q -f java/pom.xml compile exec:java -Dexec.mainClass=dev.belgavi.llm.l01.TokenCli -Dexec.args="compare data/cache/l01/tokenizers"
