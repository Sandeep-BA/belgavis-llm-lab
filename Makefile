# Author: Sandeep Belgavi
PY ?= python3

.PHONY: setup test test-java test-python lint probe l01-cpu l02-cpu

setup:
	$(PY) -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -e "python[dev,l01,l02]"

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

# Episode 2 (transformer): the toy model and the PyTorch check, Python then Java. CPU only.
l02-cpu:
	PYTHONPATH=python/src $(PY) -m belgavi_llm.l02_transformer.demo
	mvn -B -ntp -q -f java/pom.xml compile
	java -cp java/target/classes dev.belgavi.llm.l02.Lesson02Demo
