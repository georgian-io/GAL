# TMLS - LLM Evaluation

In this talk we'll cover various aspects of LLM evaluation including:

* Traditional evaluation, human evaluation, evaluating LLMs
* Evaluation metrics and considerations
* Hands-on end-to-end examples


## Environment Setup & Installation

1. Setup your environment

If you use Conda:

```bash
conda create -n llm-evaluation python=3.10
conda activate llm-evaluation
pip install -r requirements.txt
```

If you use Poetry:

```bash
poetry shell
poetry install
```

If you use pure Python:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

2. Create private environment file (this will not be committed!)
```
cp .env-template .env
```

Add any necessary API keys there following the given format. Specifically, you will need a key for your LLM (this demo uses OpenAI).

## Quick-Start:

The `notebooks` directory contains a step-by-step, end-to-end example of a Q&A app with RAG. it consists of the following:

1. `01-llm-app-setup.ipynb`: Set up the LLM app
2. `02-dataset-creation.ipynb`: Create evaluation dataset
3. `03-metrics-definition.ipynb`: Define metrics to use
4. `04-app-evaluation.ipynb`: Evaluate the app

### Required: Using an LLM

We use langchain for all the code in the notebooks. By default we are using an OpenAI model and thus require an OpenAI key. The entire example will cost around $25 with OpenAI. 
