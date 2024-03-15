# Transferred Learnings Workshop - LLM Evaluation

In this stream we'll cover various aspects of LLM evaluation including:

* Traditional evaluation, human evaluation, evaluating LLMs
* Evaluation metrics and considerations
* Evaluation tool and platform comparison, including open-sourced alternatives
* Hands-on end-to-end examples

To get the most out of the Evaluation stream, please come prepared with a data set and a model to evaluate

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

Add any necessary API keys there following the given format. Specifically, you will need a key for your LLM (if you're using an API), a key for [LangSmith](https://smith.langchain.com/) (optional; it's called `LANGCHAIN_API_KEY` in the config), and a key for [Langfuse](https://langfuse.com/) (optional). Both LangSmith and Langfuse offer free API keys which can be obtained on their respective websites by creating an account. You can see the section below for mroe details.

## Quick-Start:

The `notebooks` directory contains a step-by-step, end-to-end example of a Q&A app with RAG. it consists of the following:

1. `01-llm-app-setup.ipynb`: Set up the LLM app
2. `02-dataset-creation.ipynb`:Create evaluation dataset
3. `03-metrics-definition.ipynb`: Define metrics to use
4. `04-app-evaluation.ipynb`: Evaluate the app
5. `05-optional-langsmith.ipynb`: [Optional] Using LangSmith for tracing and evaluation
6. `06-optional-langfuse.ipynb`: [Optional] Using Langfuse for tracing and evaluation

### Required: Using an LLM
We use langchain for all the code in the notebooks. By default we are using an OpenAI model and thus require an OpenAI key. However, you can change the LLM being used and the code should still work. You can refer to the [LangChain documentation](https://python.langchain.com/docs/integrations/llms/) to change the LLM being used. The entire example will cost around $25 with OpenAI.

### Optional: Using LangSmith or Langfuse
LangSmith and Langfuse are observability tools to help evaluate LLM during development or monitoring during production. LangSmith is a tool provided by LangChain. As of February 2024, they provide the first 3000 traces per month for free, and each additional trace is $0.005.

Langfuse is an open-sourced observability tool. You can optionally self-host the solution for free and the code is available on [their github repo](https://github.com/langfuse/langfuse). If you choose to use their managed solution, as of February 2024, they offer the first 50,000 traces per month for free.

If you'd like to integrate the evaluation with [LangSmith](https://smith.langchain.com/) or [Langfuse](https://langfuse.com/), please sign up on their platform and get the API keys.

The demo should consume less than a few hundred traces/observations for each tool. 