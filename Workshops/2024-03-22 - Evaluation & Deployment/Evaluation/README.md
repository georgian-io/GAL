# LLM Evaluation Workshop Example
## Quick start guide

1. Environment management options
   
    a)  Poetry: ```poetry shell```
    
    b) Conda: create and activate a conda env for this project:
```bash
conda create -n llm-evaluation python=3.10
conda activate llm-evaluation
```

2. Install package
```
poetry install
```
3. Check installation worked by running 
```
pytest .
```

4. Create private environment file (this will not be committed!)
```
cp .env-template .env
```
Add any necessary API keys there following the given format.

5. Start from `notebooks/start_here.ipynb`

## Optional: Using LangSmith or Langfuse
LangSmith and Langfuse are observability tools to help evaluate LLM during development or monitoring during production. LangSmith is a tool provided by LangChain. As of February 22nd, 2024, they provide the first 3000 traces free per month, and each additional trace is $0.005.

Langfuse is an open-sourced observability tool. You can optionally self-host the solution for free and the code is available on [their github repo](https://github.com/langfuse/langfuse). If you choose to use their managed solution, as of February 22nd, 2024, they allow for the first 50,000 traces free per month.

If you'd like to integrate the evaluation with [LangSmith](https://smith.langchain.com/) or [Langfuse](https://langfuse.com/), please sign up on their platform and get the API keys.

## Repo Info
### Poetry
We use [poetry](https://python-poetry.org/) as our dependency manager.
The link above has great documentation but there is a TL;DR.

- Install the package: `poetry install`
- Add a dependency: `poetry add <python-lib>`
- Where are dependencies specified? `pyproject.toml` include the high level requirements. The latests exact versions installed are in `poetry.lock`.

