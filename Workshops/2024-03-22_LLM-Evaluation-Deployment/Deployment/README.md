# Transferred Learnings Workshop - LLM Deployment

In this stream we'll cover the steps to deploy and serve your proof-of-concept including:

* How to use open-source (Mistral, LLaMa etc.) vs. closed-source (OpenAI, Anthropic etc.) models
* The basics of serving models using tools and libraries like TGI, vLLM and Triton.
* Advanced use cases such as multiple LoRAs, asynchronous inference, multimodal models.
* Deep dive into examples and cost analysis

To participate in the Deployment stream, any trained or fine-tuned model should suffice. 

# Environment Setup & Installation

## Tools you need to install

Most of this can be run locally. Note that `Lorax` requires an NVIDIA GPU, so you may not be able to run it without an instance. If you run into any issues, reach out to us via Slack or open up an issue on Github.

- [python](https://www.python.org/)
- [docker](https://docs.docker.com/engine/install/)
- [homebrew](https://brew.sh/)
  - If you are on a Mac
- [kind](https://kind.sigs.k8s.io/docs/user/quick-start/)
  - `brew install kind`
  - The link above offers installation instructions for other operating systems.
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
  - `brew install kubectl`
  - The link above offers installation instructions for other operating systems.
- [k9s](https://k9scli.io/topics/install/)
  - `brew install derailed/k9s/k9s`
  - The link above offers installation instructions for other operating systems.
- [Optional] AWS Users: [aws-cli](https://github.com/aws/aws-cli)
  - `pip install awscli`
- [Optional] GCP Users: [gcloud CLI](https://cloud.google.com/sdk/docs/install)
  - Please follow the instructions on the webpage linked above.

> Note: You can run `install_deps.sh` to install all the packages mentioned above. This assumes you have Python, Docker, and Homebrew installed.

## Services you need access to

- [Hugging Face Token](https://huggingface.co/docs/hub/en/security-tokens)
  - Note: This is a free token you can acquire by creating a Hugging Face account.
  - Login to your account, go to [this link](https://huggingface.co/settings/tokens), and create a `User Access Token` under the `Access Tokens` option.
  - Once you have the token, run the following code in the terminal after substituting your token:
  - ```export HUGGING_FACE_HUB_TOKEN=hf_your_token_here```
- [Optional] AWS Users:
  - [AWS SageMaker](https://aws.amazon.com/sagemaker/)
  - [AWS SageMaker Role](https://docs.aws.amazon.com/sagemaker/latest/dg/role-manager.html)
- [Optional] GCP Users:
  - [GCP Vertex AI](https://cloud.google.com/vertex-ai?hl=en)

## Setup your environment

If you use Conda:

```bash
conda create -n llm-deployment python=3.10
conda activate llm-deployment
pip install -r requirements.txt
```

If you use pure Python:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

If you have access to AWS Sagemaker, run the following after setting up your environment above.

```bash
pip install -r aws_requirements.txt
```

# Workshop Code

## Text Generation Interface

### 1. Run Docker

Chose one of the options below (GPU  vs CPU) depending on your hardware.

> **_NOTE:_**  If using an M1 or M2 Macbook, you might have to additionally set this env variable before running the commands below: `export DOCKER_DEFAULT_PLATFORM=linux/amd64`

#### GPU

```bash
docker run --gpus all --shm-size 1g -p 8080:80 -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN -v $PWD:/data ghcr.io/huggingface/text-generation-inference:1.4 --model-id google/gemma-2b
```

#### CPU

```bash
docker run -it --shm-size 1g -p 8080:80 -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN -v $PWD:/data ghcr.io/huggingface/text-generation-inference:1.4.3 --model-id google/flan-t5-small
```

### 2. Run the Client 

```bash
example_input="Generate SQL for this user query: How many heads of the departments are older than 56 ? for next table schema: CREATE TABLE head (age INTEGER)"
python llm_inference/clients.py tgi-requests-client $example_input --host=http://0.0.0.0:8080
```

### 3. Benchmarking

```bash
locust -f llm_inference/load_tgi.py
```
Then visit `http://0.0.0.0:8089/` to perform benchmarking.

Reference: 

- https://github.com/huggingface/text-generation-inference


## Lorax

Remember to generate a Hugging Face token [here](https://huggingface.co/settings/tokens) and run ```export HUGGING_FACE_HUB_TOKEN=hf_your_token_here```

Note: Lorax **requires** an Nvidia GPU (Ampere generation or above) to run.

### Run on GPU 

```bash
docker run --gpus all --shm-size 1g -p 8080:80 -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN -p 8080:80 -v $PWD:/data ghcr.io/predibase/lorax:latest --model-id google/gemma-2b
```

### Run the Client 

```bash
example_input="Generate SQL for this user query: How many heads of the departments are older than 56 ? for next table schema: CREATE TABLE head (age INTEGER)"
python llm_inference/clients.py lorax-requests-client $example_input --host=http://0.0.0.0:8080
```

### Run the Client with an adapter

```bash
example_input="Generate SQL for this user query: How many heads of the departments are older than 56 ? for next table schema: CREATE TABLE head (age INTEGER)"
python llm_inference/clients.py lorax-requests-client $example_input --host=http://0.0.0.0:8080 --adapter-id=Yhyu13/phi-2-sft-alpaca_gpt4_en-ep1-lora
```

### Benchmarking

```bash
locust -f llm_inference/load_tgi.py
```

Reference: 

- https://github.com/predibase/lorax

# Deploy

## K8S

### Create a cluster 

```bash
kind create cluster --name llm-inference 
```

### Deploy an endpoint and forwarding
> Pick whichever model you want to use:

#### Small

```bash
kubectl create -f ./k8s/serving-custom-model-flan-small.yaml
kubectl port-forward --address 0.0.0.0 svc/flan-small 8888:80
```

#### Base

```bash
kubectl create -f ./k8s/serving-custom-model-flan-base.yaml
kubectl port-forward --address 0.0.0.0 svc/flan-base 8888:80
```

### Run UI for cluster 

```bash
k9s -A 
```

### Access the model

```bash
curl localhost:8888/generate \
    -X POST \
    -d '{
        "inputs": "Who won the 2014 FIFA World Cup?",
        "parameters": {
            "max_new_tokens": 64
        }
    }' \
    -H 'Content-Type: application/json'
```

Reference: 

- https://predibase.github.io/lorax/getting_started/kubernetes/


## AWS Sagemaker


Make sure you have access to Google's gemma-2b model [here](https://huggingface.co/google/gemma-2b). If you don't have access, you can either request access in the [link](https://huggingface.co/google/gemma-2b) or use a different model below.

Remember to setup your AWS credentials below and ensure access to Sagemaker.

```
export AWS_ACCESS_KEY_ID=""
export AWS_SECRET_ACCESS_KEY=""
export AWS_SESSION_TOKEN=""
export AWS_REGION=us-east-1
```

### Deploy the model

```
python llm_inference/sagemaker_deploy.py deploy --endpoint-name gemma-2b-endpoint --model-name google/gemma-2b --hf-token $HUGGING_FACE_HUB_TOKEN
```

### Query the model

```
example_input="Generate SQL for this user query: How many heads of the departments are older than 56 ? for next table schema: CREATE TABLE head (age INTEGER)"
python llm_inference/sagemaker_deploy.py query --endpoint-name gemma-2b-endpoint --prompt $example_input
```

### Benchmarking

Note: If you used a different endpoint name, please run `export ENDPOINT_NAME="YourEndpointNameHere"`

```
locust -f llm_inference/load_sagemaker.py
```

# FAQ

* I'm getting an error about `bitsandbytes` along the lines of ```The installed version of bitsandbytes was compiled without GPU support```.

If you're using an M1/M2/M3 Macbook, this is unfortunately expected as `bitsandbytes` does not support those Macs yet. As such you may not be able to run the Text Generation Interface. We recommend trying it on different hardware such as a virtual machine or instance. Alternatively, you may skip it for the demo!

* I'm getting a  ```docker: no matching manifest for linux/arm64/v8 in the manifest list entries.``` error (or something similar).

Run the following in your terminal: `export DOCKER_DEFAULT_PLATFORM=linux/amd64` or add `--platform linux/amd64`  to the docker command.

* I don't want to run so many `export` commands.

Create a `.env` file and paste in all the environment variables such as `HUGGING_FACE_HUB_TOKEN` and `DOCKER_DEFAULT_PLATFORM`. Then simply run `source .env`. Thanks to Armando Rosario for the suggestion!