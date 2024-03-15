# LLM Inference 

# Setup 

Tools to install: 

- [python](https://www.python.org/)
- [docker](https://docs.docker.com/engine/install/)
- [aws-cli](https://github.com/aws/aws-cli)
- [gcloud CLI](https://cloud.google.com/sdk/gcloud)
- [kind](https://kind.sigs.k8s.io/docs/user/quick-start/)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [k9s](https://k9scli.io/topics/install/)

Access to have:

- [Huggingface token to read models](https://huggingface.co/docs/hub/en/security-tokens)
- [AWS SageMaker](https://aws.amazon.com/sagemaker/)
- [AWS SageMaker Role](https://docs.aws.amazon.com/sagemaker/latest/dg/role-manager.html)
- [GCP Vertex AI](https://cloud.google.com/vertex-ai?hl=en)


```
export HUGGING_FACE_HUB_TOKEN=hf_your_token_here
```

## TGI 

GPU

```
docker run --gpus all --shm-size 1g -p 8080:80 -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN -v $PWD:/data ghcr.io/huggingface/text-generation-inference:1.4.3 --model-id google/gemma-2b
```

CPU

```
docker run -it --shm-size 1g -p 8080:80 -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN -v $PWD:/data ghcr.io/huggingface/text-generation-inference:1.4.3 --model-id google/flan-t5-small
```

Client 

```
example_input="Generate SQL for this user query: How many heads of the departments are older than 56 ? for next table schema: CREATE TABLE head (age INTEGER)"
python llm_inference/clients.py tgi-requests-client $example_input --host=http://0.0.0.0:8080
```

Benchamark

```
locust -f llm_inference/load_tgi.py
```

Reference: 

- https://github.com/huggingface/text-generation-inference


## Lorax

GPU 

```
docker run --gpus all --shm-size 1g -p 8080:80 -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN -p 8080:80 -v $PWD:/data ghcr.io/predibase/lorax:latest --model-id google/gemma-2b
```

CPU

```
docker run --shm-size 1g -p 8080:80 -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN -p 8081:80 -v $PWD:/data ghcr.io/predibase/lorax:latest --model-id microsoft/phi-2
```

Client 

```
example_input="Generate SQL for this user query: How many heads of the departments are older than 56 ? for next table schema: CREATE TABLE head (age INTEGER)"
python llm_inference/clients.py lorax-requests-client $example_input --host=http://0.0.0.0:8080
```

Client with adapted

```
example_input="Generate SQL for this user query: How many heads of the departments are older than 56 ? for next table schema: CREATE TABLE head (age INTEGER)"
python llm_inference/clients.py lorax-requests-client $example_input --host=http://0.0.0.0:8080 --adapter-id=Yhyu13/phi-2-sft-alpaca_gpt4_en-ep1-lora
```

Benchamark

```
locust -f llm_inference/load_tgi.py
```

Reference: 

- https://github.com/predibase/lorax

# Deploy

## K8S

Create cluster 

```
kind create cluster --name llm-inference 
```

Run UI for cluster 
```
k9s -A 
```

Deploy endpoint

```
kubectl create -f ./k8s/serving-custom-model-flan-base.yaml
kubectl create -f ./k8s/serving-custom-model-flan-small.yaml
```

Access model 

```
kubectl port-forward --address 0.0.0.0 svc/flan-t5-base 8888:80
kubectl port-forward --address 0.0.0.0 svc/flan-t5-small 8888:80
```

Reference: 

- https://predibase.github.io/lorax/getting_started/kubernetes/

## Sagemaker


Setup AWS creds and get Sagemaker role

```
export AWS_ACCESS_KEY_ID="***"
export AWS_SECRET_ACCESS_KEY="***"
export AWS_SESSION_TOKEN="***"
export AWS_REGION=us-east-1
```

Deploy model

```
python llm_inference/sagemaker_deploy.py deploy --endpoint-name gemma-2b-endpoint --model-name google/gemma-2b --hf-token $HUGGING_FACE_HUB_TOKEN
```

Query model

```
example_input="Generate SQL for this user query: How many heads of the departments are older than 56 ? for next table schema: CREATE TABLE head (age INTEGER)"
python llm_inference/sagemaker_deploy.py query --endpoint-name gemma-2b-endpoint --prompt $example_input
```

Benchamark


```
locust -f llm_inference/load_sagemaker.py
```