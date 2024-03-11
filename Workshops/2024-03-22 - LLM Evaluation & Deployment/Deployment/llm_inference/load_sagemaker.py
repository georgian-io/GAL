import json
import os
import random

from datasets import load_dataset
from locust import HttpUser, between, task
from requests_aws4auth import AWS4Auth

data = load_dataset("b-mc2/sql-create-context")


def get_random_sample() -> str:
    idx = random.randint(0, len(data["train"]))
    sample = data["train"][idx]
    schema = sample["context"]
    query = sample["question"]

    prompt = f"Generate SQL for this user query: {query} for next table schema: {schema}"
    return prompt


ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "gemma-2b-endpoint")
REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

aws_auth = AWS4Auth(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, REGION, "sagemaker", session_token=AWS_SESSION_TOKEN)


class SageMakerUser(HttpUser):
    wait_time = between(5, 10)  # Wait time between tasks (1-2 seconds here)

    @task
    def post_inference(self):
        prompt = get_random_sample()

        payload = {
            "inputs": prompt,
            "parameters": {
                "details": True,
                "max_new_tokens": 50,
            },
        }

        headers = {"Content-Type": "application/json"}
        response = self.client.post(
            f"https://runtime.sagemaker.{REGION}.amazonaws.com/endpoints/{ENDPOINT_NAME}/invocations",
            data=json.dumps(payload),
            headers=headers,
            auth=aws_auth,
        )

        print("Response status:", response.status_code)
        response.raise_for_status()  # To make sure we notice failed requests
