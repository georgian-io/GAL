import random

from datasets import load_dataset
from locust import HttpUser, between, task

data = load_dataset("b-mc2/sql-create-context")


def get_random_sample() -> str:
    idx = random.randint(0, len(data["train"]))
    sample = data["train"][idx]
    schema = sample["context"]
    query = sample["question"]

    prompt = f"Generate SQL for this user query: {query} for next table schema: {schema}"
    return prompt


def get_random_adapter() -> str:
    available_adapters = [
        None,
        "sohug/gemma-2b_banglo_lora",
        "DarshanDeshpande/gemma-2b-lora-commonsense-qa",
        "NickyNicky/gemma-2b-it_lora_oasst2_cluster1",
    ]
    idx = random.randint(0, len(available_adapters))
    return available_adapters[idx]


class TGILoadTest(HttpUser):
    wait_time = between(1, 5)  # Define wait time between tasks (1 to 5 seconds)
    host = "http://localhost:8080"

    @task
    def send_tgi_request(self):
        prompt = get_random_sample()
        adapter_id = get_random_adapter()
        endpoint = "/generate"  # API endpoint
        headers = {
            "Content-Type": "application/json",
        }
        data = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 25, "adapter_id": adapter_id},
        }

        # Send POST request
        self.client.post(endpoint, headers=headers, json=data)
