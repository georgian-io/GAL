from typing import Optional

import requests
import typer


def tgi_requests_client(prompt: str, host: str = "http://127.0.0.1:8080"):
    headers = {
        "Content-Type": "application/json",
    }

    data = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 20,
        },
    }

    response = requests.post(f"{host}/generate", headers=headers, json=data)
    print(response.json())


def lorax_requests_client(prompt: str, host: str = "http://127.0.0.1:8080", adapter_id: Optional[str] = None):
    headers = {
        "Content-Type": "application/json",
    }

    data = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 20, "adapter_id": adapter_id},
    }

    response = requests.post(f"{host}/generate", headers=headers, json=data)
    print(response.json())


def main():
    app = typer.Typer()
    app.command()(lorax_requests_client)
    app.command()(tgi_requests_client)
    app()


if __name__ == "__main__":
    main()
