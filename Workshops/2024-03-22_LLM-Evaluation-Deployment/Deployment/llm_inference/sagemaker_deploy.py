import json
from pprint import pprint
from typing import Optional

import boto3
import sagemaker
import typer
from sagemaker.huggingface import HuggingFaceModel, HuggingFacePredictor, get_huggingface_llm_image_uri

region_name = "us-east-1"

SAGEMAKER_ROLE = "arn:aws:iam::823217009914:role/service-role/AmazonSageMaker-ExecutionRole-20180305T161813"


def add_autoscaling(endpoint_name: str):
    auto_scaling_client = boto3.client("application-autoscaling")

    resource_id = "endpoint/" + endpoint_name + "/variant/" + "AllTraffic"
    response = auto_scaling_client.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=1,
        MaxCapacity=5,
    )

    # GPUMemoryUtilization metric
    response = auto_scaling_client.put_scaling_policy(
        PolicyName="GPUUtil-ScalingPolicy",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",  # SageMaker supports only Instance Count
        PolicyType="TargetTrackingScaling",  # 'StepScaling'|'TargetTrackingScaling'
        TargetTrackingScalingPolicyConfiguration={
            # Scale out when GPU utilization hits GPUUtilization target value.
            "TargetValue": 60.0,
            "CustomizedMetricSpecification": {
                "MetricName": "GPUUtilization",
                "Namespace": "/aws/sagemaker/Endpoints",
                "Dimensions": [
                    {"Name": "EndpointName", "Value": endpoint_name},
                    {"Name": "VariantName", "Value": "AllTraffic"},
                ],
                "Statistic": "Average",  # Possible - 'Statistic': 'Average'|'Minimum'|'Maximum'|'SampleCount'|'Sum'
                "Unit": "Percent",
            },
            "ScaleInCooldown": 600,
            "ScaleOutCooldown": 200,
        },
    )
    print(f"Created GPU scaling policy {response}")


def deploy(
    endpoint_name: str = "gemma-2b-endpoint",
    model_name: str = "google/gemma-2b",
    role: str = SAGEMAKER_ROLE,
    region_name: str = "us-east-1",
    instance_type: str = "ml.g5.4xlarge",
    number_of_gpu: int = 1,
    quantize: Optional[str] = None,
    autoscaling: bool = False,
    hf_token: Optional[str] = None,
):

    sagemaker_session = sagemaker.Session(boto3.Session(region_name=region_name))
    llm_image = get_huggingface_llm_image_uri("huggingface", version="1.4", session=sagemaker_session)
    print(f"llm image uri: {llm_image}")

    # sagemaker config
    health_check_timeout = 300

    # Define Model and Endpoint configuration parameter
    config = {
        "HF_MODEL_ID": model_name,  # model_id from hf.co/models
        "SM_NUM_GPUS": json.dumps(number_of_gpu),  # Number of GPU used per replica
        "MAX_INPUT_LENGTH": json.dumps(2048),  # Max length of input text
        "MAX_TOTAL_TOKENS": json.dumps(4096),  # Max length of the generation (including input text)
        "MAX_BATCH_TOTAL_TOKENS": json.dumps(
            8192
        ),  # Limits the number of tokens that can be processed in parallel during the generation
        "HF_API_TOKEN": hf_token,
    }
    if quantize is not None:
        config["HF_MODEL_QUANTIZE"] = quantize
    print(f"Config {config}")

    # create HuggingFaceModel with the image uri
    llm_model = HuggingFaceModel(role=role, image_uri=llm_image, env=config, sagemaker_session=sagemaker_session)

    llm = llm_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        container_startup_health_check_timeout=health_check_timeout,
        endpoint_name=endpoint_name,
    )

    if autoscaling:
        add_autoscaling(endpoint_name=endpoint_name)

    print(f"Warm model")
    print(llm.predict({"inputs": "This is a sample sentence to warm up the model"}))
    print(llm.predict({"inputs": "This is a sample sentence to warm up the model"}))


def query(endpoint_name: str = "gemma-2b-endpoint", prompt: str = "test"):
    predictor = HuggingFacePredictor(endpoint_name)

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 25,
        },
    }

    response = predictor.predict(payload)
    pprint(response[0]["generated_text"])


def main():
    app = typer.Typer()
    app.command()(deploy)
    app.command()(query)
    app()


if __name__ == "__main__":
    main()
