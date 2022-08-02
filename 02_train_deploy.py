from sagemaker import image_uris, model_uris, script_uris
from sagemaker.estimator import Estimator
from sagemaker.utils import name_from_base
import sagemaker
from sagemaker import hyperparameters
import boto3
import time

iam = boto3.client('iam')
sagemaker_role = iam.get_role(RoleName='<role-name-with-sagemaker-execution-permission>')['Role']['Arn']

train_model_id, train_model_version, train_scope = "pytorch-tabtransformerclassification-model", "*", "training"
training_instance_type = "ml.m5.2xlarge"

# Retrieve the docker image
train_image_uri = image_uris.retrieve(
    region=None,
    framework=None,
    model_id=train_model_id,
    model_version=train_model_version,
    image_scope=train_scope,
    instance_type=training_instance_type
)

# Retrieve the training script
train_source_uri = script_uris.retrieve(
    model_id=train_model_id, model_version=train_model_version, script_scope=train_scope
)

train_model_uri = model_uris.retrieve(
    model_id=train_model_id, model_version=train_model_version, model_scope=train_scope
)

sess = sagemaker.Session()
s3_bucket = sess.default_bucket()
s3_folder = "tab-transformer"
s3_path = f"s3://{s3_bucket}/{s3_folder}/"

# Retrieve the default hyper-parameters for training the model
hyperparameters = hyperparameters.retrieve_default(
    model_id=train_model_id, model_version=train_model_version
)

hyperparameters["epoch"] = "100"

# Create SageMaker Estimator instance
tabular_estimator = Estimator(
    role=sagemaker_role,
    image_uri=train_image_uri,
    source_dir=train_source_uri,
    model_uri=train_model_uri,
    entry_point="transfer_learning.py",
    instance_count=1,
    instance_type=training_instance_type,
    max_run=360000,
    hyperparameters=hyperparameters,
    output_path=s3_path
)

training_job_name = name_from_base(f"tabtransformers")

# Launch a SageMaker Training job by passing the S3 path of the training data
tabular_estimator.fit(
    {"training": s3_path},
    logs=True,
    job_name=training_job_name,
    wait=False
)

sm = boto3.client('sagemaker')

while True:
    job_status = sm.describe_training_job(TrainingJobName=training_job_name)['TrainingJobStatus']
    # print(f"Training job status: {job_status}")

    if job_status == "Completed":
        print("Training job completed successfully - deploying endpoint")
        inference_instance_type = "ml.m5.2xlarge"

        # Retrieve the inference docker container uri
        deploy_image_uri = image_uris.retrieve(
            region=None,
            framework=None,
            image_scope="inference",
            model_id=train_model_id,
            model_version=train_model_version,
            instance_type=inference_instance_type,
        )
        # Retrieve the inference script uri
        deploy_source_uri = script_uris.retrieve(
            model_id=train_model_id, model_version=train_model_version, script_scope="inference"
        )

        endpoint_name = name_from_base("tabtransformers-ep")

        # Use the estimator from the previous step to deploy to a SageMaker endpoint
        tabular_estimator.deploy(
            initial_instance_count=1,
            instance_type=inference_instance_type,
            entry_point="inference.py",
            image_uri=deploy_image_uri,
            source_dir=deploy_source_uri,
            endpoint_name=endpoint_name,
        )

        print("\nEndpoint deployed successfully")
        break
    elif job_status == "Failed":
        print("Training job failed")
        break
    else:
        print("Training job is not yet complete. Waiting...")
        time.sleep(60)
