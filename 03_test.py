import boto3
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import json

DELETE_ENDPOINT = False

sm = boto3.client('sagemaker')

# retrieve latest endpoint
endpoints = sm.list_endpoints()
endpoint_name = endpoints['Endpoints'][0]['EndpointName']
print("Endpoint:", endpoint_name)

content_type = "text/csv"
client = boto3.client("runtime.sagemaker")


def query_endpoint(encoded_tabular_data):
    response = client.invoke_endpoint(
        EndpointName=endpoint_name, ContentType=content_type, Body=encoded_tabular_data
    )
    return response


def parse_response(query_response):
    model_predictions = json.loads(query_response["Body"].read())
    predicted_probabilities = model_predictions["probabilities"]
    return np.array(predicted_probabilities)


df_test = pd.read_csv('data/X_test.csv')

query_response_batch = query_endpoint(
    df_test.iloc[:, :].to_csv(header=False, index=False).encode("utf-8")
)

predict_prob = parse_response(query_response_batch)
predict_labels = np.argmax(predict_prob, axis=1)
test_labels = pd.read_csv('data/y_test.csv')

conf_matrix = confusion_matrix(y_true=test_labels, y_pred=predict_labels)

# Measure the prediction results quantitatively.
eval_accuracy = accuracy_score(test_labels, predict_labels)
eval_f1 = f1_score(test_labels, predict_labels)

newline, bold, unbold = "\n", "\033[1m", "\033[0m"

print(
    f"{bold}Evaluation result on test data{unbold}:{newline}"
    f"{bold}{accuracy_score.__name__}{unbold}: {eval_accuracy}{newline}"
    f"{bold}F1 {unbold}: {eval_f1}{newline}"
    f"{newline}{bold}Confusion Matrix{unbold}{newline}"
    f"{conf_matrix}"
)

if DELETE_ENDPOINT:
    sm.delete_endpoint(EndpointName=endpoint_name)
