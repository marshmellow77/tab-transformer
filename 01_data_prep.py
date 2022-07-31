import sagemaker
import boto3
import pandas as pd
import json
from fast_ml.model_development import train_valid_test_split
import os

# download the data
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
df = pd.read_csv(data_url, sep=" ", header=None)

# rename the columns
df.columns = [f"Feature_{i}" for i in range(1, df.shape[1])] + ["Target"]

# re-map target values to 0 and 1
df["Target"] = df["Target"].map({1: 1, 2: 0})

# split the data into train, validation and test
X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(df, target='Target',
                                                                            train_size=0.8, valid_size=0.1,
                                                                            test_size=0.1, random_state=42)

# create training data & validation data
df_train = pd.concat([y_train, X_train], axis=1)
df_valid = pd.concat([y_valid, X_valid], axis=1)
df_test = pd.concat([y_test, X_test], axis=1)

sess = sagemaker.Session()
s3_bucket = sess.default_bucket()
s3_folder = "tab-transformer"

# upload the data to S3
df_train.to_csv(f"s3://{s3_bucket}/{s3_folder}/train/data.csv", index=False, header=False)
df_valid.to_csv(f"s3://{s3_bucket}/{s3_folder}/validation/data.csv", index=False,
                header=False)

# check if local director "data" exists - if not create it
if not os.path.exists("data"):
    os.makedirs("data")

# save test file locally
X_test.to_csv(f"data/X_test.csv", index=False, header=False)
y_test.to_csv(f"data/y_test.csv", index=False, header=False)

# create json file to indicate categorical features based on
# https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
cat_index_dict = {}
cat_cols = [1, 3, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 19, 20]
cat_index_dict["cat_index"] = cat_cols
with open('cat_index.json', 'w', encoding="utf-8") as file:
    json.dump(cat_index_dict, file)

# upload the json file to S3
boto3.client("s3").upload_file("cat_index.json", s3_bucket, f"{s3_folder}/cat_index.json")
os.remove("cat_index.json")
