import sagemaker
from sagemaker.sklearn import SKLearn

# Initialize the Sagemaker session
sagemaker_session = sagemaker.Session()
role = 'your-aws-iam-role'

# Define the SKLearn estimator
sklearn_estimator = SKLearn(
    entry_point='train.py',
    role=role,
    instance_type='ml.m5.large',
    framework_version='0.23-1',
    py_version='py3',
    sagemaker_session=sagemaker_session
)

# Train the model
sklearn_estimator.fit({'train': 's3://your-bucket/your-training-data/'})
