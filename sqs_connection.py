from secret import AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY
from secret import AWS_SQS_REGION, AWS_REQUEST_SQS_NAME, AWS_RESPONSE_SQS_NAME
import boto3
import logging
from botocore.exceptions import ClientError


def get_request_queue():
    aws_session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_SQS_REGION)
    sqs = aws_session.resource('sqs')

    queue = sqs.get_queue_by_name(QueueName=AWS_REQUEST_SQS_NAME)
    return queue


def get_response_queue():
    aws_session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_SQS_REGION)
    sqs = aws_session.resource('sqs')

    queue = sqs.get_queue_by_name(QueueName=AWS_RESPONSE_SQS_NAME)
    return queue