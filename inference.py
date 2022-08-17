## Beforehairshop 폴더 아래에 있는 파일
from genericpath import isfile
import subprocess
import boto3
import json
import os
from PIL import Image
from secret import AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET_NAME, AWS_S3_BUCKET_REGION
from secret import AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY
from secret import AWS_SQS_REGION, AWS_REQUEST_SQS_NAME, AWS_RESPONSE_SQS_NAME
from secret import AWS_REQUEST_SQS_URL, AWS_RESPONSE_SQS_URL
from secret import USER_INPUT_IMAGE_PATH
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


# Getting Response Queue from SQS
response_queue = get_response_queue()

# Getting Request Queue from SQS
request_queue = get_request_queue()

# Getting S3 Bucket
session = boto3.Session(
    aws_access_key_id = AWS_ACCESS_KEY,
    aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
    region_name = AWS_S3_BUCKET_REGION
)
s3_resource = session.resource('s3')

logger = logging.getLogger(__name__)

def download_image_from_s3(user_id):
    # S3로부터 이미지 다운로드 받는다.
    s3_resource.meta.client.download_file(Bucket=AWS_S3_BUCKET_NAME, Key='/' + str(user_id) + '/' + 'profile.jpg', Filename=USER_INPUT_IMAGE_PATH)


# def e4e_encoding():
#     # RGB+a => RGB로 바꾼 뒤,
#     # 유저 이미지 path 이용해서, face_alignment해준다.
#     # 그리고나서, e4e 모델로 latent vector 구해서 파일로 저장한다.
#     print()

# def hairclip_inference(latent_vector):
#     # HairCLIP model inference
#     # s3에 이미지 업로드
#     print()

def main():
    # 전체 코드는 while True문 안에서 계속 돈다.

    while True:
        # messages = response_queue.receive_messages(QueueUrl=AWS_REQUEST_SQS_URL)
        messages = response_queue.meta.client.receive_message(
            QueueUrl=AWS_REQUEST_SQS_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=2,
            MessageAttributeNames=['All']
        )

        # 메시지 없었으면 그냥 continue해준다.
        # if len(messages) == 0:
        #     continue

        # 버전2
        if 'Messages' not in messages:
            continue

        
        # for message in messages:
        for message in messages['Messages']:
            data = message['Body']
            data = json.loads(data)

            logger.info("Message from Request Queue : ", data)

            param_user_id = data["user_id"]

            request_queue.meta.client.delete_message(
                QueueUrl=AWS_REQUEST_SQS_URL,
                ReceiptHandle=message['ReceiptHandle']
            )
            
            # AWS S3로부터 이미지 다운로드
            download_image_from_s3(param_user_id)

            # e4e 파일 실행
            # 작업 디렉터리 변경
            os.chdir("/home/ubuntu/BeforeHairshop/encoder4editing/")

            subprocess.call("/home/ubuntu/BeforeHairshop/encoder4editing/e4e_encoding.py", shell=True)
            
            # e4e 모델로 임베딩됐는지 확인 (파일 생성 안됐으면, fail return)
            if os.path.isfile('/home/ubuntu/BeforeHairshop/HairCLIP/base_face.pt') == False:
                fail_body_json = {
                    'result' : 'fail',
                    'user_id' : param_user_id
                }
                fail_message_body_str = json.dumps(fail_body_json)
                try:
                    # Send message to Request Queue
                    response_queue.send_message(MessageBody=fail_message_body_str, QueueUrl=AWS_RESPONSE_SQS_URL)
                    logger.info("Send message failed! (Message body : { result : %s, user_id : %s })", "fail", param_user_id)
                    continue
                except ClientError as error:
                    logger.exception("Send message failed! (Message body : { result : %s, user_id : %s })", "fail", param_user_id)

                    raise error                


            ### hairclip inference 실행 (색깔 별로 inference)

            # HairCLIP 폴더로 작업 디렉토리 변경
            os.chdir("/home/ubuntu/BeforeHairshop/HairCLIP/mapper/")

            # black command
            hair_color_list = ["black", "brown", "red", "blue", "orange", "yellow"]
            hair_style_list = ["afro", "bowl cut", "braid", "caesar cut", "chignon", "dreadlocks", "fauxhawk", "jewfro", "perm", "pixie cut"
            , "psychobilly wedge", "regular taper cut", "shingle bob", "short hair", "slicked-back"]

            for color in hair_color_list:
                inference_style_cmd = 'python scripts/inference.py \
                --exp_dir=/home/ubuntu/BeforeHairshop/HairCLIP/inference_result \
                --checkpoint_path=/home/ubuntu/BeforeHairshop/HairCLIP/pretrained_models/hairclip.pt \
                --latents_test_path=/home/ubuntu/BeforeHairshop/HairCLIP/base_face.pt \
                --editing_type=both \
                --input_type=text \
                --hairstyle_description="hairstyle_list.txt" \
                --color_description="{}" \
                --end_index=1'.format(color)

                subprocess.call(inference_style_cmd, shell=True)

                inference_color_cmd = 'python scripts/inference.py \
                --exp_dir=/home/ubuntu/BeforeHairshop/HairCLIP/inference_result \
                --checkpoint_path=/home/ubuntu/BeforeHairshop/HairCLIP/pretrained_models/hairclip.pt \
                --latents_test_path=/home/ubuntu/BeforeHairshop/HairCLIP/base_face.pt \
                --editing_type=color \
                --input_type=text \
                --end_index=1 \
                --color_description="{}"'.format(color)

                subprocess.call(inference_color_cmd, shell=True)

            # style-inference 결과 AWS S3에 업로드시킨다. (EX => afro-black, afro-brown, afro-red, afro-yellow, afro-)
            hair_style_cnt = 0
            for hair_style in hair_style_list:
                for hair_color in hair_color_list:
                    # result 이미지 => S3에 올린다
                    destination_path = hair_style + " hairstyle-" + hair_color + " hair.jpg"
                    result_image_path = "/home/ubuntu/BeforeHairshop/HairCLIP/inference_result/both/text/00000-" + str(hair_style_cnt).zfill(4) + "-" + destination_path

                    s3_resource.meta.client.upload_file(result_image_path, AWS_S3_BUCKET_NAME, '/' + str(param_user_id) + "/both/" + destination_path)
                hair_style_cnt = hair_style_cnt + 1
            
            # color-inference 결과를 AWS S3에 업로드시킨다.
            for hair_color in hair_color_list:
                destination_path = hair_color + " hair.jpg"
                result_image_path = "/home/ubuntu/BeforeHairshop/HairCLIP/inference_result/color/text/00000-0000-" + destination_path

                s3_resource.meta.client.upload_file(result_image_path, AWS_S3_BUCKET_NAME, '/' + str(param_user_id) + "/color/" + destination_path)

            # (덮어쓰기 가능하면 생략) HairCLIP/result 폴더 삭제
            delete_cmd = 'rm -rf /home/ubuntu/BeforeHairshop/HairCLIP/inference_result/'
            subprocess.call(delete_cmd, shell=True)
            
            logger.info("inference of user_id : " + str(param_user_id) +  "success!! ")

            # 추론(inference) 이후에는 latent_code 정보 삭제
            delete_latent_cmd = 'rm /home/ubuntu/BeforeHairshop/HairCLIP/base_face.pt'
            subprocess.call(delete_latent_cmd, shell=True)

            success_msg_json = {
                'result' : 'success',
                'user_id' : param_user_id
            }

            success_message_body_str = json.dumps(success_msg_json)

            try:
                # Send message to Request Queue
                response_queue.send_message(MessageBody=success_message_body_str, QueueUrl=AWS_RESPONSE_SQS_URL)
                logger.info("Send message failed! (Message body : { user_id : %s })", param_user_id)
            except ClientError as error:
                logger.exception("Send message failed! (Message body : { user_id : %s })", param_user_id)

                raise error
            
            
    
    ## SQS(request queue)
    # 1. request queue로부터 메시지 읽어들인다.

    ## preprocessing
    # 1. s3로부터 이미지 가져옴 (RGB+a => RGB) 
    # 2. face_alignment
    # 3. e4e encoding

    ## inference
    # 1. inference (결과는 s3에 업로드한다)

    ## SQS(response queue)
    # 1. sqs(response queue)로 메시지 보낸다.


if __name__ == "__main__":
	main()