import boto3
import logging
import json
from botocore.exceptions import ClientError


while True:
    # Request queue에서 message 가져옴.
    
    # === data preprocessing ===
    # S3에서 유저 프로필 이미지 가져옴.
    # face align 
    # e4e encoding 해서, latent_vector 가져옴
    # ==========================
    
    # === inference ===
    # HairCLIP inference
    # 결과 이미지 S3에 저장
    # =================

    # request message(user_id, hair_style, hair_color) 그대로 Return함
