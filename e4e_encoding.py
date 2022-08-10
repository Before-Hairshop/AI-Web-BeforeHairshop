## e4e 폴더 내에 있는 파일

from argparse import Namespace
import time
import os
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

sys.path.append(".")
sys.path.append("..")

from utils.common import tensor2im
from models.psp import pSp  


EXPERIMENT_DATA_ARGS = {
    "ffhq_encode": {
        "model_path": "/home/ubuntu/BeforeHairshop/encoder4editing/e4e_ffhq_encode.pt",
        
         "image_path": "/home/ubuntu/BeforeHairshop/user_image/input_image.jpg"
    },
    
    
}
experiment_type = 'ffhq_encode'

# === Face alignment ===
def run_alignment(image_path):
  import dlib
  from utils.alignment import align_face
  predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
  aligned_image = align_face(filepath=image_path, predictor=predictor)
  print("Aligned image has shape: {}".format(aligned_image.size))
  return aligned_image 


def run_on_batch(inputs, net):
    images, latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
    if experiment_type == 'cars_encode':
        images = images[:, :, 32:224, :]
    return images, latents


if __name__ == '__main__':
    # Setup required image transformations
    EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]
    if experiment_type == 'cars_encode':
        EXPERIMENT_ARGS['transform'] = transforms.Compose([
                transforms.Resize((192, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        resize_dims = (256, 192)
    else:
        EXPERIMENT_ARGS['transform'] = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        resize_dims = (256, 256)

    model_path = EXPERIMENT_ARGS['model_path']
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    # pprint.pprint(opts)  # Display full options used
    # update the training options
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    print('Model successfully loaded!')

    # === RGB+알파 => RGB value로 변경 ===
    image_path = EXPERIMENT_DATA_ARGS[experiment_type]["image_path"]
    original_image = Image.open(image_path)
    original_image = original_image.convert("RGB")
    # ==================================



    if experiment_type == "ffhq_encode":
        input_image = run_alignment(image_path)
        input_image = input_image.convert('RGB')
    else:
        input_image = original_image

    input_image.resize(resize_dims)
    # ==================================


    # ===== e4e 인코딩 ======
    img_transforms = EXPERIMENT_ARGS['transform']
    transformed_image = img_transforms(input_image)


    with torch.no_grad():
        tic = time.time()
        images, latents = run_on_batch(transformed_image.unsqueeze(0), net)
        result_image, latent = images[0], latents[0]
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))

    torch.save(latents, '/home/ubuntu/BeforeHairshop/HairCLIP/base_face.pt')

