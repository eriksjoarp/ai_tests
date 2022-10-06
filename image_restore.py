#conda activate python39
#
'''
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge opencv

conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

installera exakt nummer from req_image_restore.txt

ToDo
look at GAN parameters
other backend to detect faces
upscaling projects or ideas

outscale


'''
#
#
#

import os, time
import cv2
#import gradio as gr
import torch, torchvision
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer

from helper import erik_functions_files
from helper import erik_functions_remote

from ai_helper import constants_dataset as c_d
from ai_helper import dataset_load

#############################################################
'''
models:
realesr-general-x4v3.pth
GFPGANv1.2.pth
GFPGANv1.3.pth
GFPGANv1.4.pth
RestoreFormer.pth
CodeFormer.pth
'''
GPU_ID = 0
CUDA_STRING = 'cuda:' + str(GPU_ID)


MODEL_WEIGHTS = r'realesr-general-x4v3.pth'
VERSION = 'v1.4'
#VERSION = 'RestoreFormer'
DIR_IMAGES = r'C:\Users\erikw\Pictures\gan'
DIR_SAVE = os.path.join(DIR_IMAGES, 'restored')
UPSCALE = 8
DO_IMAGES_COUNT = 0

LOOP = False                                    #   to loop many tests
UPSCALE_LOOP = [2,4]
VERSIONS_LOOP = ['v1.2', 'v1.3', 'v1.4']

#############################################################


def restore_image(dir_images, dir_save, path_weights=c_d.URLS_WEIGHTS_GFP_GAN[3] , scale=2):            # default v1.4 weights
    # download all weights if necessarry
    dataset_load.download_weights_realesr_gan()
    path_images = erik_functions_files.get_filenames_in_dir(dir_images)




exit(0)


if __name__== '__main__':
    pass



# background enhancer with RealESRGAN
print('load model')
model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
path_model = os.path.join(weights_dir, MODEL_WEIGHTS)

if os.path.exists(path_model):
    print('model found in directory')
else:
    print('cannot find model in model_path')
    exit(0)

half = True if torch.cuda.is_available() else False
device = torch.device(CUDA_STRING if torch.cuda.is_available() else "cpu")

upsampler = RealESRGANer(scale=4, model_path=path_model, model=model, tile=0, tile_pad=10, pre_pad=0, half=half)

if torch.cuda.is_available():
    print('using cuda')
    model.to(device)
    #upsampler.to(device)
else:
    print('cannot find cuda, using cpu')

os.makedirs('output', exist_ok=True)

# def inference(img, version, scale, weight):
def inference(img, version, scale, dir_save):
    # weight /= 100
    path_image = img
    print('path : ' + img + '  version : ' + version + '  scale : ' + str(scale))
    try:
        extension = os.path.splitext(os.path.basename(str(img)))[1]
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        elif len(img.shape) == 2:  # for gray inputs
            img_mode = None
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_mode = None

        h, w = img.shape[0:2]
        if h < 300:
            img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

        if version == 'v1.2':
            face_enhancer = GFPGANer(
            model_path=os.path.join(weights_dir, 'GFPGANv1.2.pth'), upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)
        elif version == 'v1.3':
            face_enhancer = GFPGANer(
            model_path=os.path.join(weights_dir, 'GFPGANv1.3.pth'), upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)
        elif version == 'v1.4':
            face_enhancer = GFPGANer(
            model_path=os.path.join(weights_dir, 'GFPGANv1.4.pth'), upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)
        elif version == 'RestoreFormer':
            face_enhancer = GFPGANer(
            model_path=os.path.join(weights_dir, 'RestoreFormer.pth'), upscale=2, arch='RestoreFormer', channel_multiplier=2, bg_upsampler=upsampler)
        # elif version == 'CodeFormer':
        #     face_enhancer = GFPGANer(
        #     model_path='CodeFormer.pth', upscale=2, arch='CodeFormer', channel_multiplier=2, bg_upsampler=upsampler)

        try:
            # _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True, weight=weight)
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        except RuntimeError as error:
            print('Error', error)

        try:
            if scale != 2:
                interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
                h, w = img.shape[0:2]
                output = cv2.resize(output, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)
        except Exception as error:
            print('wrong scale input.', error)
        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            extension = 'png'
        else:
            extension = 'jpg'
        base_name, _ = erik_functions_files.file_name_from_path(path_image)

        # save restored image
        filename = f'{base_name}_{scale}_{version}.{extension}'
        save_path = os.path.join(dir_save, filename)
        cv2.imwrite(save_path, output)

        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return output, save_path
    except Exception as error:
        print('global exception', error)
        return None, None



'''

title = "GFPGAN: Practical Face Restoration Algorithm"
description = r"""Gradio demo for <a href='https://github.com/TencentARC/GFPGAN' target='_blank'><b>GFPGAN: Towards Real-World Blind Face Restoration with Generative Facial Prior</b></a>.<br>
It can be used to restore your **old photos** or improve **AI-generated faces**.<br>
To use it, simply upload your image.<br>
If GFPGAN is helpful, please help to ⭐ the <a href='https://github.com/TencentARC/GFPGAN' target='_blank'>Github Repo</a> and recommend it to your friends 😊
"""
article = r"""

[![download](https://img.shields.io/github/downloads/TencentARC/GFPGAN/total.svg)](https://github.com/TencentARC/GFPGAN/releases)
[![GitHub Stars](https://img.shields.io/github/stars/TencentARC/GFPGAN?style=social)](https://github.com/TencentARC/GFPGAN)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2101.04061)

If you have any question, please email 📧 `xintao.wang@outlook.com` or `xintaowang@tencent.com`.

<center><img src='https://visitor-badge.glitch.me/badge?page_id=akhaliq_GFPGAN' alt='visitor badge'></center>
<center><img src='https://visitor-badge.glitch.me/badge?page_id=Gradio_Xintao_GFPGAN' alt='visitor badge'></center>
"""
 


demo = gr.Interface(
    inference, [
        gr.inputs.Image(type="filepath", label="Input"),
        # gr.inputs.Radio(['v1.2', 'v1.3', 'v1.4', 'RestoreFormer', 'CodeFormer'], type="value", default='v1.4', label='version'),
        gr.inputs.Radio(['v1.2', 'v1.3', 'v1.4', 'RestoreFormer'], type="value", default='v1.4', label='version'),
        gr.inputs.Number(label="Rescaling factor", default=2),
        # gr.Slider(0, 100, label='Weight, only for CodeFormer. 0 for better quality, 100 for better identity', default=50)
    ], [
        gr.outputs.Image(type="numpy", label="Output (The whole image)"),
        gr.outputs.File(label="Download the output image")
    ],
    title=title,
    description=description,
    article=article,
    # examples=[['AI-generate.jpg', 'v1.4', 2, 50], ['lincoln.jpg', 'v1.4', 2, 50], ['Blake_Lively.jpg', 'v1.4', 2, 50],
    #           ['10045.png', 'v1.4', 2, 50]]).launch()
    examples=[[r'gfp_gan/AI-generate.jpg', 'v1.4', 2], [r'gfp_gan/lincoln.jpg', 'v1.4', 2], [r'gfp_gan/Blake_Lively.jpg', 'v1.4', 2],
              [r'gfp_gan/10045.png', 'v1.4', 2]])
demo.queue(concurrency_count=4)
demo.launch(share=False)

'''