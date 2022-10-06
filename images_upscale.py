import torch
from PIL import Image
import numpy as np
#from RealESRGAN import RealESRGAN
from realesrgan import utils
import os, time
import image_restore



SCALE = 4

work_dir = os.getcwd()
weights_dir = os.path.join(work_dir, 'gfp_gan')
weights_file = os.path.join((weights_dir), 'RealESRGAN_x4plus.pth')

DIR_IMAGES = r'C:\Users\erikw\Pictures\gan'
DIR_SAVE = os.path.join(DIR_IMAGES, 'restored_realesrgan')
path_images = image_restore.get_filenames_in_dir(DIR_IMAGES, full_path=True)
print(path_images)
print(weights_file)

START_TIME = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = RealESRGANer(device, scale=4)
#model = RealESRGANer(SCALE, weights_file)
model = RealESRGANer(SCALE)
#model.load_weights(weights_file, download=True)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

path_image = path_images[5]

image = Image.open(path_image).convert('RGB')

sr_image = model.predict(image)

file_name, file_ext = image_restore.file_name_from_path(path_image)
filename_new = file_name + '_scale_' + str(SCALE) + '.' + file_ext

path_new = os.path.join(DIR_SAVE, filename_new)
sr_image.save(path_new)

DURATION = time.time() - START_TIME
print(str(DURATION) + ' seconds')

