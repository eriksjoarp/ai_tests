# requirements
'''
min-dalle
numpy==1.23.0
pillow==9.2.0
requests==2.28.1

python -m pip install -r requirements
conda install pytorch torchvision torchaudio pandas cudatoolkit=11.3 -c pytorch
'''

from min_dalle import MinDalle
import torch
from IPython.display import display, update_display
import time
import cv2
import os
import PIL

text = 'ultrahigh resolution. photorealistic style. hurricane in the ocean. high dover cliffs. church next to the ocean.'

print('__CUDA VERSION:', torch.version.cuda)
print('__CUDNN VERSION:', torch.backends.cudnn.version())

dev = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = MinDalle(
    models_root='./pretrained',
    dtype=torch.bfloat16,       # original float32
    #dtype=torch.float32,
    device='cuda',
    is_mega=True,              # Set to True for best quality
    is_reusable=True
)


for i in range(0,10):

    image = model.generate_image(
        text=text,
        seed=-1,
        grid_size=1,
        is_seamless=False,
        temperature=1,
        top_k=256,
        supercondition_factor=32,
        is_verbose=False
    )


    #print(type(image))
    #image.show()

    #print(type(images))

    #images = images.to('cpu').numpy()

    file = os.path.join('results', 'save_test' + str(i) + '.png')
    image.save(file)

exit(0)

#cv2.imshow("Sheep", image)
img = images.fromarray(images)
img.save('image_{}.png'.format(1))

display(image)

time.sleep(10)

exit(0)

img = cv2.imread("sheep.png", cv2.IMREAD_ANYCOLOR)



while True:
    cv2.imshow("Sheep", img)
    cv2.waitKey(0)
    sys.exit()  # to exit from all the processes

cv2.destroyAllWindows()  # destroy all windows



images = images.to('cpu').numpy()

image = Image.fromarray(images[i])
image.save('image_{}.png'.format(i))

image_stream = model.generate_image_stream(
    text='Dali painting of WALL·E',
    seed=-1,
    grid_size=3,
    progressive_outputs=True,
    is_seamless=False,
    temperature=1,
    top_k=256,
    supercondition_factor=16,
    is_verbose=False
)

for image in image_stream:
    display(image)


