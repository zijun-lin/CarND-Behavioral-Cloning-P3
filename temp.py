import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# image = mpimg.imread('origin_data/IMG/center_2016_12_01_13_30_48_287.jpg')
# # image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
# print("Image1: ", image.shape)
# image_pro1 = image[70:135, :]
# print("Image2: ", image_pro1.shape)
# image_pro2 = cv2.resize(image_pro1, (200, 66))
# print("Image3: ", image_pro2.shape)
#
#
# f, ax = plt.subplots(2, 2, figsize=(9, 6))
# f.tight_layout()
# ax[0, 0].imshow(image)
# ax[1, 0].imshow(image_pro1)
# ax[1, 1].imshow(image_pro2)
#
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.show()

# a = np.tan(np.pi/4)

# image = mpimg.imread('data/center_2.jpg')
#
# f, ax = plt.subplots(1, 2, figsize=(12, 4))
# f.tight_layout()
# flip_image = cv2.flip(image, 1)
# ax[0].imshow(image)
# ax[0].set_title('Original Image')
# ax[1].imshow(flip_image)
# ax[1].set_title('Filpped Image')
# plt.show()

########################################################################
import imageio
import glob


def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return



def main():
    image_list = [
                  'gif/11.jpg', 'gif/12.jpg', 'gif/13.jpg', 'gif/14.jpg', 'gif/15.jpg', 'gif/16.jpg', 'gif/17.jpg', 'gif/18.jpg', 'gif/19.jpg', 'gif/20.jpg',
                  'gif/21.jpg', 'gif/22.jpg', 'gif/23.jpg', 'gif/24.jpg', 'gif/25.jpg', 'gif/26.jpg', 'gif/27.jpg', 'gif/28.jpg', 'gif/29.jpg', 'gif/30.jpg',
                  'gif/31.jpg', 'gif/32.jpg', 'gif/33.jpg', 'gif/34.jpg', 'gif/35.jpg', 'gif/36.jpg', 'gif/37.jpg', 'gif/38.jpg', 'gif/39.jpg', 'gif/40.jpg',
                  'gif/41.jpg', 'gif/42.jpg', 'gif/43.jpg', 'gif/44.jpg', 'gif/45.jpg', 'gif/46.jpg', 'gif/47.jpg', 'gif/48.jpg', 'gif/49.jpg', 'gif/50.jpg']
    gif_name = 'gif/side.gif'
    duration = 0.35
    create_gif(image_list, gif_name, duration)


if __name__ == '__main__':
    main()
