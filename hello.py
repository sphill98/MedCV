import numpy as np
from PIL import Image
import os

if __name__ == '__main__':
    file_list = os.listdir('/Users/siyeol/2023-2/software_college_project/png_images_seg/coronal/')

    file_list = sorted(file_list)

    for i, file in enumerate(file_list):
        seg_img = Image.open(f'/Users/siyeol/2023-2/software_college_project/png_images_seg/coronal/{file}').convert('L')
        raw_img = Image.open(f'/Users/siyeol/2023-2/software_college_project/png_images_raw/coronal/{file}').convert('L')

        seg_arr = np.array(seg_img)
        raw_arr = np.array(raw_img)

        for j in range(seg_arr.shape[0]):
            for k in range(seg_arr.shape[1]):
                if seg_arr[j, k] > 0:
                    raw_arr[j, k] = 255

        img = Image.fromarray(raw_arr, 'L')
        img.save(f'/Users/siyeol/2023-2/software_college_project/png_images_overlay/coronal/{file}')

        




