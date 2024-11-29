from PIL import Image
import os
import random
import cv2
import numpy as np


def process_file(file_folder, ratio=0.8):
    image_list = []
    image_new_name_list = []
    index = 0
    sub_folder_list = ["barrett\\image", "cancer\\image", "inflamm\\image", "normal\\image"]
    for sub_folder_name in sub_folder_list:
        sub_folder = os.path.join(file_folder, sub_folder_name)
        for file in os.listdir(sub_folder):
            image_name = os.path.join(sub_folder, file)
            try:
                img = Image.open(image_name)
                image_list.append(img)
                image_new_name_list.append(f"{sub_folder_name[:-6]}_{index}.bmp")
                index += 1
            except IOError:
                print(f"Error opening image: {image_name}")

    train_folder = "train"
    test_folder = "test"
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    num_elements_to_select = int(ratio * len(image_list))
    random_indices = random.sample(range(len(image_list)), num_elements_to_select)
    rest_indices = [index for index in range(len(image_list)) if index not in random_indices]
    train_image_dict = {
        "image": [image_list[i] for i in random_indices],
        "image_name": [image_new_name_list[i] for i in random_indices]
    }
    test_image_dict = {
        "image": [image_list[i] for i in rest_indices],
        "image_name": [image_new_name_list[i] for i in rest_indices]
    }

    for image, image_name in zip(train_image_dict["image"], train_image_dict["image_name"]):
        image.save(os.path.join(train_folder, image_name))
    for image, image_name in zip(test_image_dict["image"], test_image_dict["image_name"]):
        image.save(os.path.join(test_folder, image_name))


# 旋转图像
def rotate_image(image):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    angle = random.randint(0, 360)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


# 对图像进行镜像翻转
def flip_image(image):
    flip_choices = [0, 1, -1, None]  # 0: 水平翻转, 1: 垂直翻转, -1: 同时水平和垂直翻转, None: 不翻转
    flip_code = random.choice(flip_choices)
    return cv2.flip(image, flip_code)


# 进行数据增强
def data_augmentation(folder_path:str):
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        image_original = cv2.imread(file_path)
        image_rotated = rotate_image(image_original)
        image_flipped = flip_image(image_original)
        new_image_list = [image_original, image_rotated, image_flipped]
        new_path_list = ["train_cloned//" + file[:-4] + f"_{i}.bmp" for i in range(3)]
        for image, path in zip(new_image_list, new_path_list):
            cv2.imwrite(path, image)


if __name__ == "__main__":
    train_folder_path = "train"
    data_augmentation(train_folder_path)

