import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image


# 对图像进行线性归一化处理
def min_max_normalize(tensor, eps=1e-6):
    image_normal_list = []
    for channel in tensor:
        print(channel.shape)
        max_0 = torch.max(channel)
        print(max_0)
        min_0 = torch.min(channel)
        print(min_0)
        channel_0 = (channel - min_0.item() * torch.ones_like(channel)) * (1 / (max_0.item() - min_0.item() + eps))
        image_normal_list.append(channel_0)
    return torch.stack(image_normal_list)


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将PIL Image转换为torch.Tensor
    ])
    image_path = "train\\barrett_3.bmp"
    image = Image.open(image_path)
    image_tensor = transform(image)
    image_0 = Image.open("test\\barrett_3.bmp")
    image_tensor_0 = transform(image_0)
    print(image_tensor)
    print(image_tensor_0)
    print(image_tensor_0 == image_tensor)

