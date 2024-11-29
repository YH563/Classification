import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image


# 生成标签
def generate_label(file_name: str):
    if file_name[0] == "b":
        return 0
    if file_name[0] == "c":
        return 1
    if file_name[0] == "i":
        return 2
    if file_name[0] == "n":
        return 3


# 创建数据集
class image_dataset(Dataset):
    def __init__(self, file_folder: str):
        super(image_dataset, self).__init__()
        transform = transforms.Compose([
            transforms.ToTensor(),  # 将PIL Image转换为torch.Tensor
        ])
        image_list = []
        label_list = []
        for file in os.listdir(file_folder):
            file_path = os.path.join(file_folder, file)
            image = Image.open(file_path)
            image_list.append(transform(image))
            label_list.append(generate_label(file))
        self.image = torch.stack(image_list)
        self.label = torch.tensor(label_list)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, item):
        return self.image[item], self.label[item]
