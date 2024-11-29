from torch.utils.data import DataLoader
import seaborn as sns
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import model
import dataset


# 修改标签格式
def change_label(label_test):
    if isinstance(label_test, torch.Tensor):
        _, max_indices = torch.max(label_test, dim=1)
        return max_indices
    else:
        raise TypeError("Input should be a torch.Tensor")


# 绘制热力图
def draw_heatmap(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    classes = ['Barrett', 'Cancer', 'Inflamm', 'Normal']

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    # 添加标题和标签
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix Heatmap')
    # 显示图表中的所有标签，防止它们被截断
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    # 显示图表
    plt.show()


def test(c_model):
    # 初始参数
    file_folder = "test"
    batch_size = 20

    test_dataset = dataset.image_dataset(file_folder)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True, drop_last=False)

    # 进行输出
    real_list = []
    pred_list = []
    c_model.eval()
    with torch.no_grad():
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")  # 将数据移到GPU上进行加速训练
        print("使用的设备是：", device)
        c_model.to(device)
        for image, label in test_dataloader:
            image = image.to(device)
            label = label.to(device)
            output = c_model(image)

            real_list.append(label)
            pred_list.append(change_label(output))

    real_labels = np.array(torch.cat(real_list, dim=0).cpu())
    pred_labels = np.array(torch.cat(pred_list, dim=0).cpu())
    acc = 0
    for real, pred in zip(real_labels, pred_labels):
        if real == pred:
            acc += 1
    acc /= len(real_labels)
    print(f"正确率为{acc:4f}")
    draw_heatmap(real_labels, pred_labels)


if __name__ == "__main__":
    classification_model = model.ClassificationModel()
    model_state = torch.load("model_state//classification_40.pkl")
    classification_model.load_state_dict(model_state)
    test(classification_model)
