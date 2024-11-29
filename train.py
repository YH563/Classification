from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import model
import dataset
import time


if __name__ == "__main__":
    # 初始参数
    file_folder = "train"
    batch_size = 20
    num_epochs = 40
    lr = 0.0001
    betas = (0.9, 0.999)

    # 创建模型，加载数据，创建优化器和损失函数
    c_model = model.ClassificationModel()
    train_dataset = dataset.image_dataset(file_folder)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(c_model.parameters(), lr=lr, betas=betas)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")  # 将数据移到GPU上进行加速训练
    print("使用的设备是：", device)

    # 记录训练时间
    prev_time = time.time()

    # 记录损失函数值，用于绘图
    loss_list = []
    accuracy_list = []  # 新增准确率列表
    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0  # 记录正确预测的数量
        total_samples = 0  # 记录样本总数
        for image, label in train_dataloader:
            c_model.train()
            c_model.to(device)
            image = image.to(device)
            label = label.to(device)
            output = c_model(image)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 计算正确预测
            _, predicted = torch.max(output, 1)  # 获取预测类别
            correct_predictions += (predicted == label).sum().item()  # 统计正确预测的数量
            total_samples += label.size(0)  # 统计样本总数

        total_loss = total_loss / len(train_dataloader)
        accuracy = correct_predictions / total_samples  # 计算准确率
        loss_list.append(total_loss)
        accuracy_list.append(accuracy)  # 记录准确率
        print(f"Epoch:{epoch + 1}, Loss:{total_loss}, Accuracy:{accuracy:.4f}")
        if (epoch + 1) % 20 == 0:
            # 保存模型参数
            path_model = f"model_state//classification_{epoch + 1}.pkl"
            model_state_dict = c_model.state_dict()
            torch.save(model_state_dict, path_model)
            print(f"Save Model Successfully on path:{path_model}")

        # if (epoch + 1) == num_epochs:
        #     path_model = "classification.pkl"
        #     model_state_dict = c_model.state_dict()
        #     torch.save(model_state_dict, path_model)
        #     print(f"Save Model Successfully on path:{path_model}")

    # 记录结束时间
    end_time = time.time()
    total_time = end_time-prev_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"训练总时间为{int(hours)}时{int(minutes)}分{int(seconds)}秒")

    # 绘制损失函数图
    loss_array = np.array(loss_list)
    acc_array = np.array(accuracy_list)
    epoch_array = np.array([i for i in range(1, num_epochs + 1)])
    plt.plot(epoch_array, loss_array, 'b', label="Loss")
    plt.plot(epoch_array, acc_array, 'r', label="Accuracy")
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Classification Loss and Accuracy")
    plt.show()
