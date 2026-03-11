import torch

# 加载模型的 state_dict
state_dict = torch.load('../checkpoints/endofm_lv.pth')['teacher']

# 打印所有参数的键
for key in state_dict.keys():
    print(key)


# import pickle
#
# # 指定 .pkl 文件的路径
# path = 'train_val_paths_labels.pkl'
#
# # 以二进制读取模式打开文件
# with open(path, 'rb') as file:
#     data = pickle.load(file)
#
# partial_data = data[:1]
#
# # 输出部分数据内容
# print(partial_data)