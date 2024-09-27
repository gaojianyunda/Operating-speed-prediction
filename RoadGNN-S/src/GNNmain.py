import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

# 定义模型
class GNNModel(nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = pyg_nn.GINConv(nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64)))
        self.conv2 = pyg_nn.GINConv(nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 128)))
        self.conv3 = pyg_nn.GATConv(128, 128, heads=4)  # 输出维度应与输入维度匹配
        self.fc = nn.Linear(128 * 4, 512)  # 4个头，每个头128维
        self.output = nn.Linear(512, 4)  # 最终输出层

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x, attn_weights = self.conv3(x, edge_index, return_attention_weights=True)

        x = pyg_nn.global_mean_pool(x, data.batch)
        x = self.fc(x)  
        x = torch.relu(x)  # 可选择添加激活函数
        x = self.output(x)  # 输出层
        return x.squeeze(1), attn_weights