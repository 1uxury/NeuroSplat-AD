import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer

# ==========================================
# 1. BNTT: 时序批量归一化模块 (防Bug重构版)
# ==========================================
class BNTT2d(nn.Module):
    def __init__(self, num_features, time_steps):
        super().__init__()
        self.time_steps = time_steps
        bn_list = tuple(nn.BatchNorm2d(num_features) for _ in range(time_steps))
        self.bns = nn.ModuleList(bn_list)

    def forward(self, x):
        out = list() 
        for t in range(self.time_steps):
            bn_t = self.bns.__getitem__(t)
            x_t = x.select(0, t)
            out.append(bn_t(x_t))
        return torch.stack(out, dim=0)

class BNTT1d(nn.Module):
    def __init__(self, num_features, time_steps):
        super().__init__()
        self.time_steps = time_steps
        bn_list = tuple(nn.BatchNorm1d(num_features) for _ in range(time_steps))
        self.bns = nn.ModuleList(bn_list)

    def forward(self, x):
        out = list()
        for t in range(self.time_steps):
            bn_t = self.bns.__getitem__(t)
            x_t = x.select(0, t)
            out.append(bn_t(x_t))
        return torch.stack(out, dim=0)

# ==========================================
# 2. Spiking T-Net: 微型脉冲空间对齐网络
# ==========================================
class SpikingTNet(nn.Module):
    def __init__(self, time_steps=4):
        super().__init__()
        self.time_steps = time_steps
        
        self.conv1 = layer.SeqToANNContainer(nn.Conv1d(3, 64, 1))
        self.lif1 = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m')
        self.conv2 = layer.SeqToANNContainer(nn.Conv1d(64, 128, 1))
        self.lif2 = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m')
        self.conv3 = layer.SeqToANNContainer(nn.Conv1d(128, 1024, 1))
        self.lif3 = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m')

        self.fc1 = layer.SeqToANNContainer(nn.Linear(1024, 512))
        self.lif4 = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m')
        self.fc2 = layer.SeqToANNContainer(nn.Linear(512, 256))
        self.lif5 = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m')
        
        self.fc3 = layer.SeqToANNContainer(nn.Linear(256, 9))

    def forward(self, pos):
        x_seq = pos.unsqueeze(0).repeat(self.time_steps, 1, 1, 1) 
        
        x = self.lif1(self.conv1(x_seq))
        x = self.lif2(self.conv2(x))
        x = self.lif3(self.conv3(x))
        
        x, _ = torch.max(x, dim=3) 
        
        x = self.lif4(self.fc1(x))
        x = self.lif5(self.fc2(x))
        x = self.fc3(x) 
        
        x = x.mean(0).view(-1, 3, 3)
        iden = torch.eye(3, device=x.device).view(1, 3, 3).repeat(x.size(0), 1, 1)
        return x + iden


# ==========================================
# 工具函数：相对坐标 KNN 图构建 (防Bug版)
# ==========================================
def knn_graph(pos, k=16):
    inner = -2 * torch.matmul(pos.transpose(2, 1), pos)
    xx = torch.sum(pos ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    idx = pairwise_distance.topk(k=k, dim=-1).__getitem__(1)   
    return idx

def get_graph_feature(x, k=16):
    batch_size = x.size(0)
    num_points = x.size(2)
    x_view = x.view(batch_size, -1, num_points)

    pos = x_view.narrow(dim=1, start=0, length=3)
    idx = knn_graph(pos, k=k) 

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx_flat = idx.view(-1)

    x_transposed = x_view.transpose(2, 1).contiguous()
    x_flat = x_transposed.view(batch_size * num_points, -1)
    
    feature_flat = x_flat.index_select(dim=0, index=idx_flat)
    feature = feature_flat.view(batch_size, num_points, k, -1) 
    
    x_center = x_transposed.view(batch_size, num_points, 1, -1).repeat(1, 1, k, 1) 

    feature = torch.cat((x_center, feature - x_center), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


# ==========================================
# 3. 融合了 T-Net 和 BNTT 的主编码器
# ==========================================
class GaussianSpikingEncoder(nn.Module):
    def __init__(self, in_channels=14, latent_dim=1024, time_steps=4, k=16):
        super().__init__()
        self.time_steps = time_steps
        self.k = k

        self.t_net = SpikingTNet(time_steps=time_steps)

        self.conv1 = layer.SeqToANNContainer(nn.Conv2d(in_channels * 2, 64, kernel_size=1, bias=False))
        self.bn1 = BNTT2d(64, time_steps)
        self.lif1 = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m')

        self.conv2 = layer.SeqToANNContainer(nn.Conv2d(64, 128, kernel_size=1, bias=False))
        self.bn2 = BNTT2d(128, time_steps)
        self.lif2 = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m')

        self.conv3 = layer.SeqToANNContainer(nn.Conv1d(128, 256, kernel_size=1, bias=False))
        self.bn3 = BNTT1d(256, time_steps)
        self.lif3 = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m')

        self.conv4 = layer.SeqToANNContainer(nn.Conv1d(256, latent_dim, kernel_size=1, bias=False))
        self.bn4 = BNTT1d(latent_dim, time_steps)
        self.lif4 = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m')

    def forward(self, x):
        pos = x.narrow(dim=1, start=0, length=3)
        trans_matrix = self.t_net(pos) 
        
        pos_aligned = torch.bmm(pos.transpose(2, 1), trans_matrix).transpose(2, 1)
        
        other_feats = x.narrow(dim=1, start=3, length=11)
        x_aligned = torch.cat((pos_aligned, other_feats), dim=1) 
        
        x_graph = get_graph_feature(x_aligned, k=self.k) 

        x_seq = x_graph.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)

        out = self.lif1(self.bn1(self.conv1(x_seq)))
        out = self.lif2(self.bn2(self.conv2(out)))

        out, _ = torch.max(out, dim=4) 

        out = self.lif3(self.bn3(self.conv3(out)))
        out = self.lif4(self.bn4(self.conv4(out)))

        global_feat, _ = torch.max(out, dim=3) 

        return global_feat.mean(0) 

# ==========================================
# 4. 解码器与主网络 (AutoEncoder)
# ==========================================
class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim=1024, num_points=4096, out_channels=14):
        super().__init__()
        self.num_points = num_points
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_points * out_channels)
        )
        
    def forward(self, z):
        reconstruction = self.fc(z)
        return reconstruction.view(-1, 14, self.num_points)

class GaussianSNNAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GaussianSpikingEncoder()
        # 初始化解码器，隐变量维度必须和编码器保持一致(1024)
        self.decoder = SimpleDecoder(latent_dim=1024)
        
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon