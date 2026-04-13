import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from spikingjelly.activation_based import functional
import torch.nn.functional as F

# 引入你之前写好的模块
from dataset import MVTecGaussianDataset
from model import GaussianSNNAutoEncoder

# ================= 配置区域 =================
DATASET_ROOT = r"F:/download/mvtec_3d_anomaly_detection"  # 你的数据集路径
CATEGORY = "bagel"                         # 训练类别
BATCH_SIZE = 4                             # 显存不够就改小 (2 或 1)
NUM_POINTS = 4096                          # 点云数量
LR = 1e-3                                  # 学习率
EPOCHS = 200                               # 总训练轮数

START_EPOCH = 0                           
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===========================================

def feature_aware_chamfer_loss(x, y):
    """
    【特征感知倒角距离】：
    1. 仅依赖前 3 维(XYZ)寻找空间中对应的物理点。
    2. 计算这对匹配点在全部 14 维特征上的绝对误差(L1 Loss)。
    """
    x_t = x.transpose(1, 2) # 形状: (B, N, 14)
    y_t = y.transpose(1, 2) # 形状: (B, N, 14)
    
    # 1. 仅用 XYZ 计算物理空间距离矩阵
    x_xyz = x_t[:, :, :3]
    y_xyz = y_t[:, :, :3]
    dist_mat = torch.cdist(x_xyz, y_xyz, p=2)
    
    # 2. 找到空间上的最近邻索引 (匹配点)
    _, min_idx_x2y = torch.min(dist_mat, dim=2) # x找y (B, N)
    _, min_idx_y2x = torch.min(dist_mat, dim=1) # y找x (B, N)
    
    # 3. 根据索引，收集匹配点的 14 维全特征
    num_dims = x_t.shape[2] # 应该是 14
    
    idx_x2y_expanded = min_idx_x2y.unsqueeze(-1).expand(-1, -1, num_dims)
    y_matched = torch.gather(y_t, 1, idx_x2y_expanded)
    
    idx_y2x_expanded = min_idx_y2x.unsqueeze(-1).expand(-1, -1, num_dims)
    x_matched = torch.gather(x_t, 1, idx_y2x_expanded)
    
    # 4. 计算 14 维的综合重构误差 
    # 使用 L1 误差对异常点更鲁棒，梯度也比 L2 平稳
    loss_x2y = F.l1_loss(x_t, y_matched, reduction='mean')
    loss_y2x = F.l1_loss(y_t, x_matched, reduction='mean')
    
    return loss_x2y + loss_y2x

def train():
    gaussian_data_path = "F:/download/LGM-main/LGM-main/gaussian_data_optimized" 
    
    print(f"正在加载数据: {CATEGORY}...")
    try:
        train_dataset = MVTecGaussianDataset(
            root_dir=gaussian_data_path, 
            category=CATEGORY, 
            num_points=NUM_POINTS
        )
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 【核心修复 1】：将 num_workers 改回 0，解决 Windows 系统下 Pickle 数据截断的问题
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        drop_last=True,
        num_workers=0,       
        pin_memory=True      
    )
    print(f"✅ 加载成功，共 {len(train_dataset)} 个样本")

    model = GaussianSNNAutoEncoder().to(DEVICE)
    
    current_start_epoch = 0
    if START_EPOCH > 0:
        load_path = f"snn_model_{CATEGORY}_{START_EPOCH}.pth"
        if os.path.exists(load_path):
            model.load_state_dict(torch.load(load_path, map_location=DEVICE))
            current_start_epoch = START_EPOCH
            print(f"🔄 成功加载断点模型: {load_path}")
        else:
            print(f"❌ 找不到模型文件: {load_path}，将从头开始训练。")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 【核心修复 2】：使用 PyTorch 官方推荐的新版 AMP 写法，消除黄色警告
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"🚀 开始在 {DEVICE} 上极速训练升级版 SNN 自编码器...")
    model.train()
    
    for epoch in range(current_start_epoch, EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        
        for batch_idx, data in enumerate(progress_bar):
            data = data.to(DEVICE, non_blocking=True) 
            
            functional.reset_net(model)
            optimizer.zero_grad()
            
            # 使用新版 autocast 写法
            with torch.amp.autocast('cuda'):
                recon = model(data)
                loss = feature_aware_chamfer_loss(data, recon)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1:03d}/{EPOCHS} 完成 | 平均 DCD Loss: {avg_loss:.6f}")
        
        if (epoch + 1) % 10 == 0:
            save_path = f"snn_model_{CATEGORY}_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"💾 模型已保存: {save_path}")



if __name__ == "__main__":
    gaussian_data_path = "F:/download/LGM-main/LGM-main/gaussian_data_optimized"
    if not os.path.exists(gaussian_data_path):
        print(f"❌ 错误：找不到gaussian_data 文件夹：{gaussian_data_path}")
    else:
        train()