import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from spikingjelly.activation_based import functional

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

def chamfer_distance_loss(x, y):
    """
    【彻底修复】：回归最纯粹、最稳定的标准倒角距离 (Standard Chamfer Distance)！
    因为我们的数据已经通过 RANSAC 和 SOR 彻底清洗，不再有极端噪点，
    使用标准 CD 不仅不会变形，还能提供最完美的全局梯度，彻底解决指数函数导致的梯度消失问题！
    """
    x_xyz = x[:, :3, :].transpose(1, 2) 
    y_xyz = y[:, :3, :].transpose(1, 2) 
    
    dist_mat = torch.cdist(x_xyz, y_xyz, p=2)
    min_dist_x, _ = torch.min(dist_mat, dim=2)
    min_dist_y, _ = torch.min(dist_mat, dim=1)
    
    # 直接使用距离求均值，提供稳定、持续的梯度！
    return torch.mean(min_dist_x) + torch.mean(min_dist_y)

def train():
    gaussian_data_path = "F:/download/LGM-main/LGM-main/gaussian_data" 
    
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
    
    current_start_epoch = 30
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
                loss = chamfer_distance_loss(data, recon)
            
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
    gaussian_data_path = "F:/download/LGM-main/LGM-main/gaussian_data"
    if not os.path.exists(gaussian_data_path):
        print(f"❌ 错误：找不到gaussian_data 文件夹：{gaussian_data_path}")
    else:
        train()