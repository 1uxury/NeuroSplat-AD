import torch
import numpy as np
import open3d as o3d
from pathlib import Path
from sklearn.metrics import roc_auc_score
from spikingjelly.activation_based import functional
from torch.utils.data import DataLoader

# 引入你的模型和数据集加载器
from model import GaussianSNNAutoEncoder
from dataset import MVTecGaussianDataset

# ================= 配置 =================
CATEGORY = "bagel"
MODEL_PATH = "snn_model_bagel_200.pth" # 确保这是你最新训练好的模型
NUM_POINTS = 4096
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =======================================

def feature_aware_chamfer_score(x, y, top_k_ratio=0.10):
    """
    【特征感知异常打分】：提取前 10% 参数畸变最严重的点作为异常分数
    """
    x_t = x.transpose(1, 2)
    y_t = y.transpose(1, 2)
    
    x_xyz = x_t[:, :, :3]
    y_xyz = y_t[:, :, :3]
    dist_mat = torch.cdist(x_xyz, y_xyz, p=2)
    
    _, min_idx_x2y = torch.min(dist_mat, dim=2)
    _, min_idx_y2x = torch.min(dist_mat, dim=1)
    
    num_dims = x_t.shape[2]
    idx_x2y_expanded = min_idx_x2y.unsqueeze(-1).expand(-1, -1, num_dims)
    y_matched = torch.gather(y_t, 1, idx_x2y_expanded)
    
    idx_y2x_expanded = min_idx_y2x.unsqueeze(-1).expand(-1, -1, num_dims)
    x_matched = torch.gather(x_t, 1, idx_y2x_expanded)
    
    # 计算每个点在 14 维上的平均误差
    error_x = torch.mean(torch.abs(x_t - y_matched), dim=-1) # (B, N)
    error_y = torch.mean(torch.abs(y_t - x_matched), dim=-1) # (B, N)
    
    num_points = error_x.shape[1]
    k = max(1, int(num_points * top_k_ratio))
    
    topk_x, _ = torch.topk(error_x, k, dim=1)
    topk_y, _ = torch.topk(error_y, k, dim=1)
    
    return max(torch.mean(topk_x).item(), torch.mean(topk_y).item())

if __name__ == "__main__":
    # 1. 加载最新训练的模型
    model = GaussianSNNAutoEncoder().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    # 开启多时间步推理 (提升 SNN 稳定性)
  
    model.eval()
    print(f"✅ 模型已加载，开启 T=8 多时间步推理")

    y_true = list()
    y_scores = list()
    
    # 2. 遍历测试集各个缺陷文件夹
    test_root = Path(f"F:/download/LGM-main/LGM-main/gaussian_data/{CATEGORY}/test")
    
    with torch.no_grad():
        for defect_type in test_root.iterdir():
            if not defect_type.is_dir(): continue
            
            label = 0 if defect_type.name == "good" else 1
            print(f"  - 正在测试 {defect_type.name}...")
            
            # 【核心对接】：复用 MVTecGaussianDataset，确保测试集也被 PCA对齐 和 归一化
            # 这里的 root_dir 巧妙地指向了 test 目录的上一层，category 指向具体的缺陷类型
            test_dataset = MVTecGaussianDataset(
                root_dir=str(test_root), 
                category=defect_type.name, 
                num_points=NUM_POINTS
            )
            
            # 使用 DataLoader 逐个读取
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            
            for data in test_loader:
                data = data.to(DEVICE)
                
                # SNN 状态重置
                functional.reset_net(model)
                recon = model(data)
                
                score = feature_aware_chamfer_score(data, recon)
                
                y_true.append(label)
                y_scores.append(score)

    # 3. 计算最终的 AUROC 指标
    if len(y_true) > 0 and sum(y_true) > 0: 
        auroc = roc_auc_score(y_true, y_scores)
        print(f"\n🏆 最终结果 (I-AUROC): {auroc:.4f}")
    else:
        print("\n⚠️ 数据不足，无法计算 AUROC。")
