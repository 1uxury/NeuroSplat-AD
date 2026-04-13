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

def chamfer_score_symmetric(x, y, top_k_ratio=0.10):
    """
    计算双向异常分数，取前 10% 的极值误差以兼顾抗噪和敏感度。
    """
    # 同样必须只取前 3 维计算距离！
    x_xyz = x[:, :3, :].transpose(1, 2) 
    y_xyz = y[:, :3, :].transpose(1, 2)
    
    dist_mat = torch.cdist(x_xyz, y_xyz)
    min_dist_x, _ = torch.min(dist_mat, dim=2) 
    min_dist_y, _ = torch.min(dist_mat, dim=1) 
    
    num_points = min_dist_x.shape[1]
    k = max(1, int(num_points * top_k_ratio))
    
    topk_x, _ = torch.topk(min_dist_x, k, dim=1)
    topk_y, _ = torch.topk(min_dist_y, k, dim=1)
    
    # 只要 Input多出异物(x) 或 Recon缺失实体(y)，取其最大值报警
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
                
                score = chamfer_score_symmetric(data, recon)
                
                y_true.append(label)
                y_scores.append(score)

    # 3. 计算最终的 AUROC 指标
    if len(y_true) > 0 and sum(y_true) > 0: 
        auroc = roc_auc_score(y_true, y_scores)
        print(f"\n🏆 最终结果 (I-AUROC): {auroc:.4f}")
    else:
        print("\n⚠️ 数据不足，无法计算 AUROC。")