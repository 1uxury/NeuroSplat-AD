import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path
from spikingjelly.activation_based import functional

# 引入你的模型和数据集加载器
from model import GaussianSNNAutoEncoder
from dataset import MVTecGaussianDataset

# ================= 配置区域 =================
CATEGORY = "bagel"
MODEL_PATH = "snn_model_bagel_200.pth"  # 确保填写最新训练好的权重
NUM_POINTS = 4096
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===========================================

def apply_3d_gaussian_smoothing(points, errors, sigma=0.05):
    """
    对点云上的标量误差场进行 3D 高斯平滑，消除高频椒盐噪声。
    """
    pts_t = torch.tensor(points, dtype=torch.float32, device=DEVICE)
    err_t = torch.tensor(errors, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    
    dist_mat = torch.cdist(pts_t, pts_t)
    weights = torch.exp(-(dist_mat ** 2) / (2 * sigma ** 2))
    weights = weights / torch.sum(weights, dim=1, keepdim=True)
    
    smoothed_errors = torch.mm(weights, err_t).squeeze().cpu().numpy()
    return smoothed_errors

def get_jet_colors(smoothed_error):
    """
    将平滑后的误差映射为 Jet 伪彩色 (蓝->正常, 红->异常)
    """
    vmin = np.percentile(smoothed_error, 2)
    vmax = np.percentile(smoothed_error, 98)
    error_norm = np.clip((smoothed_error - vmin) / (vmax - vmin + 1e-6), 0, 1)
    
    cmap_rgba = plt.get_cmap("jet")(error_norm)
    # 安全删除 Alpha 通道，保留 RGB
    colors_rgb = np.delete(cmap_rgba, 3, axis=1) 
    return colors_rgb

def main():
    # 1. 加载模型
    model = GaussianSNNAutoEncoder().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ 模型已加载 (保持训练时的时序配置)")

    # 2. 读取带有裂纹的测试样本
    test_root = f"F:/download/LGM-main/LGM-main/gaussian_data/{CATEGORY}/test"
    defect_type = "hole"  
    
    try:
        dataset = MVTecGaussianDataset(root_dir=test_root, category=defect_type, num_points=NUM_POINTS)
    except Exception as e:
        print(f"❌ 找不到测试数据，请检查路径。")
        return
        
    print(f"正在可视化 {defect_type} 类别...")
    data = dataset.__getitem__(0).unsqueeze(0).to(DEVICE) 
    
    with torch.no_grad():
        functional.reset_net(model)
        recon = model(data)
        
        # 1. 维度转换 (B, 14, N) -> (B, N, 14)
        x_t = data.transpose(1, 2)    
        y_t = recon.transpose(1, 2)   
        
        # 2. 仅依靠前 3 维(XYZ)计算物理空间的距离矩阵
        x_xyz = x_t[:, :, :3]
        y_xyz = y_t[:, :, :3]
        dist_mat = torch.cdist(y_xyz, x_xyz, p=2) 
        
        # 3. 找到重构体 y 上每个点对应的输入体 x 的最近物理点
        _, min_idx_y2x = torch.min(dist_mat, dim=2)
        
        # 4. 根据索引拉取输入点云的 14 维真实特征
        num_dims = x_t.shape[2]
        idx_y2x_expanded = min_idx_y2x.unsqueeze(-1).expand(-1, -1, num_dims)
        x_matched = torch.gather(x_t, 1, idx_y2x_expanded)
        
        # 5. 热力图依据：全面评估 3DGS 参数的畸变程度！
        # 权重分配：XYZ(1.0), RGB(0.5), Scale(2.0), Rotation(1.0), Opacity(1.0)
        weights = torch.tensor([1.0]*3 + [0.5]*3 + [2.0]*3 + [1.0]*4 + [1.0]*1, device=DEVICE)
        
        # 计算 14 维加权绝对误差
        diff = torch.abs(y_t - x_matched) * weights
        error_y = torch.mean(diff, dim=-1).squeeze().cpu().numpy()
        
        # 获取用于最终绘图的 3D 坐标
        raw_points_recon = y_xyz.squeeze().cpu().numpy()

    # 3. 对误差进行 3D 高斯平滑
    raw_points_recon = y.squeeze().cpu().numpy()
    smoothed_err_y = apply_3d_gaussian_smoothing(raw_points_recon, error_y, sigma=0.05)
    
    # 4. 获取 Jet 热力图颜色
    colors_y = get_jet_colors(smoothed_err_y)
    
    # 5. 构建重构点云
    pcd_recon = o3d.geometry.PointCloud()
    pcd_recon.points = o3d.utility.Vector3dVector(raw_points_recon)
    pcd_recon.colors = o3d.utility.Vector3dVector(colors_y)
    
    # 视角修复：绕 X 轴翻转 180 度，确保贝果表面正对相机
    R = pcd_recon.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    pcd_recon.rotate(R, center=(0, 0, 0))
    
    # 不再需要 translate 平移，直接居中显示
    
    print("🚀 正在打开单体可视化窗口...")
    print("👉 【高亮红色】代表网络重构时发现的精准物理裂纹！")
    
    o3d.visualization.draw_geometries([pcd_recon], window_name="网络重构缺陷热力图 (精准定位裂纹)")
if __name__ == "__main__":
    main()
