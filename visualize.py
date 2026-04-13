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
        
        # 安全提取前3维 (XYZ 坐标)
        x = data.narrow(dim=1, start=0, length=3).transpose(1, 2)
        y = recon.narrow(dim=1, start=0, length=3).transpose(1, 2)
        
        # 计算距离矩阵
        dist_mat = torch.cdist(x, y)
        
        # 【核心逻辑】：只保留 Recon -> Input 的误差，专门捕捉“缺失的裂纹”
        error_y = torch.min(dist_mat, dim=1).values.squeeze().cpu().numpy()

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