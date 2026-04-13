import os
import torch
import torch.nn.functional as F
import numpy as np
import tifffile
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from plyfile import PlyData, PlyElement

# 引入 gsplat (pip install gsplat)
from gsplat import rasterization

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_gaussian_ply(path, means, colors, scales, quats, opacities):
    """将优化后的高斯参数保存为标准的 3DGS PLY 格式"""
    pts = means.detach().cpu().numpy()
    # 颜色转换回球谐函数的基础项 (f_dc)
    f_dc = (colors.detach().cpu().numpy() - 0.5) / 0.28209479
    s = scales.detach().cpu().numpy()
    q = quats.detach().cpu().numpy()
    o = opacities.detach().cpu().numpy()

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
             ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
             ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
             ('opacity', 'f4')]
    
    elements = np.empty(pts.shape[0], dtype=dtype)
    elements['x'], elements['y'], elements['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    elements['f_dc_0'], elements['f_dc_1'], elements['f_dc_2'] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    elements['scale_0'], elements['scale_1'], elements['scale_2'] = s[:, 0], s[:, 1], s[:, 2]
    elements['rot_0'], elements['rot_1'], elements['rot_2'], elements['rot_3'] = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    elements['opacity'] = o.squeeze()

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def optimize_single_bagel(tiff_path, rgb_path, output_ply_path, iterations=300):
    # 1. 读取基础数据 (Ground Truth)
    xyz_map = tifffile.imread(tiff_path)
    rgb_map = np.array(Image.open(rgb_path)) / 255.0
    H, W = rgb_map.shape[:2]

    # 展平并过滤无效背景点 (MVTec 中无效深度通常是 0)
    pts = xyz_map.reshape(-1, 3)
    colors = rgb_map.reshape(-1, 3)
    valid_mask = pts[:, 2] > 0.001 
    
    gt_pts = pts[valid_mask]
    gt_colors = colors[valid_mask]
    
    if len(gt_pts) == 0:
        return

    # 将 Ground Truth 转为 Tensor，准备计算 Loss
    gt_rgb_tensor = torch.tensor(rgb_map, dtype=torch.float32, device=DEVICE)
    # 提取深度通道作为 GT
    gt_depth_tensor = torch.tensor(xyz_map[..., 2], dtype=torch.float32, device=DEVICE)

    # 2. 初始化高斯参数
    num_points = len(gt_pts)
    means = torch.tensor(gt_pts, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    colors_p = torch.tensor(gt_colors, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    # 初始化为一个较小的各向同性体积 (基于 MVTec 的物理尺度，比如 1mm)
    scales = torch.full((num_points, 3), -5.0, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    # 初始化为单位四元数 [w, x, y, z]
    quats = torch.zeros((num_points, 4), dtype=torch.float32, device=DEVICE)
    quats[:, 0] = 1.0
    quats.requires_grad_(True)
    # 初始化透明度 (经过 sigmoid 激活前的值)
    opacities = torch.zeros((num_points, 1), dtype=torch.float32, device=DEVICE).requires_grad_(True)

    # 3. 设置虚拟相机 (正交近似：将焦距设得极大，模拟工业相机的垂直俯拍)
    focal = 10000.0 
    K = torch.tensor([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]], dtype=torch.float32, device=DEVICE)
    viewmat = torch.eye(4, dtype=torch.float32, device=DEVICE)

    # 4. 优化器设置
    optimizer = torch.optim.Adam([
        {'params': [means], 'lr': 0.0001, "name": "xyz"},      # 位置微调
        {'params': [colors_p], 'lr': 0.005, "name": "color"},  # 颜色
        {'params': [scales], 'lr': 0.005, "name": "scale"},    # 核心：形状形变
        {'params': [quats], 'lr': 0.001, "name": "quat"},      # 核心：旋转姿态
        {'params': [opacities], 'lr': 0.05, "name": "opacity"} # 透明度
    ])

    # 5. 训练循环
    for step in range(iterations):
        optimizer.zero_grad()

        # 激活参数
        act_scales = torch.exp(scales)
        act_quats = F.normalize(quats, p=2, dim=-1)
        act_opacities = torch.sigmoid(opacities)
        act_colors = torch.sigmoid(colors_p)

        # 渲染 RGB 和 Depth
        renders, _, _ = rasterization(
            means=means,
            quats=act_quats,
            scales=act_scales,
            opacities=act_opacities.squeeze(-1),
            colors=act_colors,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=W,
            height=H,
            render_mode="RGB+ED" # 渲染 RGB 和 Expected Depth
        )
        
        render_rgb = renders[0, ..., :3]
        render_depth = renders[0, ..., 3]

        # 计算 Loss：强迫高斯变形以填补裂纹和表面凹凸
        loss_rgb = F.l1_loss(render_rgb, gt_rgb_tensor)
        
        # 只在有物体的区域计算深度 Loss
        mask = gt_depth_tensor > 0.001
        loss_depth = F.l1_loss(render_depth[mask], gt_depth_tensor[mask])

        # 联合 Loss，深度占主导地位，因为异常检测最看重几何形变
        total_loss = loss_rgb + 2.0 * loss_depth

        total_loss.backward()
        optimizer.step()

    # 6. 保存带有真实形变参数的 3DGS 点云
    save_gaussian_ply(output_ply_path, means, torch.sigmoid(colors_p), scales, F.normalize(quats, p=2, dim=-1), opacities)


if __name__ == "__main__":
    # 使用你之前 3d.py 中的遍历逻辑，把对 create_faithful_ply 的调用
    # 替换为对 optimize_single_bagel(tiff_path, rgb_path, output_ply_path) 的调用
    # 下面是一个测试单例：
    # optimize_single_bagel("test_xyz.tiff", "test_rgb.png", "output_gaussian.ply")
    print("准备就绪，可以开始批量 3DGS 拟合...")
