import os
import torch
import torch.nn.functional as F
import numpy as np
import tifffile
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from plyfile import PlyData, PlyElement
import open3d as o3d
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
    # 1. 读取基础数据
    xyz_map = tifffile.imread(tiff_path)
    rgb_map = np.array(Image.open(rgb_path)) / 255.0
    H, W = rgb_map.shape[:2]

    pts_flat = xyz_map.reshape(-1, 3)
    colors_flat = rgb_map.reshape(-1, 3)
    valid_mask_1d = pts_flat[:, 2] > 0.001
    
    # 2. 复用你的完美 RANSAC 清洗
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_flat[valid_mask_1d])
    pcd.colors = o3d.utility.Vector3dVector(colors_flat[valid_mask_1d])
    
    if len(pcd.points) == 0: return

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.003, ransac_n=3, num_iterations=1000)
    dists = pcd.compute_point_cloud_distance(pcd.select_by_index(inliers))
    dists = np.asarray(dists)
    
    bagel_idx_in_valid = np.where(dists > 0.001)[0]
    bagel_pcd = pcd.select_by_index(bagel_idx_in_valid)
    bagel_pcd, sor_indices = bagel_pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.5)
    
    gt_pts = np.asarray(bagel_pcd.points)
    gt_colors = np.asarray(bagel_pcd.colors)
    
    if len(gt_pts) == 0: return

    # 3. 获取只属于贝果本体的 Mask
    foreground_mask_flat = np.zeros(H * W, dtype=bool)
    valid_indices = np.where(valid_mask_1d)[0]
    bagel_absolute_indices = valid_indices[bagel_idx_in_valid][sor_indices]
    foreground_mask_flat[bagel_absolute_indices] = True
    foreground_mask_2d = torch.tensor(foreground_mask_flat.reshape(H, W), device=DEVICE)
    
    gt_rgb_tensor = torch.tensor(rgb_map, dtype=torch.float32, device=DEVICE)
    gt_depth_tensor = torch.tensor(xyz_map[..., 2], dtype=torch.float32, device=DEVICE)

    # 4. 相机矩阵计算
    X = gt_pts[:, 0]
    Z = gt_pts[:, 2]
    cx, cy = W / 2.0, H / 2.0
    u_idx = bagel_absolute_indices % W
    valid_x = np.abs(X) > 0.01 
    focal_x = float(np.median(np.abs((u_idx[valid_x] - cx) * Z[valid_x] / X[valid_x]))) if np.sum(valid_x) > 0 else 5000.0

    K = torch.tensor([[focal_x, 0, cx], [0, focal_x, cy], [0, 0, 1]], dtype=torch.float32, device=DEVICE)
    viewmat = torch.eye(4, dtype=torch.float32, device=DEVICE)

    num_points = len(gt_pts)
    init_scale = float(np.log(max((np.max(X) - np.min(X)) / W, 1e-5)))
    colors_logit = np.log(np.clip(gt_colors, 1e-4, 1 - 1e-4) / (1 - np.clip(gt_colors, 1e-4, 1 - 1e-4)))

    # ==========================================
    # 【核心：彻底冻结 XYZ，去除一切多余惩罚】
    # ==========================================
    means = torch.tensor(gt_pts, dtype=torch.float32, device=DEVICE).requires_grad_(False) # 绝对锁死！
    
    colors_p = torch.tensor(colors_logit, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    scales = torch.full((num_points, 3), init_scale, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    quats = torch.zeros((num_points, 4), dtype=torch.float32, device=DEVICE)
    quats[:, 0] = 1.0
    quats.requires_grad_(True)
    opacities = torch.full((num_points, 1), 2.0, dtype=torch.float32, device=DEVICE).requires_grad_(True)

    optimizer = torch.optim.Adam([
        {'params': [colors_p], 'lr': 0.01, "name": "color"},
        {'params': [scales], 'lr': 0.005, "name": "scale"},
        {'params': [quats], 'lr': 0.005, "name": "quat"},
        {'params': [opacities], 'lr': 0.01, "name": "opacity"}
    ])

    for step in range(iterations):
        optimizer.zero_grad()

        act_scales = torch.exp(scales)
        act_quats = F.normalize(quats, p=2, dim=-1)
        act_opacities = torch.sigmoid(opacities)
        act_colors = torch.sigmoid(colors_p)

        renders, _, _ = rasterization(
            means=means, quats=act_quats, scales=act_scales, opacities=act_opacities.squeeze(-1),
            colors=act_colors, viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0), width=W, height=H, render_mode="RGB+ED"
        )
        
        render_rgb = renders[0, ..., :3]
        render_depth = renders[0, ..., 3]

        # 最纯粹的损失：只算贝果上的点，微微压制一下 Scale，其余全靠优化器自由发挥
        loss_rgb = F.l1_loss(render_rgb[foreground_mask_2d], gt_rgb_tensor[foreground_mask_2d])
        loss_depth = F.l1_loss(render_depth[foreground_mask_2d], gt_depth_tensor[foreground_mask_2d])
        loss_scale = torch.mean(act_scales)

        total_loss = loss_rgb + 0.5 * loss_depth + 0.1 * loss_scale
        total_loss.backward()
        optimizer.step()

    save_gaussian_ply(output_ply_path, means, torch.sigmoid(colors_p), scales, F.normalize(quats, p=2, dim=-1), opacities)

if __name__ == "__main__":
    # 1. 设定根目录 (保持你原本 3d.py 里的路径配置)
    DATASET_ROOT = r"F:/download/mvtec_3d_anomaly_detection"
    CATEGORY = "bagel"
    # 【建议】换一个新的输出文件夹名字，比如加上 _optimized，以免和你之前生成的普通点云混淆
    OUTPUT_ROOT = r"F:/download/LGM-main/LGM-main/gaussian_data_optimized" 

    # MVTec 3D-AD 包含 train, validation 和 test 文件夹
    category_dir = Path(DATASET_ROOT) / CATEGORY
    
    print(f"🚀 开始批量 3DGS 拟合处理类别: {CATEGORY}")
    
    # 2. 遍历 category 下的所有子文件夹 (train, validation, test)
    for split_dir in category_dir.iterdir():
        if not split_dir.is_dir():
            continue
            
        # 3. 遍历 split 下的具体分类 (good, crack, hole, combined 等)
        for type_dir in split_dir.iterdir():
            if not type_dir.is_dir():
                continue
                
            xyz_dir = type_dir / "xyz"
            rgb_dir = type_dir / "rgb"
            
            # 确保这不是一个空文件夹
            if not xyz_dir.exists() or not rgb_dir.exists():
                continue
                
            # 4. 在输出目录中镜像创建同样的文件夹结构
            out_dir = Path(OUTPUT_ROOT) / CATEGORY / split_dir.name / type_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # 5. 扫描所有的tiff文件并处理
            tiff_files = sorted(list(xyz_dir.glob("*.tiff")))
            
            # 使用 tqdm进度条
            with tqdm(total=len(tiff_files), desc=f"{split_dir.name}/{type_dir.name}", unit="file") as pbar:
                for tiff_path in tiff_files:
                    # 寻找同名的rgb图片
                    rgb_path = rgb_dir / f"{tiff_path.stem}.png"
                    
                    # 兼容部分扩展名大写的情况
                    if not rgb_path.exists():
                        rgb_path = rgb_dir / f"{tiff_path.stem}.PNG"
                        
                    if rgb_path.exists():
                        output_ply_path = out_dir / f"{tiff_path.stem}.ply"
                        
                        # 增加断点保护：如果这个文件已经生成过了，就跳过
                        if not output_ply_path.exists():
                            # ==========================================
                            # 【这里的核心发生了改变】
                            # 不再是简单的 create_faithful_ply，而是调用我们刚写的 3DGS 联合优化！
                            # ==========================================
                            optimize_single_bagel(str(tiff_path), str(rgb_path), str(output_ply_path), iterations=300)
                        else:
                            pbar.set_postfix({'skip': output_ply_path.name})
                    else:
                        pbar.set_postfix({'error': f"RGB not found for {tiff_path.stem}"})
                    
                    pbar.update(1)

    print("\n✅ 批量 3DGS 拟合处理完成!")