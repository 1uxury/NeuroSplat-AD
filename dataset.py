import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import open3d as o3d

class MVTecGaussianDataset(Dataset):
    def __init__(self, root_dir, category, split='train', num_points=4096):
        self.num_points = num_points
        self.files = sorted(list(Path(root_dir).joinpath(category).rglob("*.ply")))
        if len(self.files) == 0:
            raise ValueError(f"未找到.ply 文件")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ply_path = str(self.files[idx])
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        num_pts = len(points)
        
        if num_pts == 0:
            return torch.zeros((14, self.num_points), dtype=torch.float32)

        # ==========================================
        # 核心修正：仅做去中心化和缩放，绝对不搞 PCA 旋转！
        # ==========================================
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if max_dist > 0:
            points = points / max_dist

        # 扩充为 14 通道
        f_dc = (colors.astype(np.float32) - 0.5) / 0.28209479
        scales = np.full((num_pts, 3), -3.0, dtype=np.float32)
        rots = np.zeros((num_pts, 4), dtype=np.float32)
        rots[:, 0] = 1.0 # 默认无旋转
        opacities = np.full((num_pts, 1), 4.6, dtype=np.float32)
        
        features = np.concatenate([points, f_dc, scales, rots, opacities], axis=1).astype(np.float32)
        
        if num_pts >= self.num_points:
            choice = np.random.choice(num_pts, self.num_points, replace=False)
        else:
            choice = np.random.choice(num_pts, self.num_points, replace=True)
            
        features = features[choice, :]
        return torch.from_numpy(features).transpose(0, 1)