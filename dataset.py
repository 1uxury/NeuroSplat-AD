import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from plyfile import PlyData  # 引入 plyfile

class MVTecGaussianDataset(Dataset):
    def __init__(self, root_dir, category, split='train', num_points=4096):
        self.num_points = num_points
        self.files = sorted(list(Path(root_dir).joinpath(category).rglob("*.ply")))
        if len(self.files) == 0:
            raise ValueError(f"未找到.ply 文件，请检查路径: {Path(root_dir).joinpath(category)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ply_path = str(self.files[idx])
        
        # 1. 使用 plyfile 读取完整的 3DGS 参数
        plydata = PlyData.read(ply_path)
        v = plydata['vertex']
        
        x, y, z = np.asarray(v['x']), np.asarray(v['y']), np.asarray(v['z'])
        points = np.vstack((x, y, z)).T
        num_pts = len(points)
        
        if num_pts == 0:
            return torch.zeros((14, self.num_points), dtype=torch.float32)

        # 2. 读取真实的 3DGS 优化参数 (不再造假！)
        f_dc = np.vstack((v['f_dc_0'], v['f_dc_1'], v['f_dc_2'])).T
        scales = np.vstack((v['scale_0'], v['scale_1'], v['scale_2'])).T
        rots = np.vstack((v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3'])).T
        opacities = np.asarray(v['opacity']).reshape(-1, 1)

        # 3. 几何去中心化和缩放 (仅对 XYZ 操作)
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if max_dist > 0:
            points = points / max_dist

        # 4. 拼接真正的 14 通道特征
        features = np.concatenate([points, f_dc, scales, rots, opacities], axis=1).astype(np.float32)
        
        # 5. 随机采样对齐点数
        if num_pts >= self.num_points:
            choice = np.random.choice(num_pts, self.num_points, replace=False)
        else:
            choice = np.random.choice(num_pts, self.num_points, replace=True)
            
        features = features[choice, :]
        return torch.from_numpy(features).transpose(0, 1)