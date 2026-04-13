import open3d as o3d
import tifffile
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def create_faithful_ply(tiff_path, rgb_path, output_ply_path):
    # 1. 读取基础数据
    xyz = tifffile.imread(tiff_path)
    rgb = np.array(Image.open(rgb_path)) / 255.0
    
    points = xyz.reshape(-1, 3)
    colors = rgb.reshape(-1, 3)
    
    # 2. 基础清理：只排除无效值
    valid_mask = (points[:, 2] > 0)
    points = points[valid_mask]
    colors = colors[valid_mask]

    # 3. 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 4. [核心改进] 找到底板的准确高度并切除
    # RANSAC 找到平面方程
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.003, ransac_n=3, num_iterations=1000)
    
    # 我们不直接用 inliers，而是计算所有点到底板平面的距离
    # 距离大于 0.001 (1mm) 的点我们才认为是贝果，这样可以保留贝果底部的所有细节
    dists = pcd.compute_point_cloud_distance(pcd.select_by_index(inliers))
    dists = np.asarray(dists)
    bagel_pcd = pcd.select_by_index(np.where(dists > 0.001)[0])

    # 5. [核心改进2] 极其温和的去噪
    # 增加邻居数到 100，但把标准差系数放宽到 2.5
    # 这样只会删掉极其离谱的黑点，不会动贝果的边缘
    bagel_pcd, _ = bagel_pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.5)
    
    # 6. 法线估计（必做，否则看起来像油画）
    bagel_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    bagel_pcd.orient_normals_towards_camera_location(np.array([0., 0., 0.]))

    # 7. 导出
    o3d.io.write_point_cloud(output_ply_path, bagel_pcd)


if __name__ == "__main__":
    # 1. 设定根目录
    DATASET_ROOT = r"F:/download/mvtec_3d_anomaly_detection"
    CATEGORY = "bagel"
    OUTPUT_ROOT = r"F:/download/depthsplat-main/gaussian_data"

    # MVTec 3D-AD 包含 train, validation 和 test 文件夹
    category_dir = Path(DATASET_ROOT) / CATEGORY
    
    print(f"开始批量处理类别: {CATEGORY}")
    
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
                        # 这样即使电脑关机或中断，重新运行也会从没做完的地方继续
                        if not output_ply_path.exists():
                            create_faithful_ply(tiff_path, rgb_path, output_ply_path)
                        else:
                            pbar.set_postfix({'skip': output_ply_path.name})
                    else:
                        pbar.set_postfix({'error': f"RGB not found for {tiff_path.stem}"})
                    
                    pbar.update(1)

    print("\n✅ 批量处理完成!")
