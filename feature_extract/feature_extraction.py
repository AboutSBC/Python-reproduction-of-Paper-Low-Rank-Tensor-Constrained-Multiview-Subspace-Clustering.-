import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
import scipy.io as sio

# 路径配置
path = 'coil-20-proc/'
num_view = 3  # 视图数量
nCls = 20     # 类别数
num_sample = 1440  # 总样本数

# 初始化数据结构
X = [None] * num_view  # 多视图数据容器
gt = np.zeros((num_sample, 1))  # 真实标签

# 初始化特征矩阵（注意Python是行优先，MATLAB是列优先）
intensity_feature_matrix = np.zeros((1024, num_sample))  # 强度特征
LBP_feature_matrix = np.zeros((3304, num_sample))       # LBP特征
Gabor_feature_matrix = np.zeros((6750, num_sample))     # Gabor特征

count = 0  # 样本计数器

def extract_intensity(img, block_size):
    """提取图像强度特征（分块均值）"""
    h, w = img.shape
    bh, bw = block_size
    features = []
    
    # 分块处理
    for i in range(0, h, bh):
        for j in range(0, w, bw):
            block = img[i:i+bh, j:j+bw]
            features.append(np.mean(block))
    
    return np.array(features).flatten()

def extract_lbp(img, grid_size, win_size, radius, neighbors):
    """提取LBP特征"""
    h, w = img.shape
    gh, gw = grid_size
    wh, ww = win_size
    
    features = []
    
    # 计算滑动步长
    step_h = max(1, (h - wh) // (gh - 1))
    step_w = max(1, (w - ww) // (gw - 1))
    
    # 网格化提取特征
    for i in range(gh):
        for j in range(gw):
            y = min(i * step_h, h - wh)
            x = min(j * step_w, w - ww)
            window = img[y:y+wh, x:x+ww]
            
            # 计算LBP特征（使用uniform模式对应MATLAB的u2）
            lbp = local_binary_pattern(window, neighbors, radius, method='uniform')
            hist, _ = np.histogram(lbp, bins=neighbors+2, range=(0, neighbors+2), density=True)
            features.extend(hist)
    
    return np.array(features).flatten()

def extract_gabor(img, scales, orientations, frequencies, bandwidths):
    """提取Gabor特征"""
    features = []
    
    for scale in scales:
        # 尺度缩放
        resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for theta in orientations:
            for freq in frequencies:
                for bw in bandwidths:
                    # 计算Gabor滤波响应
                    filt_real, filt_imag = gabor(resized, frequency=freq, 
                                                theta=np.deg2rad(theta), 
                                                bandwidth=bw)
                    features.append(filt_real.mean())
                    features.append(filt_imag.mean())
    
    return np.array(features).flatten()

# 主处理循环
for i in range(1, 21):  # 20个类别
    data_path = os.path.join(path, f'obj{i}__')
    for j in range(72):  # 每个类别72个样本
        img_path = os.path.join(data_path, f'{j}.png')
        
        # 读取图像并归一化
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        
        # 提取三种特征
        intensity_feature_matrix[:, count] = extract_intensity(img, (32, 32))
        LBP_feature_matrix[:, count] = extract_lbp(img, (16, 16), (112, 128), 1, 8)
        Gabor_feature_matrix[:, count] = extract_gabor(img, [4], [0, 45, 90, 135], [30, 25], [90, 75])
        
        # 设置标签
        gt[count] = i
        count += 1

# 组织多视图数据
X[0] = intensity_feature_matrix
X[1] = LBP_feature_matrix
X[2] = Gabor_feature_matrix

# 保存为MATLAB格式文件
sio.savemat('test_coil20.mat', {'X': X, 'gt': gt})