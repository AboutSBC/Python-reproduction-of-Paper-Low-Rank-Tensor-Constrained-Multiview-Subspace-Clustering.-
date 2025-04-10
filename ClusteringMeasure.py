import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.special import comb

def Accuracy(pred, gt):
    #pred_C是经过最优映射后的预测分类标签，确保参数顺序是（真实标签，预测标签）
    pred_C = bestMap(gt, pred)
    
    #统计与正式标签一致的数量并计算acc
    ACC = np.sum(gt == pred_C) / len(gt)
    return ACC

def bestMap(gt, pred):
    #将真实标签和预测标签展平
    gt = np.array(gt).flatten()
    pred = np.array(pred).flatten()

    #确保两个标签的维度完全一致
    if gt.shape != pred.shape:
        raise ValueError('size(gt) must == size(pred)')
    
    #获取两种标签各自包含的类别数量并采用两者最大的类别数量
    Label_gt = np.unique(gt)
    nC_gt = len(Label_gt)
    Label_pred = np.unique(pred)
    nC_pred = len(Label_pred)
    nC = max(nC_pred, nC_gt)
    
    #创建混淆矩阵
    G = np.zeros((nC, nC))
    for i in range(nC_gt):
        for j in range(nC_pred):
            G[i, j] = np.sum((gt == Label_gt[i]) & (pred == Label_pred[j]))
    
    #使用贪婪算法获取最优匹配
    row_ind, col_ind = linear_sum_assignment(-G)
    
    #将L2作为预测标签，将预测标签与真实标签统一
    newpred = np.zeros_like(pred)
    for i in range(nC_pred):
        idx = np.where(col_ind == i)[0]
        if len(idx) > 0:
            c = row_ind[idx[0]]
            newpred[pred == Label_pred[i]] = Label_gt[c]
            
    return newpred

def compute_f(gt, pred):
    #确保两个标签的维度完全一致
    if len(gt) != len(pred):
        print("Size mismatch:", np.shape(gt), np.shape(pred))
    
    #初始化参数
    N = len(gt)
    num_gt = 0
    num_pred = 0
    num_I = 0

    #初始化指标
    precision = 1
    recall = 1
    fscore = 1
    
    #计算fscore指标所需变量
    for n in range(N):
        gt_n = (gt[n+1:] == gt[n])
        pred_n = (pred[n+1:] == pred[n])
        
        num_gt += np.sum(gt_n)
        num_pred += np.sum(pred_n)
        num_I += np.sum(gt_n * pred_n)
    
    #指标计算
    if num_pred > 0:
        precision = num_I / num_pred
    if num_gt > 0:
        recall = num_I / num_gt
    #这里使用F1
    if (precision + recall) == 0:
        fscore = 0
    else:
        fscore = 2 * precision * recall / (precision + recall)
    
    return fscore, precision, recall

def compute_nmi(gt, pred):
    #获取两种标签各自包含的类别数量
    N = len(gt)
    Label_gt = np.unique(gt)
    Label_pred = np.unique(pred)
    nC_gt = len(Label_gt)
    nC_pred = len(Label_pred)
    
    #初始化参数
    mi = 0
    avgent = 0
    A = np.zeros((nC_pred, nC_gt)) 
    miarr = np.zeros((nC_pred, nC_gt))
    D = np.zeros(nC_gt)
    B = np.zeros(nC_pred)
    
    #计算MI
    for i in range(nC_pred):
        index_pred = (pred == Label_pred[i])
        B[i] = np.sum(index_pred)
        for j in range(nC_gt):
            index_gt = (gt == Label_gt[j])
            D[j] = np.sum(index_gt)
            A[i, j] = np.sum(index_gt * index_pred)
            if A[i, j] != 0:
                miarr[i, j] = A[i, j] / N * np.log2(N * A[i, j] / (B[i] * D[j]))
                avgent = avgent - (B[i] / N) * (A[i, j] / B[i]) * np.log2(A[i, j] / B[i])
            else:
                miarr[i, j] = 0      
            mi = mi + miarr[i, j]
    
    #计算真实分类和预测分类的熵
    gt_ent = 0
    for i in range(nC_gt):
        if D[i] > 0:
            gt_ent = gt_ent + D[i] / N * np.log2(N / D[i])
    pred_ent = 0
    for i in range(nC_pred):
        if B[i] > 0:
            pred_ent = pred_ent + B[i] / N * np.log2(N / B[i])
    
    #计算NMI
    nmi = 2 * mi / (gt_ent + pred_ent)
    
    return A, nmi, avgent

def Contingency(M1, M2):
    #检验输入是否满足要求并将输入转化为numpy数组
    if len(np.shape(M1)) > 1 and min(np.shape(M1)) > 1 or len(np.shape(M2)) > 1 and min(np.shape(M2)) > 1:
        raise ValueError('Contingency: Requires two vector arguments')
    M1 = np.array(M1, dtype=int)
    M2 = np.array(M2, dtype=int)
    
    #计算可能性矩阵
    Cont = np.zeros((np.max(M1) + 1, np.max(M2) + 1), dtype=int)
    for i in range(len(M1)):
        Cont[M1[i], M2[i]] += 1
    
    return Cont

def RandIndex(c1, c2):
    #检验输入是否合适
    if len(c1) != len(c2) or np.ndim(c1) > 1 or np.ndim(c2) > 1:
        raise ValueError('RandIndex: Requires two vector arguments')
    
    #构建可能性矩阵并计算各项指标
    C = Contingency(c1, c2)
    n = np.sum(C)
    nis = np.sum(np.sum(C, axis=1) ** 2)
    njs = np.sum(np.sum(C, axis=0) ** 2)
    t1 = comb(n, 2)
    t2 = np.sum(C ** 2)
    t3 = 0.5 * (nis + njs)
    nc = (n * (n**2 + 1) - (n + 1) * nis - (n + 1) * njs + 2 * (nis * njs) / n) / (2 * (n - 1))
    A = t1 + t2 - t3
    D = -t2 + t3
    if t1 == nc:
        AR = 0 
    else:
        AR = (A - nc) / (t1 - nc)
    RI = A / t1
    MI = D / t1
    HI = (A - D) / t1
    
    return AR, RI, MI, HI