import numpy as np
from scipy.linalg import inv
from lrr_utils import spectral_clustering,solve_l1l2
from TensorRank import updateG_tensor
from ClusteringMeasure import compute_nmi,Accuracy,compute_f,RandIndex
def lt_msc(X,gt,gamma):
    #初始化变量
    V = len(X)
    cls_num = len(np.unique(gt))
    K = len(X)
    N = X[0].shape[1]
    Z = [np.zeros((N, N)) for _ in range(K)]
    W = [np.zeros((N, N)) for _ in range(K)]
    G = [np.zeros((N, N)) for _ in range(K)]
    E = [np.zeros((X[k].shape[0], N)) for k in range(K)]
    Y = [np.zeros((X[k].shape[0], N)) for k in range(K)]
    w = np.zeros((N * N * K,1))
    g = np.zeros((N * N * K,1))
    dim1, dim2, dim3 = N, N, K
    myNorm = 'tSVD_1'
    sX = (N, N, K)
    parOP = False
    ABSTOL = 1e-6
    RELTOL = 1e-4
    Isconverg = 0
    epson = 1e-7
    ModCount = 3
    para_ten = [gamma] * ModCount
    iter = 0
    mu = 10e-5
    max_mu = 10e10
    pho_mu = 2
    rho = 10e-5
    max_rho = 10e12
    pho_rho = 2
    Z_tensor = np.stack(Z, axis=2)
    G_tensor = np.stack(G, axis=2)
    W_tensor = np.stack(W, axis=2)
    WT = [W_tensor] * ModCount
    history = {'norm_Z': [], 'norm_Z_G': []}

    #lt_msc
    while Isconverg == 0:
        print(f'----processing iter {iter + 1}--------')
        for k in range(K):
            #更新Z
            tmp = (X[k].T @ Y[k] + mu * X[k].T @ X[k] - mu * X[k].T @ E[k] - W[k]) / rho + G[k]
            Z[k] = inv(np.eye(N) + (mu / rho) * X[k].T @ X[k]) @ tmp

            #更新E
            F = np.vstack([X[v] - X[v] @ Z[v] + Y[v] / mu for v in range(K)])
            Econcat = solve_l1l2(F,gamma / mu)

            beg_ind = 0
            end_ind = 0
            for v in range(K):
                if v > 0:
                    beg_ind += X[v - 1].shape[0]
                else:
                    beg_ind = 0
                end_ind += X[v].shape[0]
                E[v] = Econcat[beg_ind:end_ind, :]

            #更新Y
            Y[k] = Y[k] + mu * (X[k] - X[k] @ Z[k] - E[k])

        #更新G
        Z_tensor = np.stack(Z, axis=2)
        W_tensor = np.stack(W, axis=2)
        z = Z_tensor.flatten()
        w = W_tensor.flatten()

        for umod in range(ModCount):
            G_tensor = updateG_tensor(WT[umod],Z,sX,mu,para_ten,V,umod)
            WT[umod] = WT[umod] + mu * (Z_tensor - G_tensor)  

        #设定停止标准
        Isconverg = 1
        for k in range(K):
            if np.linalg.norm(X[k] - X[k] @ Z[k] - E[k],np.inf) > epson:
                history['norm_Z'].append(np.linalg.norm(X[k] - X[k] @ Z[k] - E[k],np.inf))
                Isconverg = 0
            G[k] = G_tensor[:, :, k]
            W_tensor = np.stack(WT, axis=2)
            W_tensor = W_tensor.reshape(W_tensor.shape[0],W_tensor.shape[1],-1)
            W[k] = W_tensor[:, :, k]
            if np.linalg.norm(Z[k] - G[k], np.inf) > epson:
                history['norm_Z_G'].append(np.linalg.norm(Z[k] - G[k],np.inf))
                Isconverg = 0
        if iter > 50:
            Isconverg = 1
        iter += 1
        mu = min(mu * pho_mu,max_mu)
        rho = min(rho * pho_rho,max_rho)

    #计算分类结果
    S = np.sum([np.abs(Z[k]) + np.abs(Z[k].T) for k in range(K)], axis=0)
    C = spectral_clustering(S,cls_num)
    gt = gt.reshape(-1)

    #计算各项指标
    A,nmi,avgent=compute_nmi(gt,C)
    acc=Accuracy(C,gt.astype(float))
    f,p,r=compute_f(gt,C)
    ar,ri,MI,HI=RandIndex(gt,C)
    
    return nmi,acc,ar,f,p,r,C,ri