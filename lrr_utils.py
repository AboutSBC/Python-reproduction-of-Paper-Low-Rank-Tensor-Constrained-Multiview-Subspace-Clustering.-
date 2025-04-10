import numpy as np
from scipy.linalg import svd
from scipy.sparse import csr_matrix,diags,eye
from scipy.linalg import orth
from sklearn.cluster import KMeans

def solve_l1l2(W,lamda):
    #对输入矩阵W的每一列进行L2优化处理。
    n = W.shape[1]
    E = W.copy()
    for i in range(n):
        E[:, i] = solve_l2(W[:, i], lamda)
    
    return E

def solve_l2(w,lamda):
    #对输入向量w进行L2优化处理。
    nw = np.linalg.norm(w)
    if nw > lamda:
        x = (nw - lamda) * w / nw
    else:
        x = np.zeros_like(w)
    
    return x

def exact_alm_lrr_l1v2(D,A,lamda = None,tol = 1e-7,maxIter = 1000,display = False):
    #初始化变量
    m,n = D.shape
    k = A.shape[1]
    n_max = max(m,n)
    if lamda is None:
        lamda = 1 / np.sqrt(n_max)
    maxIter_primal = 10000
    Y = np.sign(D)
    norm_two = np.linalg.norm(Y,2)
    norm_inf = np.linalg.norm(Y,np.inf) / lamda
    dual_norm = max(norm_two,norm_inf)
    Y = Y / dual_norm
    W = np.zeros((k,n))
    Z_hat = np.zeros((k,n))
    E_hat = np.zeros((m,n))
    if m >= k:
        inv_ata = np.linalg.inv(np.eye(k) + A.T @ A)
    else:
        inv_ata = np.eye(k) - A.T @ np.linalg.inv(np.eye(m) + A @ A.T) @ A
    
    #可调整超参数定义
    mu = 0.5 / norm_two
    rho = 6 

    #归一化与投影
    dnorm = np.linalg.norm(D,'fro')
    tolProj1 = 1e-6 * dnorm
    anorm = np.linalg.norm(A,2)
    tolProj2 = 1e-6 * dnorm / anorm

    #ALM
    iter = 0
    while iter < maxIter:
        iter += 1
        primal_iter = 0
        while primal_iter < maxIter_primal:
            primal_iter += 1
            temp_Z = Z_hat.copy()
            temp_E = E_hat.copy()

            #更新J
            temp = temp_Z + W / mu
            U, S, Vt = svd(temp,full_matrices=False)
            svp = np.sum(S > 1 / mu)
            S = np.maximum(0,S - 1 / mu)
            if svp < 0.5:
                svp = 1
            J_hat = U[:, :svp] @ np.diag(S[:svp]) @ Vt[:svp, :]

            #更新Z
            temp = J_hat + A.T @ (D - temp_E) + (A.T @ Y - W) / mu
            Z_hat = inv_ata @ temp

            #更新E
            temp = D - A @ Z_hat + Y / mu
            E_hat = np.maximum(0,temp - lamda / mu) + np.minimum(0,temp + lamda / mu)
            if np.linalg.norm(E_hat - temp_E,'fro') < tolProj1 and np.linalg.norm(Z_hat - temp_Z) < tolProj2:
                break
        H1 = D - A @ Z_hat - E_hat
        H2 = Z_hat - J_hat
        Y = Y + mu * H1
        W = W + mu * H2
        mu = rho * mu

        #设定停止标准
        stopC = max(np.linalg.norm(H1,'fro') / dnorm, np.linalg.norm(H2,'fro') / dnorm * anorm)
        if display:
            print(f'LRR: Iteration {iter} ({primal_iter}), mu {mu}, |E|_0 {np.sum(np.abs(E_hat) > 0)}, stopCriterion {stopC}')
        if stopC < tol:
            break

    return Z_hat,E_hat

def exact_alm_lrr_l21v2(D,A,lamda = None,tol = 1e-7,maxIter = 1000,display = False):
    #初始化变量
    m,n = D.shape
    k = A.shape[1]
    n_max = max(m,n)
    if lamda is None:
        lamda = 1 / np.sqrt(n_max)
    maxIter_primal = 10000
    Y = np.sign(D)
    norm_two = np.linalg.norm(Y,2)
    norm_inf = np.linalg.norm(Y,np.inf) / lamda
    dual_norm = max(norm_two,norm_inf)
    Y = Y / dual_norm
    W = np.zeros((k,n))
    Z_hat = np.zeros((k,n))
    E_hat = np.zeros((m,n))
    if m >= k:
        inv_ata = np.linalg.inv(np.eye(k) + A.T @ A)
    else:
        inv_ata = np.eye(k) - A.T @ np.linalg.inv(np.eye(m) + A @ A.T) @ A

    #可调整超参数定义
    mu = 0.5 / norm_two
    rho = 6

    #归一化与投影
    dnorm = np.linalg.norm(D,'fro')
    tolProj1 = 1e-6 * dnorm
    anorm = np.linalg.norm(A,2)
    tolProj2 = 1e-6 * dnorm / anorm

    #ALM
    iter = 0
    while iter < maxIter:
        iter += 1
        primal_iter = 0
        while primal_iter < maxIter_primal:
            primal_iter += 1
            temp_Z = Z_hat.copy()
            temp_E = E_hat.copy()

            #更新J
            temp = temp_Z + W / mu
            U, S, Vt = svd(temp, full_matrices=False)
            svp = np.sum(S > 1 / mu)
            S = np.maximum(0,S - 1 / mu)
            if svp < 0.5:
                svp = 1
            J_hat = U[:, :svp] @ np.diag(S[:svp].flatten()) @ Vt[:svp, :]

            #更新Z
            temp = J_hat + A.T @ (D - temp_E) + (A.T @ Y - W) / mu
            Z_hat = inv_ata @ temp

            #更新E
            temp = D - A @ Z_hat + Y / mu
            E_hat = solve_l1l2(temp,lamda/mu)
            if np.linalg.norm(E_hat - temp_E,'fro') < tolProj1 and np.linalg.norm(Z_hat - temp_Z) < tolProj2:
                break
        H1 = D - A @ Z_hat - E_hat
        H2 = Z_hat - J_hat
        Y = Y + mu * H1
        W = W + mu * H2
        mu = rho * mu

        #设定停止标准
        stopC = max(np.linalg.norm(H1,'fro') / dnorm, np.linalg.norm(H2,'fro') / dnorm * anorm)
        if display:
            print(f'LRR: Iteration {iter} ({primal_iter}), mu {mu}, |E|_2,0 {np.sum(np.sum(E_hat**2,axis=0) > 0)}, stopCriterion {stopC}')
        if stopC < tol:
            break

    return Z_hat,E_hat

def inexact_alm_lrr_l1(X,A,lamda = None,tol = 1e-7,maxIter = 1000,display = False):
    #初始化变量
    m, n = X.shape
    k = A.shape[1]
    n_max = max(m,n)
    if lamda is None:
        lamda = 1 / np.sqrt(n_max)
    atx = A.T @ X
    inv_a = np.linalg.inv(A.T @ A + np.eye(k))
    J = np.zeros((k, n))
    Z = np.zeros((k, n))
    E = csr_matrix((m, n))  
    Y1 = np.zeros((m, n))
    Y2 = np.zeros((k, n))

    #可调整超参数定义
    rho = 1.1
    max_mu = 1e10
    mu = 1e-6
    
    #ALM
    iter = 0
    if display:
        print(f"initial, rank={np.linalg.matrix_rank(Z)}")
    while iter < maxIter:
        iter += 1
        
        #更新J
        temp = Z + Y2 / mu
        U, sigma, Vt = svd(temp, full_matrices=False)
        svp = np.sum(sigma > 1 / mu)  
        if svp >= 1:
            sigma = sigma[:svp] - 1 / mu
        else:
            svp = 1
            sigma = np.array([0])
        J = U[:, :svp] @ np.diag(sigma) @ Vt[:svp, :]

        #更新Z
        Z = inv_a @ (atx - A.T @ E + J + (A.T @ Y1 - Y2) / mu)

        #更新E
        xmaz = X - A @ Z
        temp = xmaz + Y1 / mu
        E = np.maximum(0,temp - lamda / mu) + np.minimum(0,temp + lamda / mu)

        #设定停止标准
        leq1 = xmaz - E
        leq2 = Z - J
        stopC = max(np.max(np.abs(leq1)), np.max(np.abs(leq2)))
        if display and (iter == 1 or iter % 50 == 0 or stopC < tol):
            print(f"Iter {iter}, mu={mu:.1e}, rank={np.linalg.matrix_rank(Z, tol=1e-3 * np.linalg.norm(Z, 2))}, stopALM={stopC:.3e}")
        if stopC < tol:
            break
        else:
            Y1 = Y1 + mu * leq1
            Y2 = Y2 + mu * leq2
            mu = min(max_mu, mu * rho)

    return Z, E

def inexact_alm_lrr_l21(X,A,lamda = None,tol = 1e-7,maxIter = 1000,display = False):
    #初始化变量
    m, n = X.shape
    k = A.shape[1]
    n_max = max(m,n)
    if lamda is None:
        lamda = 1 / np.sqrt(n_max)
    atx = A.T @ X
    inv_a = np.linalg.inv(A.T @ A + np.eye(k))
    J = np.zeros((k,n))
    Z = np.zeros((k,n))
    E = csr_matrix((m,n))  
    Y1 = np.zeros((m,n))
    Y2 = np.zeros((k,n))

    #可调整超参数定义
    rho = 1.1
    max_mu = 1e10
    mu = 1e-6
    
    #ALM
    iter = 0
    if display:
        print(f"initial, rank={np.linalg.matrix_rank(Z)}")
    while iter < maxIter:
        iter += 1
        
        #更新J
        temp = Z + Y2 / mu
        U, sigma, Vt = svd(temp,full_matrices = False)
        svp = np.sum(sigma > 1 / mu)  
        if svp >= 1:
            sigma = sigma[:svp] - 1 / mu
        else:
            svp = 1
            sigma = np.array([0])
        J = U[:, :svp] @ np.diag(sigma) @ Vt[:svp, :]

        #更新Z
        Z = inv_a @ (atx - A.T @ E + J + (A.T @ Y1 - Y2) / mu)

        #更新E
        xmaz = X - A @ Z
        temp = xmaz + Y1 / mu
        E = solve_l1l2(temp,lamda/mu)

        #设定停止标准
        leq1 = xmaz - E
        leq2 = Z - J
        stopC = max(np.max(np.abs(leq1)), np.max(np.abs(leq2)))
        if display and (iter == 1 or iter % 50 == 0 or stopC < tol):
            print(f"Iter {iter}, mu={mu:.1e}, rank={np.linalg.matrix_rank(Z, tol=1e-3 * np.linalg.norm(Z, 2))}, stopALM={stopC:.3e}")
        if stopC < tol:
            break
        else:
            Y1 = Y1 + mu * leq1
            Y2 = Y2 + mu * leq2
            mu = min(max_mu, mu * rho)

    return Z, E

def solve_lrr(X,A,lamda,reg_type=0,alm_type=1,display=1):
    #正交化字典矩阵
    Q = orth(A.T)
    B = A @ Q

    #选择优化策略
    if reg_type == 0:
        if alm_type == 0:
            Z,E = exact_alm_lrr_l21v2(X,B,lamda,display = display)
        else:
            Z,E = inexact_alm_lrr_l21(X,B,lamda,display = display)
    else:
        if alm_type == 0:
            Z,E = exact_alm_lrr_l1v2(X,B,lamda,display = display)
        else:
            Z,E = inexact_alm_lrr_l1(X,B,lamda,display = display)

    # 更新结果
    Z = Q @ Z
    
    return Z, E

def normalize_data(X):
    nFea, nSmp = X.shape
    ProcessData = np.zeros_like(X)
    for i in range(nSmp):
        norm = np.linalg.norm(X[:, i])
        ProcessData[:, i] = X[:, i] / max(1e-12,norm)
    
    return ProcessData

def spectral_clustering(CKSym, n):
    #可调整超参数定义
    MAXiter = 1000 
    REPlic = 20 
    
    #初始化变量
    N = CKSym.shape[0]
    DN = np.diag(1. / np.sqrt(np.sum(CKSym,axis=1) + np.finfo(float).eps))
    LapN = eye(N).toarray() - DN @ CKSym @ DN
    uN, sN, vN = svd(LapN)
    vN = vN.T
    kerN = vN[:, N-n:N]
    kerNS = np.zeros_like(kerN)
    
    #使用Kmeans进行谱聚类
    for i in range(N):
        kerNS[i, :] = kerN[i, :] / (np.linalg.norm(kerN[i, :]) + np.finfo(float).eps)
    kmeans = KMeans(n_clusters = n, max_iter = MAXiter, n_init = REPlic, init = 'k-means++')
    groups = kmeans.fit_predict(kerNS)
    
    return groups