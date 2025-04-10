import numpy as np 

def Matrix2Vector(M, unfolding_mode, dim1, dim2, dim3, K):
    if unfolding_mode == 1:
        v = M.reshape(dim1 * dim2 * dim3, 1)
    elif unfolding_mode == 2:
        v_list = []
        for k in range(K):
            block = M[k*dim1:(k+1)*dim1, 0:dim1]
            v_list.append(block.reshape(dim1*dim2,1))
            v = np.vstack(v_list)
    elif unfolding_mode == 3:
        M_transposed = M.T
        v = M_transposed.reshape((dim1 * dim2 * dim3, 1))
    else:
        raise ValueError("Invalid unfolding mode")
        
    return v

def Tensor2Matrix(T, unfolding_mode, dim1, dim2, dim3):
    blocks = []
    if unfolding_mode == 1:
        for t in T:
            blocks.append(t)
        m = np.hstack(blocks) if blocks else np.array([])
    elif unfolding_mode == 2:
        for t in T:
            blocks.append(t.T)
        m = np.vstack(blocks) if blocks else np.array([])
    elif unfolding_mode == 3:
        for t in T:
            t_reshaped = t.reshape(dim1*dim2, 1)
            blocks.append(t_reshaped)
        m = np.hstack(blocks) if blocks else np.array([])
    else:
        raise ValueError("Invalid unfolding mode")

    return m

def Tensor2Vector(T, dim1, dim2, dim3, K):
    blocks = []
    for t in T:
        t_reshaped = t.T.reshape((dim1*dim2, 1))
        blocks.append(t_reshaped)

    return np.vstack(blocks) if blocks else np.array([])

def Vector2Tensor(v, dim1, dim2, dim3, K):
    v_flat = v.flatten()
    L = len(v_flat) // K
    t = []
    for k in range(K):
        start = k * L
        end = (k+1) * L
        block = v_flat[start:end]
        tensor = block.reshape((dim1, dim2))
        t.append(tensor)

    return np.stack(t, axis = 0)

def softth(F, lambda_):
    U, S, V = np.linalg.svd(F, full_matrices = False)
    svp = np.sum(S > lambda_)
    diagS = np.maximum(S - lambda_, 0)
    if svp<0.5:
        svp=1
    U_sub = U[:,:svp]
    S_sub = np.diag(diagS[:svp])
    V_sub = V[:svp,:]
    
    return U_sub @ S_sub @ V_sub


def updateG_tensor(WT, K, sX, mu, gamma, V, mode = 2):
    W = []
    for v in range(V):
        W.append(WT[:,:,v])
    w = Tensor2Vector(W,sX[0],sX[1],sX[2],V)
    k = Tensor2Vector(K,sX[0],sX[1],sX[2],V)
    wk = k + w / mu
    WKten = Vector2Tensor(wk,sX[0],sX[1],sX[2],V)
    WK = Tensor2Matrix(WKten, mode+1,sX[0],sX[1],sX[2])
    WKsoft = softth(WK, gamma[mode]/mu)
    tensor_out = WKsoft.reshape((sX[0],sX[1],sX[2]), order = 'F')

    return tensor_out