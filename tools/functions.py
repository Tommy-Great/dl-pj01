import numpy as np


def span_y(y,class_size,pos_val=1,neg_val=-1):
    N=len(y)
    y_mat=np.ones([N,class_size])*neg_val
    y_mat[np.arange(N),y]=pos_val
    return y_mat

def square_error_solution(x,y_mat):
    cf_solu=np.linalg.solve(x.T@x,x.T@y_mat)
    return cf_solu

def my_softmax(x):
    x=np.exp(x-np.max(x,axis=x.ndim-1,keepdims=True)[0])
    return x/np.sum(x,axis=x.ndim-1,keepdims=True)

def my_convol(x, weight, transpose):
    N, H, W = x.shape
    wid1, wid2 = weight.shape
    if transpose:
        weight=weight.T
        row_cent = wid1 // 2
        col_cent = wid2 // 2
    else:
        row_cent = (wid1 - 1) // 2
        col_cent = (wid2 - 1) // 2

    out = np.zeros_like(x)
    
    # t1=time.time()
    for i in range(wid1):
        for j in range(wid2):
            row_bias = i - row_cent
            col_bias = j - col_cent
            row_sta1, row_end1 = max(row_bias, 0), min(H, H + row_bias)
            col_sta1, col_end1 = max(col_bias, 0), min(W, W + col_bias)
            row_sta2, row_end2 = max(-row_bias, 0), min(H, H - row_bias)
            col_sta2, col_end2 = max(-col_bias, 0), min(W, W - col_bias)
            out[:, row_sta2:row_end2, col_sta2:col_end2] += \
                weight[i, j] * x[:, row_sta1:row_end1, col_sta1:col_end1]
    # t2 = time.time()
    # print(f'core {t2-t1}')
    
    return out


def sheer_mul(x, dout, weight):
    N, H, W = x.shape
    wid1, wid2 = weight.shape
    row_cent = (wid1 - 1) // 2
    col_cent = (wid2 - 1) // 2
    dw = np.zeros_like(weight)
    for i in range(wid1):
        for j in range(wid2):
            row_bias = i - row_cent
            col_bias = j - col_cent
            row_sta1, row_end1 = max(row_bias, 0), min(H, H + row_bias)
            col_sta1, col_end1 = max(col_bias, 0), min(W, W + col_bias)
            row_sta2, row_end2 = max(-row_bias, 0), min(H, H - row_bias)
            col_sta2, col_end2 = max(-col_bias, 0), min(W, W - col_bias)
            dw[i, j] = np.sum(np.multiply(x[:, row_sta1:row_end1, col_sta1:col_end1],
                                           dout[:, row_sta2:row_end2, col_sta2:col_end2]))
    return dw



