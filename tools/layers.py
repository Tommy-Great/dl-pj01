import numpy as np


class Conv2D:
    def __init__(self, in_ch, out_ch, kernel=5, stride=1, pad=0):
        self.W = np.random.randn(out_ch, in_ch, kernel, kernel) * 0.1
        self.b = np.zeros(out_ch)
        self.stride = stride
        self.pad = pad
        self.cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        F, _, HH, WW = self.W.shape

        # 计算输出尺寸 
        H_out = (H + 2 * self.pad - HH) // self.stride + 1
        W_out = (W + 2 * self.pad - WW) // self.stride + 1

        # 添加Padding 
        x_pad = np.pad(x, ((0, 0), (0, 0), (self.pad, self.pad),
                           (self.pad, self.pad)), mode='constant')

        out = np.zeros((N, F, H_out, W_out))

        # 卷积计算（优化循环顺序）
        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        window = x_pad[n, :, h_start:h_start + HH, w_start:w_start + WW]
                        out[n, f, i, j] = np.sum(window * self.W[f]) + self.b[f]

        self.cache = (x, x_pad, H_out, W_out)
        return out

    def backward(self, dout):
        x, x_pad, H_out, W_out = self.cache
        N, C, H, W = x.shape
        F, _, HH, WW = self.W.shape

        dx_pad = np.zeros_like(x_pad).astype('float64')
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        # 反向传播核心计算 
        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.stride
                        w_start = j * self.stride

                        # 梯度累加 
                        window = x_pad[n, :, h_start:h_start + HH, w_start:w_start + WW]
                        dW[f] += window * dout[n, f, i, j]
                        db[f] += dout[n, f, i, j]
                        dx_pad[n, :, h_start:h_start + HH, w_start:w_start + WW] += self.W[f] * dout[n, f, i, j]

        # 去除Padding 
        dx = dx_pad[:, :, self.pad:H + self.pad, self.pad:W + self.pad]
        return dx, dW / N, db / N


class MaxPool2D:
    def __init__(self, pool=2, stride=2):
        self.pool = pool
        self.stride = stride
        self.cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        H_out = (H - self.pool) // self.stride + 1
        W_out = (W - self.pool) // self.stride + 1

        out = np.zeros((N, C, H_out, W_out))
        max_idx = np.zeros_like(x)

        # 记录最大值位置 
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        window = x[n, c, h_start:h_start + self.pool, w_start:w_start + self.pool]
                        out[n, c, i, j] = np.max(window)
                        h, w = np.unravel_index(window.argmax(), window.shape)
                        max_idx[n, c, h_start + h, w_start + w] = 1

        self.cache = (x.shape, max_idx)
        return out

    def backward(self, dout):
        orig_shape, max_idx = self.cache
        dx = np.zeros(orig_shape)

        # 梯度分配 
        for n in range(dout.shape[0]):
            for c in range(dout.shape[1]):
                for i in range(dout.shape[2]):
                    for j in range(dout.shape[3]):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        dx[n, c, h_start:h_start + self.pool, w_start:w_start + self.pool] += \
                            max_idx[n, c, h_start:h_start + self.pool, w_start:w_start + self.pool] * dout[n, c, i, j]
        return dx


class Dense:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2. / in_dim)
        self.b = np.zeros(out_dim)
        self.cache = None

    def forward(self, x):
        self.cache = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        x = self.cache
        dx = np.dot(dout, self.W.T)
        dW = np.dot(x.T, dout)
        db = np.sum(dout, axis=0)
        return dx, dW / x.shape[0], db / x.shape[0]


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class Flatten:
    def forward(self, x):
        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.orig_shape)


class SoftmaxWithCELoss:
    def __init__(self):
        self.loss = None
        self.y_pred = None  # softmax输出 
        self.y_true = None  # 真实标签（one-hot）

    def forward(self, x, y):
        """
        x : 全连接层输出 (N, C)
        y : 真实标签 (N,) 或 (N,C) 
        """
        # 数值稳定化处理 
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        self.y_pred = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        # 自动处理标签格式 
        if y.ndim == 1:
            self.y_true = np.eye(x.shape[1])[y]  # 转one-hot 
        else:
            self.y_true = y.copy()

            # 计算交叉熵损失 
        eps = 1e-7  # 防止log(0)
        self.loss = -np.sum(self.y_true * np.log(self.y_pred + eps)) / x.shape[0]
        return self.loss
    
        
    def backward(self, dout=1):
        """
        反向传播梯度计算 
        数学推导：∂L/∂x = (y_pred - y_true) / batch_size 
        """
        batch_size = self.y_true.shape[0]
        dx = (self.y_pred - self.y_true) * dout / batch_size
        return dx

# class SoftmaxWithCELoss:
#     def forward(self, x, y):
#         self.y = y
#         exps = np.exp(x - np.max(x, axis=1, keepdims=True))
#         self.probs = exps / np.sum(exps, axis=1, keepdims=True)
#         return -np.mean(y * np.log(self.probs + 1e-8))
#     
#     def predict(self, x):
#         exps = np.exp(x - np.max(x, axis=1, keepdims=True))
#         probs = exps / np.sum(exps, axis=1, keepdims=True)
#         return probs
#     
#     def backward(self):
#         return (self.probs - self.y) / self.probs.shape[0] 


