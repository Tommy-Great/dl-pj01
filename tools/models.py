from tools.layers import *


class MyCNN:
    def __init__(self, dim_in, dim_out, l_r):
        self.l_r = l_r
        
        self.layers = [
            # Conv2D(dim_in, 6, kernel=5), # 32->28
            # ReLU(),
            # MaxPool2D(2, 2),  # 28->14
            Flatten(),
            Dense(3 * 32 * 32, 1024),
            ReLU(),
            Dense(1024, 512),
            ReLU(),
            Dense(512, dim_out)
            # dense2,
            # ReLU(),
            # dense3
        ]
        self.softmax = SoftmaxWithCELoss()

        
        
    def forward(self, x, y):
        # 前向传播 
        for layer in self.layers:
            x = layer.forward(x)
        self.loss = self.softmax.forward(x, y)
        return self.loss
    
    def get_predict(self):
        probs = self.softmax.y_pred
        return probs

    def backward(self):
        # 反向传播 
        dout = self.softmax.backward()
        for layer in reversed(self.layers):
            if isinstance(layer, Dense):
                dout, dw, db = layer.backward(dout)
                layer.W -= self.l_r * dw  # 简单SGD更新 
                layer.b -= self.l_r * db
            elif isinstance(layer, Conv2D):
                dout, dw, db = layer.backward(dout)
                layer.W -= self.l_r * dw
                layer.b -= self.l_r * db
            else:
                dout = layer.backward(dout) 

