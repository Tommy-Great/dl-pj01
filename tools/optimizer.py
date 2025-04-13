import numpy as np


class MomentumOpt(object):
    def __init__(self,learning_rate, model, beta=0.9, reg2=None,
                 decay_period=None,decay_rate=0.5):
        self.learning_rate = learning_rate
        self.model = model
        self.beta=beta
        self.reg2=reg2
        self.decay_period = decay_period
        self.decay_rate = decay_rate
        self.steps=0
        self.moments = []
        for param in self.model.params:
            self.moments.append(np.zeros_like(param))
        
    def update(self):
        model = self.model
        reg2 = self.reg2
        self.check_decay()
        l_r=self.learning_rate
        for param, grad, moment in zip(model.params, model.grads, self.moments):
            moment*=self.beta
            if reg2 is not None:
                grad += 2*reg2 * param
            moment -= l_r * grad
            param += moment
        self.steps+=1
    
    def check_decay(self):
        steps=self.steps
        decay_period=self.decay_period
        if steps>0 and decay_period is not None and \
                decay_period>0 and steps%decay_period==0:
            self.learning_rate*=self.decay_rate
    
    # def clear_grads(self):
    #     for grad in self.model.grads:
    #         grad[:]=0


class SGDopt(object):
    def __init__(self, learning_rate, model, reg2=None,
                 decay_period=None,decay_rate=0.5):
        self.learning_rate = learning_rate
        self.model = model
        self.reg2 = reg2
        self.decay_period = decay_period
        self.decay_rate = decay_rate
        self.steps=0

    def update(self):
        reg2 = self.reg2
        self.check_decay()
        for param,grad in zip(self.model.params,self.model.grads):
            if reg2 is not None:
                grad+=2*reg2*param
            param -= self.learning_rate * grad
        self.steps += 1
            
    
    def clear_grads(self):
        for grad in self.model.grads:
            grad[:]=0
    
    def check_decay(self):
        steps=self.steps
        decay_period=self.decay_period
        if steps>0 and decay_period is not None and \
                decay_period>0 and steps%decay_period==0:
            self.learning_rate*=self.decay_rate



class AdamOpt(object):
    def __init__(self, learning_rate, model, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.model = model
        self.beta1 = beta1
        self.beta2 = beta2
        self.moments = []
        
        for param in self.model.params:
            self.moments.append(np.zeros_like(param))
        self.velocity = 0.0

    def update(self):

        self.velocity *= self.beta2
        model=self.model
        for param,grad,moment in zip(model.params,model.grads,self.moments):
            g =grad
            moment*=self.beta1
            moment += (1 - self.beta1) * g
            self.velocity += (1 - self.beta2) * (np.linalg.norm(g) ** 2)
        eps = 1e-8
        l_r = self.learning_rate / (np.sqrt(self.velocity) + eps)
        
        for param,moment in zip(self.model.params,self.moments):
            param -= l_r * moment
    
    def clear_grads(self):
        for grad in self.model.grads:
            grad[:]=0

