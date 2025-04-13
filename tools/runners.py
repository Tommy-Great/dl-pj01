
class Runner(object):
    def __init__(self, model, optimizer, metric):
        self.model = model
        self.optimizer = optimizer
        self.metric = metric
        self.step_losses = []
        self.epoch_losses = []
        self.dev_losses = []
        self.dev_scores = []
        self.global_step=0

    def train(self, train_loader, dev_loader, test_loader=None,**kwargs):
        num_epochs = kwargs.get('num_epochs', 0)
        eval_steps = kwargs.get('eval_steps', 0)
        log_steps = kwargs.get('log_steps', 0)
        log_first = kwargs.get('log_first', False)
        eval_first = kwargs.get('eval_first', False)
        eval_last = kwargs.get('eval_last', False)
        test_log = kwargs.get('test_log', True)
        
        
        all_steps = len(train_loader) * num_epochs
        
        if 0< eval_steps <1:
            eval_steps=int(eval_steps * all_steps)
        eval_steps=int(eval_steps)
        if 0<log_steps<1 :
            log_steps=int(log_steps * all_steps)
        log_steps = int(log_steps)

        global_step = self.global_step
        all_steps += global_step
        
        if eval_first:
            self.sub_evaluate(dev_loader, 0, num_epochs, global_step, all_steps)

        for epoch in range(1,num_epochs+1):
            epoch_loss = 0
            for data in train_loader:
                # self.model.train()
                global_step += 1
                X, y = data
                step_loss = self.model.forward(X,y)

                epoch_loss += step_loss
                self.step_losses.append(step_loss)
                
                # self.optimizer.clear_grads()
                self.model.backward()
                # self.optimizer.update()
                

                if log_steps > 0 and ((log_first and global_step == 1)or global_step % log_steps == 0):
                    print(f'[Train] epochs: {epoch}/{num_epochs}, steps: {global_step}/{all_steps} \n'
                          f'\ttrain loss: {step_loss:.4f}')

                if eval_steps > 0 and global_step % eval_steps == 0:
                    self.sub_evaluate(dev_loader, epoch, num_epochs, global_step, all_steps)

                self.global_step += 1
            self.epoch_losses.append(epoch_loss / len(train_loader))
        self.global_step = global_step

        if eval_steps > 0 and eval_last and all_steps % eval_steps != 0:
            self.sub_evaluate(dev_loader, num_epochs, num_epochs, global_step, all_steps)

        test_loss, test_score=None,None
        if test_loader is not None:
            test_loss, test_score = self.evaluate(test_loader)
            if test_log:
                print(' ')
                print(f'[Test] test_loss: {test_loss:.4f}, test_score: {test_score:.4f}')
                print('training done!\n')
        return test_loss, test_score
    
    def sub_evaluate(self,dev_loader,epoch,num_epochs,global_step,all_steps):
        dev_loss, dev_score = self.evaluate(dev_loader)
        self.dev_losses.append((global_step, dev_loss))
        self.dev_scores.append((global_step, dev_score))
        # print(f'[Dev] epochs: {epoch}/{num_epochs}, steps: {global_step}/{all_steps} \n'
        #       f'\tdev_loss: {dev_loss:.4f}, dev_score: {dev_score:.4f}')
        print(f'[Dev] steps: {global_step}/{all_steps}, dev_score: {dev_score:.4f}')

    def evaluate(self, eval_loader):
        eval_loss = 0
        for data in eval_loader:
            # self.model.eval()
            X, y = data
            step_loss = self.model.forward(X,y)
            preds = self.model.get_predict()
            eval_loss += step_loss
            self.metric.judge(preds, y)
        eval_loss = eval_loss / len(eval_loader)
        eval_score = self.metric.summary()

        return eval_loss, eval_score
    

        