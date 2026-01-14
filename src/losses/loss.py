import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        pass
    #when on activation is done
    def compute_loss_softmax(self, y_true, y_pred):
        m = y_true.shape[0]
        p = self._softmax(y_pred)
        log_likelihood = -np.log(p[range(m), y_true.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backward_with_repect_softmax(self ,y_true):
        samples =len(y_true)
        self.dinput = self.output.copy()
        self.dinput[range(samples) , y_true] -=1
        self.dinput /= samples
        return self.dinput
      
    
    def forward(self, y_true, y_pred):

        simples = len(y_pred)

        #Clip data to prevent dividion by 0
        #Clip both side to not drag mean toward any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) ==1:
            correct_confidences = y_pred_clipped[
                range(simples),
                y_true
            ]
        #one-hot encodeed labels
        elif len(y_true.shape) ==2:
            correct_confidences =np.sum(
                y_pred_clipped*y_true,
                axis =1
            )

        loss_per_row = -np.log(correct_confidences)
        return loss_per_row


    def calculate(self , y_true ,y_pred):

        simple_losses = self.forward(y_true , y_pred)

        #calculate mean
        average_loss =np.mean(simple_losses)

        return average_loss
