class StochasticGradientDescent:
    def __init__(self , learning_rate = 0.1 , decay =0.0 ,momentum =0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations =0
        self.momentum = momentum



    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate =(self.learning_rate * \
                1. /(1. + self.decay * self.iterations )
            )
        self.iterations +=1
    
    def update_params(self , layer):
        layer.weights += -self.current_learning_rate * layer.dweights
        layer.biases += -self.current_learning_rate * layer.dbiases