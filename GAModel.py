from keras.models import Model 
from Evolutionary_Optimizers import NGA

class GAModel(Model):
    def __init__(self, input, output, is_genetic = False, *args, **kwargs):
        super(GAModel, self).__init__(input, output, *args, **kwargs)
        #self.input_shape = inputs
        #self.output_shape = output
        #self.is_genetic = is_genetic

    def compile(self, optimizer, loss, metrics, *args, **kwargs ):
        if hasattr('self.optimizer', 'is_genetic'):
            self.optimizer = optimizer
            self.loss = loss
            self.metrics = metrics
            super().compile(optimizer='adam', loss=loss, metrics=metrics)
        else:
            super().compile(optimizer=optimizer, loss=loss, metrics=metrics)


    def fit(self, x=None, y=None, epochs=10, *args, **kwargs):
        if hasattr('self.optimizer', 'is_genetic'):
            self.optimizer.import_model()
            for epoch in range(epochs):
                self.weights_size = self.optimizer.get_shape()
                mutant = self.optimizer.create_mutants()
                out = self.optimizer.evaluate_mutants()
                self.optimizer.kill_mutants()
        else: 
            super().fit(x, y, *args, **kwargs)



""" 
class GAmodel(Model):

    def compile(self, is_genetic = False, ):
        if is_genetic:
            self.is_genetic = True
            self.optimizer = optimizer
            self.model = import_model()




    def fit(self):
        if self.is_not_genetic:
            super().fit()
            return


        self.optimizer.create_mutants(model = self)
        self.optimizer.kill_mutants(model = self)
 """
