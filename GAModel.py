# How to deal with MetaModel?

from keras.models import Model 
import keras.optimizers as keras_opt
import Evolutionary_Optimizers as evo_opt

class GAModel(Model):
          
    optimizers={
        "sgd": keras_opt.SGD,
        "rmsprop": keras_opt.RMSprop,
        "adagrad": keras_opt.Adagrad,
        "adadelta": keras_opt.Adadelta,
        "adam": keras_opt.Adam,
        "adamax": keras_opt.Adamax,
        "nadam": keras_opt.Nadam,
        "ga": evo_opt.GA,
        "nga": evo_opt.NGA,
        "cma": evo_opt.CMA,
    }

    def __init__(self, input_tensor, output_tensor, **kwargs):
        super(GAModel, self).__init__(input_tensor, output_tensor, **kwargs)
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def compile(self, loss, metrics, optimizer, is_genetic = False, **kwargs):
        optimizer = optimizer.lower()
        self.myopt = self.optimizers[optimizer]
        self.is_genetic = hasattr(self.myopt, 'is_genetic')
        super().compile(optimizer='rmsprop', loss=loss, metrics=metrics)
             
    def fit(self, x=None, y=None, epochs=10, **kwargs):
        if self.is_genetic:
            for epoch in range(epochs):
                model = self.myopt.import_model(self)
                weights_size = self.myopt.get_shape(self, model=model)
                mutant = self.myopt.create_mutants()
                out = self.myopt.evaluate_mutants()
                self.myopt.kill_mutants()
        else: 
            history = super().fit(x, y, **kwargs)
            return history

    def evaluate(self, x=None, y=None, **kwargs):
        out = super().evaluate(x, y, **kwargs)
        return out