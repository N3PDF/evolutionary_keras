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

    def compile(self, loss, metrics, optimizer, is_genetic = False, **kwargs):
        optimizer = optimizer.lower()
        self.myopt = self.optimizers[optimizer]
        self.is_genetic = hasattr(self.myopt, 'is_genetic')
        super().compile(optimizer='rmsprop', loss=loss, metrics=metrics)
             
    def fit(self, x_train=None, y_train=None, epochs=10, **kwargs):
        if self.is_genetic:
            self.myopt.initialize_training(self=self.myopt, training_model=self, x_train=x_train, y_train=y_train)
            weights_size = self.myopt.get_shape(self=self.myopt, model = self)
            for epoch in range(epochs):
                mutant = self.myopt.create_mutants(self=self.myopt, training_model = self, weight_size=weights_size)
                out_fit = self.myopt.evaluate_mutants(self=self.myopt, training_model = self, mutant = mutant, x_train=x_train, y_train=y_train)
                self.set_weights(out_fit[1])
                print('epoch: ', epoch, '/', epochs, ', accuracy: ', out_fit[0][1]  )             
        else: 
            out_fit = super().fit(x_train, y_train, **kwargs)
        return out_fit[0]

    def evaluate(self, x=None, y=None, **kwargs):
        out_eval = super().evaluate(x=x, y=y, **kwargs)
        return out_eval
