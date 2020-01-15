from keras.models import Model 
import keras.optimizers as keras_opt
import Evolutionary_Optimizers 
import numpy as np


# Dictionary of the new evolutionay optimizers 
optimizers={
    "ga" : Evolutionary_Optimizers.GA,
    "nga" : Evolutionary_Optimizers.NGA,
    "cma" : Evolutionary_Optimizers.CMA,
    "bfgs" : Evolutionary_Optimizers.BFGS,
    "ceressolver" : Evolutionary_Optimizers.CeresSolver
}


class GAModel(Model):


    """  
    GAModel forewards all tasks to keras if the optimizer is NOT genetic. In case the optimizer is genetic, fitting methods 
    from Evolutionary_Optimizers.py are being used. 
    """


    # Initialization is not being at this point and hence superfluous 
    def __init__(self, input_tensor, output_tensor, **kwargs):
        super(GAModel, self).__init__(input_tensor, output_tensor, **kwargs)

    
    def compile(self, optimizer, **kwargs): # can we remove anything but optimizer from these inputs?
        
        # Checks wether the optimizer is genetic or not and creates an optimizer instance in case a string type was given as input
        self.is_genetic = False
        # Checks (if the optimizer input is a string) whether it is in the 'optimizers' dictionary
        if isinstance(optimizer, str):
            optimizer = optimizer.lower()
            if optimizer in optimizers.keys():
                optimize = optimizers[optimizer]()
                self.is_genetic = True
        # Checks if the optimizer is an evolutionary strategy    
        elif isinstance(optimizer, Evolutionary_Optimizers.EvolutionaryStragegies):
            optimize = optimizer
            self.is_genetic = True
        self.optimizer_instance = optimize

        # If the optimizer is genetic, compile using keras while setting a random (keras supported) gradient descent optimizer
        if self.is_genetic:
            super().compile(optimizer='rmsprop', **kwargs)
            self.optimizer_instance.prepare_during_compile(model = self)
        else: 
            super().compile(optimizer=optimizer, **kwargs)
    
   
    # If the optimizer is genetic the fitting precedure consists of executing run_stop for the given number of epochs
    def fit(self, x=None, y=None, validation_data=None, epochs=1, verbose = 0, **kwargs):
        if self.is_genetic:
            # Validation data is currently not being used!!
            if validation_data is not None:
                x_val = validation_data[0]
                y_val = validation_data[1]

            for epoch in range(epochs):
                score, best_mutant = self.optimizer_instance.run_step( model = self, x=x, y=y )
                self.set_weights(best_mutant)

                if epoch == 0:
                    # use numpy array becuase list can only give one type of score at each epoch 
                    # (no different values in the same row of the list (probably just something I don't know how) )
                    history_temp = np.zeros(( len(score), epochs ))
                if verbose == 1:
                    print('epoch: ', epoch+1, '/', epochs, ', train_accuracy: ', score[1], 'sigma:', self.optimizer_instance.sigma ) 
                
                for i in range( len(score) ):
                    history_temp[i][epoch] = score[i]

            # keras fit outputs history as dict with dict.values list type, so we convert numpy->list as well
            history_temp = history_temp.tolist()
            history = {}
            for i in range ( len(score) ):
                history[ self.metrics_names[i] ] = history_temp[i] 
            
            return returnvalues(self, history, epochs, validation_data)
          
        else: 
            # if not is_gentic, let keras deal with the fit.
            return super().fit(x=x, y=y, validation_data=validation_data, epochs=epochs, verbose=verbose, **kwargs)


    # Evaluate is done by keras
    def evaluate(self, *args, **kwargs):
        return super().evaluate(*args, **kwargs)


class returnvalues:
    def __init__(self, model=None, history=None, epochs=None, validation_data=None, params=None):
        self.history = history
        self.epoch = [*range(epochs)]
        self.model = model
        self.validation_data = validation_data
        self.params = params