""" Implementation of GA Model """

import logging
from keras.models import Model
from keras.callbacks.callbacks import History
import evolutionary_keras.optimizers as Evolutionary_Optimizers
from evolutionary_keras.utilities import parse_eval

log = logging.getLogger(__name__)

# Dictionary of the new evolutionay optimizers
optimizer_dict = {
    "ga": Evolutionary_Optimizers.GA,
    "nga": Evolutionary_Optimizers.NGA,
    "cma": Evolutionary_Optimizers.CMA,
    "bfgs": Evolutionary_Optimizers.BFGS,
    "ceressolver": Evolutionary_Optimizers.CeresSolver,
}

class EvolModel(Model):
    """
    EvolModel forewards all tasks to keras if the optimizer is NOT genetic.
    In case the optimizer is genetic, fitting methods
    from Evolutionary_Optimizers.py are being used.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_genetic = False
        self.opt_instance = None
        self.history = History()

    def parse_optimizer(self, optimizer):
        """ Checks whether the optimizer is genetic
        and creates and optimizer instance in case a string was given
        as input """
        # Checks (if the optimizer input is a string)
        # and whether it is in the 'optimizers' dictionary
        if isinstance(optimizer, str) and optimizer in optimizer_dict.keys():
            opt = optimizer_dict.get(optimizer.lower())
            # And instanciate it with default values
            optimizer = opt()
        # Check whether the optimizer is an evolutionary
        # optimizer
        if isinstance(optimizer, Evolutionary_Optimizers.EvolutionaryStrategies):
            self.is_genetic = True
            self.opt_instance = optimizer
            optimizer.on_compile(self)

    def compile(self, optimizer="rmsprop", **kwargs):
        """ Compile """
        self.parse_optimizer(optimizer)
        # If the optimizer is genetic,
        # compile using keras while setting a random (keras supported) gradient descent optimizer
        if self.is_genetic:
            super().compile(optimizer="rmsprop", **kwargs)
        else:
            super().compile(optimizer=optimizer, **kwargs)

    def perform_genetic_fit(
        self, x=None, y=None, epochs=1, verbose=0, validation_data=None
    ):
        # Prepare the history for the initial epoch
        self.history.on_train_begin()
        # Validation data is currently not being used!!
        if validation_data is not None:
            log.warning(
                "Validation data is not used at the moment by the Genetic Algorithms!!"
            )
        #             x_val = validation_data[0]
        #             y_val = validation_data[1]

        metricas = self.metrics_names
        for epoch in range(epochs):
            # Generate the best mutant
            score, best_mutant = self.opt_instance.run_step(x=x, y=y)
            # Ensure the best mutant is the current one
            self.set_weights(best_mutant)
            if verbose == 1:
                loss = parse_eval(score)
                sigma = self.opt_instance.sigma
                information = f" > epoch: {epoch+1}/{epochs}, {loss} {sigma}"
                log.info(information)
            # Fill keras history
            try:
                history_data = dict(zip(metricas, score))
            except TypeError as e:
                # Maybe score was just one number, evil Keras
                if parse_eval(score) == score:
                    score = [score, score]
                    history_data = dict(zip(metricas, score))
                else:
                    raise e
            self.history.on_epoch_end(epoch, history_data)
        return self.history

    def fit(self, x=None, y=None, validation_data=None, epochs=1, verbose=0, **kwargs):
        """ If the optimizer is genetic, the fitting
        procedure consists on executing `run_step` for the given
        number of epochs """
        if self.is_genetic:
            result = self.perform_genetic_fit(
                x=x,
                y=y,
                epochs=epochs,
                verbose=verbose,
                validation_data=validation_data,
            )
        else:
            result = super().fit(
                x=x,
                y=y,
                validation_data=validation_data,
                epochs=epochs,
                verbose=verbose,
                **kwargs,
            )
        return result
