""" Implementation of GA Model """

import logging

from tensorflow.keras.callbacks import History
from tensorflow.keras.models import Model

import evolutionary_keras.optimizers as Evolutionary_Optimizers

log = logging.getLogger(__name__)

# Dictionary of the new evolutionay optimizers
optimizer_dict = {
    "nga": Evolutionary_Optimizers.NGA,
    "cma": Evolutionary_Optimizers.CMA,
}


class EvolModel(Model):
    """
    EvolModel forwards all tasks to keras if the optimizer is NOT genetic. In case the optimizer is
    genetic, fitting methods from Evolutionary_Optimizers.py are being used.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_genetic = False
        self.opt_instance = None
        self.history_info = History()

    def parse_optimizer(self, optimizer):
        """ Checks whether the optimizer is genetic and creates and optimizer instance in case a
        string was given as input.
        """
        # Checks (if the optimizer input is a string) and whether it is in the 'optimizers'
        # dictionary

        if isinstance(optimizer, str) and optimizer.lower() in optimizer_dict.keys():
            opt = optimizer_dict.get(optimizer.lower())
            # And instanciate it with default values
            optimizer = opt()
            optimizer.on_compile(self)
        # Check whether the optimizer is an evolutionary optimizer
        if isinstance(optimizer, Evolutionary_Optimizers.EvolutionaryStrategies):
            self.is_genetic = True
            self.opt_instance = optimizer
            optimizer.on_compile(self)

    def compile(self, optimizer="rmsprop", **kwargs):
        """ When the optimizer is genetic, compiles the model in keras setting an arbitrary
        keras supported optimizer """
        self.parse_optimizer(optimizer)
        self.history_info.set_model(self)
        if self.is_genetic:
            super().compile(optimizer="rmsprop", **kwargs)
        else:
            super().compile(optimizer=optimizer, **kwargs)

    def perform_genetic_fit(
        self, x=None, y=None, epochs=1, verbose=0, validation_data=None
    ):
        """ 
        Parameters
        ----------
            x: array or list of arrays
                input data
            y: array or list of arrays
                target values
            epochs: int
                number of generations of mutants
            verbose: int
                verbose, prints to log.info the loss per epoch
        """
        # Prepare the history for the initial epoch
        self.history_info.on_train_begin()
        # Validation data is currently not being used!!
        if validation_data is not None:
            log.warning(
                "Validation data is not used at the moment by the Genetic Algorithms!!"
            )

        if isinstance(self.opt_instance, Evolutionary_Optimizers.CMA) and epochs != 1:
            epochs = 1
            log.warning(
                "The optimizer determines the number of generations, epochs will be ignored."
            )

        for epoch in range(epochs):
            # Generate the best mutant
            score, best_mutant = self.opt_instance.run_step(x=x, y=y)

            training_metric = next(iter(score))

            # Ensure the best mutant is the current one
            self.set_weights(best_mutant)
            if verbose == 1:
                loss = score[training_metric]
                information = f" > epoch: {epoch+1}/{epochs}, {loss} "
                log.info(information)

            # Fill keras history
            history_data = score
            self.history_info.on_epoch_end(epoch, history_data)

        return self.history_info

    def fit(self, x=None, y=None, validation_data=None, epochs=1, verbose=0, **kwargs):
        """ If the optimizer is genetic, the fitting procedure consists on executing `run_step` for
        the given number of epochs.
        """
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
                # **kwargs,
            )
        return result
