"""
    This module contains different Evolutionary Optimizers
"""

from abc import abstractmethod
import numpy as np
import math
from copy import deepcopy
from keras.optimizers import Optimizer


class EvolutionaryStrategies(Optimizer):
    """ Parent class for all Evolutionary Strategies
    """

    def __init__(self, *args, **kwargs):
        self.model = None
        self.shape = None

    @abstractmethod
    def get_shape(self):
        """ Gets the shape of the weights to train """
        shape = None
        return shape


    def on_compile(self, model):
        """ Function to be called by the model during compile time.
        Register the model `model` with the optimizer.
        """
        # Here we can perform some checks as well
        self.model = model
        self.shape = self.get_shape()

    def get_updates(self, loss, params):
        pass


class GA(EvolutionaryStrategies):
    pass


class NGA(EvolutionaryStrategies):
    """
    The Nodal Genetic Algorithm (NGA) is similar to the regular GA, but this time a number
    of nodes (defined by the mutation_rate variable) are selected at random and
    only the weights and biases corresponding to the selected nodes are mutated by
    adding normally distributed values with normal distrubtion given by sigma.

    Parameters
    ----------
        `sigma_original`: 
        `population_size`:
        `mutation_rate`:
    """

    # In case the user wants to adjust sigma_original, population_size or mutation_rate parameters the NGA method has to be initiated
    def __init__(
        self, sigma_original=15, population_size=80, mutation_rate=0.05, *args, **kwargs
    ):
        self.sigma_original = sigma_original
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.has_init_variables = False
        self.sigma = sigma_original
        self.n_nodes = 0

        super(NGA, self).__init__(*args, **kwargs)

    # Only works if all non_trainable_weights come after all trainable_weights
    # perhaps part of the functionality (getting shape) can be moved to ES
    def get_shape(self):
        # Initialize number of nodes
        self.n_nodes = 0
        # Get trainable weight from the model and their shapes
        trainable_weights = self.model.trainable_weights
        weight_shapes = [weight.shape.as_list() for weight in trainable_weights]
        # TODO: eventually we should save here a reference to the layer and their
        # corresponding weights, since the nodes are the output of the layer
        # and the weights the corresponding to that layer
        for layer in self.model.layers:
            # It is necessary to check whether the layer is trainable
            # AND whether it has any weights which is trainable
            if layer.trainable and layer.trainable_weights:
                # The first dimension is always the batch size so
                # this is a good proxy for the number of nodes
                output_nodes = layer.get_output_shape_at(0)[1:]
                self.n_nodes += sum(output_nodes)
        # TODO related to previous TODO: non trianable weights should not be important
        self.non_training_weights = self.model.non_trainable_weights
        return weight_shapes
       
    def create_mutants(self, shape, change_both_weights_and_biases_of_a_node=True):
        """
        Takes a single mutant as input and creates a new generation by performing random nodal mutations.
        To change node and corresponding weights with probability, set change_both_weights_and_biases_of_a_node=True if change_both_weights_and_biases_of_a_node=False, either a node or the biases of a node will be mutated. """
        model = self.model
        mutant = []
        mutant.append(model.get_weights())
        for k in range(self.population_size):

            mutant.append(deepcopy(mutant[0]))
            mutated_nodes = []

            # select random nodes to mutate
            for i in range(math.floor(self.n_nodes * self.mutation_rate)):
                mutated_nodes.append(math.floor(self.n_nodes * np.random.rand()))

            for i in mutated_nodes:

                # find the nodes in their respective layers
                count_nodes = 0
                layer = 1
                # HERE WEIGHT-BIAS-WEIGHT-BIAS IS ALSO ASSUMED BY THE +=2
                while count_nodes <= i:
                    count_nodes += shape[layer][0]
                    layer += 2
                layer -= 2
                count_nodes -= shape[layer][0]
                node_in_layer = i - count_nodes

                # mutate weights and biases by adding values from random distrubution
                sigma_eff = self.sigma * (self.N_generations ** (-np.random.rand()))
                if change_both_weights_and_biases_of_a_node:
                    randn_mutation = sigma_eff * np.random.randn(shape[layer - 1][0])
                    mutant[k + 1][layer - 1][:, node_in_layer] += randn_mutation
                    mutant[k + 1][layer][node_in_layer] += sigma_eff * np.random.randn()
                else:
                    bias_or_weights = math.floor(2 * np.random.rand())
                    if bias_or_weights == 0:
                        randn_mutation = sigma_eff * np.random.randn(
                            shape[layer - 1][0]
                        )
                        mutant[k + 1][layer - 1][:, node_in_layer] += randn_mutation
                    else:
                        mutant[k + 1][layer][node_in_layer] += (
                            sigma_eff * np.random.randn()
                        )

                # add non_training weights such that 'mutant' now parametrizes the entire model
                mutant[k + 1].append(self.non_training_weights)

        return mutant

    # RESULTS DO IMPROVE WHEN TRAINING BASED ON SELECTING HIGHEST ACCURACY, BUT NOT IF SELECTION IS BASED ON LOWEST LOSS
    # ALSO, THE ACCURACY CHANGES AFTER THE WEIGHTS OF THE BEST PERFROMING MODEL ARE LOADED INTO THE MODEL AGAIN, IS THERE
    # SOME RANDOMIZATION DONE BY TENSORFLOW? BUT THE MODEL DOES NOT INCLUDE THINGS SUCH AS DROPOUT LAYER. SHOULD WE USE DIFFERENT
    # METHOD TO SAVE AND LOAD MODELS?

    # Evalutates all mutantants of a generationa and ouptus loss and the single best performing mutant of the generation
    def evaluate_mutants(
        self, mutant, x=None, y=None,
    ):
        model = self.model

        # most_accurate_model=0 corresponds to setting the input model as most accurate
        most_accurate_model = 0
        for i in range(self.population_size + 1):
            # replace weights of the input model by weights generated using the create_mutants function
            model.set_weights(mutant[i])
            output = model.evaluate(verbose=False, x=x, y=y)
            if self.loss_is_list_type:
                loss_new = output[0]
            else:
                loss_new = output

            if loss_new < self.loss:
                self.loss = loss_new
                most_accurate_model = i

        # if none of the mutants have performed better on the training data than the original mutant, reduce sigma
        if most_accurate_model == 0:
            self.N_generations += 1
            self.sigma = self.sigma_original / self.N_generations

        return output, mutant[most_accurate_model]

    # --------------------- only the functions below are called in GAModel ---------------------

    # Run during compilation to get objects that are unchanged at fit()
    def prepare_during_compile(self, model):
        self.shape = self.get_shape(model=model)

    # Collects the functions defined above to form the fitting step that is to be repeated for a number of epochs
    def run_step(self, x, y):

        # Initialize training paramters
        if self.has_init_variables != True:
            self.N_generations = 1
            self.loss = self.model.evaluate(x=x, y=y, verbose=0)
            if isinstance(self.loss, list):
                self.loss = self.loss[0]
                self.loss_is_list_type = True
            self.has_init_variables = True

        mutant = self.create_mutants(shape=self.shape)
        score, selected_parent = self.evaluate_mutants(mutant=mutant, x=x, y=y)

        return score, selected_parent


class CMA(EvolutionaryStrategies):
    pass


class BFGS(EvolutionaryStrategies):
    pass


class CeresSolver(EvolutionaryStrategies):
    pass
