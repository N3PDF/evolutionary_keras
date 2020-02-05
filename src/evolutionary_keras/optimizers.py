"""
    This module contains different Evolutionary Optimizers
"""

from abc import abstractmethod
from copy import deepcopy
import numpy as np
from keras.optimizers import Optimizer
from evolutionary_keras.utilities import get_number_nodes, parse_eval
import cma


class EvolutionaryStrategies(Optimizer):
    """ Parent class for all Evolutionary Strategies
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.shape = None
        self.non_training_weights = []

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

    @abstractmethod
    def run_step(self, x, y):
        """ Performs a step of the optimizer.
        Returns the current score of the best mutant
        and its new weights """
        score = 0
        selected_parent = None
        return score, selected_parent

    def get_updates(self, loss, params):
        """ Capture Keras get_updates method """
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
        `sigma_original`: int
            Allows adjusting the original sigma
        `population_size`: int
            Number of mutants to be generated per iteration
        `mutation_rate`: float
            Mutation rate
    """

    # In case the user wants to adjust sigma_original
    # population_size or mutation_rate parameters the NGA method has to be initiated
    def __init__(
        self, *args, sigma_original=15, population_size=80, mutation_rate=0.05, **kwargs
    ):
        self.sigma_original = sigma_original
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.sigma = sigma_original
        self.n_nodes = 0
        self.n_generations = 1

        super(NGA, self).__init__(*args, **kwargs)

    # Only works if all non_trainable_weights come after all trainable_weights
    # perhaps part of the functionality (getting shape) can be moved to ES
    def get_shape(self):
        """ Study the model to get the shapes of all trainable weight as well
        as the number of nodes. It also saves a reference to the non-trainable weights
        in the system.

        Returns
        -------
            `weight_shapes`: a list of the shapes of all trainable weights
        """
        # Initialize number of nodes
        self.n_nodes = 0
        # Get trainable weight from the model and their shapes
        trainable_weights = self.model.trainable_weights
        weight_shapes = [weight.shape.as_list() for weight in trainable_weights]
        # TODO: eventually we should save here a reference to the layer and their
        # corresponding weights, since the nodes are the output of the layer
        # and the weights the corresponding to that layer
        for layer in self.model.layers:
            self.n_nodes += get_number_nodes(layer)
        # TODO related to previous TODO: non trianable weights should not be important
        self.non_training_weights = self.model.non_trainable_weights
        return weight_shapes

    def create_mutants(self, change_both_wb=True):
        """
        Takes the current state of the network as the starting mutant and creates a new generation
        by performing random nodal mutations.
        By default, from a layer dense layer, only weights or biases will be mutated.
        In order to mutate both set `change_both_wb` to True
        """
        # TODO here we should get only trainable weights
        parent_mutant = self.model.get_weights()
        # Find out how many nodes are we mutating
        nodes_to_mutate = int(self.n_nodes * self.mutation_rate)

        mutants = [parent_mutant]
        for _ in range(self.population_size):
            mutant = deepcopy(parent_mutant)
            mutated_nodes = []
            # Select random nodes to mutate for this mutant
            # TODO seed numpy random at initialization time
            # Note that we might mutate the same node several times for the same mutant
            for _ in range(nodes_to_mutate):
                mutated_nodes.append(np.random.randint(self.n_nodes))

            for i in mutated_nodes:
                # Find the nodes in their respective layers
                count_nodes = 0
                layer = 1
                # TODO HERE WEIGHT-BIAS-WEIGHT-BIAS IS ALSO ASSUMED BY THE +=2
                # Once again, this is related to the previous TODO
                while count_nodes <= i:
                    count_nodes += self.shape[layer][0]
                    layer += 2
                layer -= 2
                count_nodes -= self.shape[layer][0]
                node_in_layer = i - count_nodes

                # Mutate weights and biases by adding values from random distributions
                sigma_eff = self.sigma * (self.n_generations ** (-np.random.rand()))
                if change_both_wb:
                    randn_mutation = sigma_eff * np.random.randn(
                        self.shape[layer - 1][0]
                    )
                    mutant[layer - 1][:, node_in_layer] += randn_mutation
                    mutant[layer][node_in_layer] += sigma_eff * np.random.randn()
                else:
                    change_weight = np.random.randint(2, dtype="bool")
                    if change_weight:
                        randn_mutation = sigma_eff * np.random.randn(
                            self.shape[layer - 1][0]
                        )
                        mutant[layer - 1][:, node_in_layer] += randn_mutation
                    else:
                        mutant[layer][node_in_layer] += sigma_eff * np.random.randn()
            mutants.append(mutant)
        return mutants

    # RESULTS DO IMPROVE WHEN TRAINING BASED ON SELECTING HIGHEST ACCURACY, BUT NOT IF SELECTION IS BASED ON LOWEST LOSS
    # ALSO, THE ACCURACY CHANGES AFTER THE WEIGHTS OF THE BEST PERFROMING MODEL ARE LOADED INTO THE MODEL AGAIN, IS THERE
    # SOME RANDOMIZATION DONE BY TENSORFLOW? BUT THE MODEL DOES NOT INCLUDE THINGS SUCH AS DROPOUT LAYER. SHOULD WE USE DIFFERENT
    # METHOD TO SAVE AND LOAD MODELS?

    # Evalutates all mutantants of a generationa and ouptus loss and the single best performing mutant of the generation
    def evaluate_mutants(self, mutants, x=None, y=None, verbose=0):
        """ Evaluates all mutants of a generation and select the best one.

        Parameters
        ----------
            `mutants`: list of all mutants for this generation

        Returns
        -------
            `loss`: loss of the best performing mutant
            `best_mutant`: best performing mutant
        """
        best_loss = self.model.evaluate(x=x, y=y, verbose=verbose)
        best_loss_val = parse_eval(best_loss)
        best_mutant = mutants[0]
        new_mutant = False
        for mutant in mutants[1:]:
            # replace weights of the input model by weights generated
            # TODO related to the other todos, eventually this will have to be done
            # in a per-layer basis
            self.model.set_weights(mutant)
            loss = self.model.evaluate(x=x, y=y, verbose=False)
            loss_val = parse_eval(loss)

            if loss_val < best_loss_val:
                best_loss_val = loss_val
                best_loss = loss
                best_mutant = mutant
                new_mutant = True

        # if none of the mutants have performed better on the training data than the original mutant
        # reduce sigma
        if not new_mutant:
            self.n_generations += 1
            self.sigma = self.sigma_original / self.n_generations
        return best_loss, best_mutant

    # --------------------- only the functions below are called in EvolModel ---------------------
    def run_step(self, x, y):
        """ Wrapper to run one single step of the optimizer"""
        # Initialize training paramters
        mutants = self.create_mutants()
        score, selected_parent = self.evaluate_mutants(mutants, x=x, y=y)

        return score, selected_parent


class CMA(EvolutionaryStrategies):
    """
    From http://cma.gforge.inria.fr/:
    "The CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is an evolutionary algorithm for 
    difficult non-linear non-convex black-box optimisation problems in continuous domain."
    The work-horse of this class is the cma package developed and maintained by Nikolaus Hansen
    (see https://pypi.org/project/cma/), this class allows for convenient implementation within 
    the keras environment. 

    Parameters
    ----------
        `sigma_init`: int
            Allows adjusting the initial sigma
    """

    def __init__(self, sigma_init=1, *args, **kwargs):
        self.sigma_init = sigma_init
        self.shape = None
        self.length_flat_layer = None
        self.weightscopy = None
        self.trainable_weights_names = None
        super(CMA, self).__init__(*args, **kwargs)

    def get_shape(self):
        # we do all this to keep track of the position of the trainable weights
        self.trainable_weights_names = [
            weights.name for weights in self.model.trainable_weights
        ]

        if self.trainable_weights_names == []:
            raise TypeError("The model does not have any trainable weights!")

        self.shape = []
        self.weightscopy = deepcopy(self.model.weights)
        remove_these_weights = []
        for i in range(len(self.model.weights)):
            weight = self.model.weights[i]
            if weight.name not in self.trainable_weights_names:
                remove_these_weights.append(i)
        number_to_remove = len(remove_these_weights)
        for i in range(number_to_remove):
            del self.weightscopy[remove_these_weights[number_to_remove - 1 - i]]
        self.shape = [weight.shape.as_list() for weight in self.weightscopy]
        return self.shape

    """ 
    'weights_per_layer' creates 'self.lengt_flat_layer' which is a list conatining
    the numer of weights in each layer of the network. 
    """

    def weights_per_layer(self):
        """
        The first values of 'self.length_flat_layer' is set to 0 which is helpful
        in determining the range of weights in the function 'undo_flatten'. 
        """
        self.length_flat_layer = [0]
        for layer in range(len(self.shape)):
            flat_layer = np.reshape(self.weightscopy[layer].numpy(), [-1])
            self.length_flat_layer.append(len(flat_layer))

    """
    'flatten' returns a 1 dimensional list of all weights in the keras model 
    """

    def flatten(self):
        weights = self.weightscopy
        flattened_weights = []
        for layer in range(len(weights)):
            flattened_weights.append(np.reshape(weights[layer].numpy(), [-1]))
        flattened_weights = np.concatenate(flattened_weights)
        return flattened_weights

    """ 
    'undo_flatten' does the inverse of 'flatten': it takes a 1 dimensional input
    and returns a weight structure that can be loaded into the model. 
    """

    def undo_flatten(self, flattened_weights):
        new_weights = []
        for i in range(len(self.shape)):
            flat_layer = flattened_weights[
                self.length_flat_layer[i] : self.length_flat_layer[i]
                + self.length_flat_layer[i + 1]
            ]
            new_weights.append(np.reshape(flat_layer, self.shape[i]))

        ordered_names = [
            weight.name for layer in self.model.layers for weight in layer.weights
        ]

        new_parent = deepcopy(self.model.get_weights())
        i = 0
        for weight in self.trainable_weights_names:
            location_weight = ordered_names.index(weight)
            new_parent[location_weight] = new_weights[i]
            i += 1

        return new_parent

    """     
    As one might have noticed, 'CMA' does not allow the user to set a number of
    epochs, as this is dealth with by 'cma'. The default 'epochs' in EvolModel
    is one, meaning 'run step' is only called once during training. 
    """

    def run_step(self, x, y):
        """ Wrapper to the optimizer"""

        """ 
        Get the nubmer of weights in each keras layer 
        """
        self.weights_per_layer()

        """ 
        The function that 'cma' aims to minimize 
        """

        def minimizethis(flattened_weights):
            weights = self.undo_flatten(flattened_weights)
            self.model.set_weights(weights)
            loss = parse_eval(self.model.evaluate(x=x, y=y, verbose=0))
            return loss

        """     
        Run the minimization and return the ultimatly selected 1 dimensional
        layer of weights 'xopt'. 
        """
        xopt, es = cma.fmin2(minimizethis, self.flatten(), self.sigma_init)

        """ 
        Transform 'xopt' to the models' weight shape. 
        """
        selected_parent = self.undo_flatten(xopt)

        """ 
        Determine the ultimatly selected mutants' performance on the training data. 
        """
        self.model.set_weights(selected_parent)
        score = self.model.evaluate(x=x, y=x, verbose=0)
        score = parse_eval(score)

        return score, selected_parent


class BFGS(EvolutionaryStrategies):
    pass


class CeresSolver(EvolutionaryStrategies):
    pass
