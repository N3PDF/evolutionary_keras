"""
    This module contains different Evolutionary Optimizers
"""

from abc import abstractmethod
from copy import deepcopy

import cma
import numpy as np
from tensorflow.keras.optimizers import Optimizer

from evolutionary_keras.utilities import compatibility_numpy, get_number_nodes


class EvolutionaryStrategies(Optimizer):
    """ Parent class for all Evolutionary Strategies
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.model = None
        self.shape = []
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

    def _resource_apply_dense(self):
        """ Override """

    def _resource_apply_sparse(self):
        """ Override """



class NGA(EvolutionaryStrategies):
    """
    The Nodal Genetic Algorithm (NGA) is similar to the regular GA, but this time a number
    of nodes (defined by the mutation_rate variable) are selected at random and
    only the weights and biases corresponding to the selected nodes are mutated by
    adding normally distributed values with normal distrubtion given by sigma.

    Parameters
    ----------
        `sigma_init`: int
            Allows adjusting the original sigma
        `population_size`: int
            Number of mutants to be generated per iteration
        `mutation_rate`: float
            Mutation rate
    """

    # In case the user wants to adjust sigma_init
    # population_size or mutation_rate parameters the NGA method has to be initiated
    def __init__(
        self,
        name="NGA",
        sigma_init=15,
        population_size=80,
        mutation_rate=0.05,
        **kwargs
    ):
        self.sigma_init = sigma_init
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.sigma = sigma_init
        self.n_nodes = 0
        self.n_generations = 1

        super(NGA, self).__init__(name, **kwargs)

    def get_config(self):
        config = super(NGA, self).get_config()
        config.update(
            {
                "sigma_init": self.sigma_init,
                "population_size": self.population_size,
                "mutation_rate": self.mutation_rate,
            }
        )
        return config

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

    # Evalutates all mutantants of a generationa and ouptus loss and the single best performing
    # mutant of the generation
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
        best_loss = self.model.evaluate(x=x, y=y, verbose=verbose, return_dict=True)
        training_metric = next(iter(best_loss))
        best_loss_val = best_loss[training_metric]
        best_mutant = mutants[0]
        new_mutant = False
        for mutant in mutants[1:]:
            # replace weights of the input model by weights generated
            # TODO related to the other todos, eventually this will have to be done
            # in a per-layer basis
            self.model.set_weights(mutant)
            loss = self.model.evaluate(x=x, y=y, verbose=False, return_dict=True)
            loss_val = loss[training_metric]

            if loss_val < best_loss_val:
                best_loss_val = loss_val
                best_loss = loss
                best_mutant = mutant
                new_mutant = True

        # if none of the mutants have performed better on the training data than the original mutant
        # reduce sigma
        if not new_mutant:
            self.n_generations += 1
            self.sigma = self.sigma_init / self.n_generations
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
        `population_size`: int
            Number of mutants to be generated per iteration
        `target_value`: float
            Stops the minimizer if the target loss is achieved
        `max_evaluations`: int
            Maximimum total number of mutants tested during optimization
    """

    def __init__(
        self,
        name="CMA",
        sigma_init=0.3,
        target_value=None,
        population_size=None,
        max_evaluations=None,
        verbosity=1,
        **kwargs
    ):
        """
        `CMA` does not allow the user to set a number of generations (epochs),
        as this is dealth with by the `cma` package.
        The default `epochs` in EvolModel is 1, meaning `run step` called once during training.
        """
        self.sigma_init = sigma_init
        self.length_flat_layer = None
        self.trainable_weights_names = None
        self.verbosity = verbosity
        if verbosity == 0:
            self.verbosity = -9
        else:
            self.verbosity = 1
        self.max_evaluations = max_evaluations

        # These options do not all work as advertised
        self.options = {"verb_log": 0, "verbose": self.verbosity, "verb_disp": 1000}
        if target_value:
            self.options["ftarget"] = target_value
        if population_size:
            self.options["popsize"] = population_size

        super(CMA, self).__init__(name, **kwargs)

    def get_config(self):
        config = super(CMA, self).get_config()
        config.update(
            {
                "sigma_init": self.sigma_init,
                "target_value": self.target_value,
                "population_size": self.population_size,
                "max_evaluations": self.max_evaluations,
            }
        )
        return config

    def on_compile(self, model):
        """ Function to be called by the model during compile time. Register the model `model` with
        the optimizer.
        """
        # Here we can perform some checks as well
        self.model = model
        self.shape = self.get_shape()

    def get_shape(self):
        # we do all this to keep track of the position of the trainable weights
        self.trainable_weights_names = [
            weights.name for weights in self.model.trainable_weights
        ]

        if self.trainable_weights_names == []:
            raise TypeError("The model does not have any trainable weights!")

        self.shape = [weight.shape.as_list() for weight in self.model.trainable_weights]
        return self.shape

    def weights_per_layer(self):
        """
        'weights_per_layer' creates 'self.lengt_flat_layer' which is a list conatining the numer of
        weights in each layer of the network.
        """

        # The first values of 'self.length_flat_layer' is set to 0 which is helpful in determining
        # the range of weights in the function 'undo_flatten'.
        self.length_flat_layer = [
            len(np.reshape(weight.numpy(), [-1]))
            for weight in self.model.trainable_weights
        ]
        self.length_flat_layer.insert(0, 0)

    def flatten(self):
        """
        'flatten' returns a 1 dimensional list of all weights in the keras model.
        """
        # The first values of 'self.length_flat_layer' is set to 0 which is helpful in determining
        # the range of weights in the function 'undo_flatten'.
        flattened_weights = []
        self.length_flat_layer = []
        self.length_flat_layer.append(0)
        for weight in self.model.trainable_weights:
            a = np.reshape(compatibility_numpy(weight), [-1])
            flattened_weights.append(a)
            self.length_flat_layer.append(len(a))

        flattened_weights = np.concatenate(flattened_weights)

        return flattened_weights

    def undo_flatten(self, flattened_weights):
        """
        'undo_flatten' does the inverse of 'flatten': it takes a 1 dimensional input and returns a
        weight structure that can be loaded into the model.
        """
        new_weights = []
        for i, layer_shape in enumerate(self.shape):
            flat_layer = flattened_weights[
                self.length_flat_layer[i] : self.length_flat_layer[i]
                + self.length_flat_layer[i + 1]
            ]
            new_weights.append(np.reshape(flat_layer, layer_shape))

        ordered_names = [
            weight.name for layer in self.model.layers for weight in layer.weights
        ]

        new_parent = deepcopy(self.model.get_weights())
        for i, weight in enumerate(self.trainable_weights_names):
            location_weight = ordered_names.index(weight)
            new_parent[location_weight] = new_weights[i]

        return new_parent

    def run_step(self, x, y):
        """ Wrapper to the optimizer"""

        # Get the nubmer of weights in each keras layer
        x0 = self.flatten()

        # If max_evaluations is not set manually, use the number advised in arXiv:1604.00772
        if self.max_evaluations is None:
            self.options["maxfevals"] = 1e3 * len(x0) ** 2
        else:
            self.options["maxfevals"] = self.max_evaluations

        # minimizethis is function that 'cma' aims to minimize
        def minimizethis(flattened_weights):
            weights = self.undo_flatten(flattened_weights)
            self.model.set_weights(weights)
            loss = self.model.evaluate(x=x, y=y, verbose=0, return_dict=True)
            training_metric = next(iter(loss))
            return loss[training_metric]

        # Run the minimization and return the ultimatly selected 1 dimensional layer of weights
        # 'xopt'.
        xopt = (
            cma.CMAEvolutionStrategy(x0, self.sigma_init, self.options)
            .optimize(minimizethis)
            .result[0]
        )

        # Transform 'xopt' to the models' weight shape.
        selected_parent = self.undo_flatten(xopt)

        # Determine the ultimatly selected mutants' performance on the training data.
        self.model.set_weights(selected_parent)
        loss = self.model.evaluate(x=x, y=y, verbose=0, return_dict=True)
        return loss, selected_parent
