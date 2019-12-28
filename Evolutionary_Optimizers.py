import numpy as np
import math
import copy

from keras.models import Model
from keras.optimizers import Optimizer


class GA(Optimizer):
    pass


class NGA(Optimizer):

    """
    The Nodal Genetic Algorithm (NGA) is similar to the GA, but this time a number
    of nodes (defined by the mutation_rate variable) are selected at random and 
    only the weights and biases corresponding to the selected nodes are mutated by 
    adding normally distributed values with normal distrubtion given by sigma. 
    """

    def __init__(self, population_size = 80, mutation_rate = 0.05, sigma_original = 15, *args, **kwargs):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.sigma_original = sigma_original
        super(NGA, self).__init__(*args, **kwargs)

    is_genetic = True


    def get_updates(self, loss, params):
        pass


    def initialize_training(self, training_model, population_size = 10, mutation_rate = 0.05, sigma_original = 10, x_train=None, y_train=None):
        if hasattr(self, 'self.population_size') != True:
            self.population_size = population_size
            self.mutation_rate = mutation_rate
            self.sigma_original = sigma_original

        self.N_generations = 1
        self.sigma=self.sigma_original
        self.accuracy = training_model.evaluate(x_train,y_train)[0]


    def get_shape(self, model):
        training_model = model
        original_weights = []
        for i in range(6):
            original_weights.append(training_model.get_weights()[i])

        """ self.non_training_weights = []
        for i in range(6, len( training_model.get_weights() ) ):
            self.non_training_weights.append( training_model.get_weights()[i] ) """

        weight_size = []
        self.N_nodes = 0
        for i in range( len( original_weights ) ):
            if i%2 == 1:                                                    # Biases
                a1 = len( original_weights[i] )  
                self.N_nodes += a1  
                weight_size.append( a1 )            
            if i%2 == 0:                                                    # Weights
                c = len( original_weights[i] ) 
                r = len( original_weights[i][0] )
                weight_size.append( [ c, r ] )
        return weight_size


    def create_mutants(self, training_model, weight_size, ):
        mutant = []
        mutant.append(training_model.get_weights())

        for k in range(self.population_size):
            mutant.append(copy.deepcopy( mutant[0] ) )
            mutated_nodes = []
            for i in range( math.floor(self.N_nodes * self.mutation_rate ) ):
                mutated_nodes.append( math.floor( self.N_nodes*np.random.rand() ) )
    
            for i in mutated_nodes:
                count_nodes = 0
                layer = 1
                while count_nodes <= i:
                    count_nodes += weight_size[layer] 
                    layer += 2
                layer -= 2
                count_nodes -= weight_size[layer]
                node_in_layer = i - count_nodes 

                sigma_eff = self.sigma * ( self.N_generations ** ( -np.random.rand() ) )
                bias_or_weights = math.floor(2 * np.random.rand() )
                if bias_or_weights == 0:
                    randn_mutation = sigma_eff * np.random.randn( weight_size[layer-1][0] ) 
                    mutant[k+1][layer-1][:,node_in_layer] += randn_mutation
                else:
                    mutant[k+1][layer][node_in_layer] += sigma_eff * np.random.randn() 
                
                """ mutant[k+1].append(self.non_training_weights) """

        return mutant


    def evaluate_mutants(self, training_model, mutant, x_train=None, y_train=None):
        most_accurate_model = 0   
        for i in range(self.population_size+1):
                training_model.set_weights(mutant[i])
                accuray_new=training_model.evaluate(verbose=False, x=x_train, y=y_train)[0]
                if(accuray_new < self.accuracy):
                    self.accuracy = accuray_new
                    most_accurate_model = i


        if(most_accurate_model == 0):
            self.N_generations += 1
            self.sigma = self.sigma_original/self.N_generations
 
        training_model.set_weights(mutant[most_accurate_model])

        out = training_model.evaluate(verbose=False, x=x_train, y=y_train)

        return out, mutant[most_accurate_model]

    # Not being used 
    def kill_mutants(self):
        mutant = []
        return mutant



class CMA(Optimizer):
    pass