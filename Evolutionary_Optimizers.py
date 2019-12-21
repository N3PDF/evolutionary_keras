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

    is_genetic = True

    def __init__(self, population_size = 80, sigma_init = 10, mutation_rate = 0.05, *args, **kwargs):
        self.population_size = population_size
        self.sigma_init = sigma_init
        self.mutation_rate = mutation_rate


    def get_updates(self):
        return params


    def import_model(self):
        self.model = Model
        return model


    def get_shape(self, model):
        original_weights = []
        for i in range(6):
            original_weights.append(training_model.get_weights()[i])

        non_training_weights = []
        for i in range(6, len( training_model.get_weights() ) ):
            non_training_weights.append( training_model.get_weights()[i] )

        self.weight_size = []
        N_nodes = 0
        for i in range( len( original_weights ) ):
            if i%2 == 1:                                                    # Biases
                a1 = len( original_weights[i] )  
                N_nodes += a1  
                self.weight_size.append( a1 )            
            if i%2 == 0:                                                    # Weights
                c = len( original_weights[i] ) 
                r = len( original_weights[i][0] )
                self.weight_size.append( [ c, r ] )
        return self.weight_size


    def create_mutants(self, model):
        mutant.append(training_model.get_weights())

        for k in range(population_size):
            mutant.append(copy.deepcopy( mutant[0] ) )
            mutated_nodes = []
            for i in range( math.floor(N_nodes * mutation_rate ) ):
                mutated_nodes.append( math.floor( N_nodes*np.random.rand() ) )
    
            for i in mutated_nodes:
                count_nodes = 0
                layer = 1
                while count_nodes <= i:
                    count_nodes += weight_size[layer] 
                    layer += 2
                layer -= 2
                count_nodes -= weight_size[layer]
                node_in_layer = i - count_nodes 

                sigma_eff = sigma * ( N_generations ** ( -np.random.rand() ) )
                bias_or_weights = math.floor(2 * np.random.rand() )
                if bias_or_weights == 0:
                    randn_mutation = sigma_eff * np.random.randn( weight_size[layer-1][0] ) 
                    mutant[k+1][layer-1][:,node_in_layer] += randn_mutation
                else:
                    mutant[k+1][layer][node_in_layer] += sigma_eff * np.random.randn() 
                
                mutant[k+1].append(non_training_weights)

        return mutant


    def evaluate_mutants(self):
        for i in range(population_size+1):
                training_model.set_weights(mutant[i])
                accuray_new=training_model.evaluate(verbose=False)["loss"]
                if(accuray_new < accuracy):
                    accuracy = accuray_new
                    most_accurate_model = i

        if(most_accurate_model == 0):
            N_generations += 1
            sigma = sigma_original/N_generations
 
        training_model.set_weights(mutant[most_accurate_model])

        out = training_model.evaluate(verbose=False)

        return out


    def kill_mutants(self):
        mutant = []



class CMA(Optimizer):
    pass



# Aliases
ga = GA
nga = NGA
cma = CMA

all_classes={}
all_classes['ga'] = GA()
all_classes['nga'] = NGA()