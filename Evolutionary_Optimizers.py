import numpy as np
import math
from copy import deepcopy
from keras.optimizers import Optimizer


class ES(Optimizer):

    """ 
    'ES' contains functions shared by the Evolutionary Strategies (ES) in this file. 
    """
    
    #Can we remove this?
    def get_updates(self, loss, params):
        pass


class GA(ES):
    pass


class NGA(ES):

    """
    The Nodal Genetic Algorithm (NGA) is similar to the regular GA, but this time a number
    of nodes (defined by the mutation_rate variable) are selected at random and 
    only the weights and biases corresponding to the selected nodes are mutated by 
    adding normally distributed values with normal distrubtion given by sigma. 
    """

    # In case the user wants to adjust sigma_original, population_size or mutation_rate parameters the NGA method has to be initiated 
    def __init__(self, sigma_original = 15, population_size = 80, mutation_rate = 0.05, *args, **kwargs):
        self.sigma_original = sigma_original
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.has_init_variables = False
        self.sigma = sigma_original

        super(NGA, self).__init__(*args, **kwargs)


    # Only works if all non_trainable_weights come after all trainable_weights
    # perhaps part of the functionality (getting shape) can be moved to ES
    def get_shape(self, model):
        original_weights = []
        for i in range( len(model.trainable_weights) ):
            original_weights.append(model.get_weights()[i])

        self.non_training_weights = model.non_trainable_weights

        weight_size = []
        self.N_nodes = 0
        for i in range( len( original_weights ) ):
            if i%2 == 1: # Biases
                a1 = len( original_weights[i] )  
                self.N_nodes += a1  
                weight_size.append( a1 )            
            if i%2 == 0: # Weights
                c = len( original_weights[i] ) 
                r = len( original_weights[i][0] )
                weight_size.append( [ c, r ] )
        return weight_size
        

    # Takes a single mutant as input and creates a new generation by performing random nodal mutations
    # To change node and corresponding weights with probability, set change_both_weights_and_biases_of_a_node=True
    # if change_both_weights_and_biases_of_a_node=False, either a node or the biases of a node will be mutated. 
    def create_mutants(self, model, shape, change_both_weights_and_biases_of_a_node = True):
        mutant = []
        mutant.append(model.get_weights())
        for k in range(self.population_size):

            mutant.append(deepcopy( mutant[0] ) )
            mutated_nodes = []

            # select random nodes to mutate
            for i in range( math.floor(self.N_nodes * self.mutation_rate ) ):
                mutated_nodes.append( math.floor( self.N_nodes*np.random.rand() ) )

            for i in mutated_nodes:

                # find the nodes in their respective layers
                count_nodes = 0
                layer = 1
                while count_nodes <= i:
                    count_nodes += shape[layer] 
                    layer += 2
                layer -= 2
                count_nodes -= shape[layer]
                node_in_layer = i - count_nodes 

                # mutate weights and biases by adding values from random distrubution
                sigma_eff = self.sigma * ( self.N_generations ** ( -np.random.rand() ) )
                if change_both_weights_and_biases_of_a_node:
                    randn_mutation = sigma_eff * np.random.randn( shape[layer-1][0] ) 
                    mutant[k+1][layer-1][:,node_in_layer] += randn_mutation
                    mutant[k+1][layer][node_in_layer] += sigma_eff * np.random.randn() 
                else:
                    bias_or_weights = math.floor(2 * np.random.rand() )
                    if bias_or_weights == 0:
                        randn_mutation = sigma_eff * np.random.randn( shape[layer-1][0] ) 
                        mutant[k+1][layer-1][:,node_in_layer] += randn_mutation
                    else:
                        mutant[k+1][layer][node_in_layer] += sigma_eff * np.random.randn() 
                
                # add non_training weights such that 'mutant' now parametrizes the entire model
                mutant[k+1].append(self.non_training_weights)
                
        return mutant


    # Evalutates all mutantants of a generationa and ouptus loss and the single best performing mutant of the generation 
    def evaluate_mutants(self, model, mutant, x_train=None, y_train=None, ):

        # most_accurate_model=0 corresponds to setting the input model as most accurate
        most_accurate_model = 0   
        for i in range(self.population_size+1):
                # replace weights of the input model by weights generated using the create_mutants function
                model.set_weights(mutant[i])
                output=model.evaluate(verbose=False, x=x_train, y=y_train)
                accuracy_new=output[1]
                if(accuracy_new > self.accuracy):
                    self.accuracy = accuracy_new
                    most_accurate_model = i
        # if none of the mutants have performed better on the training data than the original mutant, reduce sigma
        if(most_accurate_model == 0):
            self.N_generations += 1
            self.sigma = self.sigma_original/self.N_generations
 
        model.set_weights(mutant[most_accurate_model])

        #out = model.evaluate(verbose=False, x=x_train, y=y_train)

        return output, mutant[most_accurate_model]

    # --------------------- only the functions below are called in GAModel ---------------------

    # Run during compilation to get objects that are unchanged at fit()
    def prepare_during_compile(self, model):
       self.shape = self.get_shape( model=model ) 


    # Collects the functions defined above to form the fitting step that is to be repeated for a number of epochs
    def run_step(self, model, x_train, y_train):
        
        # Initialize training paramters
        if self.has_init_variables is not True:
            self.N_generations = 1
            self.accuracy = model.evaluate(x_train, y_train, verbose=0)[0]
            self.has_init_variables = True     

        mutant = self.create_mutants( model=model, shape=self.shape )
        score, selected_parent = self.evaluate_mutants(model=model, mutant=mutant, x_train=x_train, y_train=y_train)
        model.set_weights(selected_parent)
        return score



class CMA(ES):
    pass


class BFGS(ES):
    pass


class CeresSolver(ES):
    pass