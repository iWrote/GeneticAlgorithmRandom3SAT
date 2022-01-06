"""
@author: 2018B5A70869G_VARUN
"""
from CNF_Creator import *
import time
import numpy as np
import random


def clauses_to_bitstrings(sentence):
    '''Converts integer triplets to P, N bitstring pairs.
    
    Parameters
    ----------
    sentence: list
        List of integer triplets representing input sentence.    
    '''
    
    num_clauses = len(sentence)
    positive_literals = np.zeros(num_clauses, dtype = np.int64)
    negative_literals = np.zeros(num_clauses, dtype = np.int64)
    
    for i in range(0, num_clauses):
        for literal in sentence[i]:
            if literal > 0:
                positive_literals[i] |= (1 << (literal-1))
            else:
                negative_literals[i] |= (1 << (-literal-1))
                
   
    return positive_literals, negative_literals

def seeding_schema(P, N):
    '''Computes pS and nS bitstrings which identify pure positive and negative literals
    
    Parameters
    ----------
    P, N : numpy arrays of 64-bit integers 
        Contain bitstrings representing clauses in the sentence.
    '''
    
    present_as_positive = np.bitwise_or.reduce(P)
    present_as_negative = np.bitwise_or.reduce(N)
    
    present_as_positive_only = np.bitwise_and(present_as_positive , np.bitwise_not(present_as_negative))
    present_as_negative_only = np.bitwise_and(present_as_negative , np.bitwise_not(present_as_positive))
    
    
    return present_as_positive_only, present_as_negative_only
            
def enforce_schema(V, pS, nS):
    '''Fixes pure literals to 0 or 1 in valuation.
    
    Note: Fixing a pure literal satisfies all clauses it appears in. 
    Mutation only affects literals in failing clauses. So, this 
    function needs to be called only once for seeding schema in the
    starting population.
    '''
    
    V = np.bitwise_or(V, pS)
    V = np.bitwise_not(V)
    V = np.bitwise_or(V, nS)
    V = np.bitwise_not(V)
    
    return V

def bitstring_to_numlist(v):
    '''Converts bitstring (64-bit integer) to a list of integers for display.
    '''
    numlist = []
    for i in range(1, 51):
       if v & (1 << (i-1)) == 0:
           numlist.append(-i)
       else:
           numlist.append(i)
           
           
    return numlist

def display(sentence,v,f,t):
    print('\n\n')
    print('Roll No : 2018B5A70869G')
    print('Number of clauses in CSV file : ',len(sentence))
    print('Best model : ', v)
    print(f"Fitness value of best model : {f}%")
    print(f"Time taken : {t} seconds")
    print('\n\n')
    


def GA(P, N, pS, nS, sentence):
    """Improved genetic algorithm.
    
    Parameters
    ----------
    sentence: list
        List of integer triplets representing input sentence.
    P, N : numpy arrays of 64-bit integers 
        Contain bitstrings representing clauses in the sentence.
    pS, nS : numpy 64-bit integers
        Set bits denote pure positive / negative literals. Input to enforce_schema(...)
        
    """
        
    
    
    m = len(P)
    
    fitness = lambda v : np.count_nonzero(np.bitwise_or(np.bitwise_and(P,v), np.bitwise_and(N,np.bitwise_not(v))))
    
    population_size = 100
    mutation_probability = 0.2
    time_limit = 44
    plateau_time_limit = 8     
    next_gen = np.zeros(population_size, dtype = np.int64)
    
    current_gen = np.random.randint(0, 2**50, size = population_size, dtype = np.int64)    
    current_gen = enforce_schema(current_gen, pS, nS)
    
    current_fitness = np.vectorize(fitness)(current_gen)
    
    best_fit = 0
    start_time = time.time()    
    
    restart_count= 0

    while time.time() - start_time < time_limit and best_fit < m:       
        
        improvement_time = time.time()
        
        while (time.time() - start_time) < time_limit and best_fit < m and (time.time() - improvement_time) < plateau_time_limit:
            
            for i in range(population_size):
                #SELECTION
                idx1, idx2 = random.choices(range(population_size), weights=current_fitness, k=2)
                parent1 = current_gen[idx1]
                parent2 = current_gen[idx2]
                
                #CROSSOVER
                cross_mask = (1 << np.random.randint(0, 50)) - 1
                child = (parent1 & cross_mask) | (parent2 & ~cross_mask)
                
                #MUTATION
                if (random.random() < mutation_probability):
                    sat = np.bitwise_or(np.bitwise_and(P, child),np.bitwise_and(N, ~child))
                    fc = np.where(sat == 0)[0] #failing clause
                    if len(fc) > 0:
                        to_flip = sentence[fc[random.choice(range(len(fc)))]][random.choice(range(3))]
                        child = child ^ (1 << abs(to_flip)) #flip 1 of 3 literals in failing clause
                
                next_gen[i] = child
            
            #ELITISM
            next_gen[population_size-1] = current_gen[np.argmax(current_fitness)] 
            # TODO: repeated copy (how to avoid in python?)
            current_gen[:] = next_gen[:] 
            current_fitness = np.vectorize(fitness)(current_gen)
            
        
            if max(current_fitness) > best_fit:
                best_idx = np.argmax(current_fitness)
                best_v = current_gen[best_idx]
                best_fit = max(current_fitness)
                improvement_time = time.time()
            
        
                
        if restart_count == 2:
            break
        
        #RESTART
        next_gen = np.zeros(population_size, dtype = np.int64)
        current_gen = np.random.randint(0, 2**50, size = population_size, dtype = np.int64)    
        current_fitness = np.vectorize(fitness)(current_gen)        
        restart_count += 1
        
        
    end_time = time.time()
    
    return best_v, best_fit/m * 100.0 , (end_time - start_time)


def run(sentence):
    P, N = clauses_to_bitstrings(sentence)
    pS, nS = seeding_schema(P, N)
    v, f, t = GA(P, N, pS, nS, sentence)
    vlist = bitstring_to_numlist(v)
    display(sentence, vlist, f, t)


    
def main():
    cnfC = CNF_Creator(n=50)
    sentence = cnfC.ReadCNFfromCSVfile()
    run(sentence)
    
    
if __name__=='__main__':
    main()
    
    
    




