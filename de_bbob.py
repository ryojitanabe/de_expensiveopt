#!/usr/bin/env python
"""
This runs a DE configuration on all the 2-, 3-, 5-, 10-, 20-, 40-dimensional 24 BBOB test functions. This is based on "example_experiment_for_beginners.py" originally provided by the COCO framework (https://github.com/numbbo/coco).

The following is an example to run a DE with the synchronous model and the hand-tuned parameters:

python de_bbob.py -de_alg 'syn_de' -out_folder 'Syn' -archive_rate '1.0' -de_cr '0.9' -de_sf '0.5' -de_strategy 'rand_to_pbest_1' -p_best_rate '0.05' -pop_size_rate '13.0' -subset_size_rate '0.0' -children_size_rate '0.0'
"""

#import cocoex, cocopp  # experimentation and post-processing modules
import cocoex  # only experimentation module
import numpy as np
import sys
import argparse
import random

from de import SynchronousDE
from de import AsynchronousDE
from de import MuPlusLambdaDE
from de import WIDE
from de import SubToSubDE

if __name__ == '__main__':
    np.random.seed(seed=1)
    random.seed(1)
    
    ### input
    parser = argparse.ArgumentParser()
    parser.add_argument('-pop_size_rate', type=float)    
    parser.add_argument('-de_cr', type=float)
    parser.add_argument('-de_sf', type=float)
    parser.add_argument('-de_strategy')
    parser.add_argument('-p_best_rate', type=float)
    parser.add_argument('-archive_rate', type=float)
    parser.add_argument('-de_alg')
    parser.add_argument('-out_folder')
    parser.add_argument('-children_size_rate', type=float)
    parser.add_argument('-subset_size_rate', type=float)
    #    parser.add_argument('-run_mode')
    args = parser.parse_args()

    de_alg = args.de_alg
    pop_size_rate = args.pop_size_rate
    de_sf = args.de_sf
    de_cr = args.de_cr
    de_strategy = args.de_strategy
    p_best_rate = 0.05
    archive_rate = 1.0
    if de_strategy == 'current_to_pbest_1' or de_strategy == 'rand_to_pbest_1':
        p_best_rate = args.p_best_rate
        archive_rate = args.archive_rate

    output_folder = args.out_folder
    #output_folder = "tmp"
    suite_name = "bbob"
    # the maximum number of function evaluations is "budget_multiplier * dimensionality"
    budget_multiplier = 100

    ### prepare
    suite = cocoex.Suite(suite_name, "", "")
    observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
    minimal_print = cocoex.utilities.MiniPrint()

    ### go
    for problem in suite:  # this loop will take several minutes or longer
        problem.observe_with(observer)  # generates the data for cocopp post-processing
        remaining_evals = problem.dimension * budget_multiplier
    
        pop_size =  int(np.floor(pop_size_rate * np.log(problem.dimension)))
        pop_size = max(pop_size, 6)

        num_children = 0        
        if de_alg == 'plus_de' or de_alg == 'wi_de':
            num_children = int(np.floor(pop_size * args.children_size_rate))
            num_children = max(num_children, 1)

        subset_size = 0
        if de_alg == 'sts_de':
            subset_size = int(np.floor(pop_size * args.subset_size_rate))
            subset_size = max(subset_size, 2)    
        
        de = None    
        if de_alg == 'syn_de':        
            de = SynchronousDE(problem, problem.dimension, problem.lower_bounds, problem.upper_bounds, remaining_evals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate)    
        elif de_alg == 'asy_de':
            de = AsynchronousDE(problem, problem.dimension, problem.lower_bounds, problem.upper_bounds, remaining_evals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate)
        elif de_alg == 'wi_de':
            de = WIDE(problem, problem.dimension, problem.lower_bounds, problem.upper_bounds, remaining_evals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, num_children)    
        elif de_alg == 'plus_de':        
            de = MuPlusLambdaDE(problem, problem.dimension, problem.lower_bounds, problem.upper_bounds, remaining_evals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, num_children)    
        elif de_alg == 'sts_de':        
            de = SubToSubDE(problem, problem.dimension, problem.lower_bounds, problem.upper_bounds, remaining_evals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, subset_size)
        else:
            raise Exception('Error: %s is not defined' % de_alg)
        de.run()       
        minimal_print(problem, final=problem.index == len(suite) - 1)
