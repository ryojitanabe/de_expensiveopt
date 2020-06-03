#!/usr/bin/env python
"""
This runs a DE configuration on one of the 28 CEC2013 test functions f1, ..., f28. Please note that this is not stand-alone. This requires the wrapper class "cec2013single " implemented by Dr. Daniel Molina (https://github.com/dmolina/cec2013single). The wrapper calls the CEC2013 test functions implemented in the original C source code. In addition, some data files (e.g., cec2013_data) are needed.

The following is an example to run a DE with the synchronous model and the hand-tuned parameters on the 10-dimensional f4 function, where the random seed = 1:

python de_cec2013.py -func_id 4 -dim 10 -seed 1 -de_alg 'syn_de' -archive_rate '1.0' -de_cr '0.9' -de_sf '0.5' -de_strategy 'rand_to_pbest_1' -p_best_rate '0.05' -pop_size_rate '13.0' -subset_size_rate '0.0' -children_size_rate '0.0'
"""

import numpy as np
import argparse
import random

from cec2013single.cec2013 import Benchmark
from de import SynchronousDE
from de import AsynchronousDE
from de import MuPlusLambdaDE
from de import WIDE
from de import SubToSubDE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-func_id', type=int)
    parser.add_argument('-dim', type=int)
    parser.add_argument('-seed', type=int)
    parser.add_argument('-de_cr', type=float)
    parser.add_argument('-de_sf', type=float)
    parser.add_argument('-de_strategy')
    parser.add_argument('-pop_size_rate', type=float)    
    parser.add_argument('-p_best_rate', type=float)
    parser.add_argument('-archive_rate', type=float)
    parser.add_argument('-de_alg')
    parser.add_argument('-children_size_rate', type=float)
    parser.add_argument('-subset_size_rate', type=float)

    args = parser.parse_args()
    func_id = args.func_id
    dim = args.dim
    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    
    de_alg = args.de_alg
    pop_size =  int(np.floor(args.pop_size_rate * np.log(dim)))
    pop_size = max(pop_size, 6)
    de_sf = args.de_sf
    de_cr = args.de_cr
    de_strategy = args.de_strategy

    p_best_rate = 0.05
    archive_rate = 1.0
    if de_strategy == 'current_to_pbest_1' or de_strategy == 'rand_to_pbest_1':
        p_best_rate = args.p_best_rate
        archive_rate = args.archive_rate

    num_children = 0        
    if de_alg == 'plus_de' or de_alg == 'wi_de':
        num_children = int(np.floor(pop_size * args.children_size_rate))
        num_children = max(num_children, 1)
    subset_size = 0
    if de_alg == 'sts_de':
        subset_size = int(np.floor(pop_size * args.subset_size_rate))
        subset_size = max(subset_size, 2)    
        
    remaining_evals = 100 * dim

    fbench = Benchmark()
    info = fbench.get_info(func_id, dim)
    fun = fbench.get_function(func_id)
    lower_bounds = np.full(dim, info['lower'])
    upper_bounds = np.full(dim, info['upper'])    

    de = None    
    if de_alg == 'syn_de':        
        de = SynchronousDE(fun, dim, lower_bounds, upper_bounds, remaining_evals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate)
    elif de_alg == 'asy_de':        
        de = AsynchronousDE(fun, dim, lower_bounds, upper_bounds, remaining_evals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate)
    elif de_alg == 'wi_de':
        de = WIDE(fun, dim, lower_bounds, upper_bounds, remaining_evals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, num_children)
    elif de_alg == 'plus_de':        
        de = MuPlusLambdaDE(fun, dim, lower_bounds, upper_bounds, remaining_evals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, num_children)
    elif de_alg == 'sts_de':        
        de = SubToSubDE(fun, dim, lower_bounds, upper_bounds, remaining_evals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, subset_size)
    else:
        raise Exception('Error: %s is not defined' % de_alg)
    res = de.run()    
    print('error =', res)
