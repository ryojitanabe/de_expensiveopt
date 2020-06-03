#!/usr/bin/env python

import numpy as np
from pyDOE import lhs
import copy
import random

class DE():
    """
    Collection of functions in DE such as the mutation strategy and crossover
    """
    def __init__(self, fun, dim, lbounds, ubounds, budget, pop_size=100, de_strategy='rand_1', de_sf=0.5, de_cr=0.9, p_best_rate=0.05, archive_rate=0.05):
        self.fun = fun
        self.dim = dim
        self.lbounds = lbounds
        self.ubounds = ubounds
        self.budget = budget
        self.pop_size = pop_size
        self.de_strategy = de_strategy
        self.de_sf = de_sf
        self.de_cr = de_cr
        self.p_best_rate = p_best_rate
        self.archive_rate = archive_rate
        self.p_best_size = max(int(round(self.p_best_rate * self.pop_size)), 2)
        self.archive_size = int(np.floor(self.archive_rate * self.pop_size))
        
    # this function is implemented in each derived class
    def run(self):
        pass
        
    def differential_mutation(self, de_strategy, target_idx, best_idx, p_best_idx, idxs, pop, archive, sf=0.5):
        r1, r2, r3, r4, r5 = idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]
        if de_strategy == 'rand_1':
            v = pop[r1] + sf * (pop[r2] - pop[r3])
        elif de_strategy == 'rand_2':
            v = pop[r1] + sf * (pop[r2] - pop[r3]) + sf * (pop[r4] - pop[r5])
        elif de_strategy == 'best_1':
            v = pop[best_idx] + sf * (pop[r1] - pop[r2])
        elif de_strategy == 'best_2':
            v = pop[best_idx] + sf * (pop[r1] - pop[r2]) + sf * (pop[r3] - pop[r4])
        elif de_strategy == 'current_to_rand_1':
            # This implementation of the current-to-rand/1 strategy is not standard in the DE community.
            v = pop[target_idx] + sf * (pop[r1] - pop[target_idx]) + sf * (pop[r2] - pop[r3])
            # Traditionally, the scale factor value for the first difference vector is randomly generated in [0,1] as follows:
            #K = np.random.rand()
            #v = pop[target_idx] + K * (pop[r1] - pop[target_idx]) + sf * (pop[r2] - pop[r3])
        elif de_strategy == 'current_to_best_1':
            v = pop[target_idx] + sf * (pop[best_idx] - pop[target_idx]) + sf * (pop[r1] - pop[r2])
        elif de_strategy == 'current_to_pbest_1':
            if r2 >= self.pop_size:                
                r2 -= self.pop_size
                v = pop[target_idx] + sf * (pop[p_best_idx] - pop[target_idx]) + sf * (pop[r1] - archive[r2])
            else:
                v = pop[target_idx] + sf * (pop[p_best_idx] - pop[target_idx]) + sf * (pop[r1] - pop[r2])
        elif de_strategy == 'rand_to_pbest_1':
            if r3 >= self.pop_size:                
                r3 -= self.pop_size
                v = pop[r1] + sf * (pop[p_best_idx] - pop[r1]) + sf * (pop[r2] - archive[r3])
            else:
                v = pop[r1] + sf * (pop[p_best_idx] - pop[r1]) + sf * (pop[r2] - pop[r3])
                                            
        # This repair method is used in JADE
        # After the repair, a violated element is in the middle of the target vector and the correspoinding bound
        v = np.where(v < self.lbounds, (self.lbounds + pop[target_idx]) / 2.0, v)
        v = np.where(v > self.ubounds, (self.ubounds + pop[target_idx]) / 2.0, v)           
        return v

    def parent_idxs(self, de_strategy, target_idx, best_idx, p_best_idx, arc_size=0):
        # randomly select parent indices such that i != r1 != r2 != ...
        r1 = r2 = r3 = r4 = r5 = target_idx
        if de_strategy == 'rand_1' or de_strategy == 'current_to_rand_1':
            while r1 == target_idx:
                r1 = np.random.randint(self.pop_size)
            while r2 == target_idx or r2 == r1:
                r2 = np.random.randint(self.pop_size)
            while r3 == target_idx or r3 == r2 or r3 == r1:
                r3 = np.random.randint(self.pop_size)
        elif de_strategy == 'rand_2':
            while r1 == target_idx:
                r1 = np.random.randint(self.pop_size)
            while r2 == target_idx or r2 == r1:
                r2 = np.random.randint(self.pop_size)
            while r3 == target_idx or r3 == r2 or r3 == r1:
                r3 = np.random.randint(self.pop_size)
            while r4 == target_idx or r4 == r3 or r4 == r2 or r4 == r1:
                r4 = np.random.randint(self.pop_size)
            while r5 == target_idx or r5 == r4 or r5 == r3 or r5 == r2 or r5 == r1:
                r5 = np.random.randint(self.pop_size)
        elif de_strategy == 'best_1' or de_strategy == 'current_to_best_1':
            while r1 == target_idx or r1 == best_idx:
                r1 = np.random.randint(self.pop_size)
            while r2 == target_idx or r2 == best_idx or r2 == r1:
                r2 = np.random.randint(self.pop_size)
        elif de_strategy == 'best_2':
            while r1 == target_idx or r1 == best_idx:
                r1 = np.random.randint(self.pop_size)
            while r2 == target_idx or r2 == best_idx or r2 == r1:
                r2 = np.random.randint(self.pop_size)
            while r3 == target_idx or r3 == best_idx or r3 == r2 or r3 == r1:
                r3 = np.random.randint(self.pop_size)
            while r4 == target_idx or r4 == best_idx or r4 == r3 or r4 == r2 or r4 == r1:
                r4 = np.random.randint(self.pop_size)
        elif de_strategy == 'current_to_pbest_1':
            # while r1 == target_idx or r1 == p_best_idx:
            #     r1 = np.random.randint(self.pop_size)
            # while r2 == target_idx or r2 == p_best_idx or r2 == r1:
            #     r2 = np.random.randint(self.pop_size + arc_size)
            # This implementation allows 
            while r1 == target_idx:
                r1 = np.random.randint(self.pop_size)
            while r2 == target_idx or r2 == r1:
                r2 = np.random.randint(self.pop_size + arc_size)
        elif de_strategy == 'rand_to_pbest_1':
            # while r1 == target_idx or r1 == p_best_idx:
            #     r1 = np.random.randint(self.pop_size)
            # while r2 == target_idx or r2 == p_best_idx or r2 == r1:
            #     r2 = np.random.randint(self.pop_size)
            # while r3 == target_idx or r3 == p_best_idx or r3 == r2 or r3 == r1:
            #     r3 = np.random.randint(self.pop_size + arc_size)
            while r1 == target_idx:
                r1 = np.random.randint(self.pop_size)
            while r2 == target_idx or r2 == r1:
                r2 = np.random.randint(self.pop_size)
            while r3 == target_idx or r3 == r2 or r3 == r1:
                r3 = np.random.randint(self.pop_size + arc_size)
        else:
            raise Exception('Error: %s is not defined' % de_strategy)
        idx = [r1, r2, r3, r4, r5]
        return idx

    def binomial_crossover(self, x, v, cr=0.9):
        j_rand = np.random.randint(self.dim)
        rnd_vals = np.random.rand(self.dim)
        rnd_vals[j_rand] = 0.0                
        u = np.where(rnd_vals <= cr, v, x)                
        return u

    def exponential_crossover(self, x, v, cr=0.9):
        u = np.copy(x)        
        i = np.random.randint(self.dim)
        l = 0
        while True:
            u[i] = v[i]
            i = (i + 1) % self.dim
            l += 1
            if np.random.rand() >= cr or l >= self.dim:
                break
        return u
    
class SynchronousDE(DE):
    """
    DE with the synchronous population model, i.e., the most basic DE:
    R. Storn and K. Price. Differential Evolution - A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces. J. Glo. Opt., 11(4):341–359, 1997.
    """
    def __init__(self, fun, dim, lbounds, ubounds, budget, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate):
        super().__init__(fun, dim, lbounds, ubounds, budget, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate)
        
    def run(self):
        max_num_evals = self.budget
        num_evals = 0
        bsf_x = np.empty(self.dim)
        bsf_fit = np.inf    

        pop = []
        pop_fit = []
        archive = []
        
        # generate the initial population using Latin hypercube sampling
        # The option for 'criterion': center, maximin, centermaximin, correlation
        # https://pythonhosted.org/pyDOE/randomized.html
        lhd = lhs(self.dim, samples=self.pop_size, criterion='maximin')
        
        for i in range(self.pop_size):
            x = self.lbounds + (self.ubounds - self.lbounds) * lhd[i]
            #x = self.lbounds + (self.ubounds - self.lbounds) * np.random.rand(self.dim)
            fit = self.fun(x)
            num_evals += 1
            if fit < bsf_fit:
                bsf_fit = fit
                bsf_x = np.copy(x)
                if __name__ == '__main__':
                    print('%d %8.5e' % (num_evals, bsf_fit))                    
                    if bsf_fit < 1e-8:
                        return(bsf_fit)
            pop.append(x)
            pop_fit.append(fit)
            
        while num_evals < max_num_evals:
            children = []
            children_fit = []
            best_idx = np.argmin(pop_fit)
            p_top_idxs = np.argsort(pop_fit)[:self.p_best_size]

            for i in range(self.pop_size):
                p_best_idx = np.random.choice(p_top_idxs)
                idxs = self.parent_idxs(self.de_strategy, i, best_idx, p_best_idx, len(archive))                                           
                v = self.differential_mutation(self.de_strategy, i, best_idx, p_best_idx, idxs, pop, archive, self.de_sf)
                u = self.binomial_crossover(pop[i], v, self.de_cr)
                fit = self.fun(u)
                num_evals += 1
                if fit < bsf_fit:
                    bsf_fit = fit
                    bsf_x = np.copy(u)
                    if __name__ == '__main__':
                        print('%d %8.5e' % (num_evals, bsf_fit))                    
                        if bsf_fit < 1e-8:
                            return(bsf_fit)
                children.append(u)
                children_fit.append(fit)
            
            for i in range(self.pop_size):
                if children_fit[i] <= pop_fit[i]:
                    archive.append(np.copy(pop[i]))
                    pop_fit[i] = children_fit[i]
                    pop[i] = np.copy(children[i])

            while len(archive) > self.archive_size:
                r = np.random.randint(len(archive))
                del archive[r]

        return bsf_fit
                
class AsynchronousDE(DE):
    """
    DE with the asynchronous population model
    """
    def __init__(self, fun, dim, lbounds, ubounds, budget, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate):
        super().__init__(fun, dim, lbounds, ubounds, budget, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate)
        
    def run(self):
        max_num_evals = self.budget
        num_evals = 0
        bsf_x = np.empty(self.dim)
        bsf_fit = np.inf    

        pop = []
        pop_fit = []
        archive = []

        # generate the initial population using Latin hypercube sampling
        # The option for 'criterion': center, maximin, centermaximin, correlation
        # https://pythonhosted.org/pyDOE/randomized.html
        lhd = lhs(self.dim, samples=self.pop_size, criterion='maximin')
        
        for i in range(self.pop_size):
            x = self.lbounds + (self.ubounds - self.lbounds) * lhd[i]
            #x = self.lbounds + (self.ubounds - self.lbounds) * np.random.rand(self.dim)
            fit = self.fun(x)
            num_evals += 1
            if fit < bsf_fit:
                bsf_fit = fit
                bsf_x = np.copy(x)
                if __name__ == '__main__':
                    print('%d %8.5e' % (num_evals, bsf_fit))                    
                    if bsf_fit < 1e-8:
                        return(bsf_fit)
            pop.append(x)
            pop_fit.append(fit)
            
        while num_evals < max_num_evals:
            for i in range(self.pop_size):
                best_idx = np.argmin(pop_fit)
                p_top_idxs = np.argsort(pop_fit)[:self.p_best_size]
                p_best_idx = np.random.choice(p_top_idxs)
                idxs = self.parent_idxs(self.de_strategy, i, best_idx, p_best_idx, len(archive))                                           
                v = self.differential_mutation(self.de_strategy, i, best_idx, p_best_idx, idxs, pop, archive, self.de_sf)  
                u = self.binomial_crossover(pop[i], v, self.de_cr)
                fit = self.fun(u)
                num_evals += 1
                if fit < bsf_fit:
                    bsf_fit = fit
                    bsf_x = np.copy(u)
                    if __name__ == '__main__':
                        print('%d %8.5e' % (num_evals, bsf_fit))                    
                        if bsf_fit < 1e-8:
                            return(bsf_fit)
                            
                if fit <= pop_fit[i]:
                    archive.append(np.copy(pop[i]))
                    pop_fit[i] = fit
                    pop[i] = np.copy(u)

                while len(archive) > self.archive_size:
                    r = np.random.randint(len(archive))
                    del archive[r]
        return bsf_fit
                    
class MuPlusLambdaDE(DE):
    """
    DE with the plus-selection, i.e,. the traditional (mu+lambda)-selection model
    """
    def __init__(self, fun, dim, lbounds, ubounds, budget, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, num_children=1):
        super().__init__(fun, dim, lbounds, ubounds, budget, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate)        
        self.num_children = num_children
        
    def run(self):
        max_num_evals = self.budget
        num_evals = 0
        bsf_x = np.empty(self.dim)
        bsf_fit = np.inf    

        pop = []
        pop_fit = []
        archive = []

        # generate the initial population using Latin hypercube sampling
        # The option for 'criterion': center, maximin, centermaximin, correlation
        # https://pythonhosted.org/pyDOE/randomized.html
        lhd = lhs(self.dim, samples=self.pop_size, criterion='maximin')
        
        for i in range(self.pop_size):
            # generate the initial population using Latin hypercube sampling
            x = self.lbounds + (self.ubounds - self.lbounds) * lhd[i]
            #x = self.lbounds + (self.ubounds - self.lbounds) * np.random.rand(self.dim)
            fit = self.fun(x)
            num_evals += 1
            if fit < bsf_fit:
                bsf_fit = fit
                bsf_x = np.copy(x)
                if __name__ == '__main__':
                    print('%d %8.5e' % (num_evals, bsf_fit))                    
                    if bsf_fit < 1e-8:
                        return(bsf_fit)
            pop.append(x)
            pop_fit.append(fit)

        while num_evals < max_num_evals:
            children = []
            children_fit = []
            best_idx = np.argmin(pop_fit)
            p_top_idxs = np.argsort(pop_fit)[:self.p_best_size]

            # randomly select lambda indices of target individuals
            target_idxs = random.sample(range(self.pop_size), self.num_children)
            for i in target_idxs:
                p_best_idx = np.random.choice(p_top_idxs)
                idxs = self.parent_idxs(self.de_strategy, i, best_idx, p_best_idx, len(archive))                                           
                v = self.differential_mutation(self.de_strategy, i, best_idx, p_best_idx, idxs, pop, archive, self.de_sf)
                u = self.binomial_crossover(pop[i], v, self.de_cr)
                fit = self.fun(u)
                num_evals += 1
                if fit < bsf_fit:
                    bsf_fit = fit
                    bsf_x = np.copy(u)
                    if __name__ == '__main__':
                        print('%d %8.5e' % (num_evals, bsf_fit))                    
                        if bsf_fit < 1e-8:
                            return(bsf_fit)
                children.append(u)
                children_fit.append(fit)

            # select best mu individuals from pop and children
            union = copy.deepcopy(pop + children)
            union_fit = copy.deepcopy(pop_fit + children_fit)            
            union_top_idxs = np.argsort(union_fit)

            for i in range(self.pop_size):
                r = union_top_idxs[i]
                pop[i] = copy.deepcopy(union[r])
                pop_fit[i] = copy.deepcopy(union_fit[r])

            # how to manage the external archive in the plus-seleciton is not obvious. In this implementation, the worst lambda individuals are stored into the external archive.
            for i in range(self.pop_size, self.pop_size + self.num_children):
                r = union_top_idxs[i]
                archive.append(np.copy(union[r]))

            while len(archive) > self.archive_size:
                r = np.random.randint(len(archive))
                del archive[r]
        return bsf_fit
                
class WIDE(DE):
    """
    DE with the worst improvement model, which is the population update model in DEGD:
    M. M. Ali: Differential evolution with generalized differentials. J. Comput. Appl. Math. 235(8): 2205-2216 (2011)
    """
    def __init__(self, fun, dim, lbounds, ubounds, budget, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, num_children=1):
        super().__init__(fun, dim, lbounds, ubounds, budget, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate)
        self.num_children = num_children
                
    def run(self):
        max_num_evals = self.budget
        num_evals = 0
        bsf_x = np.empty(self.dim)
        bsf_fit = np.inf    

        pop = []
        pop_fit = []
        archive = []

        # generate the initial population using Latin hypercube sampling
        # The option for 'criterion': center, maximin, centermaximin, correlation
        # https://pythonhosted.org/pyDOE/randomized.html
        lhd = lhs(self.dim, samples=self.pop_size, criterion='maximin')
                
        for i in range(self.pop_size):
            x = self.lbounds + (self.ubounds - self.lbounds) * lhd[i]
            #x = self.lbounds + (self.ubounds - self.lbounds) * np.random.rand(self.dim)
            fit = self.fun(x)
            num_evals += 1
            if fit < bsf_fit:
                bsf_fit = fit
                bsf_x = np.copy(x)
                if __name__ == '__main__':
                    print('%d %8.5e' % (num_evals, bsf_fit))                    
                    if bsf_fit < 1e-8:
                        return(bsf_fit)
            pop.append(x)
            pop_fit.append(fit)
                               
        while num_evals < max_num_evals:
            children = []
            children_fit = []
            best_idx = np.argmin(pop_fit)
            p_top_idxs = np.argsort(pop_fit)[:self.p_best_size]

            # only the worst lambda (num_children) individuals can generate their trial vectors
            top_idxs = np.argsort(pop_fit)
            for j in range(self.pop_size - self.num_children, self.pop_size):
                i = top_idxs[j]
                
                p_best_idx = np.random.choice(p_top_idxs)
                idxs = self.parent_idxs(self.de_strategy, i, best_idx, p_best_idx, len(archive))                                           
                v = self.differential_mutation(self.de_strategy, i, best_idx, p_best_idx, idxs, pop, archive, self.de_sf)
                u = self.binomial_crossover(pop[i], v, self.de_cr)
                fit = self.fun(u)
                num_evals += 1
                if fit < bsf_fit:
                    bsf_fit = fit
                    bsf_x = np.copy(u)
                    if __name__ == '__main__':
                        print('%d %8.5e' % (num_evals, bsf_fit))                    
                        if bsf_fit < 1e-8:
                            return(bsf_fit)
                children.append(u)
                children_fit.append(fit)

            for child_idx, j in enumerate(range(self.pop_size - self.num_children, self.pop_size)):
                i = top_idxs[j]
                if children_fit[child_idx] <= pop_fit[i]:
                    archive.append(np.copy(pop[i]))
                    pop_fit[i] = children_fit[child_idx]
                    pop[i] = np.copy(children[child_idx])
                    
            while len(archive) > self.archive_size:
                r = np.random.randint(len(archive))
                del archive[r]                                
        return bsf_fit

class SubToSubDE(DE):
    """
    DE with the STS model:
    J. Guo, Z. Li, and S. Yang. Accelerating differential evolution based on a subset-to-subset survivor selection operator. Soft Comput., 23(12):4113–4130, 2019.
    """
    def __init__(self, fun, dim, lbounds, ubounds, budget, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, subset_size):
        super().__init__(fun, dim, lbounds, ubounds, budget, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate)
        self.subset_size = subset_size
                
    def run(self):
        max_num_evals = self.budget
        num_evals = 0
        bsf_x = np.empty(self.dim)
        bsf_fit = np.inf    

        pop = []
        pop_fit = []
        archive = []

        # generate the initial population using Latin hypercube sampling
        # The option for 'criterion': center, maximin, centermaximin, correlation
        # https://pythonhosted.org/pyDOE/randomized.html
        lhd = lhs(self.dim, samples=self.pop_size, criterion='maximin')
        
        for i in range(self.pop_size):
            x = self.lbounds + (self.ubounds - self.lbounds) * lhd[i]
            #x = self.lbounds + (self.ubounds - self.lbounds) * np.random.rand(self.dim)
            fit = self.fun(x)
            num_evals += 1
            if fit < bsf_fit:
                bsf_fit = fit
                bsf_x = np.copy(x)
                if __name__ == '__main__':
                    print('%d %8.5e' % (num_evals, bsf_fit))                    
                    if bsf_fit < 1e-8:
                        return(bsf_fit)
            pop.append(x)
            pop_fit.append(fit)
            
        num_subsets = int(self.pop_size / self.subset_size)
        # num_subsets should be even
        if self.pop_size % self.subset_size != 0:
            num_subsets += 1
            
        while num_evals < max_num_evals:
            children = []
            children_fit = []
            best_idx = np.argmin(pop_fit)
            p_top_idxs = np.argsort(pop_fit)[:self.p_best_size]

            for i in range(self.pop_size):
                p_best_idx = np.random.choice(p_top_idxs)
                idxs = self.parent_idxs(self.de_strategy, i, best_idx, p_best_idx, len(archive))                                           
                v = self.differential_mutation(self.de_strategy, i, best_idx, p_best_idx, idxs, pop, archive, self.de_sf)
                u = self.binomial_crossover(pop[i], v, self.de_cr)
                fit = self.fun(u)
                num_evals += 1
                if fit < bsf_fit:
                    bsf_fit = fit
                    bsf_x = np.copy(u)
                    if __name__ == '__main__':
                        print('%d %8.5e' % (num_evals, bsf_fit))                    
                        if bsf_fit < 1e-8:
                            return(bsf_fit)
                children.append(u)
                children_fit.append(fit)

            # the following looks somewhat complicated
            # the selection is based on the index-based ring topology, e.g., |0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, ...|
            # i is the start position on the index-based ring topology.
            i = np.random.randint(self.pop_size)
            for j in range(num_subsets):
                num_remains = self.pop_size - (j * self.subset_size)
                # num_replacements represents the number of individuals to be replaced in this round
                num_replacements = self.subset_size                
                if num_remains <= self.subset_size:
                    num_replacements = num_remains                    
                union = [] 
                union_fit = []
                for k in range(num_replacements):
                    idx = (i + k) % self.pop_size
                    union.append(pop[idx])
                    union_fit.append(pop_fit[idx])
                    union.append(children[idx])
                    union_fit.append(children_fit[idx])

                # the best num_replacements individuals can enter the next population
                union_top_idxs = np.argsort(union_fit)                        
                for k in range(num_replacements):
                    idx = (i + k) % self.pop_size
                    pop_fit[idx] = union_fit[union_top_idxs[k]]
                    pop[idx] = copy.deepcopy(union[union_top_idxs[k]])
                    
                for k in range(num_replacements, len(union)):
                    r = union_top_idxs[k]
                    archive.append(np.copy(union[r]))                                        
                i = (i + self.subset_size) % self.pop_size

            while len(archive) > self.archive_size:
                r = np.random.randint(len(archive))
                del archive[r]
        return bsf_fit
        
"""
The three test functions were derived from pycma (https://github.com/CMA-ES/pycma)
"""
def sphere(x):
    return sum(np.asarray(x)**2)

def rastrigin(x):
    """Rastrigin test objective function"""
    if not np.isscalar(x[0]):
        N = len(x[0])
        return [10 * N + sum(xi**2 - 10 * np.cos(2 * np.pi * xi)) for xi in x]
    # return 10*N + sum(x**2 - 10*np.cos(2*np.pi*x), axis=1)
    N = len(x)
    return 10 * N + sum(x**2 - 10 * np.cos(2 * np.pi * x))

def rosen(x, alpha=1e2):
    """Rosenbrock test objective function"""
    x = [x] if np.isscalar(x[0]) else x  # scalar into list
    x = np.asarray(x)
    f = [sum(alpha * (x[:-1]**2 - x[1:])**2 + (1. - x[:-1])**2) for x in x]
    return f if len(f) > 1 else f[0]  # 1-element-list into scalar

if __name__ == '__main__':
    np.random.seed(seed=1)
    random.seed(1)
    
    fun = sphere
    dim = 10
    lower_bounds = np.full(dim, -5)
    upper_bounds = np.full(dim, 5)    
    remaining_evals = 100 * dim
    
    # 'syn_de', 'asy_de', 'wi_de', 'plus_de', 'sts_de'
    de_alg = 'wi_de'
    pop_size =  int(np.floor(13 * np.log(dim)))
    pop_size = max(pop_size, 6)
    de_sf = 0.5
    de_cr = 0.9
    # 'rand_1', 'rand_2', 'best_1', 'best_2', 'current_to_best_1', 'current_to_pbest_1', 'rand_to_pbest_1'
    de_strategy = 'rand_to_pbest_1'

    # these are activated only when using 'current_to_pbest_1' and 'rand_to_pbest_1'
    p_best_rate = 0.05
    archive_rate = 1.0

    # num_children is only for 'plus_de' and 'wi_de'
    # num_children = int(np.floor(pop_size * 0.5))
    # num_children = max(num_children, 1)
    num_children = 1

    # subset_size is only for 'sts_de'
    subset_size = 2
    # subset_size = int(np.floor(pop_size * 0.5))
    # subset_size = max(subset_size, 2)    
        
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
    print('error = ', res)
    
