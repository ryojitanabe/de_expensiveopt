# Revisiting Population Models in Differential Evolution on a Limited Budget of Evaluations

This repository provides the Python code to reproduce experimental data shown in the following paper:

> Ryoji Tanabe, **Revisiting Population Models in Differential Evolution on a Limited Budget of Evaluations**, PPSN2020, [pdf](https://ryojitanabe.github.io/pdf/t-ppsn2020.pdf)

The performance data of the 10 DE configurations presented in our PPSN2020 paper can be downloaded from [here](https://drive.google.com/file/d/1p_DFnKPc5NWsWxG-3lFyI6sTmUBYwyZt/view?usp=sharing)

This repository is based on the COCO framework implemented in Python (https://github.com/numbbo/coco):

> Nikolaus Hansen, Anne Auger, Olaf Mersmann, Tea Tusar, Dimo Brockhoff, **COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting**, CoRR abs/1603.08785 (2016), [link](https://arxiv.org/abs/1603.08785)

To run a DE configuration on the CEC2013 test functions, this repository requires the wrapper package implemented by Dr. Daniel Molina (https://github.com/dmolina/cec2013single).

# Requirements

This code at least require Python 3, numpy, and pyDOE. To perform a benchmarking of DE on the BBOB test functions, this code require the module "cocoex" provided by COCO. The module "cocopp" is also necessary for postprocessing experimental data. For details, please see https://github.com/numbbo/coco. Ruby (>=2.5) is needed to run "run_bbob.rb", which is an optional script to perform an experiment automatically. 

# Usage

## Parameter (or configuration) files

Dat files in the folder "de_configs" provide control parameters for DE. "hand_tuned" means the hand-tuned parameters. "smac_tuned" means the automatically-tuned parameters by [SMAC](http://www.cs.ubc.ca/labs/beta/Projects/SMAC/). Details of control parameters can be found in Section 3 and Table 2 in our PPSN2020 paper.

- pop\_size\_rate controls the population size.
- de_strategy is the differential mutation strategy. This is the categorical parameter to be selected from 'rand\_1', 'rand\_2', 'best\_1', 'best\_2', 'current\_to\_best\_1', 'current\_to\_pbest\_1', 'rand\_to\_pbest\_1'.
- de_sf is the scale factor in the differential mutation
- de_cr is the crossover rate in binomial crossover
- p\_best\_rate is the pbest rate in the current-to-pbest/1 and rand-to-pbest/1 mutation strategies. 
- archive\_rate controls the size of the external archive for the current-to-pbest/1 and rand-to-pbest/1 mutation strategies.
- children\_size\_rate controls the number of children per one iteration. This is activated only for the plus selection model and worst improvement models.

## Simple example

```
$ python de.py
```

The above command just runs a DE with the worst improvement model and the hand-tuned parameters on the 10-dimensional Sphere function.

```
$ python de_bbob.py -de_alg 'wi_de' -out_folder 'WI' -archive_rate '1.0' -de_cr '0.9' -de_sf '0.5' -de_strategy 'rand_to_pbest_1' -p_best_rate '0.05' -pop_size_rate '13.0' -subset_size_rate '0.0' -children_size_rate '0.0'
```

The above command runs a DE with the worst improvement model and the hand-tuned parameters on all the 2-, 3-, 5-, 10-, 20-, 40-dimensional 24 BBOB test functions. The maximum number of function evaluations is 100 * dimensionality. Results are recorded into exdata/Syn.

## Reproduce results presented in our PPSN2020 paper

The Ruby script "run_bbob.rb" automatically performs all experiments to reproduce all results presented in our PPSN2020 paper:

```
$ ruby run_bbob.rb
```
