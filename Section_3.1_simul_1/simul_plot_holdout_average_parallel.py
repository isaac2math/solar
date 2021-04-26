import numpy as np

from simulator                       import simul
from solar_holdout_average_parallel  import solar
from tqdm                            import tqdm

##########################################
# define the class 'the simulation_plot' #
##########################################
'''
this class is used for plotting the result of Simulation 3, Section 3:

Check this before you run the code:
Plz check if you have 'joblib' 'sci-kit learn', 'numpy', 'matplotlib' and 'tqdm' installed. If not,
    1. run 'pip install scikit-learn joblib numpy matplotlib tqdm' if you use pure Python3
    2. run 'conda install scikit-learn joblib numpy matplotlib tqdm' if you use Anaconda3

Modules:
    1. from scikit-learn, we call 'LassoLarsCV' for lasso and bolasso;
    2. we use 'numpy' for matrix computation and random variable generation;
    3. for 'simulator_ic', 'solar' and 'costcom', plz see 'simulator_ic.py', 'solar.py' and 'costcom.py' for detail;
    4. 'tqdm' is used to construct the progress bar;
    5. we use 'matplotlib' to plot all figures;

Inputs:
    1. sample_size : the total sample size we generate
    2. n_dim       : the number of total variables in X
    3. n_info      : the number of informative variables in data-generating process
    4. num_rep     : the number of repeatitions in Simulation 3
    5. step_size   : step size for tuning the value of c for solar;
    6. rnd_seed    : the random seed

Outputs:
    1. bsolar3_Q_opt_c_stack : the stack of variables selected by bsolar3 in 200 repetitions
    4. bsolar3_holdout_stack : the stack of variables selected by bsolar3 + hold-out average in 200 repetitions

Remarks:
    1. simul_func() : compute the simulation.
'''

class simul_plot:

    def __init__(self, sample_size, n_dim, n_info, num_rep, step_size, rnd_seed):
    ##for convinience, we define the common variable (variables we need to use multiple times) in the class as follows (xxxx as self.xxxx)

        #define the paras
        self.sample_size     = sample_size     #sample size
        self.n_dim           = n_dim           #the number of total variables in X
        self.n_info          = n_info          #number of non-zero regression coefficients in data-generating process
        self.n_repeat        = 10
        self.num_rep         = num_rep         #the number of repeatitions in Simulation 1, 2 and 3
        self.step_size       = step_size       #step size for tuning the value of c for solar
        self.rnd_seed        = rnd_seed        #the random seed for reproduction


    def simul_func(self):
    #compute 200 repeats of solar vs cv-lars-lasso and cv-cd

        solar_Q_opt_c_stack = list() #the solar variable stack of 200 repeats
        solar_holdout_stack = list() #the "solar+hold_out" variable stack of 200 repeats
        
        #to make parallel computing replicable, set random seeds
        np.random.seed(self.rnd_seed)
        # Spawn off 200 child SeedSequences to pass to child processes.
        seeds = np.random.randint(1e8, size=self.num_rep)

        ##use for-loop to compute 200 repeats
        #use 'tqdm' in for loop to construct the progress bar
        for i in tqdm(range(self.num_rep)):

            np.random.seed(seeds[i])

            #1.  call the class 'simul' from 'simul.py'
            trial1 = simul(self.sample_size, self.n_dim, self.n_info)
            #2.  generate X and Y in each repeat
            X, Y = trial1.data_gen()

            #2. compute solar3 + hold-out on the sample
            #2a. call the class 'solar' from 'solar.py'
            trial2 = solar(X, Y, self.n_repeat, self.step_size, rnd=0, lasso=False)
            #2b. compute solar
            Q_opt_c, hold_out_active_set, _, _, _, _ = trial2.fit()
            #2c. save Q(c*) (Q_opt_c) into 'Q_opt_c_stack' (the stack of 'variables selected by solar' in 200 repeats)
            solar_Q_opt_c_stack.append(Q_opt_c)
            solar_holdout_stack.append(hold_out_active_set)

        return solar_Q_opt_c_stack, solar_holdout_stack


##################################
# test if this module works fine #
##################################

'''
this part is set up to test the functionability of the class above;
you can run all the codes in this file to test if the class works;
when you call the class from this file, the codes (even functions or classes) after " if __name__ == '__main__': " will be ingored
'''

if __name__ == '__main__':

    sample_size     = 100
    n_dim           = 12
    n_info          = 5
    step_size       = -0.02
    num_rep         = 3
    rnd_seed        = 1

    trial = simul_plot(sample_size, n_dim, n_info, num_rep, step_size, rnd_seed)

    solar_Q_opt_c_stack, solar_holdout_stack = trial.simul_func()