**Python package &quot;solarpy&quot;**

Last updated at April 15, 2021

### for details, please see that paper at https://arxiv.org/abs/2007.15707

### #0. This zip file includes

- the Python3 packages (both parallel and sequential vesions) for

  - solar;
  - solar + hold out;
  - bsolar;
  - bolasso;

- the step-by-step demostration for bolasso, bsolar and solar;
- the codes, graphs, tables, raw results and data for

  - Example 3 (Section 2.1);
  - Simulations 1, 2, 3 (Section 3.1, 3.2, 3.3);
  - Real-world application (Section 4);

### #1. Read this first carefully: package description

#### #1(a). Ananconda3

- solar, bsolar and hold-out average are developed on Ubuntu 20.04 using Anaconda3 version 2020-11. **We highly recommend using Anaconda3 since, by default, it satisfies all the dependencies (#2 below), relieving you from Python package management.**

#### #1(b). Coding style

In developing this package, we rigorously adhere to the following programming paradigm:

- **the comment-code ratio is larger than .** We only briefly list the file structure in #3 below (explaining the purpose of each file), since in each &quot;.py&quot; and &quot;.ipynb&quot; file we thoroughly explain

  - the meaning of every step;
  - the meaning of inputs and output;
  - the purpose of each function;
  - how each step corresponds to the paper.

- **the simulations and examples are in the &quot;.ipynb&quot; files; the &quot;.py&quot; files contain only the supporting functions for simulations and examples**.
- at the end of each &quot;.py&quot; file, we add a testing module for debugging. Simply run each &quot;.py&quot; file at your terminal. If no bugs are reported, the package is bug-free.
- The Python files automatically export the raw simulation/example results as &quot;.html&quot; files, which can be found in the &quot;./raw\_results&quot; folder; the numerical results are automatically saved as &quot;.p&quot; files in the &quot;numerical\_result&quot; subfolder of each simulation folder.

#### #1(c). Making the comparisons fair

**To ensure a fair sparsity/accuracy comparison between bsolar and bolasso (section 3.3),** we specifically code the simulation function &quot;simul\_plot\_parallel.py&quot; as follows:

- step 1: before simulation starts, reset the Numpy random seed to (aka the **father seed = 0** ).
- step 2: immediately after step 1,

  - use the father seed to generate **200 child seeds** via Numpy.
  - use each child seed to generate the data for one repetition in each scenario of Simulation 3, i.e., the child seed is used to generate the data in repetition for bsolar, solar and bolasso.

- Step 2 guarantees that, **even if you run Algorithm A and B in different files, Algorithm A and Algorithm B are evaluated on the same training/test/validation data in each repetition** , which is necessary for parallel or distributive computation.

**To ensure a fair runtime comparison between bsolar and bolasso** , we specifically code the competitors identically according to the following structure:

- step 1: compute the subsample selection frequency on the sample. Bsolar-3 trains 3 solar and bolasso trains 256 lasso.
- step 2: based on the subsample selection frequency of each variable, rank the variables in decreasing order.
- step 3: set the threshold (or 1) and select the variables with subsample selection frequencies larger than .
- To show that **the runtime difference is purely because we replace lasso with solar in a boostrap ensemble** ,

  - we **use the same code in steps 1 to 3** for bsolar and bolasso.
  - **the only difference** is that we use different algorithms (solar or lasso) to estimate subsample selection frequency in the for-loop of step 1.
  - we also **use the same parallel scheme (coded the same)** for bsolar and bolasso.
  - to address any claims that our parallel scheme is too aggressive, we also report **the runtime of bolasso using the default SciKit-learn parallel scheme, which is optimzed for the trainnig of lasso and, hence, bolasso.**

#### #1(d). Replication

- To replicate the simulations, **after you read though the detailed explanations and comments in each &quot;.ipynb&quot; file** , simply

  - pen each &quot;.ipynb&quot; file in Jupyter Lab,
  - click the &quot;Kernel&quot; menu
  - click &quot;Restart Kernel and Run All Cells&quot;.
  - **you may want to read the comments in &quot;.ipynb&quot; files carefully before you replicate 200-repetition bolasso simulations in Simulation 3 (since it could take very long time).**

- Python Package dependence: reported below in **#2.**
- Hardware settings: all the results are computed on a desktop with:

  - 8-core i9-9900K CPU;
  - CPU cooler: Corsair iCUE H150i ELITE CAPELLIX Liquid CPU Cooler, which prevents the CPU from thermal throttling by keeping CPU temperature below 65° Celsius at 100% load of each core.
  - 32 GB RAM;
  - GPU: _&quot;Asus GeForce RTX 3090 TUF Gaming&quot;_ for CUDA.

### #2. Package dependence

#### the package has been tested on Anaconda3 version 2020-11 and Ubuntu 20.04.1 with the following dependence:

- **Python 3.7.10** ;
- **scikit-learn 0.24.1** ;
- the C++ library **openBLAS** or, equivalently, **Intel MKL** ;

Note that

    - Intel MKL is known to be faster at matrix operations than openBLAS;
    - _2021/02/13: we found several openBLAS bugs on Apple M1 chips; there were no problems on AMD or Intel CPUs_.

- **numpy 1.19.2** or, equivalently, **jax.numpy** of **jax 0.2.10** ;

  - if you are working on large data (such as picture recognition, natural language processing or text mining with dimension ), the repeated bolasso simulations could take **days**. Hence, we strongly recommend

    - using **jax** for much quicker matrix operations in bolasso;
    - using **incomplete Cholesky decomposition** instead of the default one in Scikit-learn for bolasso.
    - if neither offer an improvement in bolasso runtime, reprogram your package using [**Numba**](http://numba.pydata.org/) **+** [**PyCuda**](https://developer.nvidia.com/pycuda) and run it with a **Nvidia GPU**. With Numba and PyCuda, the bsolar-3 runtime can be further reduced by a factor of around for .

- **jupyter 1.0.0** and **jupyterlab 3.0.11** ;
- **Matplotlib 3.3.4** ;

  - _2021/02/13: we disabled Plotly temporarily and use Matplotlib + Pandas instead; Plotly causes Javascript errors in Jupyter Lab and fails to show the graph/table when exporting ipyb files into html._

- **statsmodels 0.12.2** ;
- **tqdm 4.59.0** ;

  - _2021/02/13: among the runtime measure packages, tqdm takes the least runtime overhead (80ns) and guarantees the accuracy of runtime measurement._

- **joblib 1.0.1**.

### #3. File structure

#### Read this first:

_the detailed explanations and comments can be found at the beginning and each step of the &quot;.py&quot; or &quot;.ipynb&quot; file. We only introduce the structure here._

#### #3(a) ./raw\_results: all the raw results with detailed explanations (in .html files) of Example 3, Simulation 1 to 3 and real-world application

- **Application\_Houseprice\_2010\_linear** and **Application\_Houseprice\_2010\_log.html** : the raw result of the real-world application (Section 4) with detailed explanations.
- **Example\_Post\_solar\_test.html** : the raw results of the hold-out average (Example 3, Section 2.1) with detailed explanations.
- **solar\_demo** , **bsolar\_walkthrough** and **bolasso\_walkthrough.html** : the step-by-step demonstration of solar, bsolar and bolasso.
- **Simul\_1a** , **Simul\_1b** , **Simul\_1c** and **Simul\_1d.html** : the raw results of Simulation 1 (solar and &quot;solar + hold out&quot; vs lasso under different p/n, Section 3.1) with detailed explanations.
- **Simul\_2.html** : the raw results of Simulation 2 (solar vs lasso with different irrepresentable conditions, Section 3.2) with detailed explanations.
- **Simul\_3\_runtime\_built\_in\_parallel** and **Simul\_3\_runtime\_joblib\_parallel.html** : the raw results of the runtime comparison in Simulation 3 (Section 3.3) with detailed explanations.
- **Simul\_3\_subsample\_frequency.html** : the raw results of the subsample selection frequency comparison in Simulation 3 (bsolar vs bolasso, Section 3.3) with detailed explanations.
- **Simul\_3a** , **Simul\_3b** and **Simul\_3c.html** : the raw results of Simulation 3 (Section 3.3) with detailed explanations.

#### #3(b) ./Section\_2.1\_Holdout\_example: the Python package for Example 3 (Section 2.1).

- **./raw\_results** : the folder of all the numerical results of Example 3, saved as a &quot;.p&quot; file by the Pickle package;
- **simulator.py** : the data generating package.
- **costcom.py** : the package to compute the error of the regression.
- **solar\_parallel** and **solar.py** : the packages for solar under Joblib parallel computing and sequential computing.
- **Example\_Post\_solar\_test.ipynb** : the simulation for Example 3.

#### #3(c) ./Section\_3.1\_simul\_1: the Python package for Simulation 1 (Section 3.1).

- **./figures** : the folder of all detailed graphical results of Simulation 1, saved as &quot;.pdf&quot;;
- **./raw\_results** : same as above;
- **costcom** , **simulator** and **solar\_parallel.py** : same as above;
- **debug.sh** : (for Mac OS and Linux only) the bash file for bug testing of all &quot;.py&quot; and &quot;.ipyb&quot; files here.

  - in Mac OS or Linux, open your terminal and switch to this folder; run this bash file with the &quot;bash&quot; command;
  - this produces all the test plots, results, and tables;
  - provided you do not find any errors during the procedure and the bash file ends normally, there is no bug in any of the packages in this folder.

- **solar\_holdout\_average\_parallel.py** : the solar packages with the hold-out average under the customized Joblib parallel computing scheme;
- **simul\_plot.py** : all the simulation functions (computation and plotting functions) that solar and lasso require in Simulation 1;
- **simul\_plot\_holdout\_average\_parallel.py** : all the simulation functions that &quot;solar + hold out&quot; requires in Simulation 1;
- **Simul\_1a** , **Simul\_1b** , **Simul\_1c** and **Simul\_1d.ipynb** : the simulations of the lasso-solar comparison for , and .

#### #3(d) ./Section\_3.2\_simul\_2: the Python package for Simulation 1 (Section 3.2).

- **./figures** : same as above;
- **./raw\_results** : same as above;
- **costcom** and **solar\_paralle.py** and **debug.sh** : same as above;
- **simul\_plot\_ic.py** : all the simulation functions (computation and plotting functions) that Simulation 2 requires;
- **simulator\_ic.py** : the data generating package for Simulation 2 only.
- **Simul\_2.ipynb.ipynb** : the simulation for robustness comparison under different irrepresentable conditions.

#### #3(e) ./Section\_3.3\_simul\_3: the Python package for Simulation 3 (Section 3.3).

- **./0\_Subsample\_selection\_frequency** : the folder for the subsample selection frequency comparison in Simuation 3;

  - **./raw\_results** : same as above;
  - **costcom** , **simulator** , **solar** and **solar\_paralle.py** and **debug.sh** : same as above;
  - **bolasso\_parallel.py** : the bolasso package under the customized Joblib parallel computing scheme;
  - **bootstrap\_demo\_parallel.py** : the simulation functions required for the subsample selection frequency comparison;
  - **bsolar.py** : the bsolar package under the sequential computing scheme;
  - **Simul\_3\_subsample\_frequency.ipynb** : the simulation for the subsample selection frequency comparison.

- **./1\_Sparsity\_accuracy** : the folder for the sparsity and accuracy comparison in Simuation 3;

  - **./raw\_results** : same as above;
  - **costcom** , **simulator** , **solar** , **solar\_parallel** , **bolasso\_parallel** and **bsolar.py** and **debug.sh** : same as above;
  - **bsolar\_parallel.py** : the bsolar package under customized Joblib parallel computing scheme;
  - **simul\_plot\_parallel.py** : all the simulation functions that the sparsity comparison requires;
  - **Simul\_3a** , **Simul\_3b** and **Simul\_3c.ipynb** : the simulation for the sparsity and accuracy comparison.

- **./2\_Runtime** : the folder for runtime comparison at Simuation 3;

  - **./raw\_results** : the folder for all the numerical results, saved as a &quot;.p&quot; file by the Pickle package;
  - **costcom.py, simulator.py, debug.sh, solar.py, solar\_paralle.py, bolasso\_parallel.py, bsolar.py, bsolar\_parallel.py** : same as above;
  - **simul\_built\_in\_parallel** and **simul\_joblib\_parallel.py** : the simulation functions, respectively, for the Scikit-learn build-in parallel scheme and the customized Joblib scheme;
  - **Simul\_3\_runtime\_built\_in\_parallel** and **Simul\_3\_runtime\_joblib\_parallel.ipynb** : the simulation for the runtime comparison.

#### #3(d) ./Section\_4\_application: the result of real-world application (Section 4).

- **Data2010.csv** : data;
- **House2010\_linear** and **House2010\_log.csv** : variables in linear and log forms, based on Data2010.csv;
- **costcom.py, simulator.py, solar.py** : same as above;
- **Application\_Houseprice\_2010\_linear** and **Application\_Houseprice\_2010\_log.ipynb** : the real-world application in both linear and log forms.

#### #3(e) ./Demo: the step-by-step walkthrough of &quot;bolasso\_parallel&quot;, &quot;solar\_parallel&quot; and &quot;bsolar\_parallel.py&quot;.

- **costcom.py, simulator.py, debug.sh, solar.py, solar\_parallel.py** : same as above.
- **solar\_simul\_demo.py** : all the simulation functions that the solar demostration needs.
- **bsolar\_walkthough** , **bolasso\_walkthough** and **solar\_demo.ipynb** : the step-by-step demonstration for bsolar, bolasso and solar.
