# Subsample-ordered least-angle regression (solar): a much more stablized and accurate variable selection algorithm.
> A new regularziation algorithm for stagewise learning with ![formula](https://render.githubusercontent.com/render/math?math=L_p) shrinkage. 

## Author : Ning Xu

-----------
## Updates in next version (v1.1)
- **I merged subsample ordering into another deep learning package of mine, called  *DeepFrame* https://github.com/echushe/DeepFrame.**
- **I migrate the package to CUDA and OpenCL framework for GPU parallel computation.**
- **I introduce R and Matlab packages (internally fortran and C++)**
- **I apply subsample ordering to elastic net, adaptive lasso and graphical lasso**
- **I applied subsample ordering regularization to one of my deep learning framework, see https://github.com/echushe/DeepFrame for detail.**
------------

## About current version (v1.0)
## Subsample ordered least angel regression is a novel machine learning package for ultrahigh dimensional supervised learning, which substantially improve the sparsity and predition accuracy of lasso-type estimators in ultrahigh dimensional spaces without increasing computational load

---
## This version includes the current, stable version of the 'solarpy' package and the simulation results for solar vs CV-lars-lasso and CV-cd.
## A demonstration slides can be found under the namer "demo_slides.pdf"

## 1. Solarpy and plotting scripts
## A detailed manual with comments and explanations is available in the '.py' and '.ipynb' files as follows:

### .py files contains all functions and scripts for simulation and visualization

* "simulator.py" is the Python package for data generation in the demonstration simulation, Section 3 and simulation 1 and 2, Section 4.

  * This package depends on Python package "numpy".

* "simulator_ic.py" is the Python package for data generation in Simulation 3, Section 4.

  * This package depends on Python package "numpy".

* "costcom.py" is the Python package to compute $L_2$ error of a regression model.

  * This package depends on Python package "numpy" and "Scikit-learn".

* "solar.py" is the Python package for solar.

  * This package depends on Python packages "costcom.py", "numpy", "matplotlib", "Scikit-learn" and "tqdm".

* "solar_simul_one_shot.py" is plotting script for the demonstration simuation, Section 3.

  * This package depends on Python packages "simulator.py", "solar.py", "numpy", "matplotlib" and "tqdm".

* "simul_plot.py" is plotting script for Simuation 1, Section 4. \

  * This package depends on Python packages "simulator.py", "solar.py", "numpy", "matplotlib" and "tqdm".

* "simul_plot_p_large.py" is plotting script for Simuation 2a and 2b, Section 4.

  * This package depends on Python packages "simulator.py", "solar.py", "numpy", "matplotlib" and "tqdm".

* "simul_plot_ic.py" is plotting script for Simuation 3, Section 4.

  * This package depends on Python packages "simulator_ic.py", "solar.py", "numpy", "matplotlib" and "tqdm".

### .ipynb files generates all graphical and numerical results

* "solar_simulation_demo.ipynb" generates all the graphical and numerical results for the demonstration simuation, Section 3.

  * All graphical and numerical results will be reported in ipynb file.

  * This package depends on Python packages "solar_simul_one_shot.py".

* "solar_simulation_1.ipynb" generates all the graphical and numerical results for Simuation 1, Section 4.

  * All graphical results will be reported in ipynb file. They will also be automatically saved as pdf files in the same folder of "solar_simulation_1.ipynb".

  * All numerical results will be reported in ipynb file. Using Python packages "Pickle", numerical results will also be automatically saved as "solar_graph_n_xxx_.p" ("xxx" represent the sample size in Simulation 1).

  * This package depends on Python packages "simul_plot.py" and "Pickle".

* "solar_simulation_2a.ipynb" generates all the graphical and numerical results for Simuation 2a, Section 4.

  * All graphical results will be reported in ipynb file. They will also be automatically saved as pdf files in the same folder of "solar_simulation_2a.ipynb".

  * All numerical results will be reported in ipynb file. Using Python packages "Pickle", numerical results will also be automatically saved as "solar_graph_n_xxx_p_xxx_dell.p" ("xxx" represent the sample size and number of variables in Simulation 2a and 2b).

  * This package depends on Python packages "simul_plot_p_large.py" and "Pickle".

* "solar_simulation_2b.ipynb" generates all the graphical and numerical results for Simuation 2b, Section 4.

  * All graphical results will be reported in ipynb file. They will also be automatically saved as pdf files in the same folder of "solar_simulation_2b.ipynb".

  * All numerical results will be reported in ipynb file. Using Python packages "Pickle", numerical results will also be automatically saved as "solar_graph_n_xxx_p_xxx_dell.p" ("xxx" represent the sample size and number of variables in Simulation 2a and 2b).

  * This package depends on Python packages "simul_plot_p_large.py" and "Pickle".

* "solar_simulation_3.ipynb" generates all the graphical and numerical results for Simuation 2b, Section 4.

  * All graphical results will be reported in ipynb file. They will also be automatically saved as pdf files in the same folder of "solar_simulation_3.ipynb".

  * All numerical results will be reported in ipynb file. Using Python packages "Pickle", numerical results will also be automatically saved as "solar_graph_ic_xxx.p" ("xxx" represent the value of $\omega$ in Simulation 3).

  * This package depends on Python packages "simul_plot_ic.py" and "Pickle".


### "Supp_B_Python_package.zip" files report all the raw graphical/numerical results of simuations in HTML files. You can download and view it.

## 2. How "solarpy" works

* "solar.py" is the computation of solar, which can be used for any dataset. If you decide to copy "solar.py" to other folder and use it for other data, please also copy "costcom.py" with it and check if your Python3 library has "numpy", "matplotlib", "Scikit-learn" and "tqdm".

* At the end of each .py file we have a small script, which is coded to check if each package works (run all codes in .py files and no error is returned). However, all .py files will not produce any simulation result in our paper. The simulation results will only be produced by .ipynb files.

* When you unzip solarpy, please make sure that all .py files and .ipynb files are in the same folder (if you are not familiar with Jupyter notebook on Windows, we recommend the "Download" folder, which can be easily found in Jupyter notebook).

* When you run all codes in .ipynb to reproduce simulation results, all the graphical results will be saved in pdf and all numerical results will be saved as .p file. Please ignore all possible Windows alert.


## 3. Python environment of the development of solarpy

* IDE and dependence: solarpy is developed in Python 3.7.3 using Anaconda3 version 2019.03.

  * In Anaconda3 2019-03, we develop solarpy based on Scikit-learn 0.21.2, numpy 1.16.4, matplotlib 3.1.0 and tqdm 4.32.1.

  * All .py files are coded under Spyder 3.3.4. Hence, we highly recommend opening them in Spyder

  * All .ipynb files are coded under Jupyter lab 0.35.5. Hence, we highly recommend opening them in Jupyter Lab (or equivalently, Jupyter Notebook)

* Hardware: solarpy is developed under Debian 9.7 (equivalently, Ubuntu 18.04.2) with a 1.6 GHz i5-8250U processor and 16GB Ram.


## 4. Reproduction in Python, R and Matlab

### Reproduction in Python

* For simulation reproduction, we highly recommend using Anaconda3 2019-03 without any updates, which can be found at https://repo.continuum.io/archive/. The installation guide can be found at https://docs.anaconda.com/anaconda/install/. A quick guide on Spyder and Jupyter Notebook can be found at https://docs.anaconda.com/anaconda/user-guide/getting-started/.

* Even though .py files and .ipynb files can be opened in many other ways, we highly recommand using Spyder and Jupyter Lab/Notebook in Anaconda3 to avoid any unnecessary error and bug

###Reproduction in R and Matlab

* If necessary, solar.py can also be run in R (Rstudio, R markdown, R notebook, etc) using R package "reticulate". For detail, please see the offical page of R package reticulate' at https://rstudio.github.io/reticulate/.

* If necessary, solar.py can also be run in Matlab using the offical Python API of Matlab. For detail, please see the offical page of "Python library in Matlab" at https://www.mathworks.com/help/matlab/call-python-libraries.html?s_tid=CRUX_lftnav.

* However, due to (1) different operating system setting; (2) different version of C++, R, Rstudio and Matlab; (3) possible future updates on R, Python and Matlab, we highly recommend reproducing the simuation in Anaconda3 or Python3 to avoid any unnecessary debuging.
