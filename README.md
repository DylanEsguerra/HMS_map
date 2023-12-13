
Accounting for observation noise in equation-free forecasting: the hidden-Markov S-map


1. Ecosystems contain many species interacting with each other and the environment. Quantitative understanding of these complex systems is limited by which aspects of the ecosystem can be observed. Often relevant potential variables are left out due to them being unknown or difficult to measure. Empirical dynamic modeling and nonlinear forecasting can be used to infer unobserved ecosystem dynamics with minimal assumptions. Enthusiasm is growing for these methods across a wide range of disciplines.

2. However, as traditionally applied in ecology, EDM assumes that the available time series are observed without error. Failing to account for observation noise strongly biases estimates of Lyapunov exponents and reduces forecast accuracy. To address this limitation, we propose incorporating EDM into a hidden Markov framework and using an iterative scheme based on the Expectation Maximization (EM) algorithm to obtain filtered state and parameter estimates.

3. We evaluate the performance of this approach on simulated dynamical systems with a range of additive noise levels, as well as on insect population time series.

4. The results demonstrate improved forecast accuracy robust computation of Lyapunov exponents in the presence of observation noise.

Key Words: Hidden-Markov, S-map, Empirical dynamic modeling.


The main file contains the main HMS-map algorithm _HMSmap_lags.m_ as well as it’s dependencies in the _lag.m_ file. This file also contains the False Nearest Neighbors algorithm _f_fnn.m_ [Zhou, Y. (2019).](https://github.com/gyrheart/FNN) used to estimate the Embedding dimension and a script to calculate Lyapunov Exponents from HMS-map regression coefficients _lyapunov_QR_lags.m_.

The Simulations file contains the code necessary to replicate the results from the paper as well as a simple example _Example_Simulation.m_. Replication of the results from the paper can be done by running _Future_noise_level.m_ and modifying it to call the simulated system you are interested in (1D,2D,5D). These scripts preform smoothing, leave one out forecasting and future predictions. This file also contains code used for additional tests preformed in the appendix of the paper.

The Empirical data folder contains selected insect data from GPDD (Pendergast et al., 2010) used in (Rogers et al., 2022). The _Empirical_Example.m_ script computes the LE for each of the insect time series and can be compared with the results from (Rogers et al., 2022) in the _LE_mean.csv_ file.

The Figures file contains scripts used to create all figures from simulations in the paper. _Figures.m_ creates LE plots using _dropplot.m_ as well as plots of MSE over various noise levels. _MSE_future.m_ calculates future prediction error from the data generated by the simulation studies. _prediction_err.m_ does the same for multi step leave one out predictions. The data from these two scripts is saved in _LOO_3x3.m_ and _Future3x3.m_ which makes the corresponding figures.

The docs folder contains the code used to launch a Github pages website with instructions for how to apply HMS-map to simulated and empirical data.

[Github Repository with all code and data used in evaluation of the model](https://github.com/DylanEsguerra/HMS_map)

*Example 1* Simulation of Discrete Logistic Map with 10% observation noise
<a href="docs/Example_Simulation.html">Example 1</a>

*Example 2* Replication of case study on Global Population Dynamics Database insect Time Series
<a href="docs/Empirical_Example.html">Example 2</a>


Note that exact results will not be replicated due to random noise
