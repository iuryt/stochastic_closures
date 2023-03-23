# Stochasticity of Turbulence Closures

This is a repository for the code from my [WHOI GFD summer school](https://gfd.whoi.edu/) project.

We use the [Dedalus package](https://github.com/DedalusProject/dedalus) to solve the system of differential equations on a doubly periodic square domain with 4096 points on each side. Then we apply a stochastic Gaussian forcing centered at k=32, with width of 2 and random phase to the model. The viscosity and linear drag were chosen to fit the power spectrum within the wave numbers solved by the model. We use adaptive time stepping for the simulations, always keeping the CFL condition stable.

The computational resources used by this project were a separate challenge. We ran each pair of simulations on [MIT's Supercloud machine](https://supercloud.mit.edu/) using 16 nodes and 768 cores. This added up to a total of 200,000 CPU hours, including simulations and the analyzes.

<img src="https://github.com/iuryt/stochastic_closures/blob/main/img/spinup.png" data-canonical-src="https://github.com/iuryt/stochastic_closures/blob/main/img/spinup.png" width="800" height="auto" />

**Figure:** Spin-up time series of kinetic energy (A) and enstrophy (B). Colored vertical dashed lines mark the snapshots used to present the energy spectrum in C. D and E depict vorticity snapshots for t=0.2 and 10, respectively.


The main scientific questions explored are:
 - How different are subgrid stresses from different nearby turbulent solutions? 
 - How fast do nearby coarse-grained solutions diverge?
 - How quickly does the subgrid stresses decorrelate compared to the LES timestep?


Check out the [video](https://youtu.be/FCil6NBZCyc?vq=hd1440) of some of those simulations.
