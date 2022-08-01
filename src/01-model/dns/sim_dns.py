import numpy as np
from mpi4py import MPI
import time
import pathlib

from dedalus import public as de
from dedalus.extras import flow_tools
import param_dns as param
import forcing

import logging
logger = logging.getLogger(__name__)
np.seterr(over="raise")

# Bases and domain
x_basis = de.Fourier('x', param.N, interval=(0, param.L), dealias=3/2)
y_basis = de.Fourier('y', param.N, interval=(0, param.L), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64, mesh=param.mesh)

# Stochastic forcing
dkx = dky = 2 * np.pi / param.L
draw = forcing.incomp_shell_2d(param.kf, param.kfw, dkx, dky, param.seed)
Fx = domain.new_field()
Fy = domain.new_field()
kx = domain.elements(0)
ky = domain.elements(1)

# Problem
problem = de.IVP(domain, variables=['ψ'])
problem.parameters['L'] = param.L
problem.parameters['ν'] = param.ν
problem.parameters['α'] = param.α
problem.parameters['Fx'] = Fx
problem.parameters['Fy'] = Fy
problem.substitutions['ux'] = "dy(ψ)"
problem.substitutions['uy'] = "-dx(ψ)"
problem.substitutions['ω'] = "dx(uy) - dy(ux)"
problem.substitutions['e'] = "(ux*ux + uy*uy) / 2"
problem.substitutions['z'] = "ω*ω / 2"
problem.substitutions['Lap(a)'] = "dx(dx(a)) + dy(dy(a))"
problem.substitutions['Adv(a)'] = "ux*dx(a) + uy*dy(a)"
problem.substitutions['mean(a)'] = "integ(a) / L**2"
problem.substitutions['Fω'] = "dx(Fy) - dy(Fx)"
problem.add_equation("dt(ω) - ν*Lap(ω) + α*ω = -Adv(ω) + Fω", condition="nx != 0 or ny != 0")
problem.add_equation("ψ = 0", condition="nx == 0 and ny == 0")

# Build solver
solver = problem.build_solver(param.ts)
solver.stop_sim_time = param.stop_sim_time
solver.stop_wall_time = param.stop_wall_time
solver.stop_iteration = param.stop_iteration
logger.info('Solver built')

# Initial conditions
if pathlib.Path('restart.h5').exists():
    _, dt = solver.load_state('restart.h5', -1)
    cshape = solver.state["ψ"]['c'].shape
    noise = np.random.randn(*cshape) + 1j*np.random.randn(*cshape)
    k2 = kx**2 + ky**2
    solver.state["ψ"]["c"] += param.noise_amp * noise / ((k2==0) + k2**(3/4)) / np.sqrt(2 * np.pi)
    if not param.enable_CFL:
        dt = param.safety * dt
    solver.stop_sim_time += solver.sim_time
else:
    dt = param.dt
    
# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', iter=param.snapshots_iter, max_writes=10, mode='overwrite')
snapshots.add_task("ψ", layout='c')
scalars = solver.evaluator.add_file_handler('scalars', iter=param.scalars_iter, max_writes=100, mode='overwrite')
scalars.add_task("mean(e)", name='E')
scalars.add_task("mean(z)", name='Z')
scalars.add_task("mean(-α*2*e)", name='εα')
scalars.add_task("mean(-α*2*z)", name='ηα')
scalars.add_task("mean(ν*(ux*Lap(ux) + uy*Lap(uy)))", name='εν')
scalars.add_task("mean(ν*ω*Lap(ω))", name='ην')

# CFL
if param.enable_CFL:
    CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=param.safety,
                         max_change=1.5, min_change=0.5, max_dt=param.dt, threshold=0.05)
    CFL.add_velocities(('ux', 'uy'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=param.print_iteration)
flow.add_property("mean(e)", name='E')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        if param.enable_CFL:
            dt = CFL.compute_dt()
        # Change forcing
        Fx['c'], Fy['c'] = draw(kx, ky)
        # Project onto grid space and scale
        Fx['g'] *= (param.ε / dt)**0.5
        Fy['g'] *= (param.ε / dt)**0.5
        solver.step(dt)
        if (solver.iteration-1) % param.print_iteration == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Mean KE/M = %f' %flow.max('E'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
