import os
import random
from fenics import *
from dolfin import *
from mshr import *
import numpy as np
import sys

comm = MPI.comm_world
rank = MPI.rank(comm)

lr1 = 0.1 # learning rate
lr2 = 0.1 

A, B, C, D, E = 1.0, 1.0, 1.0, 1.0, 1.0
nx = ny = 60
LX = LY = 128
count = 0
mesh = RectangleMesh(Point(0, 0), Point(LX, LY), nx, ny)
dt = 1
time = 500
num_steps = int(time/dt)
dal = 25
lr = np.zeros(dal)

for i in range(dal - 4):
    lr[i] = lr1

for i in range(dal - 4, dal):
    lr[i] = lr2

penalty = np.zeros(dal)
Hdf = HDF5File(mesh.mpi_comm(), f'/home/saptorshi/Desktop/hpcc_codes/data_test.h5', "w")
####################################################################################################


class PeriodicBoundary(SubDomain):

    def inside(self, x, on_boundary):
        return bool((near(x[0], 0) or near(x[1], 0)) and
                    (not ((near(x[0], 0) and near(x[1], LY)) or
                          (near(x[0], LX) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], LX) and near(x[1], LY):
            y[0] = x[0] - LX
            y[1] = x[1] - LY
        elif near(x[0], LX):
            y[0] = x[0] - LX
            y[1] = x[1]
        else:   # near(x[1], 127)
            y[0] = x[0]
            y[1] = x[1] - LY


#####################################################################################################
V = VectorElement("Lagrange", mesh.ufl_cell(), 1, dim=2)
Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
MFS = FunctionSpace(mesh, MixedElement(
    [Q, V]), constrained_domain=PeriodicBoundary())
FS = FunctionSpace(mesh, Q, constrained_domain=PeriodicBoundary())
#####################################################################################################


u_star = Function(MFS, '/home/saptorshi/Desktop/hpcc_codes/solution_10000_0.08.xml')
#Hdf.write(u_star, f"u_star/Vector/vector_0")

# Split Mixed Functions


(rho_star, tau_star) = split(u_star)
###################################

w = Function(FS)


class InitialConditions_w(UserExpression):
    def __init__(self, **kwargs):
        random.seed(1 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = np.sqrt(0.08)


w_init = InitialConditions_w(degree=1)
w.interpolate(w_init)

w_all_k = np.empty([len(w.vector()[:]), num_steps])

for i in range(num_steps):
    w_all_k[:, i] = w.vector()[:]


w_star = Function(FS)
w_star.vector().set_local(w_all_k[:, 0])
#########################################################
djdw = Function(FS)
k = TestFunction(FS)

# INTIAL CONDITION GRADIENT


class InitialConditions_djdw(UserExpression):
    def __init__(self, **kwargs):
        random.seed(1 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = 0.0  # costgradient


djdw_init = InitialConditions_djdw(degree=1)
# djdw.interpolate(djdw_init)

#########################################################


# Adjoint functional


lmd = 0.8







def update_control(u_all, v_all, w_all_k, w_star):

    wt = Function(FS)
    w_trial = TrialFunction(FS)

    for i in range(num_steps):

        u_new.vector().set_local(u_all[:, i])
        v_new.vector().set_local(v_all[:, i])
        w.vector().set_local(w_all_k[:, i])

        Res5 = (wt - w)/lr[count]*k*dx + C*(w - w_star)*k*dx - \
            2*w*dot(tau_new, grad(eta_new))*k*dx - 2*w*rho_new*div(nu_new)*k*dx
        
        J = derivative(Res5, wt, w_trial)

        problem = NonlinearVariationalProblem(Res5, wt, [], J)
        solver = NonlinearVariationalSolver(problem)
        #prm = solver.parameters
    

        solver.solve()

        w.assign(wt)
        w_all_k[:, i] = wt.vector()[:]

    return w_all_k

############################################################################################

# COST CALCULATION


def cost_function(u_all, u_star, w_all_k, w_star):
    """
    cost_function(u_all)

    Function to compute the cost for a given trajectory
    """

    
    # Terminal state penalty
    u_new.vector().set_local(u_all[:, -1])

    cost_functional = (0.5 * A * (rho_new - rho_star)**2
                       + 0.5 * B * dot((tau_new - tau_star),
                                       (tau_new - tau_star))**2
                       + 0.5 * C * dot(grad(rho_new), grad(rho_new))) * dx

    terminal_cost = assemble(cost_functional)
    # stage state penalty

    stage_state_cost = 0.0
    #print("Start stage calculation")
    for i in range(u_all.shape[-1]):
        u_new.vector().set_local(u_all[:, i])
        w.vector().set_local(w_all_k[:, i])

        cost_functional = (0.5 * D * (rho_new - rho_star)**2
                           + 0.5 * E * dot((tau_new-tau_star),
                                           (tau_new-tau_star))**2
                           + 0.5 * C * (w - w_star)**2) * dx

        stage_state_cost += dt * assemble(cost_functional)

    return (terminal_cost + stage_state_cost)




Hdf.write(mesh, "mesh")

while(count < dal):

    if(rank == 0):
        print(f"count:{count}")
    

    tf = TestFunction(MFS)
    (q, v) = split(tf)

    u_new = Function(
        MFS, '/home/saptorshi/Desktop/hpcc_codes/solution_10000_0.05.xml')
    u_old = Function(
        MFS, '/home/saptorshi/Desktop/hpcc_codes/solution_10000_0.05.xml')


    temp = Function(MFS)
    temp_array = np.empty(len(u_old.vector()[:]))
    

    
      
    (rho_new, tau_new) = split(u_new)

    (rho_old, tau_old) = split(u_old)

    v_new = Function(MFS)
    v_old = Function(MFS)

    (eta_new, nu_new) = split(v_new)
    (eta_old, nu_old) = split(v_old)

    u_all = np.empty([len(u_old.vector()[:]), num_steps])


    for i in range(num_steps):
        
        if(rank == 0):
            print(f"forward:{i}")

        a2 = (1 - rho_new)

        a4 = (1 + rho_new)/pow(rho_new, 2)
        w.vector().set_local(w_all_k[:, i])

        Res_0 = (rho_new - rho_old)/dt*q*dx + \
            dot(grad(rho_new), grad(q))*dx + div(w**2*tau_new)*q*dx

        Res_1 = dot((tau_new - tau_old), v)/dt*dx + (a2 + a4*dot(tau_new, tau_new))*dot(tau_new, v)*dx + \
            dot(grad(w**2*rho_new), v)*dx + inner(nabla_grad(tau_new), nabla_grad(v))*dx - \
            lmd*(0.5*dot(grad(dot(tau_new, tau_new)), v)*dx + div(tau_new)*dot(tau_new, v)*dx -
                 dot(dot(tau_new, nabla_grad(tau_new)), v)*dx)

        Res = Res_0 + Res_1
        u_trial = TrialFunction(MFS)

        J = derivative(Res, u_new, u_trial)

        problem = NonlinearVariationalProblem(Res, u_new, [], J)
        solver = NonlinearVariationalSolver(problem)
        #prm = solver.parameters

        solver.solve()
        
        u_all[:, i] = u_new.vector()[:]
        u_old.assign(u_new)

    v_all = np.empty([len(v_old.vector()[:]), num_steps])

    u_new.vector().set_local(u_all[:, -1])
    #Hdf.write(u_new, f"u_new/Vector/vector_0")

    
    a1 = -1
    a3 = -(2 + rho_new)/rho_new**3


    temp_array = -A*(u_new.vector()[:] - u_star.vector()[:])
    temp.vector().set_local(temp_array)
    v_old = temp.copy()

    Hdf.write(v_old, f"v_old/Vector/vector_0")
    


 

    for i in range(num_steps):

        if(rank == 0):
            print(f"backward:{i}")

        j = num_steps - i - 1

        u_new.vector().set_local(u_all[:, j])
        w.vector().set_local(w_all_k[:, j])

        Res3 = ((eta_old - eta_new)*q/dt*dx - D*(rho_new - rho_star)*q*dx - dot(grad(eta_new), grad(q))
                * dx + w**2*div(nu_new)*q*dx - (a1 + a3*dot(tau_new, tau_new))*(dot(nu_new, tau_new))*q*dx)

        Res4 = (dot((nu_old - nu_new), v)/dt*dx - E*dot((tau_new - tau_star), v)*dx - inner(nabla_grad(nu_new), nabla_grad(v))*dx + w**2*dot(grad(eta_new), v)*dx
                - (a2 + a4*dot(tau_new, tau_new))*dot(nu_new, v)*dx - 2*a4*dot(nu_new, tau_new)*dot(tau_new, v)*dx +
                lmd*(-div(nu_new)*dot(tau_new, v)*dx + 2*div(tau_new)*dot(nu_new, v)*dx - dot(grad(dot(nu_new, tau_new)), v)*dx + dot(dot(tau_new, grad(nu_new)), v)*dx
                     - dot(dot(grad(tau_new), nu_new), v)*dx))

        Res_adj = Res3 + Res4
        solve(Res_adj == 0, v_new)

        v_all[:, j] = v_new.vector()[:]
        v_old.assign(v_new)

    

    penalty[count] = cost_function(u_all, u_star, w_all_k, w_star)
    #print(penalty[count])

    djdw.interpolate(djdw_init)
    w_all_k = update_control(u_all, v_all, w_all_k, w_star)



    if (count % 1 == 0):

        
        for t in range(num_steps):

            u_new.vector().set_local(u_all[:, t])
            Hdf.write(u_new, f"u_new/Vector/vector_{count}_{t}")

            w.vector().set_local(w_all_k[:, t]**2)

            Hdf.write(w, f"w/Vector/vector_{count}_{t}")
    count = count + 1

np.savez(f'/home/saptorshi/Desktop/hpcc_codes/cost.npz', J = penalty)
Hdf.close()


