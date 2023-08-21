import os
import random
from fenics import *
from dolfin import *
#from mshr import *
import numpy as np
import sys

comm = MPI.comm_world
rank = MPI.rank(comm)

set_log_level(50)

learning_rate = 0.1
A, B, C, D = 0.1, 0.1, 0.1, 1.99
c_star = 0.00001
nx, ny = 60, 100
LX, LY = 120, 200
count = 0
mesh = RectangleMesh(Point(0, 0), Point(LX, LY), nx, ny)
dt = 1

time_interval = 2
dis = 120
time_equilibrate = 1800
time = time_equilibrate + dis*time_interval
num_steps = int(time/dt)
dal = 60
epsilon = 0.04
penalty = []
learning = []


data_dir = f"/home/fenics/shared"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
nopy = False

# data_dir = f"../moving_aster/"


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

u_new = Function(MFS)
(rho_new, tau_new) = split(u_new)

u_star_all = []


w = Function(FS)


class InitialConditions_w(UserExpression):
    def __init__(self, **kwargs):
        random.seed(1 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = np.sqrt(0.05)


w_init = InitialConditions_w(degree=1)
w.interpolate(w_init)

w_all_k = []

for i in range(num_steps):
    w_all_k.append(w.copy(deepcopy = True))

w_star = Function(FS)

w_star.interpolate(w_init)
k = TestFunction(FS)



lmd = 0.8
u_new = Function(MFS)
u_old = Function(MFS)
(rho_new, tau_new) = split(u_new)
(rho_old, tau_old) = split(u_old)

v_new = Function(MFS)
v_old = Function(MFS)

(eta_new, nu_new) = split(v_new)
(eta_old, nu_old) = split(v_old)
tf = TestFunction(MFS)
(q, v) = split(tf)

u_all = []
v_all = []


data = HDF5File(comm, f"{data_dir}/data_traj_dis_{dis}_time_in_{time_interval}_time_eq_{time_equilibrate}.h5", 'r')

for i in range(num_steps):

    if(rank == 0 and i % 100 == 0):
        print(f"copying step : {i}", flush = True)

    data.read(u_new, f"u_new/Vector/vector_{i}")

    u_star_all.append(u_new.copy(deepcopy = True))

data.close()


def timestep(w_all_k):
    u_new.assign(u_star_all[0])
    u_old.assign(u_star_all[0])

    u_star = Function(MFS)
    (rho_star, tau_star) = split(u_star)

    (rho_new, tau_new) = split(u_new)
    (rho_old, tau_old) = split(u_old)

    v_new = Function(MFS)
    v_old = Function(MFS)

    (eta_new, nu_new) = split(v_new)
    (eta_old, nu_old) = split(v_old)

    u_all = []


    for i in range(num_steps):

        if(rank == 0 and i % 10 == 0):
            print(f"forward:{i}", flush = True)

        a2 = (1 - rho_new)

        a4 = (1 + rho_new)/pow(rho_new, 2)
        w.assign(w_all_k[i])

        Res_0 = (rho_new - rho_old)/dt*q*dx + \
            dot(grad(rho_new), grad(q))*dx + div(w**2*tau_new)*q*dx

        Res_1 = dot((tau_new - tau_old), v)/dt*dx + (a2 + a4*dot(tau_new, tau_new))*dot(tau_new, v)*dx + \
            dot(grad(w**2*rho_new), v)*dx + inner(nabla_grad(tau_new), nabla_grad(v))*dx

        Res = Res_0 + Res_1
        u_trial = TrialFunction(MFS)

        J = derivative(Res, u_new, u_trial)

        problem = NonlinearVariationalProblem(Res, u_new, [], J)
        solver = NonlinearVariationalSolver(problem)
        #prm = solver.parameters

        solver.solve()

        u_all.append(u_old.copy(deepcopy = True))
        u_old.assign(u_new)

    v_all = []


    a1 = -1
    a3 = -(2 + rho_new)/rho_new**3

    v_old = project( -A*(u_new - u_star), MFS)



    u_star = Function(MFS)
    (rho_star, tau_star) = split(u_star)


    for i in range(num_steps):

        a1 = -1
        a3 = -(2 + rho_new)/rho_new**3

        a2 = (1 - rho_new)
        a4 = (1 + rho_new)/pow(rho_new, 2)

        if(rank == 0 and i % 10 == 0):
            print(f"backward:{i}", flush = True)

        j = num_steps - i - 1

        u_new.assign(u_all[j])
        w.assign(w_all_k[j])
        u_star.assign(u_star_all[j])

        Res3 = ((eta_old - eta_new)*q/dt*dx - D*(rho_new - rho_star)*q*dx - dot(grad(eta_new), grad(q))
                * dx + w**2*div(nu_new)*q*dx - (a1 + a3*dot(tau_new, tau_new))*(dot(nu_new, tau_new))*q*dx)

        Res4 = (dot((nu_old - nu_new), v)/dt*dx - D*dot((tau_new - tau_star), v)*dx - inner(nabla_grad(nu_new), nabla_grad(v))*dx + w**2*dot(grad(eta_new), v)*dx
                -(a2 + a4*dot(tau_new, tau_new))*dot(nu_new, v)*dx - 2*a4*dot(nu_new, tau_new)*dot(tau_new, v)*dx)

        Res_adj = Res3 + Res4
        solve(Res_adj == 0, v_new)


        v_all.append(v_old.copy(deepcopy = True))

        v_old.assign(v_new)

    v_all.reverse()

    return u_all, v_all


def update_control(u_all, v_all, w_all_k, w_star, lr):

    wt = Function(FS)
    w_trial = TrialFunction(FS)

    updated_w = []
    for i in range(num_steps):

        u_new.assign(u_all[i])
        v_new.assign(v_all[i])
        w.assign(w_all_k[i])

        Res5 = (wt - w)/lr*k*dx + C*(w - w_star)*k*dx + c_star*dot(grad(w**2), grad(k*w))*dx - 2*w*dot(tau_new, grad(eta_new))*k*dx - 2*w*rho_new*div(nu_new)*k*dx


        J = derivative(Res5, wt, w_trial)

        problem = NonlinearVariationalProblem(Res5, wt, [], J)
        solver = NonlinearVariationalSolver(problem)
        #prm = solver.parameters


        solver.solve()

        w.assign(wt)

        updated_w.append(wt.copy(deepcopy = True))

    return updated_w



def cost_function(u_all, w_all_k, w_star):
    """
    cost_function(u_all)

    Function to compute the cost for a given trajectory
    """
    u_new = Function(MFS)
    (rho_new, tau_new) = split(u_new)
    u_star = Function(MFS)
    (rho_star, tau_star) = split(u_star)
    # Terminal state penalty
    u_new.assign(u_all[-1])
    u_star.assign(u_star_all[-1])
    cost_functional = (0.5 * A * (rho_new - rho_star)**2
                       + 0.5 * B * dot((tau_new - tau_star),
                                       (tau_new - tau_star))) * dx

    terminal_cost = assemble(cost_functional)
    # stage state penalty

    stage_state_cost = 0.0
    #print("Start stage calculation")
    for i in range(num_steps):

        u_new.assign(u_all[i])
        w.assign(w_all_k[i])
        u_star.assign(u_star_all[i])

        cost_functional = (0.5 * D * (rho_new - rho_star)**2
                           + 0.5 * D * dot((tau_new-tau_star),
                                           (tau_new-tau_star))
                           + 0.5 * C * (w - w_star)**2 + 0.25*c_star*dot(grad(w**2), grad(w**2))) * dx

        stage_state_cost += dt * assemble(cost_functional)

    return (terminal_cost + stage_state_cost)


def armijo(u_all, v_all, w_all_k, w_star):

    cost = 0.0

    gradf = C*(w - w_star) - 2*w*dot(tau_new, grad(eta_new)) - 2*w*rho_new*div(nu_new) - c_star*w*div(grad(w**2))
    for i in range(num_steps):

        u_new.assign(u_all[i])
        v_new.assign(v_all[i])
        w.assign(w_all_k[i])

        Res = -dot(gradf, gradf)
        #test = project(Res, FS)
        cost_functional = Res*dx
        cost += dt*assemble(cost_functional)

    return cost


while(count < dal):



    if(rank == 0):
        print(f"count:{count}", flush  = True)

    lr = learning_rate


    (u_all, v_all) = timestep(w_all_k)
    penalty.append(cost_function(u_all, w_all_k, w_star))

    #Backtracking implementation
    if(count != 0):

        while((penalty[count] > penalty[count - 1] + lr*epsilon*armj) and lr > 0.0001):

            lr = 0.1*lr
            print(f"Correcting learning rate to {lr}", flush = True)
            w_all_k = update_control(u_backup, v_backup, w_backup, w_star, lr)
            (u_all, v_all) = timestep(w_all_k)
            penalty[count] = cost_function(u_all, w_all_k, w_star)

        learning.append(lr)

    armj = armijo(u_all, v_all, w_all_k, w_star)
    w_backup = w_all_k.copy()
    (u_backup, v_backup) = (u_all, v_all)
    w_all_k = update_control(u_all, v_all, w_all_k, w_star, lr)

    if (count % 1 == 0 and count > 40):

        Hdf = HDF5File(mesh.mpi_comm(), f'{data_dir}/data_mpi_time_int_{time_interval}_dis_{dis}_time_eq_{time_equilibrate}_D_{D}_count_{count}.h5', "w")
        Hdf.write(mesh, "mesh")

        for t in range(num_steps):

            u_new.assign(u_all[t])
            Hdf.write(u_new, f"u_new/Vector/vector_{t}")
            w.assign(w_all_k[t])
            Hdf.write(w, f"w/Vector/vector_{t}")

        Hdf.close()

    if(rank == 0):
        np.savez(f"{data_dir}/cost_mpi_time_int_{time_interval}_dis_{dis}_time_eq_{time_equilibrate}_D_{D}.npz", J = penalty, lr = learning)
    count = count + 1



if (rank == 0):
    np.savez(f"{data_dir}/cost_mpi_time_int_{time_interval}_dis_{dis}_time_eq_{time_equilibrate}_D_{D}.npz", J = penalty, lr = learning)


