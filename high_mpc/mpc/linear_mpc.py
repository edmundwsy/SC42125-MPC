"""
Standard MPC for Passing through a dynamic gate
"""
import casadi as ca
import numpy as np
import time
from os import system
#
from high_mpc.common.quad_index import *

#
class MPC2(object):
    """
    Nonlinear MPC
    """
    def __init__(self, T, dt, so_path='./nmpc.so'):
        """
        Nonlinear MPC for quadrotor control        
        """
        self.so_path = so_path

        # Time constant
        self._T = T
        self._dt = dt
        self._N = int(self._T/self._dt)

        # Gravity
        self._gz = 9.81

        # Quadrotor constant
        self._w_max_yaw = 6.0
        self._w_max_xy = 1.0
        self._thrust_min = 0.0
        self._thrust_max = 30.0

        #
        # state dimension (px, py, pz,           # quadrotor position
        #                  vx, vy, vz,           # quadrotor linear velocity
        #                  roll, pitch           # quadrotor quaternion
        self._s_dim = 8
        # action dimensions (c_thrust, wx, wy, wz)
        self._u_dim = 4
        
        # cost matrix for tracking the goal point
        self._Q = np.diag([
            100, 100, 100,  # delta_x, delta_y, delta_z
            10, 10, 10, # delta_vx, delta_vy, delta_vz
            10, 10]) # delta_wx, delta_wy
        
        # cost matrix for the action
        self._R = np.diag([0.1, 0.1, 0.1, 0.1]) # T, wx, wy, wz

        # initial state and control action
        self._quad_s0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._quad_u0 = [9.81, 0.0, 0.0, 0.0]

        self._initDynamics()

    def _initDynamics(self,):
        # # # # # # # # # # # # # # # # # # # 
        # ---------- Input States -----------
        # # # # # # # # # # # # # # # # # # # 

        px, py, pz = ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz')
        #
        pitch, roll = ca.SX.sym('pitch'), ca.SX.sym('roll')
        #
        vx, vy, vz = ca.SX.sym('vx'), ca.SX.sym('vy'), ca.SX.sym('vz')

        # -- conctenated vector
        self._x = ca.vertcat(px, py, pz, vx, vy, vz, pitch, roll) 

        # # # # # # # # # # # # # # # # # # # 
        # --------- Control Command ------------
        # # # # # # # # # # # # # # # # # # #

        thrust, wr, wp, wz = ca.SX.sym('thrust'), ca.SX.sym('wr'), \
            ca.SX.sym('wp'), ca.SX.sym('wz')
        
        # -- conctenated vector
        self._u = ca.vertcat(thrust, wr, wp, wz)
        
        # # # # # # # # # # # # # # # # # # # 
        # --------- System Dynamics ---------
        # # # # # # # # # # # # # # # # # # #

        x_dot = ca.vertcat(
            vx,
            vy,
            vz,
            - self._gz * pitch,
            - self._gz * roll,
            thrust,
            wr,
            wp
        )

        #
        self.f = ca.Function('f', [self._x, self._u], [x_dot], ['x', 'u'], ['ode'])
                
        # # Fold
        F = self.sys_dynamics(self._dt)
        fMap = F.map(self._N, "openmp") # parallel
        
        # # # # # # # # # # # # # # # 
        # ---- loss function --------
        # # # # # # # # # # # # # # # 

        # placeholder for the quadratic cost function
        Delta_s = ca.SX.sym("Delta_s", self._s_dim)
        Delta_u = ca.SX.sym("Delta_u", self._u_dim)        
        
        #        
        cost_goal = Delta_s.T @ self._Q @ Delta_s 
        cost_u = Delta_u.T @ self._R @ Delta_u

        #
        f_cost_goal = ca.Function('cost_goal', [Delta_s], [cost_goal])
        f_cost_u = ca.Function('cost_u', [Delta_u], [cost_u])

        #
        # # # # # # # # # # # # # # # # # # # # 
        # # ---- QP Optimization -----
        # # # # # # # # # # # # # # # # # # # #
        self.nlp_w = []       # variables
        self.nlp_w0 = []      # initial guess of variables
        self.lbw = []         # lower bound of the variables, lbw <= nlp_x
        self.ubw = []         # upper bound of the variables, nlp_x <= ubw
        #
        self.mpc_obj = 0      # objective 
        self.nlp_g = []       # constraint functions
        self.lbg = []         # lower bound of constrait functions, lbg < g
        self.ubg = []         # upper bound of constrait functions, g < ubg

        u_min = [self._thrust_min, -self._w_max_xy, -self._w_max_xy, -self._w_max_yaw]
        u_max = [self._thrust_max,  self._w_max_xy,  self._w_max_xy,  self._w_max_yaw]
        x_bound = ca.inf
        x_min = [-x_bound for _ in range(self._s_dim)]
        x_max = [+x_bound for _ in range(self._s_dim)]
        #
        g_min = [0 for _ in range(self._s_dim)]
        g_max = [0 for _ in range(self._s_dim)]

        X = ca.SX.sym("X", self._s_dim, self._N+1)
        U = ca.SX.sym("U", self._u_dim, self._N)
        P = ca.SX.sym("P", self._s_dim)  # initial state
        #
        X_next = fMap(X[:, :self._N], U)
        
        # "Lift" initial conditions
        self.nlp_w += [X[:, 0]]
        self.nlp_w0 += self._quad_s0
        self.lbw += x_min
        self.ubw += x_max
        
        # # starting point.
        self.nlp_g += [X[:, 0] - P[0:self._s_dim]]
        self.lbg += g_min
        self.ubg += g_max
        
        for k in range(self._N):
            #
            self.nlp_w += [U[:, k]]
            self.nlp_w0 += self._quad_u0
            self.lbw += u_min
            self.ubw += u_max
            
            delta_s_k = X[:, k+1]
            cost_goal_k = f_cost_goal(delta_s_k)
        
            delta_u_k = U[:, k]-[self._gz, 0, 0, 0]
            cost_u_k = f_cost_u(delta_u_k)

            self.mpc_obj = self.mpc_obj + cost_goal_k + cost_u_k 

            # New NLP variable for state at end of interval
            self.nlp_w += [X[:, k+1]]
            self.nlp_w0 += self._quad_s0
            self.lbw += x_min
            self.ubw += x_max

            # Add equality constraint
            self.nlp_g += [X_next[:, k] - X[:, k+1]]
            self.lbg += g_min
            self.ubg += g_max

        # nlp objective
        nlp_dict = {'f': self.mpc_obj, 
            'x': ca.vertcat(*self.nlp_w),
            'p': P, 
            'g': ca.vertcat(*self.nlp_g) }        
        
        # # # # # # # # # # # # # # # # # # # 
        # -- qpoases            
        # # # # # # # # # # # # # # # # # # # 
        # nlp_options ={
        #     'verbose': False, \
        #     "qpsol": "qpoases", \
        #     "hessian_approximation": "gauss-newton", \
        #     "max_iter": 100, 
        #     "tol_du": 1e-2,
        #     "tol_pr": 1e-2,
        #     "qpsol_options": {"sparse":True, "hessian_type": "posdef", "numRefinementSteps":1} 
        # }
        # self.solver = ca.nlpsol("solver", "sqpmethod", nlp_dict, nlp_options)
        # cname = self.solver.generate_dependencies("mpc_v1.c")  
        # system('gcc -fPIC -shared ' + cname + ' -o ' + self.so_path)
        # self.solver = ca.nlpsol("solver", "sqpmethod", self.so_path, nlp_options)
        

        # # # # # # # # # # # # # # # # # # # 
        # -- ipopt
        # # # # # # # # # # # # # # # # # # # 
        ipopt_options = {
            'verbose': False, \
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0, 
            "print_time": False
        }
        
        self.solver = ca.nlpsol("solver", "ipopt", nlp_dict, ipopt_options)
        # jit (just-in-time compilation)
        # print("Generating shared library........")
        # cname = self.solver.generate_dependencies("mpc_v1.c")  
        # system('gcc -fPIC -shared -O3 ' + cname + ' -o ' + self.so_path) # -O3
        
        # # reload compiled mpc
        # print(self.so_path)
        # self.solver = ca.nlpsol("solver", "ipopt", self.so_path, ipopt_options)

    def solve(self, state):
        # # # # # # # # # # # # # # # #
        # -------- solve NLP ---------
        # # # # # # # # # # # # # # # #
        #
        self.sol = self.solver(
            x0=self.nlp_w0,
            p=state, 
            lbx=self.lbw, 
            ubx=self.ubw, 
            lbg=self.lbg, 
            ubg=self.ubg)
        #
        sol_x0 = self.sol['x'].full()
        opt_u = sol_x0[self._s_dim:self._s_dim+self._u_dim]

        # Warm initialization
        self.nlp_w0 = list(sol_x0[self._s_dim+self._u_dim:2*(self._s_dim+self._u_dim)]) + list(sol_x0[self._s_dim+self._u_dim:])
        
        #
        print("SOLUTION SHAPE", sol_x0.shape)
        print("OPTIMAL CONTROL", opt_u.transpose())
        print("Current state", sol_x0[:self._s_dim].transpose())
        x0_array = np.reshape(sol_x0[:-self._s_dim], newshape=(-1, self._s_dim+self._u_dim))
        
        # return optimal action, and a sequence of predicted optimal trajectory.  
        return opt_u, x0_array
    
    def sys_dynamics(self, dt):
        M = 1       # refinement
        DT = dt/M
        X0 = ca.SX.sym("X", self._s_dim)
        U = ca.SX.sym("U", self._u_dim)
        # #
        X = X0
        for _ in range(M):
            # --------- RK4------------
            # k1 =DT*self.f(X, U)
            # k2 =DT*self.f(X+0.5*k1, U)
            # k3 =DT*self.f(X+0.5*k2, U)
            # k4 =DT*self.f(X+k3, U)
            # #
            # X = X + (k1 + 2*k2 + 2*k3 + k4)/6     
            
            # --------- Euler ----------
            X = X + DT*self.f(X, U)
        # Fold
        F = ca.Function('F', [X0, U], [X])
        return F
            