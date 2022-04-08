"""
Standard MPC for Passing through a dynamic gate
"""
import scipy.linalg as la
import numpy as np
import time
from os import system
#
from high_mpc.common.quad_index import *

#
class LQR(object):
    """
    Linear Quadratic Regulator
    """
    def __init__(self, T, dt, so_path=''):
        """
        Linear MPC for quadrotor control        
        """

        # Time constant
        self._T = T
        self._dt = dt
        self._N = int(self._T/self._dt)

        # Gravity
        self._gz = 9.81

        # Quadrotor constant
        self._w_max_yaw = 6.0
        self._w_max_xy = 3.0
        self._thrust_min = 0.0
        self._thrust_max = 30.0
        self._euler_bound = np.pi/6

        #
        # state dimension (px, py, pz,           # quadrotor position
        #                  vx, vy, vz,           # quadrotor linear velocity
        #                  roll, pitch           # quadrotor quaternion
        self._s_dim = 8
        # action dimensions (c_thrust, wx, wy, wz)
        self._u_dim = 3
        
        # cost matrix for tracking the goal point
        self._Q = np.diag([
            100, 100, 100,  # delta_x, delta_y, delta_z
            0.01, 0.01, 0.01, # delta_vx, delta_vy, delta_vz
            0.01, 0.01]) # delta_wx, delta_wy
        
        # cost matrix for the action
        self._R = np.diag([0.1, 0.1, 0.1]) # T, wx, wy, wz

        # solution of the DARE        
        self._P = None
        
        # initial state and control action
        self._quad_s0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._quad_u0 = [9.81, 0.0, 0.0]

        self._initDynamics()

    def _initDynamics(self,):
        """
        x_dot = ca.vertcat(
            vx,
            vy,
            vz,
            self._gz * pitch,
            self._gz * roll,
            thrust,
            wr,
            wp
        )
        """
        
        t = self._dt
        g = self._gz
        self._A = np.array([
        [1,0,0,t,0,0,0,0],
        [0,1,0,0,t,0,0,0],
        [0,0,1,0,0,t,0,0],
        [0,0,0,1,0,0,0, g*t],
        [0,0,0,0,1,0,g*t,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,1]
        ]        )
        self._B = np.array([
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [t,0,0],
        [0,t,0],
        [0,0,t]
        ])


    def solve(self, state):
        # # # # # # # # # # # # # # # #
        # -------- solve LQR ---------
        # # # # # # # # # # # # # # # #
        
        P = [None] * (self._N + 1)
        X = [None] * (self._N + 1)
        K = [None] * self._N
        U = [None] * self._N
        
        P[self._N] = la.solve_discrete_are(self._A, self._B, self._Q, self._R)
        
        # Iterative discrete algebraic Riccati equation (IDA)
        for i in range(self._N, 0, -1):
            P[i-1] = self._Q + self._A.T @ P[i] @ self._A - (self._A.T @ P[i] @ self._B) @ np.linalg.pinv(
            self._R + self._B.T @ P[i] @ self._B) @ (self._B.T @ P[i] @ self._A)  
            
        for i in range(self._N):
            # Calculate the optimal feedback gain K
            K[i] = -np.linalg.pinv(self._R + self._B.T @ P[i+1] @ self._B) @ self._B.T @ P[i+1] @ self._A
            U[i] = K[i] @ state

        cost = 0.0
        x0_array = np.zeros((self._N, self._s_dim + self._u_dim + 1))
        X[0] = np.array(state)
        for i in range(self._N):
            X[i+1] = self._A @ X[i] + self._B @ U[i]
            cost += self._state_cost(X[i])
            cost += self._input_cost(U[i])
            x0_array[i, :] = np.concatenate((X[i], U[i], [0]))
        
        cost += self._terminal_cost(X[self._N], P[self._N])

        opt_u = np.append(U[0], 0).reshape([self._u_dim + 1, -1])
        print("OPTIMAL CONTROL", opt_u.transpose()[0])
        print("Cost", cost)
        
        # return optimal action, and a sequence of predicted optimal trajectory.  
        return opt_u, x0_array, cost
    
    def sys_dynamics(self, dt):
        M = 1       # refinement
        DT = dt/M
        X0 = ca.SX.sym("X", self._s_dim)
        U = ca.SX.sym("U", self._u_dim)
        # #
        X = X0
        for _ in range(M):
            X = X + DT*self.f(X, U)
        # Fold
        F = ca.Function('F', [X0, U], [X])
        return F
            
    
    def _state_cost(self, x):
        return x.T @ self._Q @ x
    
    def _input_cost(self, u):
        return u.T @ self._R @ u
    
    def _terminal_cost(self, x, P):
        return x.T @ P @ x