import numpy as np
#
from high_mpc.simulation.quadrotor import Quadrotor_v0
from high_mpc.simulation.box_v0 import box_v0
#
from high_mpc.common.quad_index import *

#
class Space(object):

    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.shape = self.low.shape

    def sample(self):
        return np.random.uniform(self.low, self.high)

class DynamicGap2(object):

    def __init__(self, mpc, plan_T, plan_dt):
        self.mpc = mpc
        self.plan_T = plan_T
        self.plan_dt = plan_dt

        # parameters
        self.ball_init_pos = np.array([0.0, 0.0, -0.5]) # starting point of the ball
        self.ball_init_vel = np.array([0.0, -3.5]) # starting velocity of the ball
        self.quad_init_pos = np.array([-0.3, 0.0, 0.0]) # starting point of the quadrotor
        
        # simulation parameters ....
        self.sim_T = 1.5        # Episode length, seconds
        self.sim_dt = 0.02      # simulation time step
        self.max_episode_steps = int(self.sim_T/self.sim_dt)
        
        # Simulators, a quadrotor and a pendulum
        self.quad = Quadrotor_v0(dt=self.sim_dt)
        self.ball = box_v0(self.ball_init_pos, dt=self.sim_dt)
    
        # state space
        self.observation_space = Space(
            low=np.array([-10.0, -10.0, -10.0, -2*np.pi, -2*np.pi, -2*np.pi, -10.0, -10.0, -10.0]),
            high=np.array([10.0, 10.0, 10.0, 2*np.pi, 2*np.pi, 2*np.pi, 10.0, 10.0, 10.0]),
        )

        self.action_space = Space(
            low=np.array([0.0]),
            high=np.array([2*self.plan_T])
        )

        # reset the environment
        self.t = 0
        self.reset(init_vel=self.ball_init_vel)
    
    def seed(self, seed):
        np.random.seed(seed=seed)
    
    def reset(self, init_vel=None):
        self.t = 0
        # state for ODE
        self.quad_state = self.quad.reset(self.quad_init_pos)
        if init_vel is not None:
            self.ball_state = self.ball.reset(init_vel)
        else:
            self.ball_state = self.ball.reset()
        
        # observation, can be part of the state, e.g., postion
        # or a cartesian representation of the state
        quad_obs = self.quad.get_cartesian_state()
        ball_obs = self.ball.get_cartesian_state()
        #
        obs = (quad_obs - ball_obs).tolist()
        
        return obs

    def step(self, u=0):
        self.t += self.sim_dt
        opt_t = u
        
        print("===========================================================")
        
        #
        quad_state = self.quad.get_cartesian_state()
        ball_state = self.ball.get_cartesian_state()
        # pend_state = np.array([0, 0, 3, 0, 0, 0, 0, 0, 0])
        quad_s0 = np.zeros(8)
        quad_s0[0:3] = quad_state[0:3] - ball_state[0:3]  # relative position
        quad_s0[3:6] = quad_state[6:9] + ball_state[6:9]  # relative velocity # TODO -5
        quad_s0[6:8] = quad_state[3:5]
        quad_s0 = quad_s0.tolist()
        
        # ref_traj = quad_s0 + plan_pend_traj + self.quad_sT # in mpc state, 8d+3

        # ------------------------------------------------------------
        # run liear model predictive control
        quad_act, pred_traj, cost = self.mpc.solve(quad_s0) # in relative frame
        # ------------------------------------------------------------
        
        # back to world frame
        pred_traj[:,0:3] = pred_traj[:,0:3] + self.ball.get_cartesian_state()[0:3]
        
        # run the actual control command on the quadrotor
        # if (quad_state[4] > 0.5):
        #     quad_act = np.array([12, 0, 0, 0])
        # else:
        #     quad_act = np.array([9.81, 0, 1.0, 0])
        
        
        self.quad_state = self.quad.run(quad_act)
        # simulate one step pendulum
        self.ball_state = self.ball.run()
        
        # update the observation.
        quad_obs = self.quad.get_cartesian_state()
        ball_obs = self.ball.get_cartesian_state()
        
        obs = (quad_obs - ball_obs).tolist()
        
        pred_pend_traj_cart = pred_traj  # TODO: useless
        #
        info = {
            "quad_obs": quad_obs, 
            "quad_act": quad_act, 
            "quad_axes": self.quad.get_axes(),
            "pend_obs": ball_obs,
            "pend_corners": self.ball.get_3d_corners(),
            "pred_quad_traj": pred_traj, 
            "pred_pend_traj": pred_pend_traj_cart, 
            "opt_t": opt_t, "plan_dt": self.plan_dt,
            "quad_s0": quad_s0,
            "cost": cost}
        done = False
        if self.t >= (self.sim_T-self.sim_dt):
            done = True

        return obs, 0, done, info
    
    @staticmethod
    def _is_within_gap(gap_corners, point):
        A, B, C = [], [], []    
        for i in range(len(gap_corners)):
            p1 = gap_corners[i]
            p2 = gap_corners[(i + 1) % len(gap_corners)]
            
            # calculate A, B and C
            a = -(p2.y - p1.y)
            b = p2.x - p1.x
            c = -(a * p1.x + b * p1.y)

            A.append(a)
            B.append(b)
            C.append(c)
        D = []
        for i in range(len(A)):
            d = A[i] * point.x + B[i] * point.y + C[i]
            D.append(d)

        t1 = all(d >= 0 for d in D)
        t2 = all(d <= 0 for d in D)
        return t1 or t2

    def close(self,):
        return True

    def render(self,):
        return False

    def terminal_cost(self,x,u):
        P = np.array([[ 4.62926944e+02,  1.12579780e-13,  2.71715235e-14,
         8.39543304e+01,  3.76898971e-14, -6.76638800e-15,
         1.67233208e-14,  6.27040086e+01],
       [ 1.12579780e-13,  4.62926944e+02,  2.96208280e-13,
         2.13652440e-14,  8.39543304e+01,  1.44584455e-13,
         6.27040086e+01, -1.58557492e-14],
       [ 2.71715235e-14,  2.96208280e-13,  3.61822016e+02,
        -4.79742110e-15,  2.96141426e-14,  4.73164850e+01,
         1.76132359e-15, -4.60323355e-15],
       [ 8.39543304e+01,  2.13652440e-14, -4.79742110e-15,
         2.40774426e+01,  1.22085597e-15, -6.17333244e-15,
        -2.42034579e-15,  2.27569742e+01],
       [ 3.76898971e-14,  8.39543304e+01,  2.96141426e-14,
         1.22085597e-15,  2.40774426e+01,  2.40290197e-14,
         2.27569742e+01, -1.02861327e-14],
       [-6.76638800e-15,  1.44584455e-13,  4.73164850e+01,
        -6.17333244e-15,  2.40290197e-14,  1.23884975e+01,
         2.51466214e-14, -1.05792536e-14],
       [ 1.67233208e-14,  6.27040086e+01,  1.76132359e-15,
        -2.42034579e-15,  2.27569742e+01,  2.51466214e-14,
         2.93179270e+01, -1.98758154e-14],
       [ 6.27040086e+01, -1.58557492e-14, -4.60323355e-15,
         2.27569742e+01, -1.02861327e-14, -1.05792536e-14,
        -1.98758154e-14,  2.93179270e+01]])
        cost = x.T @ P @ x
        return cost

    def cost_l(self,x,u):
        Q = np.diag([
            100, 100, 100,  # delta_x, delta_y, delta_z
            0.0, 0.0, 0.0, # delta_vx, delta_vy, delta_vz
            0.01, 0.01]) # delta_wx, delta_wy
        R = np.diag([0.1, 0.1, 0.1, 0.1])
        
        cost = x.T @ Q @ x + u.T @ Q @ u
        return cost
    