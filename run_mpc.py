import os
import time
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#
from functools import partial

#
from high_mpc.simulation.linear_env import LinearEnv
from high_mpc.mpc.linear_mpc import LinearMPC as MPC
from high_mpc.mpc.lqr import LQR
from high_mpc.simulation.animation import SimVisual

import csv
#


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_video', type=bool, default=True,
                        help="Save the animation as a video file")
    return parser


def run_mpc(env,write=False):
    #
    env.reset()
    t, n = 0, 0
    t0 = time.time()
    csv_file = "No.csv"
    caught = False
    try:
        # print(info)
        if write:
            with open(csv_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                t_temp = 0
                while t < env.sim_T:
                    t = env.sim_dt * n
                    _, _, _, info = env.step()
                    
                    relative_pos = np.array(info['quad_s0'][0:3])
                    print("Caught the ball: ", relative_pos, np.linalg.norm(relative_pos))
                    if np.linalg.norm(relative_pos) < 1e-1/2:
                        caught = True
                        break
                    
                    t_now = time.time()
                    t_temp += t_now - t0
                    print(t_now - t0)
                    #
                    t0 = time.time()
                    #
                    n += 1
                    update = False
                    if t > env.sim_T:
                        update = True
                    yield [info, t, update]
                    
                    # writer.writerow(info)
                    temp_list = []
                    temp_list.extend(info["quad_obs"])

                    flat_list = [item for sublist in info["quad_act"]
                                for item in sublist]
                    temp_list.extend(flat_list)

                    temp_list.extend(info["ball_obs"])

                    # flat_list = [item for sublist in info["pred_quad_traj"] for item in sublist]
                    # temp_list.extend(flat_list)

                    # flat_list = [item for sublist in info["pred_ball_traj"] for item in sublist]
                    # temp_list.extend(flat_list)

                    temp_list.extend(info["quad_s0"])

                    temp_list.append(t_temp)

                    temp_list.append(info["cost"])

                    # calculate costs

                    Q = env.mpc._Q

                    # cost matrix for the action
                    R = env.mpc._R  # T, wx, wy, wz

                    # solution of the DARE
                    P = env.mpc._P

                    u = np.array(info["quad_act"])
                    x = np.array(info["quad_s0"])[:, np.newaxis]
                    stage_cost = 0.5 * (x.transpose().dot(Q).dot(x) + u.transpose().dot(R).dot(u))
                    terminal_cost = 0.5 * (x.transpose().dot(P).dot(x))
                    print("stage cost", stage_cost)
                    print("terminal cost", terminal_cost)
                    temp_list.append(stage_cost[0][0])
                    temp_list.append(terminal_cost[0][0])

                    writer.writerow(temp_list)
                print("caught: ", caught)
        else:
            while t < env.sim_T:
                t = env.sim_dt * n
                _, _, _, info = env.step()
                relative_pos = np.array(info['quad_s0'][0:3])
                print("Caught the ball: ", relative_pos, np.linalg.norm(relative_pos))
                if np.linalg.norm(relative_pos) < 1e-1/2:
                        caught = True
                        break
                t_now = time.time()
                # print(t_now - t0)
                #
                t0 = time.time()
                #
                n += 1
                update = False
                if t > env.sim_T:
                    update = True
                if np.linalg.norm(relative_pos) < 1e-1:
                        caught = True
                yield [info, t, update]
            print("caught: ", caught)
    except IOError:
        print("I/O error")
    return caught


def main():
    #
    args = arg_parser().parse_args()
    #
    plan_T = 0.4   # Prediction horizon for MPC
    plan_dt = 0.05  # Sampling time step for MPC
    # saved mpc model (casadi code generation)
    so_path = ""
    #
    init_param = []
    init_param.append(np.array([0.0, 0.0, -.5])) # starting point of the ball
    init_param.append(np.array([0.0, -3])) # starting velocity of the ball
    init_param.append(np.array([-0.3, 0.0, 0.0])) # starting point of the quadrotor

    mpc = MPC(T=plan_T, dt=plan_dt, so_path=so_path)
    lqr = LQR(T=plan_T, dt=plan_dt)
    env = LinearEnv(mpc, plan_T, plan_dt, init_param)

    #
    sim_visual = SimVisual(env)

    #
    run_mpc(env)

    # Test the MPC
    # a = np.arange(-8,9,2) # p_z
    # b = np.arange(-8,9,2) # p_x
    # c = np.arange(-10,4,2) # v_z
    # caught = np.zeros((len(a), len(b), len(c)))
    # print(caught.shape)
    # for idx_i, i in enumerate(a):
    #     for idx_j, j in enumerate(b):
    #         for idx_k, k in enumerate(c):
    #             init_param = [np.array([0.0, 0.0, i]), np.array([0.0, k]), np.array([j, 0.0, 0.0])]
    #             env = DynamicGap2(mpc, plan_T, plan_dt, init_param)
    #             print(i,j,k)
    #             if run_mpc(env,write=False): # comment the yield above to proceed
    #                 caught[idx_i, idx_j, idx_k] = 1                    
    #                 print('True')
    # print(caught)
    # with open('test4.npy', 'wb') as f:
    #     np.save(f, caught)

    run_frame = partial(run_mpc, env)
    ani = animation.FuncAnimation(sim_visual.fig, sim_visual.update, frames=run_frame,
                                  init_func=sim_visual.init_animate, interval=100, blit=True, repeat=False)

    #
    if args.save_video:
        writer = animation.writers["ffmpeg"]
        writer = writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        ani.save("MPC_quad_ball.mp4", writer=writer)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
