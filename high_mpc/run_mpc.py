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
from high_mpc.simulation.dynamic_gap import DynamicGap
from high_mpc.simulation.dynamic_gap_linear import DynamicGap2
from high_mpc.mpc.mpc import MPC
from high_mpc.mpc.linear_mpc import MPC2
from high_mpc.simulation.animation import SimVisual

import csv
#


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_video', type=bool, default=False,
                        help="Save the animation as a video file")
    return parser


def run_mpc(env, init_param = [-0.5, -0.3, -5], write=False):
    #
    env.reset(np.array([0.0, init_param[2]]), np.array([init_param[0], 0, init_param[1], 1, 0, 0, 0, 0, 0, 0]))
    t, n = 0, 0
    t0 = time.time()
    print(init_param)
    if write:
        csv_file = "Names.csv"
        try:
            # print(info)
            with open(csv_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                t_temp = 0
                while t < env.sim_T:
                    t = env.sim_dt * n
                    _, _, _, info = env.step()
                    t_now = time.time()
                    t_temp += t_now - t0
                    # print(t_now - t0)
                    #
                    t0 = time.time()
                    #
                    n += 1
                    update = False
                    if t > env.sim_T:
                        update = True
                    yield [info, t, update]
                    temp_list = []
                    temp_list.extend(info["quad_obs"])

                    flat_list = [item for sublist in info["quad_act"]
                                for item in sublist]
                    temp_list.extend(flat_list)

                    temp_list.extend(info["pend_obs"])

                    # flat_list = [item for sublist in info["pred_quad_traj"] for item in sublist]
                    # temp_list.extend(flat_list)

                    # flat_list = [item for sublist in info["pred_pend_traj"] for item in sublist]
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

                    # print(Q - env.mpc._Q)
                    # print(R - env.mpc._R)
                    # print(P - env.mpc._P)

                    u = np.array(info["quad_act"])
                    x = np.array(info["quad_s0"])[:, np.newaxis]
                    stage_cost = 0.5 * (x.transpose().dot(Q).dot(x) + u.transpose().dot(R).dot(u))
                    terminal_cost = 0.5 * (x.transpose().dot(P).dot(x))
                    # print("stage cost", stage_cost)
                    # print("terminal cost", terminal_cost)
                    temp_list.append(stage_cost[0][0])
                    temp_list.append(terminal_cost[0][0])

                    writer.writerow(temp_list)
            print(env.catch_flag)
            print('Written')
        except IOError:
            print("I/O error")
    else:
        while t < env.sim_T:
            t = env.sim_dt * n
            _, _, _, info = env.step()
            t_now = time.time()
            # print(t_now - t0)
            #
            t0 = time.time()
            #
            n += 1
            update = False
            if t > env.sim_T:
                update = True
            yield [info, t, update]
        print(env.catch_flag)
    return env.catch_flag


def main():
    #
    args = arg_parser().parse_args()
    #
    plan_T = 0.45   # Prediction horizon for MPC
    plan_dt = 0.05  # Sampling time step for MPC
    # saved mpc model (casadi code generation)
    so_path = "./mpc/saved/mpc_v1.so"
    #
    mpc = MPC(T=plan_T, dt=plan_dt, so_path=so_path)
    mpc2 = MPC2(T=plan_T, dt=plan_dt, so_path=so_path)
    env1 = DynamicGap(mpc, plan_T, plan_dt)
    env2 = DynamicGap2(mpc2, plan_T, plan_dt)
    env = env2

    #
    sim_visual = SimVisual(env)

    #
    init_param =  [-0.5, -0.3, -5] # init x of quad, z of quad, vz of ball
    run_mpc(env,init_param,write=True)

    # a = np.arange(-20,12,4)
    # b = np.arange(-20,12,4)
    # c = np.arange(-20,12,4)
    # caught = np.zeros((len(a), len(b), len(c)))
    # print(caught.shape)
    # for idx_i, i in enumerate(a):
    #     for idx_j, j in enumerate(b):
    #         for idx_k, k in enumerate(c):
    #             init_param = [i, j, k]
    #             if run_mpc(env,init_param,write=False):
    #                 caught[idx_i, idx_j, idx_k] = 1
    #                 print('True')
    # print(caught)
    # with open('test4.npy', 'wb') as f:
    #     np.save(f, caught)

    run_frame = partial(run_mpc, env, init_param, write=True)
    ani = animation.FuncAnimation(sim_visual.fig, sim_visual.update, frames=run_frame,
                                init_func=sim_visual.init_animate, interval=100, blit=True, repeat=False)

    # #
    # if args.save_video:
    #     writer = animation.writers["ffmpeg"]
    #     writer = writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    #     ani.save("MPC_0.mp4", writer=writer)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
