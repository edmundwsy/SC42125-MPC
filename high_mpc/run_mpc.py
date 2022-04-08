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


def run_mpc(env):
    #
    env.reset()
    t, n = 0, 0
    t0 = time.time()
    csv_file = "Names.csv"
    try:
        # print(info)
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            t_temp = 0
            while t < env.sim_T:
                t = env.sim_dt * n
                _, _, _, info = env.step()
                
                relative_pos = np.array(info['quad_s0'][0:3])
                print("Reached the goal:", relative_pos, np.linalg.norm(relative_pos))
                if np.linalg.norm(relative_pos) < 1e-1:
                    print("!!!!!!!!!!!!!")
                    break
                
                t_now = time.time()
                t_temp += t_now - t0
                print(t_now - t0)
                #
                t0 = time.time()
                #
                n += 1
                update = False
                if t >= env.sim_T:
                    update = True
                yield [info, t, update]
                

                
                # writer.writerow(info)
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

                Q = np.diag([100, 100, 100, 0.01, 0.01, 0.01, 0.01, 0.01])

                # cost matrix for the action
                R = np.diag([0.1, 0.1, 0.1, 0.1])  # T, wx, wy, wz

                # solution of the DARE
                P = np.array([[4.62926944e+02,  1.12579780e-13,  2.71715235e-14, 8.39543304e+01,  3.76898971e-14, -6.76638800e-15,
                               1.67233208e-14,  6.27040086e+01],
                              [1.12579780e-13,  4.62926944e+02,  2.96208280e-13,
                               2.13652440e-14,  8.39543304e+01,  1.44584455e-13,
                               6.27040086e+01, -1.58557492e-14],
                              [2.71715235e-14,  2.96208280e-13,  3.61822016e+02,
                               -4.79742110e-15,  2.96141426e-14,  4.73164850e+01,
                               1.76132359e-15, -4.60323355e-15],
                              [8.39543304e+01,  2.13652440e-14, -4.79742110e-15,
                               2.40774426e+01,  1.22085597e-15, -6.17333244e-15,
                               -2.42034579e-15,  2.27569742e+01],
                              [3.76898971e-14,  8.39543304e+01,  2.96141426e-14,
                               1.22085597e-15,  2.40774426e+01,  2.40290197e-14,
                               2.27569742e+01, -1.02861327e-14],
                              [-6.76638800e-15,  1.44584455e-13,  4.73164850e+01,
                               -6.17333244e-15,  2.40290197e-14,  1.23884975e+01,
                               2.51466214e-14, -1.05792536e-14],
                              [1.67233208e-14,  6.27040086e+01,  1.76132359e-15,
                               -2.42034579e-15,  2.27569742e+01,  2.51466214e-14,
                               2.93179270e+01, -1.98758154e-14],
                              [6.27040086e+01, -1.58557492e-14, -4.60323355e-15,
                               2.27569742e+01, -1.02861327e-14, -1.05792536e-14,
                               -1.98758154e-14,  2.93179270e+01]])

                u = np.array(info["quad_act"])
                x = np.array(info["quad_s0"])[:, np.newaxis]
                stage_cost = 0.5 * (x.transpose().dot(Q).dot(x) + u.transpose().dot(R).dot(u))
                terminal_cost = 0.5 * (x.transpose().dot(P).dot(x))
                print("stage cost", stage_cost)
                print("terminal cost", terminal_cost)
                temp_list.append(stage_cost[0][0])
                temp_list.append(terminal_cost[0][0])

                writer.writerow(temp_list)
    except IOError:
        print("I/O error")


def main():
    #
    args = arg_parser().parse_args()
    #
    plan_T = 0.8   # Prediction horizon for MPC
    plan_dt = 0.1  # Sampling time step for MPC
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
    run_mpc(env)

    run_frame = partial(run_mpc, env)
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
