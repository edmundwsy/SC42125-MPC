# MPC course project: flying a quadrotor through a fast-moving gate

![img](https://img.shields.io/badge/STATUS-Building-green?style=for-the-badge)

This project is a course project for the Model Predictive Control course (SC42125). We'd like to apply MPC to a challenging practical problem: control a quadrotor flying through a fast-moving gate. With the help of knowledge learned in the course, we will linearize the control problem, use MPC to find a solution and analysis its feasibility and stability.




### Installation

Clone the repo

```
git clone git@github.com:edmundwsy/SC42125-MPC.git
```

Installation Dependencies:

```
cd high_mpc
pip install -r requirements.txt
```

Add the repo path to your PYTHONPATH by adding the following to your ~/.bashrc

```
export PYTHONPATH=${PYTHONPATH}:/path/to/high_mpc
```

### Run

Standard MPC

```
cd high_mpc
python3 run_mpc.py
```

### Thanks

The simulation codes in this project is based on [high_mpc](https://github.com/uzh-rpg/high_mpc), provided by [Robotics and Perception Group, University of Zurich](https://github.com/uzh-rpg). Their impressive work has been a great inspiration to us.