# MPC course project: A Model Predictive Control Approach for Catching Ball with Quadrotor


![img](https://img.shields.io/badge/STATUS-Building-green?style=for-the-badge)

This project is a course project for the Model Predictive Control course (SC42125). We'd like to apply Linear MPC to a challenging practical problem: catch a flying ball with a quadrotor. With the help of knowledge learned in the course, we will linearize the control problem, use MPC to find a solution and analysis its feasibility and stability with DARE and terminal set theory.




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

### Demo



https://user-images.githubusercontent.com/44539400/167367098-674c9fed-670f-4b21-9c20-bdfc391d1bd7.mp4



### Thanks

The simulation codes in this project is based on [high_mpc](https://github.com/uzh-rpg/high_mpc), provided by [Robotics and Perception Group, University of Zurich](https://github.com/uzh-rpg). Their impressive work has been a great inspiration to us.
