#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import Circle

from neural_network import BasicPolicyNetwork
from template_model import CART_LIMIT, L1, L2, compute_energies
from template_simulator import DoubleInvertedPendulumSimulator


rcParams['text.usetex'] = False
rcParams['axes.grid'] = True
rcParams['lines.linewidth'] = 2.0
rcParams['axes.labelsize'] = 'xx-large'
rcParams['xtick.labelsize'] = 'xx-large'
rcParams['ytick.labelsize'] = 'xx-large'


obstacles = [
    {'x': 0.0, 'y': 0.6, 'r': 0.3},
]

scenario = 1  # 1 = down-down start, 2 = up-up start.
if scenario == 1:
    initial_state = np.array([0.0, 0.9 * np.pi, 0.9 * np.pi, 0.0, 0.0, 0.0], dtype=float)
elif scenario == 2:
    initial_state = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
else:
    raise ValueError('Scenario not defined.')

target_position = 0.0

simulator = DoubleInvertedPendulumSimulator(state=initial_state, dt=0.04)


def pendulum_bars(state):
    state = np.asarray(state, dtype=float).flatten()
    line_1_x = np.array([state[0], state[0] + L1 * np.sin(state[1])])
    line_1_y = np.array([0.0, L1 * np.cos(state[1])])
    line_2_x = np.array([line_1_x[1], line_1_x[1] + L2 * np.sin(state[2])])
    line_2_y = np.array([line_1_y[1], line_1_y[1] + L2 * np.cos(state[2])])
    return np.stack((line_1_x, line_1_y)), np.stack((line_2_x, line_2_y))

def compute_reward(state, force):
    pos, theta1, theta2, dpos, dtheta1, dtheta2 = state

    # Upright = theta ≈ 0
    reward = 0.0

    reward -= theta1**2
    reward -= theta2**2
    reward -= 0.1 * dtheta1**2
    reward -= 0.1 * dtheta2**2
    reward -= 0.01 * force**2

    return reward


fig = plt.figure(figsize=(16, 9))
plt.ion()

ax1 = plt.subplot2grid((1, 2), (0, 0))
ax2 = plt.subplot2grid((1, 2), (0, 1))

ax2.set_ylabel('Reward')
ax2.set_xlabel('time [s]')
ax2.yaxis.set_label_position('right')
ax2.yaxis.tick_right()

ax1.axhline(0, color='black')

bar1, = ax1.plot([], [], '-o', linewidth=5, markersize=10)
bar2, = ax1.plot([], [], '-o', linewidth=5, markersize=10)

for obstacle in obstacles:
    ax1.add_artist(Circle((obstacle['x'], obstacle['y']), obstacle['r']))

ax1.set_xlim(-1.8, 1.8)
ax1.set_ylim(-1.2, 1.2)
ax1.set_axis_off()

for wall_x in (-CART_LIMIT, CART_LIMIT):
    ax1.axvline(wall_x, color='tab:red', linestyle='--', linewidth=1.5, alpha=0.7)

reward_line, = ax2.plot([], [], label='reward', color='tab:green')

fig.suptitle('NN controller active. Q or Esc to quit.', y=0.98)
fig.align_ylabels()
fig.tight_layout(rect=[0, 0, 1, 0.95])

control = {
    'current_force': 0.0,
    'max_force': 12.0,
}

policy_network = BasicPolicyNetwork(input_dim=10, hidden_dim=16, force_limit=6.0, seed=42)
use_random_policy = True


def _angle_features(theta):
    return np.sin(theta), np.cos(theta)


def build_observation(state, current_time, prev_force, target_x=0.0):
    state = np.asarray(state, dtype=float).flatten()
    pos, theta1, theta2, dpos, dtheta1, dtheta2 = state
    sin1, cos1 = _angle_features(theta1)
    sin2, cos2 = _angle_features(theta2)

    return np.array(
        [
            pos - target_x,
            sin1,
            cos1,
            sin2,
            cos2,
            dpos,
            dtheta1,
            dtheta2,
            prev_force,
            current_time,
        ],
        dtype=float,
    )


def nn_policy(observation):
    """Basic NN policy with optional random exploration output."""
    if use_random_policy:
        return policy_network.random_action()
    return policy_network.forward(observation)


def compute_control_force(state, current_time):
    obs = build_observation(
        state=state,
        current_time=current_time,
        prev_force=control['current_force'],
        target_x=target_position,
    )
    force = nn_policy(obs)
    return float(np.clip(force, -control['max_force'], control['max_force']))

time_history = [0.0]
reward_history = [compute_reward(simulator.state, 0.0)]


def on_key_press(event):
    if event.key in ('escape', 'q'):
        plt.close(fig)


fig.canvas.mpl_connect('key_press_event', on_key_press)


def refresh_plot():
    bar1.set_data(*pendulum_bars(simulator.state)[0])
    bar2.set_data(*pendulum_bars(simulator.state)[1])

    reward_line.set_data(time_history, reward_history)

    ax2.relim()
    ax2.autoscale_view(scalex=False, scaley=True)
    ax2.set_xlim(0.0, max(5.0, time_history[-1] + simulator.dt))

    ax1.set_title(f'Force: {control["current_force"]:.1f} N')

    fig.canvas.draw_idle()

plt.show(block=False)

step_times = []
while plt.fignum_exists(fig.number):
    tic = time.time()
    current_time = time_history[-1]
    control['current_force'] = compute_control_force(simulator.state, current_time)
    state = simulator.step(control['current_force'])
    elapsed = time.time() - tic

    current_time = current_time + simulator.dt
    reward = compute_reward(state, control['current_force'])

    time_history.append(current_time)
    reward_history.append(reward)
    step_times.append(elapsed)

    refresh_plot()
    plt.pause(simulator.dt)


if step_times:
    mean_step = np.round(np.mean(step_times) * 1000.0)
    std_step = np.round(np.std(step_times) * 1000.0)
    print(f'mean runtime: {mean_step} ms +- {std_step} ms for simulation step')