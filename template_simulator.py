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

import numpy as np

from template_model import CART_LIMIT, state_derivative


WALL_RESTITUTION = 0.15


class DoubleInvertedPendulumSimulator:
    def __init__(self, state=None, dt=0.04):
        self.dt = float(dt)
        self.state = np.zeros(6, dtype=float) if state is None else np.asarray(state, dtype=float).copy()

    def reset(self, state):
        self.state = np.asarray(state, dtype=float).copy()

    def step(self, force):
        force = float(force)
        state = self.state
        dt = self.dt

        k1 = state_derivative(state, force)
        k2 = state_derivative(state + 0.5 * dt * k1, force)
        k3 = state_derivative(state + 0.5 * dt * k2, force)
        k4 = state_derivative(state + dt * k3, force)

        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if next_state[0] < -CART_LIMIT:
            next_state[0] = -CART_LIMIT
            next_state[3] = abs(next_state[3]) * WALL_RESTITUTION
        elif next_state[0] > CART_LIMIT:
            next_state[0] = CART_LIMIT
            next_state[3] = -abs(next_state[3]) * WALL_RESTITUTION

        self.state = next_state
        return self.state.copy()
