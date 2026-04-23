import numpy as np


M0 = 0.6
M1 = 0.19
M2 = 0.19
L1 = 0.5
L2 = 0.5
L1_HALF = L1 / 2.0
L2_HALF = L2 / 2.0
J1 = (M1 * L1_HALF ** 2) / 3.0
J2 = (M2 * L2_HALF ** 2) / 3.0
GRAVITY = 9.80665
FRICTION_CART = 0.03
FRICTION_ANGLE = 0.01
CART_LIMIT = 1.35


def solve_accelerations(state, force):
    pos, theta1, theta2, dpos, dtheta1, dtheta2 = np.asarray(state, dtype=float)

    h1 = M0 + M1 + M2
    h2 = M1 * L1_HALF + M2 * L1
    h3 = M2 * L2_HALF
    h4 = M1 * L1_HALF ** 2 + M2 * L1 ** 2 + J1
    h5 = M2 * L2_HALF * L1
    h6 = M2 * L2_HALF ** 2 + J2
    h7 = (M1 * L1_HALF + M2 * L1) * GRAVITY
    h8 = M2 * L2_HALF * GRAVITY

    cos1 = np.cos(theta1)
    cos2 = np.cos(theta2)
    cos12 = np.cos(theta1 - theta2)
    sin1 = np.sin(theta1)
    sin2 = np.sin(theta2)
    sin12 = np.sin(theta1 - theta2)

    matrix = np.array(
        [
            [h1, h2 * cos1, h3 * cos2],
            [h2 * cos1, h4, h5 * cos12],
            [h3 * cos2, h5 * cos12, h6],
        ],
        dtype=float,
    )

    rhs = np.array(
        [
            force + h2 * dtheta1 ** 2 * sin1 + h3 * dtheta2 ** 2 * sin2 - FRICTION_CART * dpos,
            h7 * sin1 - h5 * dtheta2 ** 2 * sin12 - FRICTION_ANGLE * dtheta1,
            h5 * dtheta1 ** 2 * sin12 + h8 * sin2 - FRICTION_ANGLE * dtheta2,
        ],
        dtype=float,
    )

    return np.linalg.solve(matrix, rhs)


def state_derivative(state, force):
    _, _, _, dpos, dtheta1, dtheta2 = np.asarray(state, dtype=float)
    ddpos, ddtheta1, ddtheta2 = solve_accelerations(state, force)

    return np.array([dpos, dtheta1, dtheta2, ddpos, ddtheta1, ddtheta2], dtype=float)


def compute_energies(state):
    _, theta1, theta2, dpos, dtheta1, dtheta2 = np.asarray(state, dtype=float)

    e_kin_cart = 0.5 * M0 * dpos ** 2
    e_kin_p1 = 0.5 * M1 * (
        (dpos + L1_HALF * dtheta1 * np.cos(theta1)) ** 2
        + (L1_HALF * dtheta1 * np.sin(theta1)) ** 2
    ) + 0.5 * J1 * dtheta1 ** 2
    e_kin_p2 = 0.5 * M2 * (
        (dpos + L1 * dtheta1 * np.cos(theta1) + L2_HALF * dtheta2 * np.cos(theta2)) ** 2
        + (L1 * dtheta1 * np.sin(theta1) + L2_HALF * dtheta2 * np.sin(theta2)) ** 2
    ) + 0.5 * J2 * dtheta2 ** 2

    e_kin = e_kin_cart + e_kin_p1 + e_kin_p2
    e_pot = M1 * GRAVITY * L1_HALF * np.cos(theta1) + M2 * GRAVITY * (
        L1 * np.cos(theta1) + L2_HALF * np.cos(theta2)
    )

    return e_kin, e_pot


def obstacle_distances(state, obstacles):
    pos, theta1, theta2 = np.asarray(state, dtype=float)[:3]

    node0_x = pos
    node0_y = 0.0
    node1_x = node0_x + L1 * np.sin(theta1)
    node1_y = node0_y + L1 * np.cos(theta1)
    node2_x = node1_x + L2 * np.sin(theta2)
    node2_y = node1_y + L2 * np.cos(theta2)

    distances = []
    for obstacle in obstacles:
        radius = obstacle['r'] * 1.05
        distances.extend(
            [
                np.hypot(node0_x - obstacle['x'], node0_y - obstacle['y']) - radius,
                np.hypot(node1_x - obstacle['x'], node1_y - obstacle['y']) - radius,
                np.hypot(node2_x - obstacle['x'], node2_y - obstacle['y']) - radius,
            ]
        )

    return np.asarray(distances, dtype=float)