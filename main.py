import numpy as np
from simulator import Simulator, centerline

sim = Simulator()

# track + car constants
TRACK_LENGTH = 104.75797893910486
DS = 0.1                            # spacing b/n path points
N = int(TRACK_LENGTH / DS)
LR = 0.79                           # dist from CG to rear axle
DRAG = 6.1333333333333327e-03
A_LAT_MAX = 7.5
A_BRAKE = 6.0
A_ACCEL = 4.0
V_CAP = 26.0
STEER_LIMIT = 0.7

# sample the centerline at startup
s_grid = np.arange(N) * DS
path = centerline(s_grid)


def circular_smooth(arr, window):
    # moving average that wraps around the loop
    kernel = np.ones(window) / window
    padded = np.concatenate([arr[-window:], arr, arr[:window]])
    return np.convolve(padded, kernel, mode='same')[window:-window]

# centerline is only 40 points, smooth it out
px = circular_smooth(path[:, 0], 15)
py = circular_smooth(path[:, 1], 15)

# finite differences for 1st and 2nd derivatives
dx = (np.roll(px, -1) - np.roll(px, 1)) / (2 * DS)
dy = (np.roll(py, -1) - np.roll(py, 1)) / (2 * DS)
ddx = (np.roll(px, -1) - 2 * px + np.roll(px, 1)) / DS ** 2
ddy = (np.roll(py, -1) - 2 * py + np.roll(py, 1)) / DS ** 2

# curvature of a parametric curve
kappa = np.abs(dx * ddy - dy * ddx) / np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-9)
kappa = circular_smooth(kappa, 25)

# max speed profile based on points
v_profile = np.minimum(np.sqrt(A_LAT_MAX / np.maximum(kappa, 1e-6)), V_CAP)

for _ in range(3):
    # make sure you can brake into corners
    for i in range(N - 1, -1, -1):
        j = (i + 1) % N
        v_profile[i] = min(v_profile[i], np.sqrt(v_profile[j] ** 2 + 2 * A_BRAKE * DS))
    # make sure you can accelerate out of corners
    for i in range(N):
        j = (i - 1) % N
        v_profile[i] = min(v_profile[i], np.sqrt(v_profile[j] ** 2 + 2 * A_ACCEL * DS))

_last_idx = 0 # remember where the car is on the path


def controller(x):
    global _last_idx

    xpos = x[0]
    ypos = x[1]
    phi = np.mod(x[2], 2 * np.pi)
    v = x[3]
    theta = x[4]

    # find nearest path point
    pos = np.array([xpos, ypos])
    window = np.arange(_last_idx - 20, _last_idx + 120) % N
    _last_idx = int(window[np.argmin(np.sum((path[window] - pos) ** 2, axis=1))])

    # look further ahead when going faster
    ld = np.clip(0.55 * v + 3.5, 3.5, 14.0)
    target = path[int((_last_idx + ld / DS) % N)]

    # angle to the target point
    heading_to_target = np.arctan2(target[1] - ypos, target[0] - xpos)
    alpha = np.arctan2(np.sin(heading_to_target - phi), np.cos(heading_to_target - phi))

    # pure pursuit then slip angle compensation
    beta = np.arcsin(np.clip(2 * LR * np.sin(alpha) / ld, -0.999, 0.999))
    sat = np.tanh(STEER_LIMIT)
    steer_cmd = np.arctanh(np.clip(np.arctan(2 * np.tan(beta)), -sat, sat))

    # steer rate over steer angle so P control to the angle
    steer_rate = np.clip(5.0 * (steer_cmd - theta), -1.0, 1.0)

    # take slower of here and ahead to ensure smooth speed transitions
    preview = int((_last_idx + max(1.0 * v, 3.0) / DS) % N)
    v_target = min(v_profile[_last_idx], v_profile[preview])

    # P control on speed
    accel = np.clip(2.5 * (v_target - v) + DRAG * v ** 2, -10.0, 4.0)

    return np.array([accel, steer_rate])


sim.set_controller(controller)
sim.run()
sim.animate()
sim.plot()