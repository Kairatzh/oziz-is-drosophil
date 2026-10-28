"""
Hand-coded potential-field controller for the 2D quadrotor navigation task
(Phase 6 baseline). Classical, model-based control -- no learning, no
training data. Every gain below was tuned by hand against the same
curriculum layouts used to evaluate the RL agent, the same way a real
drone's control loops get tuned on the bench.

Two-loop cascade, mirroring how a real quadrotor autopilot is structured:
  1. OUTER (guidance) loop: an artificial potential field (Khatib, 1986)
     combines an attractive pull toward the goal with a repulsive push away
     from any nearby obstacle, producing a desired acceleration in world
     (x, z) coordinates.
  2. INNER (attitude) loop: converts that desired acceleration into a
     desired thrust VECTOR (a magnitude and a direction -- gravity must
     also be cancelled), then a PD controller on tilt angle theta steers
     the body to point its thrust that way. This is the same underactuated
     "tilt to translate" problem described in Phase 1 -- the controller
     cannot move sideways without first rotating.

Unlike the RL agent (restricted to an 8-ray lidar, PHASE5_GENERALIZATION_RESULTS.md),
this controller is given the EXACT obstacle positions/radii and the drone's
full true state -- the standard assumption for classical potential-field/PID
control, which requires a known, structured model of the world rather than
learning one from partial observations. This asymmetry is deliberate and is
called out explicitly in PHASE6_BASELINE_COMPARISON_RESULTS.md.
"""

import numpy as np

from env.quadrotor_env import QuadrotorEnv


class PotentialFieldController:
    def __init__(
        self,
        k_att: float = 1.4,
        max_att_accel: float = 3.0,
        k_vel_damp: float = 1.6,
        k_rep: float = 1.2,
        influence_margin: float = 1.2,
        max_rep_accel: float = 7.0,
        kp_theta: float = 5.0,  # tuned by hand: with INERTIA=0.05 (very light), naive
        kd_theta: float = 1.2,  # PD gains (e.g. 45/9) demand torque the rotors can't
                                 # deliver without saturating -- see PHASE6_BASELINE_COMPARISON_RESULTS.md
        max_tilt: float = 1.0,  # radians (~57 deg) -- keep thrust mostly upward
    ):
        self.k_att = k_att
        self.max_att_accel = max_att_accel
        self.k_vel_damp = k_vel_damp
        self.k_rep = k_rep
        self.influence_margin = influence_margin  # meters of clearance beyond the surface where repulsion starts
        self.max_rep_accel = max_rep_accel
        self.kp_theta = kp_theta
        self.kd_theta = kd_theta
        self.max_tilt = max_tilt

    def _attractive_accel(self, pos: np.ndarray, vel: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """PD pull toward the goal: proportional term capped so a distant
        goal never demands more tilt than the inner loop can deliver, plus
        a damping term so the drone settles instead of overshooting."""
        err = goal - pos
        a = self.k_att * err
        norm = float(np.hypot(*a))
        if norm > self.max_att_accel:
            a = a * (self.max_att_accel / norm)
        return a - self.k_vel_damp * vel

    def _repulsive_accel(self, pos: np.ndarray, obstacles) -> np.ndarray:
        """Classic Khatib repulsive field: zero beyond influence_margin from
        the obstacle SURFACE, growing as 1/d^2 * (1/d - 1/d0) as the drone
        gets closer, pointed away from the obstacle center."""
        a = np.zeros(2)
        for obs in obstacles:
            delta = pos - np.array([obs.x, obs.z])
            raw_dist = float(np.hypot(*delta))
            surf_dist = max(raw_dist - obs.radius, 0.05)  # clamp away from the singularity at the surface
            if surf_dist < self.influence_margin:
                direction = delta / (raw_dist + 1e-9)
                magnitude = self.k_rep * (1.0 / surf_dist - 1.0 / self.influence_margin) / (surf_dist ** 2)
                a += direction * magnitude
        norm = float(np.hypot(*a))
        if norm > self.max_rep_accel:
            a = a * (self.max_rep_accel / norm)
        return a

    def act(self, x, z, vx, vz, theta, omega, goal, obstacles) -> np.ndarray:
        pos, vel, goal = np.array([x, z]), np.array([vx, vz]), np.asarray(goal, dtype=np.float64)
        a_des = self._attractive_accel(pos, vel, goal) + self._repulsive_accel(pos, obstacles)

        # Desired net force = m*a_des, plus feedforward cancellation of
        # gravity and (estimated) drag -- the dynamics are
        # m*a = F_thrust + (0, -mg) - drag_coef*v, so solving for F_thrust:
        f_des = QuadrotorEnv.MASS * a_des
        f_des[1] += QuadrotorEnv.MASS * QuadrotorEnv.GRAVITY
        f_des += QuadrotorEnv.DRAG_COEF * vel

        total_thrust = float(np.hypot(*f_des))
        if total_thrust < 1e-6:
            theta_des = theta
        else:
            ux, uz = f_des / total_thrust
            # thrust direction as a function of theta is (-sin theta, cos theta) -- invert it.
            theta_des = float(np.arctan2(-ux, uz))
        theta_des = float(np.clip(theta_des, -self.max_tilt, self.max_tilt))

        angle_err = float(np.arctan2(np.sin(theta_des - theta), np.cos(theta_des - theta)))
        torque = self.kp_theta * angle_err - self.kd_theta * omega

        diff = torque / QuadrotorEnv.ARM_LENGTH
        left = (total_thrust - diff) / 2.0
        right = (total_thrust + diff) / 2.0

        left_action = np.clip(left / QuadrotorEnv.MAX_THRUST_PER_ROTOR, 0.0, 1.0)
        right_action = np.clip(right / QuadrotorEnv.MAX_THRUST_PER_ROTOR, 0.0, 1.0)
        return np.array([left_action, right_action], dtype=np.float32)
