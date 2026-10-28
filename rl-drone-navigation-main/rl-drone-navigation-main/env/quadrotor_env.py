"""
Custom 2D planar quadrotor environment, built from scratch (no OpenAI
Gym / Gymnasium dependency).

Physical model
--------------
The drone is a rigid rod of length 2*ARM_LENGTH with a rotor at each end.
Each rotor produces a thrust force that always points along the drone's
own "up" direction (perpendicular to the rod) -- it rotates together with
the body. This is what makes the system underactuated: there are only two
scalar controls (left thrust, right thrust) but three degrees of freedom
(x, z, theta). The drone cannot move sideways without first rotating to
point its thrust vector sideways, which is the central control challenge
of quadrotor flight.

Sign conventions
-----------------
theta is the body's rotation angle in the standard counter-clockwise-
positive math convention, measured from vertical. At theta = 0 the body
is level and thrust points straight up. The right rotor sits at body-frame
offset (+ARM_LENGTH, 0), the left rotor at (-ARM_LENGTH, 0).

A subtlety worth internalizing: increasing LEFT thrust relative to right
makes the drone roll so its right side dips and its thrust vector tilts
toward +x -- i.e. left > right makes it move RIGHT. This is exactly how
a real quadrotor rolls to translate, and it is validated explicitly in
tests/test_quadrotor_env.py.

Observation vector (OBS_DIM = 16)
----------------------------------
[x, z, vx, vz, theta, omega, dx, dz, lidar_0, ..., lidar_7]
The first 8 values are the drone's own state plus its offset to the goal.
The last 8 are a simple lidar-style scan: fixed world-frame rays evenly
spaced over 360 degrees, each reporting normalized distance ([0, 1], 1.0
= nothing within LIDAR_MAX_RANGE) to the nearest obstacle. This is the
agent's only source of obstacle information -- there is no other way for
it to perceive obstacles at all.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class CircleObstacle:
    x: float
    z: float
    radius: float

    def distance_to_point(self, px: float, pz: float) -> float:
        """Signed distance from the obstacle surface to (px, pz); negative = inside."""
        return float(np.hypot(px - self.x, pz - self.z) - self.radius)

    def ray_intersection_distance(self, ox: float, oz: float, dx: float, dz: float) -> Optional[float]:
        """Distance from ray origin (ox, oz) along unit direction (dx, dz) to
        the nearest intersection with this circle, or None if the ray misses
        it (or the circle is entirely behind the ray's origin)."""
        lx, lz = self.x - ox, self.z - oz
        tca = lx * dx + lz * dz
        d2 = (lx * lx + lz * lz) - tca * tca
        r2 = self.radius * self.radius
        if d2 > r2:
            return None
        thc = np.sqrt(r2 - d2)
        t0, t1 = tca - thc, tca + thc
        if t0 >= 0:
            return float(t0)
        elif t1 >= 0:
            return 0.0  # origin is inside the circle
        return None


@dataclass
class RectObstacle:
    x: float       # center x
    z: float       # center z
    width: float
    height: float

    def distance_to_point(self, px: float, pz: float) -> float:
        """Distance from the rectangle surface to (px, pz); 0 if inside."""
        half_w, half_h = self.width / 2.0, self.height / 2.0
        dx = max(abs(px - self.x) - half_w, 0.0)
        dz = max(abs(pz - self.z) - half_h, 0.0)
        return float(np.hypot(dx, dz))

    def ray_intersection_distance(self, ox: float, oz: float, dx: float, dz: float) -> Optional[float]:
        """Same contract as CircleObstacle.ray_intersection_distance, via
        the standard slab method for a ray vs. an axis-aligned box."""
        half_w, half_h = self.width / 2.0, self.height / 2.0
        xmin, xmax = self.x - half_w, self.x + half_w
        zmin, zmax = self.z - half_h, self.z + half_h

        t_min, t_max = -np.inf, np.inf
        for o, d, lo, hi in ((ox, dx, xmin, xmax), (oz, dz, zmin, zmax)):
            if abs(d) < 1e-12:
                if o < lo or o > hi:
                    return None  # parallel to this slab and outside it
                continue
            t1, t2 = (lo - o) / d, (hi - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            t_min, t_max = max(t_min, t1), min(t_max, t2)
            if t_min > t_max:
                return None

        if t_max < 0:
            return None  # box is entirely behind the ray
        return float(max(t_min, 0.0))


class QuadrotorEnv:
    """A minimal reset()/step() interface, in the same spirit as Gym's
    API, but implemented independently -- there is no external RL
    environment library involved anywhere in this class."""

    # ---- physical constants ----
    GRAVITY = 9.81            # m/s^2
    MASS = 1.0                # kg
    ARM_LENGTH = 0.2          # m, center-of-mass to each rotor
    INERTIA = 0.05            # kg*m^2, moment of inertia about center of mass
    DRAG_COEF = 0.15          # N per (m/s), simple linear drag
    DT = 0.02                 # s, integration timestep (50 Hz)

    # Thrust-to-weight ratio of 1.5 gives enough control authority to
    # both climb and tilt for lateral motion; below ~1.0 the drone
    # could never even counteract gravity.
    THRUST_TO_WEIGHT = 1.5
    MAX_THRUST_PER_ROTOR = THRUST_TO_WEIGHT * MASS * GRAVITY / 2.0
    HOVER_ACTION = (MASS * GRAVITY / 2.0) / MAX_THRUST_PER_ROTOR  # normalized action that exactly hovers

    # ---- world / task constants ----
    WORLD_WIDTH = 10.0
    WORLD_HEIGHT = 8.0
    DRONE_RADIUS = 0.15       # collision radius of the drone body
    GOAL_RADIUS = 0.3
    GOAL_SPEED_LIMIT = 0.5    # m/s -- must be "arriving", not just passing through
    MAX_STEPS = 500

    # ---- lidar-style obstacle sensing ----
    # Added after Phase 4 failure analysis showed the observation had NO
    # obstacle information at all -- the agent could only ever learn an
    # implicit, position-based memorization of a fixed layout, which
    # doesn't generalize and explains the inconsistent obstacle-avoidance
    # behavior observed (curves around one obstacle, then flies straight
    # into a second). Rays are fixed to WORLD-frame angles (not rotating
    # with the drone's body/theta) for simplicity; they sense only
    # obstacles, not world-boundary walls (out-of-bounds remains a
    # separate terminal condition, unrelated to this sensor).
    LIDAR_NUM_RAYS = 8
    LIDAR_MAX_RANGE = 3.0     # m
    OBS_DIM = 8 + LIDAR_NUM_RAYS  # [x,z,vx,vz,theta,omega,dx,dz] + lidar rays

    # ---- reward weights (see PROJECT_SPEC.md Phase 1 discussion) ----
    W_PROGRESS = 1.0     # potential-based shaping toward the goal
    W_TIME = 0.01        # per-step cost, discourages loitering
    W_EFFORT = 0.001     # penalizes jittery/max-thrust control
    W_VELOCITY = 0.05    # velocity-toward-goal shaping (see _compute_reward_and_done)
    W_DIST = 0.002       # per-step penalty proportional to remaining distance to goal
    R_GOAL = 100.0       # terminal bonus for a successful arrival
    R_CRASH = -100.0     # terminal penalty for collision / out of bounds

    # ---- obstacle-clearance shaping ----
    # Added after the lidar-at-collision diagnosis: the lidar sensor works
    # (obstacle-facing rays correctly read ~0 several steps before impact),
    # but NO reward term rewarded acting on it, so the agent flew straight
    # into obstacles it could plainly see. This adds a dense, LOCAL penalty
    # that switches on only when an obstacle is close (the minimum lidar
    # ray drops below LIDAR_CLEAR_THRESHOLD) and grows linearly as the
    # drone gets nearer -- a live "steer away" gradient before contact.
    # Sized so its magnitude near collision is ~3-4x the per-step time
    # penalty -- strong enough to be felt throughout the approach, while
    # still ~3 orders of magnitude below the +/-100 terminals so it shapes
    # the path without distorting the actual task. Zero in open space.
    # (Started at 0.04 == ~1x time penalty; raised to 0.15 after the
    # decomposition showed that was too gentle to counter the goal-pull.)
    LIDAR_CLEAR_THRESHOLD = 0.3   # normalized ray => obstacle within 0.3 * LIDAR_MAX_RANGE = 0.9 m
    W_CLEARANCE = 0.15            # penalty = W_CLEARANCE * (threshold - min_ray) below threshold

    # ---- goal-proximity bonus ----
    # A positive reward that ramps up inside GOAL_PROX_RANGE and grows as the
    # drone nears the goal -- strongest exactly where it was stalling (~0.3-
    # 0.6 m out). Fixes the near-goal "dead zone": the distance penalty's
    # gradient is only W_DIST=0.002/m and the progress/velocity terms vanish
    # at zero speed, so a halted drone felt essentially no pull to close the
    # final gap. This term supplies a per-step gradient of W_GOAL_PROX/m --
    # sized so moving 0.1 m closer beats the per-step time penalty several
    # times over. Zero beyond the range, so long-range navigation is untouched.
    #
    # Loiter-trap safety: a RAW positive proximity bonus is dangerous -- if
    # the bonus for hovering just outside the radius exceeds the discounted
    # cost of delaying the +100 terminal by one step, R_GOAL*(1-gamma)=1.0,
    # the agent is rewarded for never entering. Here the edge bonus is
    # W_GOAL_PROX*(RANGE-GOAL_RADIUS) = 0.3*1.2 = 0.36, comfortably below 1.0,
    # so entering the goal always strictly beats loitering to farm the bonus.
    GOAL_PROX_RANGE = 1.5   # m -- bonus active only within this distance of the goal
    W_GOAL_PROX = 0.3       # bonus = W_GOAL_PROX * (GOAL_PROX_RANGE - dist) inside the range
    # Line-of-sight suppression: corridor clearance (gap between the drone->goal
    # line and the nearest blocking obstacle surface) at/above which the path is
    # treated as fully clear. Below 0 the straight line passes through an
    # obstacle (goal occluded) and the bonus is fully suppressed.
    GOAL_LOS_MARGIN = 0.5   # m

    def __init__(self, obstacles: Optional[List] = None, seed: Optional[int] = None,
                 obstacle_fn=None):
        """obstacle_fn: optional callable (rng, start_xz, goal_xz) -> list of
        obstacles, invoked each reset() to place obstacles per-episode (e.g.
        the Level 1.5 curriculum step's randomly-placed single obstacle that is
        guaranteed clear of the start->goal line). When set, it overrides the
        static `obstacles` list on every reset; when None, `obstacles` is fixed
        for the env's lifetime (all other levels)."""
        self.obstacles = obstacles if obstacles is not None else []
        self.obstacle_fn = obstacle_fn
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(6, dtype=np.float64)   # [x, z, vx, vz, theta, omega]
        self.goal = np.zeros(2, dtype=np.float64)
        self.steps = 0
        self._prev_dist = 0.0

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def _sample_free_point(self, margin: float) -> Tuple[float, float]:
        """Rejection-sample a point at least `margin` away from every obstacle."""
        for _ in range(1000):
            x = self.rng.uniform(margin, self.WORLD_WIDTH - margin)
            z = self.rng.uniform(margin, self.WORLD_HEIGHT - margin)
            if all(obs.distance_to_point(x, z) > margin for obs in self.obstacles):
                return x, z
        raise RuntimeError("Could not find a free point -- obstacles too dense for this world size.")

    def reset(self) -> np.ndarray:
        # With a per-episode obstacle_fn, clear obstacles first so start and
        # goal are sampled in the open world; the fn then places obstacles
        # around the chosen start/goal (e.g. clear of their straight line).
        if self.obstacle_fn is not None:
            self.obstacles = []
        start_x, start_z = self._sample_free_point(margin=self.DRONE_RADIUS * 2)
        goal_x, goal_z = self._sample_free_point(margin=self.GOAL_RADIUS)
        self.goal = np.array([goal_x, goal_z])
        self.state = np.array([start_x, start_z, 0.0, 0.0, 0.0, 0.0])
        if self.obstacle_fn is not None:
            self.obstacles = self.obstacle_fn(self.rng, self.state[:2].copy(), self.goal.copy())
        self.steps = 0
        self._prev_dist = self._distance_to_goal()
        return self._get_obs()

    def reset_to(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Force a specific state/goal -- used by tests and the physics
        validation script, where we need exact, repeatable conditions
        rather than a random reset."""
        self.state = np.array(state, dtype=np.float64)
        self.goal = np.array(goal, dtype=np.float64)
        self.steps = 0
        self._prev_dist = self._distance_to_goal()
        return self._get_obs()

    def _distance_to_goal(self) -> float:
        x, z = self.state[0], self.state[1]
        return float(np.hypot(x - self.goal[0], z - self.goal[1]))

    def _lidar_scan(self, x: float, z: float) -> np.ndarray:
        """Casts LIDAR_NUM_RAYS rays, evenly spaced over 360 degrees in the
        WORLD frame (ray 0 points along +x, angles increase counter-
        clockwise), from (x, z). Each reading is the distance to the
        nearest obstacle along that ray, capped at LIDAR_MAX_RANGE and
        normalized to [0, 1] -- 1.0 means "nothing detected within range"
        (clear), 0.0 means an obstacle is right at the drone's position."""
        readings = np.full(self.LIDAR_NUM_RAYS, self.LIDAR_MAX_RANGE, dtype=np.float64)
        for i in range(self.LIDAR_NUM_RAYS):
            angle = 2.0 * np.pi * i / self.LIDAR_NUM_RAYS
            ray_dx, ray_dz = np.cos(angle), np.sin(angle)
            min_dist = self.LIDAR_MAX_RANGE
            for obs in self.obstacles:
                hit = obs.ray_intersection_distance(x, z, ray_dx, ray_dz)
                if hit is not None and hit < min_dist:
                    min_dist = hit
            readings[i] = min_dist
        return (readings / self.LIDAR_MAX_RANGE).astype(np.float32)

    def _goal_los_clearance(self, x: float, z: float, goal) -> float:
        """Corridor clearance along the straight line from (x, z) to the goal:
        the smallest gap between that line and any obstacle SURFACE, counting
        only obstacles that lie BETWEEN the drone and the goal (same segment/
        obstacle geometry as the failure-geometry diagnosis, but restricted to
        the drone->goal direction so obstacles behind the drone or beyond the
        goal don't count). Unlike the 8 fixed lidar rays, this looks exactly
        along the goal direction, so it catches a blocker that falls between
        rays.
          +inf -> no obstacle on the way (clear line of sight)
          > 0  -> the straight line clears the nearest blocker by this many m
          <= 0 -> the straight line passes through an obstacle (goal occluded)
        """
        gx, gz = goal[0] - x, goal[1] - z
        dist = float(np.hypot(gx, gz))
        if dist < 1e-9 or not self.obstacles:
            return float("inf")
        ux, uz = gx / dist, gz / dist              # unit vector toward the goal
        best = float("inf")
        for o in self.obstacles:
            t = (o.x - x) * ux + (o.z - z) * uz    # projection of center onto the ray
            if t <= 0.0 or t >= dist:              # obstacle behind the drone or beyond the goal
                continue
            perp = abs((o.x - x) * (-uz) + (o.z - z) * ux)  # distance from center to the line
            best = min(best, perp - o.radius)
        return best

    def _get_obs(self) -> np.ndarray:
        # The observation includes the goal *offset* (dx, dz) rather than
        # making the network infer it from two separate absolute
        # positions -- this is standard practice and makes the task
        # translation-invariant, which is much easier to learn.
        x, z, vx, vz, theta, omega = self.state
        dx = self.goal[0] - x
        dz = self.goal[1] - z
        lidar = self._lidar_scan(x, z)
        return np.concatenate([[x, z, vx, vz, theta, omega, dx, dz], lidar]).astype(np.float32)

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        left, right = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
        left_thrust = left * self.MAX_THRUST_PER_ROTOR
        right_thrust = right * self.MAX_THRUST_PER_ROTOR

        x, z, vx, vz, theta, omega = self.state

        total_thrust = left_thrust + right_thrust
        # Body "up" direction (0, 1) rotated by theta into world frame.
        fx = -total_thrust * np.sin(theta) - self.DRAG_COEF * vx
        fz = total_thrust * np.cos(theta) - self.MASS * self.GRAVITY - self.DRAG_COEF * vz
        ax = fx / self.MASS
        az = fz / self.MASS

        # Torque about the center of mass depends only on the thrust
        # difference and the arm length -- it does not depend on the
        # current orientation theta, since both rotor forces are always
        # perpendicular to the rod.
        torque = (right_thrust - left_thrust) * self.ARM_LENGTH
        alpha = torque / self.INERTIA

        # Semi-implicit (symplectic) Euler: update velocities first, then
        # use the *new* velocities to update position. Much more stable
        # than plain (explicit) Euler for oscillatory systems like this.
        vx += ax * self.DT
        vz += az * self.DT
        omega += alpha * self.DT
        x += vx * self.DT
        z += vz * self.DT
        theta += omega * self.DT

        self.state = np.array([x, z, vx, vz, theta, omega])
        self.steps += 1

        reward, done, info = self._compute_reward_and_done(left, right)
        return self._get_obs(), reward, done, info

    def _compute_reward_and_done(self, left: float, right: float) -> Tuple[float, bool, dict]:
        x, z, vx, vz, theta, omega = self.state
        dist = self._distance_to_goal()
        speed = float(np.hypot(vx, vz))

        # 1. Progress shaping (potential-based: phi(s) = -distance_to_goal).
        # Telescopes to (final_dist - start_dist) over an episode, so it
        # rewards the *outcome* without biasing the agent toward any
        # particular path.
        reward = self.W_PROGRESS * (self._prev_dist - dist)
        self._prev_dist = dist

        # 2. Time penalty -- without this, the telescoping property above
        # means loitering forever near the goal costs nothing.
        reward -= self.W_TIME

        # 3. Control-effort penalty -- discourages jittery, max-thrust
        # control in favor of smoother, more physically plausible flight.
        reward -= self.W_EFFORT * (left ** 2 + right ** 2)

        # 4. Velocity-toward-goal shaping. The progress term above only
        # pays out for distance actually closed over a step, so it
        # vanishes as speed -> 0 regardless of which way the drone is
        # pointed. This term instead rewards the *current* velocity's
        # component along the direction to the goal directly, so it
        # still gives a live, non-zero gradient even at very low speed
        # (e.g. a drone hovering near-stationary but tilted the wrong
        # way still gets pushed toward tilting the right way).
        if dist > 1e-6:
            goal_dir_x = (self.goal[0] - x) / dist
            goal_dir_z = (self.goal[1] - z) / dist
            vel_toward_goal = vx * goal_dir_x + vz * goal_dir_z
        else:
            vel_toward_goal = 0.0
        reward += self.W_VELOCITY * vel_toward_goal

        # 5. Distance penalty -- unlike the terms above, this does NOT
        # vanish as speed -> 0: it depends only on remaining distance to
        # the goal, so a drone parked far from the goal keeps paying it
        # every step regardless of whether it's moving, giving a
        # persistent incentive to close the distance rather than settle
        # anywhere.
        reward -= self.W_DIST * dist

        # 6. Obstacle-clearance shaping (see LIDAR_CLEAR_THRESHOLD / W_CLEARANCE
        # above). Reads the SAME lidar the policy observes, so the penalty is
        # a direct function of a signal the agent can act on. Off in open
        # space; grows linearly as the nearest obstacle enters the 0.9 m band.
        min_ray = float(self._lidar_scan(x, z).min())
        if min_ray < self.LIDAR_CLEAR_THRESHOLD:
            reward -= self.W_CLEARANCE * (self.LIDAR_CLEAR_THRESHOLD - min_ray)

        # 7. Goal-proximity bonus (see GOAL_PROX_RANGE / W_GOAL_PROX above),
        # OBSTACLE-AWARE. Linear ramp toward the goal (zero at the range edge,
        # growing as distance shrinks) that actively pulls the drone through
        # the last half metre where every other position signal has gone flat.
        # It is scaled by prox_factor = min(proximity, line-of-sight), so the
        # bonus is suppressed if EITHER an obstacle is close OR an obstacle
        # blocks the straight line to the goal:
        #   - proximity: (min_ray/threshold)^2 -- full outside the clearance
        #     band, falling quadratically to zero as the drone enters it.
        #   - line-of-sight: ramps from 0 (straight line passes through an
        #     obstacle -> goal occluded) to 1 (path clears by GOAL_LOS_MARGIN).
        # The LOS term damps the pull toward an occluded goal FROM THE START --
        # even while the drone is still at moderate range, before proximity
        # would react -- so the agent isn't lured into forward momentum it then
        # can't shed, and must route around instead. Full strength only when
        # the goal is both un-occluded and not right next to an obstacle.
        if dist < self.GOAL_PROX_RANGE:
            prox_near = min(1.0, min_ray / self.LIDAR_CLEAR_THRESHOLD) ** 2
            los_clear = self._goal_los_clearance(x, z, self.goal)
            prox_los = min(1.0, max(0.0, los_clear) / self.GOAL_LOS_MARGIN)
            prox_factor = min(prox_near, prox_los)
            reward += self.W_GOAL_PROX * (self.GOAL_PROX_RANGE - dist) * prox_factor

        done = False
        info = {"outcome": "running"}

        out_of_bounds = not (0.0 <= x <= self.WORLD_WIDTH and 0.0 <= z <= self.WORLD_HEIGHT)
        collided = any(obs.distance_to_point(x, z) <= self.DRONE_RADIUS for obs in self.obstacles)
        reached_goal = dist <= self.GOAL_RADIUS and speed <= self.GOAL_SPEED_LIMIT

        if reached_goal:
            reward += self.R_GOAL
            done = True
            info["outcome"] = "success"
        elif collided or out_of_bounds:
            reward += self.R_CRASH
            done = True
            info["outcome"] = "collision" if collided else "out_of_bounds"
        elif self.steps >= self.MAX_STEPS:
            done = True
            info["outcome"] = "timeout"

        return float(reward), done, info
