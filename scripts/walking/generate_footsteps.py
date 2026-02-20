from classes import *
from typing import List

def generate_footsteps(distance, step_length, foot_spread, initial_y=0.) -> List[Footstep]:
    footsteps = []
    
    footsteps.append(Footstep(x=+foot_spread, y=initial_y))
    footsteps.append(Footstep(x=-foot_spread, y=initial_y))
    
    x = foot_spread
    y = initial_y
    
    while y < distance:
        if distance - y < step_length:
            y += min(distance - y, 0.5 * step_length)
        else:
            y += step_length
        x = -x
        footsteps.append(Footstep(x=x, y=y))
    footsteps.append(Footstep(x=-x, y=y))
    return footsteps


def wrap_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi

def rotz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    
def T_world_xy_yaw(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotz(-yaw)
    T[:3, 3] = np.array([x, y, z])
    return T

class FootstepGenerator:
    def __init__(self,
                 step_length: float,
                 foot_spread: float,
                 initial_y: float = 0.0,
                 steering_strength: float = np.deg2rad(10.)):
        # assume starts at [+foot_spread, initial_y], [-foot_spread, initial_y]
        self.step_length = step_length
        self.foot_spread = foot_spread # half foot distance
        self.steering_strength = steering_strength
        
        self.ref_x = 0.
        self.ref_y = initial_y
        self.ref_theta = 0.
        
    
    def _forward(self, theta):
        # y-forward convention
        return np.array([np.sin(theta), np.cos(theta)])

    def _right(self, theta):
        return np.array([np.cos(theta), -np.sin(theta)])
    
    def get_next_footstep(self,
                      cmd: np.ndarray,
                      stance_foot: FootType) -> Footstep:
        # cmd = [fwd, lat, yaw]
        next_foot = FootType.LEFT if stance_foot == FootType.RIGHT else FootType.RIGHT

        # scale commands
        d_fwd  = float(cmd[0]) * self.step_length
        d_lat  = float(cmd[1]) * self.step_length   # or a separate lateral_step_length
        dtheta = float(cmd[2]) * self.steering_strength

        self.ref_theta = wrap_pi(self.ref_theta + dtheta)

        fwd = self._forward(self.ref_theta)   # unit vector
        rgt = self._right(self.ref_theta)     # unit vector (robot-right)

        # Pick convention:
        # If you want cmd[1] > 0 to mean "move left", subtract rgt.
        # If you want cmd[1] > 0 to mean "move right", add rgt.
        self.ref_x += d_fwd * fwd[0] - d_lat * rgt[0]
        self.ref_y += d_fwd * fwd[1] - d_lat * rgt[1]

        # place the swing foot at +/- foot_spread about the reference, in the current heading
        lat_sign = -1 if next_foot == FootType.LEFT else +1
        foot_xy = np.array([self.ref_x, self.ref_y]) + rgt * lat_sign * self.foot_spread

        T_foot = T_world_xy_yaw(foot_xy[0], foot_xy[1], self.ref_theta)
        return Footstep(T=T_foot)
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    footsteps = generate_footsteps(5.0, 0.3, 0.1)
    xs = [fs.x for fs in footsteps]
    ys = [fs.y for fs in footsteps]
    plt.scatter(xs, ys)
    plt.plot(xs, ys)
    plt.axis('equal')
    plt.show()
