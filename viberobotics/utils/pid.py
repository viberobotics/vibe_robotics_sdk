import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0
    
    def update(self, setpoint, measurement, dt=None, derivative=None):
        if type(setpoint) is np.ndarray:
            setpoint = setpoint.copy()
            measurement = measurement.copy()
        
        assert dt is not None or derivative is not None, "Either dt or derivative must be provided"
        if derivative is not None:
            dt = 1.0  # Dummy value since derivative is provided
            if type(derivative) is np.ndarray:
                derivative = derivative.copy()
        else:
            derivative = (setpoint - measurement - self.prev_error) / dt if dt > 0 else 0
        error = setpoint - measurement
        self.integral += error * dt
        self.prev_error = error
        
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        return output