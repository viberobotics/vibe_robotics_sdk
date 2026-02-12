# Controlling the robot
After installing the environment, you can begin controlling the robot.

## Calibrating the robot
Before the robot can be used, it has to first be calibrated. Calibration can be done by simply running the command:
```
python scripts/calibrate.py --mode 0
```
Under the hood, this scripts sends a zeroing command to the servo motors so it recognizes the current position as zero. `--mode` can be `2` if calibration of PWM mode is desired.

## Simple record and play
A minimal example for controlling Sunday A1 can be found in `scripts/simple_record.py`. When run, you will be prompted to put the robot in different poses, and when you are done, you can press 'q'. You can then replay the poses that was just recorded.

The first two lines imports the necessary class and function for controlling the motors.
```
from viberobotics.motor.motor_controller_manager import MotorControllerManager
from viberobotics.configs.config import load_config
```
The `MotorControllerManager` is a high level class for controlling the motor controllers equipped on the robot. `load_config` is a helper function that loads a configuration file present under `viberobotics/configs`. If you want a different modality, like adding an extra joint to the legs, simply create a new config file. No extra code is needed if extra motors are added. Here, we use the default config file for the base Sunday A1 model. The parsed config is used to initialize the motor controller manager:
```
config_name = 'sundaya1_real_config.yaml'
config = load_config(config_name)

motor_manager = MotorControllerManager(
    n_motors=config.real_config.n_motors,
    motor_mapping=config.real_config.motor_controllers,
    calibration_file=config.real_config.calibration_file,
    mode=0
)
```
The `calibration_file` is a file for PWM mode. Since the STS motors give different position readings under PWM mode than position mode, a separate file is needed to store the zero positions. After the motor controller manager is set up, you can begin controlling the robot. 
```
motor_manager.set_positions(config.default_qpos, 0, 30)
```
This line put the robot in the default pose, with a max acceleration of 30. Since this sends position commands to all motors, we then need to disable the torque on the motor so it can be manipulated.
```
motor_manager.controllers_mapping['arm'].disable_torque()
```
This line disable the torque for only the arm, so the robot can remain standing while the arms are manipulated. If instead you want to manipulate the entire robot, simply remove `['arm']` from the line. In the loop, the pose of the robot is recorded with:
```
q = motor_manager.get_state()[0]
```
The `get_state()` function returns two variables: the position of the motors and the speed of the motors.
Finally, when all poses are recorded, it can be played back with:
```
motor_manager.set_positions(pose, 0, 30)
```

## Controlling the robot in PWM mode
The `MotorControllerManager` also provides control for PWM mode (mode 2). Under this mode, the duty cycle is controlled instead of the motor positions. A minimal example of running under PWM mode can be found in `scripts/pd_stand.py`. The motor controller manager can be initialized in PWM mode by simply modifying the `mode` parameter:
```
motor_manager = MotorControllerManager(
    config.real_config.n_motors,
    config.real_config.motor_controllers, 
    calibration_file=config.real_config.calibration_file, 
    mode=2
)
```
Usually, PWM mode is desired if you want to have a custom motion profile. This script uses a classic PD controller. We provide easy PID control with the `PIDController` class. 
```
controller = PIDController(kp, 0, kd)
```
In the control loop, we can poll the motor controller manager for the position and speed of the motors
```
q, dq = motor_manager.get_state()
```
The motor state observation is used to update the PID controller to obtain duty.
```
duty = controller.update(
    setpoint=default_qpos,
    measurement=q,
    derivative=-dq,
)
```
Finally, the duty can be sent to the motors:
```
motor_manager.set_duty(duty)
```
Note: the duty should be an integer between -1000 and 1000 indicating the duty cycle. Each increment of the duty variable increments the duty cycle of the motors by 0.1%.