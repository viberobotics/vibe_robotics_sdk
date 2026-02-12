from viberobotics.motor.motor_controller_manager import MotorControllerManager
from viberobotics.configs.config import load_config

import argparse
import numpy as np
from numpy import array
import time
import threading

# from flask import Flask, jsonify, redirect, url_for

# app = Flask(__name__)

# # global variable you can poll
# GLOBAL_STATE = {"press": False}

# @app.route("/")
# def index():
#     return """
#     <html>
#         <body>
#             <h1>Button: {}</h1>
#             <form action="/press" method="post">
#                 <button style="width:100vw;height:100vh;font-size:10vw;">PRESS</button>
#             </form>
#         </body>
#     </html>
#     """.format(GLOBAL_STATE["press"])


# @app.route("/press", methods=["POST"])
# def press():
#     GLOBAL_STATE["press"] = True
#     return redirect(url_for("index"))


# @app.route("/state")
# def state():
#     return jsonify(GLOBAL_STATE)
# def run_server():
#     # IMPORTANT: disable reloader or it will spawn another process/thread
#     app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)

# def poll_state():
#     while not GLOBAL_STATE["press"]:
#         continue
#     GLOBAL_STATE["press"] = False  

if __name__ == '__main__':
    # server_thread = threading.Thread(target=run_server, daemon=True)
    # server_thread.start()
    config_name = 'sundaya1_real_config.yaml'
    config = load_config(config_name)
    
    motor_manager = MotorControllerManager(
        n_motors=config.real_config.n_motors,
        motor_mapping=config.real_config.motor_controllers,
        calibration_file=config.real_config.calibration_file,
        mode=0
    )
    # motor_manager.set_kp_kd(32, 32)
    motor_manager.set_positions(config.default_qpos, 0, 50)
    
    input('start>')
    
    
    # poses = [
    #     np.array([ 3.12625289e+00,  1.63982558e+00,  0.00000000e+00,  3.05262089e-01,
    #     3.06797028e-03,  0.00000000e+00,  0.00000000e+00,  1.53398514e-03,
    #     1.53398514e-03,  1.53398514e-03, -3.06797028e-03,  4.60195541e-03,
    #     1.53398514e-03,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #    -1.53398514e-03, -1.53398514e-03, -3.06797028e-03, -3.06797028e-03,
    #     1.53398514e-03,  1.53398514e-03, -1.53398514e-03]),
    #     np.array([ 3.12625289e+00,  1.64135957e+00,  0.00000000e+00,  3.06796074e-01,
    #     4.60195541e-03, -3.95766973e-01,  0.00000000e+00,  3.06797028e-03,
    #     1.53398514e-03,  1.53398514e-03, -3.06797028e-03,  4.60195541e-03,
    #     1.53398514e-03,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #    -1.53398514e-03, -1.53398514e-03, -3.06797028e-03, -3.06797028e-03,
    #     1.53398514e-03,  1.53398514e-03, -1.53398514e-03]),
    #     np.array([ 3.12625289e+00,  1.63982558e+00,  1.53398514e-03,  3.06796074e-01,
    #     4.60195541e-03, -7.66992569e-03,  0.00000000e+00,  3.06797028e-03,
    #     1.53398514e-03,  1.53398514e-03, -3.06797028e-03,  4.60195541e-03,
    #     3.06797028e-03,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #    -1.53398514e-03, -1.53398514e-03, -3.06797028e-03, -3.06797028e-03,
    #     1.53398514e-03,  1.53398514e-03, -1.53398514e-03]),
    #     np.array([ 1.61834979e+00,  2.30097771e-02,  1.34990215e-01, -4.27980661e-01,
    #     1.53398514e-03, -7.66992569e-03,  0.00000000e+00,  1.53398514e-03,
    #     4.60195541e-03,  1.53398514e-03, -1.53398514e-03,  4.60195541e-03,
    #     4.60195541e-03,  0.00000000e+00, -1.53398514e-03,  0.00000000e+00,
    #    -1.53398514e-03, -1.53398514e-03, -3.06797028e-03,  3.06797028e-03,
    #     3.06797028e-03,  4.60195541e-03,  0.00000000e+00]),
    #     np.array([ 0.05368924, -1.52324295, -0.06749511,  0.3558836 , -0.17487383,
    #    -0.00766993,  0.        ,  0.00306797,  0.00153399,  0.00306797,
    #    -0.00306797,  0.00460196,  0.00766993,  0.        , -0.00153399,
    #     0.        , -0.00153399, -0.00306797, -0.00306797,  0.00306797,
    #     0.00460196,  0.00460196, -0.00153399]),
    #     array([ 0.05368924, -1.52170897, -0.06596112,  0.35281563, -0.17640781,
    #    -0.17487383,  0.        ,  0.00306797,  0.00153399,  0.00306797,
    #    -0.00460196,  0.00460196,  0.00766993,  0.        , -0.00153399,
    #     0.        , -0.00153399, -0.00306797, -0.00306797,  0.00306797,
    #     0.00460196,  0.00460196, -0.00153399]),
    #     array([ 0.05368924,  0.13038826,  0.08436894, -0.04295135, -0.18100977,
    #     0.00613594,  0.00153399,  0.00306797, -0.        ,  0.00306797,
    #    -0.00460196,  0.00613594,  0.00766993,  0.        , -0.00153399,
    #     0.        , -0.00153399, -0.00460196, -0.00306797,  0.00306797,
    #     0.00460196,  0.00306797, -0.00153399])
        
    # ]
    
    poses = [
        array([-0.00306797, -0.00766993,  0.00153399,  0.        ,  0.00306797,
       -0.00306797, -0.00153399,  0.00306797,  0.00460196, -0.00306797,
        0.00306797,  0.        , -0.48780584,  0.00306797,  0.00153399,
        0.00460196,  0.23469901,  0.00306797,  0.00153399, -0.        ,
       -0.00460196, -0.00460196,  0.        ]),
        array([-0.00306797, -0.00766993,  0.00153399,  0.        ,  0.00306797,
       -0.00306797, -0.00153399,  0.00306797,  0.00460196, -0.00306797,
        0.00306797,  0.        , -0.48780584,  0.00153399,  0.00153399,
        0.00306797,  0.01380587,  0.00306797,  0.00306797, -0.        ,
       -0.00613594, -0.00613594,  0.        ]),
        array([-3.06797028e-03, -7.66992569e-03,  1.53398514e-03,  0.00000000e+00,
        3.06797028e-03, -3.06797028e-03, -1.53398514e-03,  3.06797028e-03,
        4.60195541e-03, -1.53398514e-03,  0.00000000e+00,  0.00000000e+00,
       -1.72419429e+00,  3.06797028e-03,  0.00000000e+00,  4.60195541e-03,
        1.38058662e-02,  1.53398514e-03,  1.53398514e-03, -4.60195541e-03,
       -1.53398514e-03, -1.53398514e-03,  1.53398514e-03]),
        array([-3.06797028e-03, -7.66992569e-03,  1.53398514e-03,  0.00000000e+00,
        3.06797028e-03, -3.06797028e-03,  0.00000000e+00,  3.06797028e-03,
        4.60195541e-03,  1.53398514e-03, -1.53398514e-03,  1.53398514e-03,
       -1.72112632e+00,  3.06797028e-03, -1.53398514e-03,  4.60195541e-03,
        4.00368929e-01,  1.53398514e-03,  1.53398514e-03, -4.60195541e-03,
       -1.53398514e-03,  0.00000000e+00,  1.53398514e-03])
    ]
    
    # motor_manager.disable_torque()
    # while True:
    #     input('>')
    #     q = motor_manager.get_state()[0]
    #     print('Current positions:', repr(q))
    
    for pose in poses:
        motor_manager.set_positions(pose, 0, 50)
        input('>')
    
    # timings = [
    #     0,
    #     0.5,
    #     -1,
    #     0.5,
    #     0.5,
    #     -1,
    #     0.5
    # ]
    
    # while True:
    #     for pose, timing in zip(poses, timings):
    #         if timing < 0:
    #             # input('>')
    #             poll_state()
    #         else:
    #             time.sleep(timing)
    #         motor_manager.set_positions(pose, 0, 30)
    #     motor_manager.set_positions(config.default_qpos, 0, 30)
    #     poll_state()
    # motor_manager.disable_torque()