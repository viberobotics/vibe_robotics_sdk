from viberobotics.motor.ftservo_python_sdk.scservo_sdk.port_handler import PortHandler
from viberobotics.motor.ftservo_python_sdk.scservo_sdk import sms_sts
from viberobotics.motor.ftservo_python_sdk.scservo_sdk.group_sync_read import GroupSyncRead
from viberobotics.motor.ftservo_python_sdk.scservo_sdk.group_sync_write import GroupSyncWrite
from viberobotics.motor.ftservo_python_sdk.scservo_sdk.sms_sts import *
from viberobotics.utils.math import step2rad
from viberobotics.exceptions.motor import *

import numpy as np

class MotorController:
    def __init__(self, motor_ids, port_name):
        self.motor_ids = np.array(motor_ids)
        self.port_name = port_name
    
        self._init_port_handler()
        self.packetHandler = sms_sts(self.portHandler)
        self.groupSyncRead = GroupSyncRead(self.packetHandler, SMS_STS_PRESENT_POSITION_L, 4)
        self.groupSyncRead_Current = GroupSyncRead(self.packetHandler, SMS_STS_PRESENT_CURRENT_L, 2)
        self.groupSyncRead_Load = GroupSyncRead(self.packetHandler, SMS_STS_PRESENT_LOAD_L, 2)
        
        self.n_motors = len(motor_ids)
        
        self.cached_positions = np.zeros(self.n_motors)
        self.cached_speeds = np.zeros(self.n_motors)
    
    def _init_port_handler(self):
        self.portHandler = PortHandler(self.port_name)

        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            quit()

        # Set port baudrate 1000000
        if self.portHandler.setBaudRate(1000000):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            quit()
    
    def _group_read(read_addr, read_length):
        def decorator(func):
            def wrapper(self):
                for i in range(len(self.motor_ids)):
                    scs_addparam_result = self.groupSyncRead.addParam(self.motor_ids[i])
                    if scs_addparam_result != True:
                        raise GroupAddParamFailedException()
                scs_comm_result = self.groupSyncRead.txRxPacket()
                if scs_comm_result != COMM_SUCCESS:
                    print('scs_comm_result:', scs_comm_result)
                    raise GroupSyncReadFailedException(f'failed: {self.motor_ids}')
                agg = []
                init = False
                for i in range(len(self.motor_ids)):
                    scs_data_result, scs_error = self.groupSyncRead.isAvailable(self.motor_ids[i], read_addr, read_length)
                    if scs_data_result == True:
                        ret = func(self, self.motor_ids[i])
                        if not init:
                            agg = [[] for _ in range(len(ret))]
                            init = True
                        for j in range(len(ret)):
                            agg[j].append(ret[j])
                    else:
                        raise GroupSyncReadNotAvailableException(self.motor_ids[i])
                        
                self.groupSyncRead.clearParam()
                return [np.array(a).reshape(-1) for a in agg]
            return wrapper
        return decorator
    
    @_group_read(SMS_STS_PRESENT_POSITION_L, 4)
    def _receive_raw_motor_states(self, motor_id):
        pos = self.groupSyncRead.getData(motor_id, SMS_STS_PRESENT_POSITION_L, 2),
        speed = self.groupSyncRead.getData(motor_id, SMS_STS_PRESENT_SPEED_L, 2)
        speed = self.packetHandler.scs_tohost(speed, 15)
        # print(f'Received motor {motor_id} state: pos {pos}, speed {speed}')
        return pos, speed
    
    def receive_raw_motor_states(self):
        try:
            raw_positions, raw_speeds = self._receive_raw_motor_states()
            self.cached_positions = raw_positions
            self.cached_speeds = raw_speeds
        except MotorException as e:
            print(f'Warning: Failed to read motor states: {self.motor_ids}. Using cached values.')
            raw_positions = self.cached_positions
            raw_speeds = self.cached_speeds
        return raw_positions, raw_speeds
    
    def receive_motor_states(self):
        raw_positions, raw_speeds = self.receive_raw_motor_states()
        positions = step2rad(raw_positions)
        speeds = step2rad(raw_speeds)
        return positions, speeds
    
    @_group_read(SMS_STS_PRESENT_CURRENT_L, 2)
    def receive_raw_motor_currents(self, motor_id):
        return self.groupSyncRead_Current.getData(motor_id, SMS_STS_PRESENT_CURRENT_L, 2)
    
    @_group_read(SMS_STS_PRESENT_LOAD_L, 2)
    def receive_raw_motor_loads(self, motor_id):
        return self.groupSyncRead_Load.getData(motor_id, SMS_STS_PRESENT_LOAD_L, 2)
    
    def set_mode(self, mode):
        for motor_id in self.motor_ids:
            print(f'Setting motor {motor_id} to mode {mode}')
            scs_comm_result, scs_error = self.packetHandler.write1ByteTxRx(motor_id, SMS_STS_MODE, mode)
            if scs_comm_result != COMM_SUCCESS:

                raise WriteFailedException(self.packetHandler.getTxRxResult(scs_comm_result), scs_error)
    
    def set_duty(self, torques):
        torques = np.clip((torques).astype(np.int32), -1000, 1000)

        for i in range(len(self.motor_ids)):
            servo_id = self.motor_ids[i]
            torque = torques[i]

            scs_addparam_result = self.packetHandler.SyncWritePWMTorque(servo_id, torque)
            if scs_addparam_result != True:
                raise GroupAddParamFailedException()

        scs_comm_result = self.packetHandler.groupSyncWrite_PWMTorque.txPacket()
        if scs_comm_result != COMM_SUCCESS:
           raise SyncWriteFailedException()
        self.packetHandler.groupSyncWrite_PWMTorque.clearParam()
    
    def set_kp_kd(self, kp, kd):
        kp = np.clip(np.round(kp), 0, 254).astype(np.int32)
        kd = np.clip(np.round(kd), 0, 254).astype(np.int32)

        for i in range(len(self.motor_ids)):
            servo_id = self.motor_ids[i]

            scs_addparam_result = self.packetHandler.SyncWriteKpKdEx(servo_id, kp[i], kd[i])
            if scs_addparam_result != True:
                raise GroupAddParamFailedException()

        scs_comm_result = self.packetHandler.groupSyncWrite_KpKd.txPacket()
        if scs_comm_result != COMM_SUCCESS:
              raise SyncWriteFailedException()

        self.packetHandler.groupSyncWrite_KpKd.clearParam()
    
    def send_raw_positions(self, q_pos_step, q_vel, q_acc):
        q_pos_step = np.clip(np.round(q_pos_step).astype(np.int32), 0, 4095)
        q_acc = np.clip(np.round(q_acc).astype(np.int32), 0, 254)
        q_vel = np.clip(np.round(q_vel).astype(np.int32), 0, 500)
        for i in range(len(self.motor_ids)):
            servo_id = self.motor_ids[i]
            motor_q_pos = q_pos_step[i]
            motor_q_acc = q_acc[i]
            motor_q_vel = q_vel[i]  # Updated to use motor_q_vel[i]
            print(f'sending to motor {servo_id}: pos {motor_q_pos}, vel {motor_q_vel}, acc {motor_q_acc}')
            scs_addparam_result = self.packetHandler.SyncWritePosEx(servo_id, motor_q_pos, motor_q_vel, motor_q_acc)
            if scs_addparam_result != True:
                raise GroupAddParamFailedException()

        scs_comm_result = self.packetHandler.groupSyncWrite.txPacket()
        
        if scs_comm_result != COMM_SUCCESS:
            raise SyncWriteFailedException()
        self.packetHandler.groupSyncWrite.clearParam()
    
    def zero_motors(self):
        for i in range(len(self.motor_ids)):
            servo_id = self.motor_ids[i]

            scs_addparam_result = self.packetHandler.SyncTorqueOffCalPos(servo_id, np.uint8(128))
            if scs_addparam_result != True:
                raise GroupAddParamFailedException()

        scs_comm_result = self.packetHandler.groupSyncWrite_TorqueOffCalPos.txPacket()
        if scs_comm_result != COMM_SUCCESS:
           raise SyncWriteFailedException()

        self.packetHandler.groupSyncWrite_TorqueOffCalPos.clearParam()
    
    def disable_torque(self):
        for i in range(len(self.motor_ids)):
            servo_id = self.motor_ids[i]

            scs_addparam_result = self.packetHandler.SyncTorqueOffCalPos(servo_id, np.uint8(0))
            if scs_addparam_result != True:
                raise GroupAddParamFailedException()

        scs_comm_result = self.packetHandler.groupSyncWrite_TorqueOffCalPos.txPacket()
        if scs_comm_result != COMM_SUCCESS:
            print(scs_comm_result)
            raise SyncWriteFailedException()

        self.packetHandler.groupSyncWrite_TorqueOffCalPos.clearParam()
    