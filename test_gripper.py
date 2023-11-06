import numpy as np
from numpy import pi
from mitsubishi_arm_student_interface.mitsubishi_robots import Mitsubishi_robot

if __name__=='__main__':
    robot = Mitsubishi_robot()

    # Open gripper
    robot.set_gripper('open')

    # Close gripper
#    robot.set_gripper('close')
