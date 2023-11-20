import numpy as np
from numpy import pi
from mitsubishi_arm_student_interface.mitsubishi_robots import Mitsubishi_robot

if __name__=='__main__':
    # Create interace with moveit, ROS and others
    robot = Mitsubishi_robot()

    # Set the robot relative maximal speed (decrease for debugging is recommended)
    robot.set_max_speed(0.05)

    # get position
    x, y, z, roll, pitch, yaw = robot.get_position()

    print(roll, pitch, yaw)

    wps = [
        [x, y, z, roll, pitch, yaw],
        [x, y, z, roll, pitch, yaw+0.1],
        [x, y, z, roll, pitch, yaw],
    ]

    robot.execute_cart_trajectory(wps)

