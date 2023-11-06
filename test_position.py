import rospy
from numpy import pi
from mitsubishi_arm_student_interface.mitsubishi_robots import Mitsubishi_robot
 
 
if __name__=='__main__':
 
    robot = Mitsubishi_robot()
 
    # Set maximal relative speed (it is recomended to decrease the speed for testing)
    robot.set_max_speed(0.3);
 
    # Move to to position given in joint coordinates
    print 'Moving to base position'
    robot.set_joint_values([0,0,pi/2,0,pi/2,0])
 
    # Move a bit more
    print 'Move'
    robot.set_joint_values([pi/4,0,pi/2,0,pi/2,0])
    robot.set_joint_values([-pi/4,0,pi/2,0,pi/2,0])
    robot.set_joint_values([0,0,pi/2,0,pi/2,0])
