#!/usr/bin/env python3
import rospy
import numpy as np
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

def callback(data):
	rospy.loginfo(f"{rospy.get_name()} received {data}")

def listener():
	rospy.init_node('audio_processor_node')

	rospy.Subscriber("audio_signal", numpy_msg(Floats), callback)

	rospy.spin()

if __name__ == '__main__':
	listener()
