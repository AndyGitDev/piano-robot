#!/usr/bin/env python3
import rospy
import numpy as np

import sounddevice as sd
sd.default.samplerate = 44100
sd.default.channels = 1
sd.default.dtype = np.float32

from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import Empty

publisher = rospy.Publisher('audio_signal', numpy_msg(Floats), queue_size=10)
rospy.init_node("mic_node")

def recordSignal(empty):
	chunkSize = 8192
	samplingRate = 44100

	rospy.sleep(0.5)
	recording = sd.rec(frames=chunkSize, samplerate=samplingRate, channels=1)
	sd.wait()
	amplitude = np.max(recording)
	rospy.loginfo(f"Recorded amplitude: {amplitude}")

	publisher.publish(recording)

rospy.Subscriber("record_signal", Empty, recordSignal)

def talker():
	rospy.loginfo("Initialising node")

	chunkSize = 8192
	samplingRate = 44100

	rate = rospy.Rate(1)

	while not rospy.is_shutdown():
		# rospy.sleep(0.5)
		recording = sd.rec(frames=chunkSize, samplerate=samplingRate, channels=1)
		sd.wait()
		amplitude = np.max(recording)
		rospy.loginfo(f"Recorded amplitude: {amplitude}")
		if amplitude > 0.45:
			publisher.publish(recording)

		rate.sleep()

	rospy.spin()

if __name__ == "__main__":
	talker()
