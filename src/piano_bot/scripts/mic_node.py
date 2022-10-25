#!/usr/bin/env python3
import rospy
import pyaudio
import struct
import numpy as np

from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

def initAudioStream(samplingRate = 44100, chunkSize = 4096, devIndex = 0):
	dataFormat = pyaudio.paInt16

	audio = pyaudio.PyAudio()
	audioStream = audio.open(format = dataFormat, rate=samplingRate, channels=1, input_device_index = devIndex, input=True, frames_per_buffer=chunkSize)

	return audioStream

def talker():
	chunkSize = 4096
	samplingRate = 44100

	audioStream = initAudioStream(samplingRate=samplingRate, chunkSize=chunkSize, devIndex=0)

	publisher = rospy.Publisher('audio_signal', numpy_msg(Floats), queue_size=10)
	rospy.init_node("mic_node")
	rate = rospy.Rate(0.1)

	while not rospy.is_shutdown():
		data = audioStream.read(chunkSize)
		decoded = struct.unpack(str(chunkSize) + 'h', data)
		amplitude = np.max(decoded)

		if amplitude > 3500:
			publisher.publish(decoded)

		rate.sleep()

if __name__ == "__main__":
	talker()
