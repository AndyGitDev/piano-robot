#!/usr/bin/env python3
import rospy
import numpy as np

from math import ceil, floor
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import String

_FREQ = []
_SAMPLING_FREQUENCY = 44100
_NOTE_FREQ = {
	32.703 : "C1",
	34.6476 : "C#1",
	36.7079 : "D1",
	38.8906 : "D#1",
	41.2031 : "E1",
	43.653 : "F1",
	46.2486 : "F#1",
	48.9986 : "G1",
	51.912 : "G#1",
	54.9986 : "A1",
	58.2687 : "A#1",
	61.7332 : "B1",
	65.4036 : "C2",
	69.2923 : "C#2",
	73.4122 : "D2",
	77.777 : "D#2",
	82.4013 : "E2",
	87.3005 : "F2",
	92.4909 : "F#2",
	97.9899 : "G2",
	103.8159 : "G#2",
	109.9881 : "A2",
	116.5274 : "A#2",
	123.4554 : "B2",
	130.7952 : "C3",
	138.5714 : "C#3",
	146.81 : "D3",
	155.5382 : "D#3",
	164.7854 : "E3",
	174.5824 : "F3",
	184.9617 : "F#3",
	195.9581 : "G3",
	207.6083 : "G#3",
	219.951 : "A3",
	233.0275 : "A#3",
	246.8815 : "B3",
	261.559 : "C4",
	277.1091 : "C#4",
	293.5837 : "D4",
	311.0377 : "D#4",
	329.5293 : "E4",
	349.1203 : "F4",
	369.8759 : "F#4",
	391.8655 : "G4",
	415.1623 : "G#4",
	439.8441 : "A4",
	465.9933 : "A#4",
	493.697 : "B4",
	523.0478 : "C5",
	554.1434 : "C#5",
	587.0877 : "D5",
	621.9905 : "D#5",
	658.9682 : "E5",
	698.1443 : "F5",
	739.6494 : "F#5",
	783.622 : "G5",
	830.2088 : "G#5",
	879.5652 : "A5",
	931.8557 : "A#5",
	987.255 : "B5",
	1045.9478 : "C6",
	1108.1298 : "C#6",
	1174.0085 : "D6",
	1243.8038 : "D#6",
	1317.7484 : "E6",
	1396.0889 : "F6",
	1479.0869 : "F#6",
	1567.019 : "G6",
	1660.1787 : "G#6",
	1758.8768 : "A6",
	1863.4424 : "A#6",
	1974.2245 : "B6",
	2091.5926 : "C7",
	2215.9382 : "C#7",
	2347.6762 : "D7",
	2487.246 : "D#7",
	2635.1132 : "E7",
	2791.7711 : "F7",
	2957.7423 : "F#7",
	3133.5806 : "G7",
	3319.8724 : "G#7",
	3517.2392 : "A7",
	3726.3395 : "A#7",
	3947.8708 : "B7",
	4182.5722 : "C8",
}

note = ""
notes = []

def goertzel(inputSignal, samplingFreq, numTerms, freqIndex):
	# Goertzel constants
	K = (_FREQ[freqIndex]*numTerms)/samplingFreq
	A = 2*np.pi*(K/numTerms)
	cw = np.cos(A)
	sw = np.sin(A)
	
	c = 2*cw

	# Initialize State Variables
	s = [0, 0, 0]

	# Main Algorithm
	for i in range(numTerms-1):
		s[0] = inputSignal[i] + c*s[1] - s[2]
		s[2] = s[1]
		s[1] = s[0]

	complexVal = [(sw*s[1]), (cw*s[1] - s[2])]

	pow = (complexVal[0]**2) + (complexVal[1]**2)

	return pow

def runGoertzel(inputSignal, samplingFreq, numTerms):
	freqBin = []
	for freqIndex in range(len(_FREQ)):
		pow = goertzel(inputSignal, samplingFreq, numTerms, freqIndex)
		freqBin.append(pow)

	freqBin = 2*np.array(freqBin)/numTerms

	return freqBin

def detectNotes(freqBin):

	threshHold = 0.9*np.max(freqBin)

	indices = np.where(freqBin > threshHold)[0]
	possibleNotes = []
	for index in indices:
		possibleNotes.append(_NOTE_FREQ[_FREQ[index]])

	if len(possibleNotes) > 1:
		if len(possibleNotes) % 2 == 0:
			detectedNote = possibleNotes[ceil(len(possibleNotes)/2)]
		else:
			detectedNote = possibleNotes[floor(len(possibleNotes)/2)]

	else:
		detectedNote = possibleNotes[0]

	return detectedNote, possibleNotes

def callback(data):
	global detectedNotes
	
	rospy.loginfo(f"{rospy.get_name()} received signal")
	freqBin = runGoertzel(inputSignal=np.array(data.data), samplingFreq=_SAMPLING_FREQUENCY, numTerms=len(data.data))
	detectedNote, possibleNotes = detectNotes(freqBin)
	rospy.loginfo(f"Detected: {detectedNote} Possible notes: {possibleNotes}")
	note = detectedNote

	notePublisher.publish(note)

rospy.Subscriber("audio_signal", numpy_msg(Floats), callback)
notePublisher = rospy.Publisher('detected_note', String, queue_size=10)

def audioProcessor():
	global _FREQ, detectedNotes
	_FREQ = np.load("/home/odroid/catkin_ws/src/static/pianoKeyFrequencies.npy")

	rospy.loginfo("Initialised audio processing node.")

	rospy.init_node('audio_processor_node')

	rospy.spin()

if __name__ == '__main__':
	audioProcessor()
