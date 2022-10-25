#!/usr/bin/env python3
import rospy
import numpy as np

from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

def radFreq(freq):
	return freq * 2*np.pi

def generateSignal(testFrequency, samplingFrequency, numTerms):
    """
    Generates sinusoidal signal at a given frequency

    @param testFrequency: Desired frequency of signal
    @param samplingFrequency: Desired samplingFrequency
    @param numTerms: Number of terms in signal
    """
    timeSteps = np.linspace(0, numTerms*(1/samplingFrequency), numTerms)
    signal = np.sin(radFreq(freq=testFrequency)*timeSteps)

    return signal, timeSteps

def addGaussianNoise(inputSignal, standardDeviation):
    """
    Adds Gaussian white noise to a given input signal where the noise is centred on the given bias.

    @param inputSignal: Input signal to add noise to
    @param standardDeviation: The standard deviation to be used for adding the gaussian noise

    @return noisy: The noisy signal
    """
    gaussianNoise = np.random.normal(0, standardDeviation, len(inputSignal))
    noisy = [sum(value) for value in zip(inputSignal, gaussianNoise)]

    return noisy

def generateNoise(inputSignal, timeSteps):
    noiseSignals = []
    noiseFrequencies = []
    for i in range(5):
        noiseAmp = np.random.uniform(0.05, 0.15)
        noiseFrequencies.append(np.random.randint(1, 10000))
        noiseSignals.append(addGaussianNoise(noiseAmp * np.sin(radFreq(noiseFrequencies[i])*timeSteps), 0.1))

    noisy = inputSignal.copy()
    for i in range(5):
        noisy = [sum(value) for value in zip(noisy, noiseSignals[i])]

    return noisy, noiseFrequencies

def generateTestSignal(freq, samplingFreq, numTerms):
    signal, timeSteps = generateSignal(freq, samplingFreq, numTerms)
    noisySignal, noiseFrequencies = generateNoise(signal, timeSteps)

    return np.array(noisySignal, dtype=np.float32)

def talker():
    _FREQ = np.load("/home/odroid/catkin_ws/src/static/pianoKeyFrequencies.npy")

    publisher = rospy.Publisher('audio_signal', numpy_msg(Floats), queue_size=10)
    rospy.init_node("test_node")
    rate = rospy.Rate(0.1)
    while not rospy.is_shutdown():
        randomFrequency = np.random.choice(_FREQ)
        testArray = generateTestSignal(randomFrequency, 44100, 4096)

        rospy.loginfo(f"Generated signal with frequency of {randomFrequency} Hz")

        publisher.publish(testArray)

        rate.sleep()


if __name__ == "__main__":
    talker()
