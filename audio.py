# Loading libraries
import numpy as np
import scipy
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.io.wavfile import write


# PlaySound
# from playsound import playsound
# playsound("degraded.wav")


def zero_append(filter_size):
    if filter_size % 2 != 0:
        n_zero = int((filter_size - 1) / 2)
        return np.pad(inp_list, (n_zero, n_zero), 'constant', constant_values=(0, 0))
    else:
        print(" Please give odd number for filter size ")

# Function of median filter
def median(padded_list):
    median_list = []
    for i in range(len(padded_list) - filter_size + 1):
        sorted_list = np.sort(padded_list[i: i + filter_size])
        median = (sorted_list[int((filter_size)/2)])
        median_list.append(median)
    return median_list

def threshold(threshold):
    err_up = np.where(data > threshold)
    err_down = np.where(data < -threshold)
    detection_signal = np.concatenate((err_up, err_down), axis=None)
    return detection_signal

def plot(data):
    length = data.shape[0] / samplerate
    time = np.linspace(0., length, data.shape[0])
    plt.figure(figsize=(15, 5))
    plt.plot(time, data, label="Degraded Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    return plt.show()

# Reading data and sample rate
samplerate, data = wavfile.read("degraded.wav")


# Plotting the input data
inp_waveform = plot(data)

# Error signal
detection_signal = threshold(19600)

# Important parameters

filter_size = 3
z = int((filter_size - 1)/2)

ok_flag = False
# Median filter for each value
k = 0
recovered_signal =[]
for k in range(len(detection_signal)):
    i = detection_signal[k]
    inp_list = data[i - z: i + (z + 1)]
    test_list = scipy.signal.medfilt(inp_list, kernel_size=filter_size)
    padded_list1 = zero_append(filter_size)
    median_list1 = np.array(median(padded_list1))
    

    if(np.array_equal(median_list1, test_list)):
        ok_flag = True
    else:
        ok_flag = False
    data[i - z: i + (z + 1)] = median_list1

if(ok_flag):
    print("OK")
else:
    print("NOT OK")

# Plotting the restored value
out_waveform = plot(data)

# from playsound import playsound
# playsound("restored_beats.wav")