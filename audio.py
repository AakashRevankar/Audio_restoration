# Loading libraries
import numpy as np
import scipy
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.io.wavfile import write


# PlaySound
from playsound import playsound
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


# Reading data and sample rate
samplerate, data = wavfile.read("degraded.wav")

# Plotting the data
length = data.shape[0] / samplerate
time = np.linspace(0., length, data.shape[0])
plt.figure(figsize=(15, 5))
plt.plot(time, data, label="Degraded Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

# Error signal
error_signal1 = np.where(data > 19600)
error_signal2 = np.where(data < -19600)
detection_signal = np.concatenate((error_signal1, error_signal2), axis=None)
print(detection_signal)

# Important parameters

filter_size = 3
z = int((filter_size - 1)/2)

# Median filter for each value
k = 0
recovered_signal =[]
for k in range(len(detection_signal)):
    i = detection_signal[k]
    inp_list = data[i - z: i + (z + 1)]
    padded_list1 = zero_append(filter_size)
    median_list1 = median(padded_list1)
    data[i - z: i + (z + 1)] = median_list1
length = data.shape[0] / samplerate
time = np.linspace(0., length, data.shape[0])
plt.figure(figsize=(15, 5))
plt.plot(time, data, label="Recovered Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()
write("recovered2.wav", samplerate, data.astype(np.int16))


for k in range(len(detection_signal)):
    i = detection_signal[k]
    inp_list = data[i - z: i + (z + 1)]
    test_list = scipy.signal.medfilt(inp_list, kernel_size=3)

check = np.array_equal(median_list1, test_list)
print(check)