# Loading libraries
import numpy as np
import scipy
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from sklearn.metrics import mean_squared_error
from datetime import datetime
from time import sleep
from tqdm import tqdm


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


def plot(data):
    length = data.shape[0] / samplerate
    time = np.linspace(0., length, data.shape[0])
    plt.figure(figsize=(15, 5))
    plt.plot(time, data, label="Degraded Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    return plt.show()

# Reading data and sample rate
samplerate, data = wavfile.read("degraded_less.wav")


# Plotting the input data
inp_waveform = plot(data)

#############################################################
# Reading clicks
# Loading click point from matlab
click_point = scipy.io.loadmat('error_points.mat')

# Taking keys from matlab
print(click_point.keys())

# Extracting error signal key
error_key = click_point['error_signal']

# Searching the click
click = np.where(error_key == 1)

# Finding the actual click (converting tuple to an array)
actual_click = click[0]
# print(actual_click)

# The number of click
click_num = len(actual_click)
print(click_num)
#######################################################################################

# Important parameters

filter_size = 5
z = int((filter_size - 1)/2)


start_time = datetime.now()

for z in tqdm(range(100)):
  sleep(0.05)


# ok_flag = False
# Median filter for each value
k = 0
recovered_signal =[]
for k in range(click_num):
    i = actual_click[k]
    inp_list = data[i - z: i + (z + 1)]
    test_list = scipy.signal.medfilt(inp_list, kernel_size=filter_size)
    padded_list1 = zero_append(filter_size)
    median_list1 = np.array(median(padded_list1))
    # print(median_list1)
    if(np.array_equal(median_list1, test_list)):
        ok_flag = True
    else:
        ok_flag = False
    data[i - z: i + (z + 1)] = median_list1
# print(data)
end_time = datetime.now()
durationTime = end_time - start_time
print("The duration for the median filter is" + str(durationTime))

# if(ok_flag):
#     print("OK")
# else:
#     print("NOT OK")

write("restored_less.wav", samplerate, data.astype(np.int16)) 

# Plotting the restored value
out_waveform = plot(data)

# from playsound import playsound
# playsound("restored_beats.wav")

samplerate_new, data_new = wavfile.read("orginal.wav")

mse = (np.square(np.subtract(data, data_new)).mean())
print(mse)