# upload error_points.mat, degraded.wav, original.wav

# ---------------------------------Importing libraries----------------------------------------------
from sklearn.metrics import mean_squared_error
from scipy.io import wavfile
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import scipy.io
from scipy.io.wavfile import write
from scipy.interpolate import CubicSpline
from datetime import datetime
from time import sleep
from tqdm import tqdm
from playsound import playsound

# -----------------------------------Defining functions ---------------------------------------------


def plot(data, samplerate):
    '''

    Takes data and sample rate of the given signal and returns the plotted graph of the signal

    Args:
        data(array) : an array which contains the audio data to be plotted

    Returns:
        plt.show() : the matplotlib function which displays the graph of the data

    '''
    # '''length and breadth of the graph is calculated'''
    length = data.shape[0] / samplerate
    time = np.linspace(0., length, data.shape[0])

    # '''The size of the figure is mentioned'''
    plt.figure(figsize=(15, 5))

    # '''The labels and display value is mentioned'''
    plt.plot(time, data, label="Degraded Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    return plt.show()


# -----------------------------Reading all the signals--------------------------------------------------
samplerate_new, data_new = wavfile.read("original.wav")
samplerate_deg, data_deg = wavfile.read("degraded.wav")

# '''------------------------ Reading clicks from matlab file ------------------------------------------'''

# ''' Plotting the input data '''
inp_waveform = plot(data_deg, samplerate_deg)

# Loading click point from matlab
click_point = scipy.io.loadmat('error_points.mat')

# Taking keys from matlab
# print(click_point.keys())

# Extracting error signal key
error_key = click_point['error_signal']

# Searching the click
click = np.where(error_key == 1)

# Finding the actual click (converting tuple to an array)
actual_click = click[0]
# print(actual_click)

# The number of click
click_num = len(actual_click)
# print(click_num)

# Assigning index for the degraded signal
index_deg = np.arange(len(data_deg))
# print(index_deg)

# Deleting the clicks from the degraded data array
y = np.delete(data_deg, actual_click)
# plt.plot(y)

# Creating index without clicks
x = np.delete(index_deg, actual_click)
# print(x)

# '''-----------------------------------CUBIC SPLINED FILTER--------------------------------------'''

# '''Initiating the counter'''
start_time = datetime.now()

# Dupicating degraded data
cubic_splined_data = data_deg

# '''The progess bar for median filter'''
for z in tqdm(range(100)):
    # Applying the cubic spline function
    cs = CubicSpline(x, y, bc_type='natural')

# Training the clicked data with prediction of cubic_splined data
for i in range(click_num):
    cubic_splined_data[actual_click[i]] = cs(actual_click)[i]

# ''' Terminating the counter's count'''
end_time = datetime.now()
durationTime = end_time - start_time
print('Done')
print("The duration for the cubic spline filter is " + str(durationTime))

# Plotting the cubic spline data
cs_data = plot(cubic_splined_data, samplerate_new)

# '''Creating and playing the restored audio'''
write("rest_c.wav", samplerate_deg, cubic_splined_data.astype(np.int16))


# Calculating mean square error
mse = (np.square(np.subtract(cubic_splined_data, data_new)).mean())
print(mse)

# '''--------------------------------Playing the audio --------------------------------------------------'''

# '''Playing degraded signal'''

print("Playing degraded audio")
playsound("degraded_less.wav")

# '''Playing restored signal'''

print("Playing restored audio from cubic spline")
playsound("rest_c.wav")

# '''-----------------------------------------------------END----------------------------------------------'''
