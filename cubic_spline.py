#Importing libraries
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

def plot(data):
    length = data.shape[0] / samplerate_new
    time = np.linspace(0., length, data.shape[0])
    plt.figure(figsize=(15, 5))
    plt.plot(time, data, label="Degraded Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    return plt.show()

#Reading all the signals
samplerate_new, data_new = wavfile.read("orginal.wav")
samplerate_rec, data_rec = wavfile.read("restored_less.wav")
samplerate_deg, data_deg = wavfile.read("degraded_less.wav")

#Calculating mean square error of original and restored signal from median filter
mse = (np.square(np.subtract(data_new, data_rec)).mean())
print(mse)

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


start_time = datetime.now()

# Dupicating degraded data 
cubic_splined_data = data_deg

for z in tqdm(range(100)):
  # Applying the cubic spline function
  cs = CubicSpline(x, y, bc_type = 'natural')

#Training the clicked 
for i in range(click_num):
  cubic_splined_data[actual_click[i]] = cs(actual_click)[i]


end_time = datetime.now()
durationTime = end_time - start_time
print('Done')
print("The duration for the cubic spline filter is" + str(durationTime))

#Plotting the cubic spline data
plt.plot(cubic_splined_data)
write("restored_cubic.wav", samplerate_new, cubic_splined_data.astype(np.int16)) 

#Calculating mean square error
mse = (np.square(np.subtract(cubic_splined_data, data_new)).mean())
print(mse)