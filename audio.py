import numpy as np
import scipy
from scipy.io import wavfile
import scipy.signal
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from sklearn.metrics import mean_squared_error
from datetime import datetime
from time import sleep
from tqdm import tqdm

def plot(data):
    length = data.shape[0] / samplerate
    time = np.linspace(0., length, data.shape[0])
    plt.figure(figsize=(15, 5))
    plt.plot(time, data, label="Degraded Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    return plt.show()

def median_filter():
  # Reading clicks
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

  def median(padded_list):
    median_list = []
    for i in range(len(padded_list) - filter_size + 1):
        sorted_list = np.sort(padded_list[i: i + filter_size])
        median = (sorted_list[int((filter_size)/2)])
        median_list.append(median)
    return median_list

  half_f = int((filter_size - 1)/2)
  for k in range(click_num):
      i = actual_click[k]
      inp_list = data[i - half_f: i + (half_f + 1)]
      padded_list = np.pad(inp_list, (half_f, half_f), 'constant', constant_values=(0, 0))
      median_list1 = np.array(median(padded_list))
      filtered_data[i - half_f: i + (half_f + 1)]= median_list1
  return filtered_data

# Important parameters

filter_size = 5

# Reading data and sample rate
samplerate, data = wavfile.read("degraded_less.wav")
filtered_data = data

# Plotting the input data
inp_waveform = plot(data)

start_time = datetime.now()
for z in tqdm(range(100)):
  sleep(0.05)
restored_signal = median_filter()
plot(restored_signal)
end_time = datetime.now()
durationTime = end_time - start_time
print("Done...")
print("The duration for the median filter is" + str(durationTime))

samplerate_new, data_new = wavfile.read("orginal.wav")

mse = (np.square(np.subtract(data, data_new)).mean())
print(mse)