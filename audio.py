''' ---------------------------------Importing libraries-------------------------------------------------'''
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
from playsound import playsound
import unittest

'''Playing degraded signal'''

print("Playing degraded audio")
playsound("degraded_less.wav")

''' Function to create padded array'''
def zero_append(filter_size, half_f):
  if filter_size % 2 != 0:

    '''This np.pad creates a require padded array for the given filter size '''
    return np.pad(inp_list, (half_f, half_f), 'constant', constant_values=(0, 0))

  else:
    print(" Please give odd number for filter size ")

'''Function to find median and create a median list'''
def median(padded_list):
  median_list = []
  for i in range(len(padded_list) - filter_size + 1):

    '''sorted_list sorts the padded_list'''
    sorted_list = np.sort(padded_list[i: i + filter_size])

    '''median finds the median value in the sorted list'''
    median = (sorted_list[int((filter_size)/2)])

    '''median list collects all the median value'''
    median_list.append(median)

  return median_list

'''Function that plots the given data'''
def plot(data):
  '''length and breadth of the graph is calculated'''
  length = data.shape[0] / samplerate
  time = np.linspace(0., length, data.shape[0])

  '''The size of the figure is mentioned'''
  plt.figure(figsize=(15, 5))

  '''The labels and display value is mentioned'''
  plt.plot(time, data, label="Degraded Signal")
  plt.xlabel("Time [s]")
  plt.ylabel("Amplitude")
  return plt.show()

'''Reading data and sample rate'''
samplerate, data = wavfile.read("degraded_less.wav")

''' Creating replica of data files as not to get shuffled with different values'''
data2 = data
myfilter_data = data
sysfilter_data = data


''' Plotting the input data '''
inp_waveform = plot(data)

'''------------------------ Reading clicks from matlab file --------------------------------------'''

'''Loading click point from matlab'''
click_point = scipy.io.loadmat('error_points.mat')

'''Taking keys from matlab'''
# print(click_point.keys())

'''Extracting error signal key'''
error_key = click_point['error_signal']

'''Searching the click'''
click = np.where(error_key == 1)

'''Finding the actual click (converting tuple to an array)'''
actual_click = click[0]
# print(actual_click)

''' The total number of clicks is counted '''
click_num = len(actual_click)
# print(click_num)

'''-----------------------------Important parameters----------------------------------------------'''

filter_size = 3
half_f = int((filter_size - 1)/2)
'''half_f is pad width and is used to extract data with given filter size'''

'''Initiating the counter'''
start_time = datetime.now()

'''The progess bar for median filter'''
for s in tqdm(range(100)):
    sleep(0.05)

'''----------------------This is the main file which acts like median filter-------------------------'''

for k in range(click_num):
    '''inp_list is the input list created through data'''
    inp_list = data[actual_click[k] - half_f: actual_click[k] + (half_f + 1)]

    '''padded_array creates pads with zero and determines whether the filter size is input or odd'''
    padded_array = zero_append(filter_size, half_f)

    '''median_array creates the output of the median number across the clicks'''
    median_array = np.array(median(padded_array))

    '''The median filtered data is applied back to the signal'''
    myfilter_data[actual_click[k] -
                  half_f: actual_click[k] + (half_f + 1)] = median_array

''' Terminating the counter's count'''

print("done")
end_time = datetime.now()
durationTime = end_time - start_time
print("The duration for the median filter is" + str(durationTime))

'''Plotting the restored value '''
out_waveform = plot(myfilter_data)

'''Creating and playing the restored audio'''
write("restored_less.wav", samplerate, myfilter_data.astype(np.int16))

print("Playing restored audio")
playsound("restored_less.wav")

'''------------------------------------Calculating MSE  -------------------------------------------'''

'''Reading the original file'''
samplerate_new, data_new = wavfile.read("orginal.wav")

mse = (np.square(np.subtract(data_new, myfilter_data)).mean())
print("The Mean square error between the restored signal and original signal is ", + mse)

''' -----------------------------This is inbuit median filter--------------------------------------------- '''
for s in tqdm(range(100)):
  sleep(0.05)

for y in range(click_num):
    inp_list = data2[actual_click[y] - half_f: actual_click[y] + (half_f + 1)]

    '''kernel_size is similar as filter_size'''
    test_list = scipy.signal.medfilt(inp_list, kernel_size=filter_size)

    '''The systems median filtered data is applied back to the signal'''
    sysfilter_data[actual_click[y] -
                   half_f: actual_click[y] + (half_f + 1)] = test_list
print('done')
'''--------------------------------Test cases -----------------------------------------------------------'''
class TestFilter(unittest.TestCase):
  def test_length(self):
    length1 = len(myfilter_data)
    length2 = len(sysfilter_data)
    self.assertEqual(length1,length2)

  def test_data(self):
    check = np.array_equal(median_array, test_list)

if __name__ == '__main__' :
  unittest.main()