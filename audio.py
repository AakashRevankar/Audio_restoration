# upload error_points.mat, degraded.wav, original.wav

# ---------------------------------Importing libraries-------------------------------------------------
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

# -----------------------------------Defining all functions ---------------------------------------------

#''' Function to create padded array'''


def zero_append(filter_size, half_f, inp_list):
    '''
    Takes in filter_size, inp_list checks whether the filter size is odd or even and builds a padded array

    Args:
        filter_size(int) : a natural number
        half_f(int) : a natural number and number of zeros added to inp_list depending on filter_size
        inp_list(array) : an array which contains the the degraded data value

    Returns:
        zero_padded array if the filter size is even or throws an error if filter_size is odd

    '''
    if filter_size % 2 != 0:

        #'''This np.pad creates a require padded array for the given filter size '''
        return np.pad(inp_list, (half_f, half_f), 'constant', constant_values=(0, 0))

    else:
        print(" Please give odd number for filter size ")

#'''Function to find median and create a median list'''


def median(padded_list):
    '''
    Takes in padded_list and returns the padded_list

    Args:
        padded_list(array) : an array which contains the padded array returned from zero_append function

    Returns:
        median_list(array) : an array which contains the median value of the given input_list

    '''
    median_list = []
    for i in range(len(padded_list) - filter_size + 1):

        # '''sorted_list sorts the padded_list'''
        sorted_list = np.sort(padded_list[i: i + filter_size])

        # '''median finds the median value in the sorted list'''
        median = (sorted_list[int((filter_size)/2)])

        # '''median list collects all the median value'''
        median_list.append(median)

    return median_list

# Function of my median filter


def my_median(data, actual_click, click_num, filter_size):
    myfilter_data = data
    for k in range(click_num):

        '''
        Takes data, actual clicks on data, number of clicks on data and filter size given by the user, returns the my_filtered_data 
        which is restored signal from the clicks

        Args:
            data(array) : an array which contains the degraded audio
            actual_click(array) : an array which contains the actual clicks on the data file, which is obtained from the matlab file
            click_num(array): a number which has the number of clicks in the degraded audio
            filter_size(int) : a natural number

        Returns:
            myfilter_data(array) : an array which contains audio with clicks removed

        '''
        # '''inp_list is the input list created through data'''
        inp_list = data[actual_click[k] -
                        half_f: actual_click[k] + (half_f + 1)]

        # '''padded_array creates pads with zero and determines whether the filter size is input or odd'''
        padded_array = zero_append(filter_size, half_f, inp_list)

        # '''median_array creates the output of the median number across the clicks'''
        median_array = np.array(median(padded_array))

        # '''The median filtered data is applied back to the signal'''
        myfilter_data[actual_click[k] -
                      half_f: actual_click[k] + (half_f + 1)] = median_array

    return myfilter_data


# '''Function that plots the given data'''


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


# '''---------------------Taking input from the signal ------------------------------------------------'''
# '''Reading data and sample rate'''
samplerate, data = wavfile.read("degraded.wav")

# ''' Creating replica of data files as not to get shuffled with different values'''
data2 = data
sysfilter_data = data


# ''' Plotting the input data '''
inp_waveform = plot(data, samplerate)

# '''------------------------ Reading clicks from matlab file ------------------------------------------'''

# '''Loading click point from matlab'''
click_point = scipy.io.loadmat('error_points.mat')

# '''Taking keys from matlab'''
# print(click_point.keys())

# '''Extracting error signal key'''
error_key = click_point['error_signal']

# '''Searching the click'''
click = np.where(error_key == 1)

# '''Finding the actual click (converting tuple to an array)'''
actual_click = click[0]
# print(actual_click)

# ''' The total number of clicks is counted '''
click_num = len(actual_click)
# print(click_num)

# '''-----------------------------Important parameters------------------------------------------------'''

filter_size = 3
half_f = int((filter_size - 1)/2)
# '''half_f is pad width and is used to extract data with given filter size'''

# '''Initiating the counter'''
start_time = datetime.now()

# '''The progess bar for median filter'''
for s in tqdm(range(100)):
    sleep(0.05)

# '''--------------------------------Calling the filter -----------------------------------------------'''

myfilter_data = my_median(data, actual_click, click_num, filter_size)


# ''' Terminating the counter's count'''

print("done")
end_time = datetime.now()
durationTime = end_time - start_time
print("The duration for the median filter is" + str(durationTime))

# '''Plotting the restored value '''
out_waveform = plot(myfilter_data, samplerate)

# '''Creating and playing the restored audio'''
write("restored.wav", samplerate, myfilter_data.astype(np.int16))

# '''--------------------------------Playing the audio --------------------------------------------------'''


# '''Playing degraded signal'''

print("Playing degraded audio")
playsound("degraded.wav")

# '''Playing restored signal'''

print("Playing restored audio")
playsound("restored.wav")

# '''------------------------------------Calculating MSE  -----------------------------------------------'''

# '''Reading the original file'''
samplerate_new, data_new = wavfile.read("original.wav")

mse = (np.square(np.subtract(data_new, myfilter_data)).mean())
print("The Mean square error between the restored signal and original signal is ", + mse)

# '''--------------------------------Test cases -----------------------------------------------------------'''


class TestFilter(unittest.TestCase):
    '''
    Test_Filter is defined as a subclass of unittest.TestCase
    '''

    def test_length(self):
        '''
        A method named test_length is defined on TestFilter 

        Args:
            self is pointing the argumensts
            length1(integer) : an integer number which contains the my_filter_data
            length2(integer): an integer number which contais the sys_filter_data (inbuit filtered data)

        Returns:
            Asserts OK if the length of my_median_filtered_data and system_inbuit_filtered_data

        '''
        length1 = len(myfilter_data)
        length2 = len(sysfilter_data)
        self.assertEqual(length1, length2)

    def test_valueOfData(self):
        '''
        A method named test_valueOfData is defined on TestFilter 

        Args:
            self is pointing the argumensts
            inp_list(array) : an array which contains the degraded audio data

        Returns:
            Checks whether the each value of my_filtered_data is equal system_filtered_data

        '''
        half_f = int((filter_size - 1)/2)

        for y in range(click_num):
            inp_list = data2[actual_click[y] -
                             half_f: actual_click[y] + (half_f + 1)]

            # '''kernel_size filter_size of inbuit system filter data'''
            test_list = scipy.signal.medfilt(inp_list, kernel_size=filter_size)

            # '''The systems median filtered data is applied back to the signal'''
            sysfilter_data[actual_click[y] -
                           half_f: actual_click[y] + (half_f + 1)] = test_list

        # checks the each value of my_filtered_data and system_filtered_data
        check = np.array_equal(myfilter_data, sysfilter_data)


# Executes the testcase
if __name__ == '__main__':
    unittest.main()

# '''-----------------------------------------------------------END---------------------------------------------------'''
