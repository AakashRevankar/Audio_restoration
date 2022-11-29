# Audio restoration using Median filter and Cubic splines

## High-level Description of the project
This project is built on Assignment 1 (Trinity College Dublin) of Computational Methods in the M.Sc Electronic Information Engineering Course. Assignment 1 is about degrading a clean signal, detecting the clicks and filtering it using Autoregressive methods.

For more details about the project check [here](https://www.google.ie/webhp) to read the project report of Assignmnet 1.

---
The '''degraded signal.wav''' , ```detected_signal_clicks.mat``` are extracted using the program mentioned here.

- median filtering
- cubic splines

---

## Installation and Execution

Provide details on the Python version and libraries (e.g. numpy version) you are using. One easy way to do it is to do that automatically:
```sh                                 
pip3 install pipreqs

pipreqs $project/path/requirements.txt
```
For more details check [here](https://github.com/bndr/pipreqs)


Afer installing all required packages you can run the demo file simply by typing:
```sh
python demo_audio_restoration.py
```
---

## Methodology and Results
Describe here how you have designed your code, e.g. a main script/routine that calls different functions, is the unittesting included in the main routine? 



**Results**

1. For the median filter, different lengths were explored to test the effectiveness of the restoration. In particular, XXXX were tested and XXX was observed to deliver the lowest MSE, as shown in the figure below.

<img src="MedianFilter_MSEvsLength.png" width="350">

The restored waveform <output_medianFilter.wav> with the optimal filter length is given below:



2. Using the cubic splines, we observe ....

The restored waveform <output_cubicSplines.wav> with the optimal filter length is given below:


3. Comparing the two different interpolation methods, we notice that method X achieves a lower MSE. The runtime of XX method is .....

After listening to the two restored files, we notice ...


---
## Credits

This code was developed for purely academic purposes by XXXX (add github profile name) as part of the module ..... 

Resources:
- XXXX
- XXX





