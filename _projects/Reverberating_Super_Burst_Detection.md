---
title: Tutorial - Reverberating Super Burst Detection Workflow
description: A breakdown of how RSB detection and analysis works.
date: 2023-09-24
layout: page
img: 
category: neuroscience
related_publications: Pradeepan2023.09.12.557388

authors:
  - name: Kartik Pradeepan

toc:
  - name: Install necessary packages
  - name: Import libraries
  - name: Feature Calculation

---

This is a tutorial on how reverberating super bursts were detected in Pradeepan et al., 2023.

DISCLAIMER: The approach used in the manuscript relies upon the generation of spike density functions - to convert a discrete, analog signal (i.e., 1s and 0s) into a continuous one (i.e., spike density function). By doing so, we are theoretically changing the signal into something it is not. However, through careful interrogation, we have assured the questions we asked are not biased by this step.

To navigate around this issue, I have also developed a different approach. This "secondary approach" avoids the convolution of a kernel and relies upon my implementation of the Max Interval Burst Detection algorithm combined with a histogram-based approach to detecting network events. By combining these two approaches, we're able to cut the run time of the analysis by up to 95%, however the trade off is the accuracy of burst boundaries. Nonetheless, the results are still comparable. This secondary approach will be shared in this document as a hyperlink, including a separate tutorial at another time.


---


[Click here to see how the code performs on non-reverberating networks (i.e., ordinary network bursts)](/projects/Non-Reverberating_Detection/)


## Install necessary packages

Requirements: Pandas, Numpy, Matplotlib, Seaborn, Math, Scipy, Sci-kit learn

## Import libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from scipy.stats import norm
from scipy.signal import convolve
from scipy.signal import find_peaks
from scipy.signal import resample
from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans
from scipy.stats import skew
from scipy.stats import mode


```


```python
filename = "/content/drive/MyDrive/Colab Notebooks/spike_data.csv"
fs = 12500
```

## Convert Axion Biosystem's spike_list.csv file into a dataframe


```python
def organize_spikelist(filepath):
    # Create neuralMetric dataframe
    spikelist_file = pd.read_csv(filepath, sep='\r\n', header=None)
    spikelist_file = spikelist_file[0].str.split(',', expand=True)
    spikelist_file.columns = spikelist_file.iloc[0]
    spikelist_file = spikelist_file.iloc[1:]

    spiketimes = spikelist_file[["Time (s)", "Electrode"]]
    spiketimes = spiketimes.iloc[:-8]
    spiketimes = spiketimes.dropna()
    spiketimes = spiketimes[spiketimes["Electrode"].str.contains("_")]
    spiketimes = spiketimes.astype({"Time (s)": float})
    spiketimes["Well"] = [electrode.split("_")[0] for electrode in list(spiketimes["Electrode"].values)]
    spiketimes["Channel"] = [electrode.split("_")[1] for electrode in list(spiketimes["Electrode"].values)]

    r,c = np.where(spikelist_file == "Well Information")
    metadata_slice = spikelist_file.iloc[r[0]:].transpose().reset_index()
    metadata_slice.columns = metadata_slice.iloc[0]
    metadata_slice = metadata_slice.iloc[1:]
    metadata = {}
    label = []
    for i, well in enumerate(metadata_slice["Well"]):
        if well != None:
            metadata[well] = metadata_slice.iloc[i]["Treatment"]
            label.append(str(well)+"-"+str(metadata_slice.iloc[i]["Treatment"]))

    return spiketimes, metadata, label

spiketimes, metadata, label = organize_spikelist(filename)
```

    <ipython-input-3-2eeb288d4c56>:3: ParserWarning: Falling back to the 'python' engine because the 'c' engine does not support regex separators (separators > 1 char and different from '\s+' are interpreted as regex); you can avoid this warning by specifying engine='python'.
      spikelist_file = pd.read_csv(filepath, sep='\r\n', header=None)



```python
spiketimes.head()
```





  <div id="df-320bd06f-04c2-4d7f-944a-099895dbfc8e" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time (s)</th>
      <th>Electrode</th>
      <th>Well</th>
      <th>Channel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.00184</td>
      <td>A1_76</td>
      <td>A1</td>
      <td>76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.00824</td>
      <td>C1_51</td>
      <td>C1</td>
      <td>51</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.01160</td>
      <td>B1_47</td>
      <td>B1</td>
      <td>47</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.01984</td>
      <td>A1_42</td>
      <td>A1</td>
      <td>42</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.02224</td>
      <td>C1_41</td>
      <td>C1</td>
      <td>41</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-320bd06f-04c2-4d7f-944a-099895dbfc8e')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-320bd06f-04c2-4d7f-944a-099895dbfc8e button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-320bd06f-04c2-4d7f-944a-099895dbfc8e');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-83e9c6f7-6f26-482d-b5f5-c64fb09efa26">
  <button class="colab-df-quickchart" onclick="quickchart('df-83e9c6f7-6f26-482d-b5f5-c64fb09efa26')"
            title="Suggest charts."
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-83e9c6f7-6f26-482d-b5f5-c64fb09efa26 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
spiketimes["Well"].unique()
```




    array(['A1', 'C1', 'B1', 'A4', 'C2', 'B2', 'B4', 'A2', 'C3', 'C4', 'B3',
           'A3'], dtype=object)




```python
spiketimes["Channel"].unique()
```




    array(['76', '51', '47', '42', '41', '16', '26', '13', '18', '31', '84',
           '44', '82', '38', '85', '32', '88', '78', '11', '72', '87', '75',
           '48', '62', '24', '65', '63', '25', '15', '43', '81', '61', '46',
           '58', '83', '55', '57', '52', '12', '73', '22', '56', '67', '77',
           '66', '53', '36', '74', '86', '45', '23', '33', '21', '14', '71',
           '17', '37', '35', '28', '54', '27', '34', '68', '64'], dtype=object)




```python
duration = int(math.ceil(max(spiketimes["Time (s)"])/100.0))*100
print(f"It appears the duration of these recordings was {duration} seconds.")
```

    It appears the duration of these recordings was 300 seconds.


The data that we'll be dealing with are spike times from a 12-well multielectrode array system. Each well contains 64 channels (arranged in an 8x8 grid). Channels are labeled according to a cartesian coordinate system where the first digit represents x (or the row) and the second digit represents y (or the column).

## Select a well to analyze
This is where you would start wrapping the below code into a loop that iterates through each well, each plate/recording.


```python
well_id = "A3"
```

## Quick look at the network activity
Here we are plotting a histogram of spike times from the well we selected, using a bin size of 0.1 seconds (300 second duration/3000 bins).


```python
plt.figure(figsize=(10,5))
plt.hist(spiketimes[spiketimes["Well"]==well_id]["Time (s)"], bins=3000);
sns.despine()
plt.xlabel("Time (s)")
plt.ylabel("Total number of spikes in network (Bin size:1s)")
plt.title("Sort of like a PSTH - representing network activity")
```




    Text(0.5, 1.0, 'Sort of like a PSTH - representing network activity')




    
<img width="100%" alt="image" src="../Reverberating_Super_Burst_Detection_files/Reverberating_Super_Burst_Detection_16_1.png">
    


It looks like there is some interesting network bursting activity! Are these reverberating super bursts (RSB) like the ones mentioned in Pradeepan et al., 2023? 😂

Jokes aside, the code below will help identify whether this is or isn't an RSB. But for now, we'll plot a close up of a smaller window.


```python
# Definitely do not keep plotting histograms with large bin sizes if you want your code to run fast
plt.figure(figsize=(5,5))
plt.hist(spiketimes[spiketimes["Well"]==well_id]["Time (s)"], bins=3000);
sns.despine()
plt.xlim(146, 157)
plt.xlabel("Time (s)")
plt.ylabel("Total number of spikes in network (Bin size:1s)")
plt.title("Multiple peaks within a network event")
```




    Text(0.5, 1.0, 'Multiple peaks within a network event')




    
<img width="75%" alt="image" src="../Reverberating_Super_Burst_Detection_files/Reverberating_Super_Burst_Detection_18_1.png">
    


## Sort data frame containing mixed-bag of electrode and spike times based on channel name



```python
channel_ids = [i for j in range(1, 9) for i in range(j * 10 + 1, j * 10 + 9)]
'''
The above produces the following:
  [11, 12, 13, 14, 15, 16, 17, 18,
  21, 22, 23, 24, 25, 26, 27, 28,
  31, 32, 33, 34, 35, 36, 37, 38,
  41, 42, 43, 44, 45, 46, 47, 48,
  51, 52, 53, 54, 55, 56, 57, 58,
  61, 62, 63, 64, 65, 66, 67, 68,
  71, 72, 73, 74, 75, 76, 77, 78,
  81, 82, 83, 84, 85, 86, 87, 88]
'''

def generate_raster(spiketimes, well_id, duration, channel_ids):
  '''
  well_data: Data frame in the format seen above
  duration: Length of recording in seconds (in this case 300 seconds)
  channel_ids: Name of all channels in the format seen in the data frame "channel" column
  '''
  raster = []
  # Loop through each channel regardless of activity
  well_spiketimes = spiketimes[spiketimes["Well"] == well_id].reset_index(drop=True)
  for channel in channel_ids:
    # If the channel is found in my data (i.e., it had at least 1 spike)
    if str(channel) in well_spiketimes["Channel"].to_list():
      # Find spike times associated with that channel
      spiketrain = well_spiketimes.loc[well_spiketimes["Channel"]==str(channel), well_spiketimes.columns.str.contains("Time")].reset_index(drop=True)
      # Convert spike times that occur before the end of the recording (i.e., duration) to an array
      spiketrain = spiketrain.to_numpy().flatten()[spiketrain.to_numpy().flatten() <= duration]
      # Store array
      raster.append(spiketrain)
    # If the channel is not found in my data, I still want to store that it was empty
    else:
      raster.append([])
  return raster

raster = generate_raster(spiketimes, well_id, duration, channel_ids)
```

## Visualize network activity (as a raster plot)


```python
plt.figure(figsize=(10,5))
plt.rcParams.update({'font.size': 12})
plt.rc('axes', linewidth=1)
plt.eventplot(raster, color='black', linelengths=0.5, linewidths=0.75, alpha=0.35);
plt.ylabel("Channels")
plt.xlabel("Time (s)")
```




    Text(0.5, 0, 'Time (s)')




    
<img width="100%" alt="image" src="../Reverberating_Super_Burst_Detection_files/Reverberating_Super_Burst_Detection_22_1.png">
    


We did it with the histogram, might as well do it with the raster. 🤷


```python
plt.figure(figsize=(10,5))
plt.rcParams.update({'font.size': 12})
plt.rc('axes', linewidth=1)
plt.eventplot(raster, color='black', linelengths=0.5, linewidths=0.75, alpha=0.35);
plt.xlim(146, 157)
plt.ylabel("Channels")
plt.xlabel("Time (s)")
```




    Text(0.5, 0, 'Time (s)')




    
<img width="100%" alt="image" src="../Reverberating_Super_Burst_Detection_files/Reverberating_Super_Burst_Detection_24_1.png">
    




# Signal Conditioning

## Approach 1: Generate spike density function

### Gaussian Kernel
Note: You should be using a bandwidth optimized kernel. Refer to Shimazaki & Shinomoto, 2010)


```python
def generate_gaussian_kernel(sigma=0.1, fs=12500):
    '''
    :param sigma: Width of kernel
    :param fs: Sampling frequency in Hz
    :return: Gaussian kernel
    '''
    edges = np.arange(-3 * sigma, 3 * sigma, 1 / fs)
    kernel = norm.pdf(edges, 0, sigma)
    kernel = kernel / max(kernel)
    return kernel
```

### Binary spike matrix
Create a matrix of nChannels x Y time points, where Y is the duration * sampling frequency.
Each cell will contain either a 0 or 1 depending on whether a spike was found i.e., 1 or not i.e., 0.


```python
def generate_spikematrix(spiketrain, fs=12500, duration=300):
    '''
    :param spiketrain: Takes spikes times from a single channel
    :param fs: Sampling frequency in Hz
    :param duration: Duration of recording in seconds
    :return: spikematrix: Binary matrix containing spikes
    '''
    spiketimes = np.array(spiketrain)
    spiketimes = spiketimes[spiketimes <= duration]  # Ensure recording is desired length
    spikematrix = [0] * (duration * fs)  # Generate empty spike matrix with appropriate number of bins
    for spike in spiketimes:
        spikematrix[int(spike * fs)] = 1
    return spikematrix
```

### Convolve gaussian kernel and spike matrix



```python
def generate_sdf(spikematrix, gaussian_kernel):
    '''
    :param spikematrix: Binary matrix containing spikes
    :param gaussian_kernel: Gaussian kernel to be convolved with spikematrix
    :return: sdf: Continuous timeseries representing probability distribution of activity
    '''
    sdf_tmp = convolve(spikematrix, gaussian_kernel)
    sdf = sdf_tmp[int((len(sdf_tmp)-len(spikematrix))/2):int(len(sdf_tmp)-((len(sdf_tmp)-len(spikematrix))/2))]
    sdf = sdf/max(gaussian_kernel)
    return sdf
```

### Iterate through each channel, and generate the channel's SDF
Convolution takes some time because the spike matrix is quite large. There is a way to avoid spike density function generation by relying solely on histograms (like you saw above) but the timing of network event boundaries will be less precise.


```python
kernel = generate_gaussian_kernel(sigma=0.075)

maxFR, channel_sdf = [], []
for n, channel in enumerate(channel_ids):
  spiketrain = raster[n]
  channel_spikematrix = generate_spikematrix(spiketrain, fs=fs, duration=duration)
  sdf_tmp = generate_sdf(channel_spikematrix, kernel)
  channel_sdf.append(sdf_tmp)
  maxFR.append(max(sdf_tmp))
```


```python
network_sdf = np.mean(channel_sdf, axis=0)
weighted_network_sdf = np.average(channel_sdf, axis=0, weights=maxFR)
```


```python
plt.figure(figsize=(10,5))
t = np.arange(0,duration,1/12500)
plt.plot(t,network_sdf)
plt.plot(t,weighted_network_sdf)
plt.xlim(146, 157)
plt.ylabel("Channels/Firing Rate (Hz)")
plt.xlabel("Time (s)")
plt.legend(["Network SDF", "Weighted Network SDF"])
```




    <matplotlib.legend.Legend at 0x7ed80051fbe0>




    
<img width="100%" alt="image" src="../Reverberating_Super_Burst_Detection_files/Reverberating_Super_Burst_Detection_37_1.png">
    


As you can see, the general shape of the spike density function (which is an estimate of the network's instantaneous firing rate) is the same between the non-maxFR-weighted SDF and the maxFR-weighted SDF. You can use whichever one, but I prefer to use the maxFR-weighted SDF because it helps improve signal:noise for subsequent analysis.

## Visualize raster plot and spike density function


```python
plt.figure(figsize=(10,5))
plt.rcParams.update({'font.size': 12})
plt.rc('axes', linewidth=1)
plt.eventplot(raster, color='black', linelengths=0.5, linewidths=0.75, alpha=0.35);
t = np.arange(0,duration,1/fs)
plt.plot(t,weighted_network_sdf, color='black')
plt.xlim(146, 157)
plt.ylabel("Channels/Firing Rate (Hz)")
plt.xlabel("Time (s)")
```




    Text(0.5, 0, 'Time (s)')




    
<img width="100%" alt="image" src="../Reverberating_Super_Burst_Detection_files/Reverberating_Super_Burst_Detection_40_1.png">
    


As you can see, generally the spike density function does a really good job of representing the average activity across the network. You may notice that the "ramp-up" prior to each network burst starts before activity has truly started. This is largely an artifact of the gaussian kernel. To avoid this, you can use an alpha function or any function that you feel represents the probability of an action potential the best. For this example, and generally most purposes, a gaussian kernel will suffice.

# Reverberating Network Detection

## Did this network have a sufficiently high firing rate to warrant further analysis?


```python
isBursting = False
BURST_THRESHOLD = 5
if max(weighted_network_sdf) >= BURST_THRESHOLD:
  isBursting = True

print(f"Is Network Bursting? {isBursting}")
```

    Is Network Bursting? True


## Peak detection


```python
def detect_burst_peaks(sdf, delta=0.5, fs=12500):
    '''
    :param sdf: Spike density function
    :param fr: Minimum average firing rate (amplitude in SDF) required to be considered a burst
    :return: Burst peak times
    '''
    burst_peaks, _ = find_peaks(sdf, prominence=delta)
    return burst_peaks/fs
```


```python
burst_peak_times = []
prime_burst_peak_times = []
if isBursting:
  burst_peak_times = detect_burst_peaks(weighted_network_sdf)
  prime_burst_peak_times = detect_burst_peaks(weighted_network_sdf, delta=max(weighted_network_sdf)*0.5)
print(f"Detected {len(burst_peak_times)} potential network bursts and {len(prime_burst_peak_times)} likely initiation bursts.")
```

    Detected 145 potential network bursts and 30 likely initiation bursts.



```python
plt.figure(figsize=(10,5))
plt.rcParams.update({'font.size': 12})
plt.rc('axes', linewidth=1)
plt.eventplot(raster, color='black', linelengths=0.5, linewidths=0.75, alpha=0.35);
t = np.arange(0,duration,1/fs)
plt.plot(t,weighted_network_sdf, color='black')
plt.xlim(135,165)
plt.ylabel("Channels/Firing Rate (Hz)")
plt.xlabel("Time (s)")
sns.despine()

plt.plot(burst_peak_times, weighted_network_sdf[(burst_peak_times*fs).astype(int)], 'v')
```




    [<matplotlib.lines.Line2D at 0x7ed7ee9285b0>]




    
<img width="100%" alt="image" src="../Reverberating_Super_Burst_Detection_files/Reverberating_Super_Burst_Detection_48_1.png">
    


As you can see, our really basic peak detection was able to identify the location of all the network bursts in this recording. Depending on how noisy your activity is, you may need a more complicated approach.

## Cluster burst peaks along inter-burst-peak-interval and firing rate dimensions

### Visualize data


```python
ibpi = np.diff(burst_peak_times)
fr = weighted_network_sdf[(burst_peak_times*fs).astype(int)]
sns.scatterplot(x=ibpi,y=fr[1:])
sns.despine()
plt.xlabel("Inter-burst-peak-interval (s)")
plt.ylabel("Firing rate (Hz)")
```




    Text(0, 0.5, 'Firing rate (Hz)')




    
<img width="75%" alt="image" src="../Reverberating_Super_Burst_Detection_files/Reverberating_Super_Burst_Detection_52_1.png">
    


This approach ignores the first burst. We can deal with that later. But the important thing is that there appears to be two main clusters where expected.

### K-means clustering


```python
def k_means(ibpi, fr):
  features = np.transpose([ibpi, fr[1:]])
  #scaler = StandardScaler()
  #scaled_features = scaler.fit_transform(features)

  kmeans_kwargs = {
      "init": "random",
      "n_init": 10,
      "max_iter": 250,
      "random_state": 0
  }
  kmeans = KMeans(n_clusters=2, **kmeans_kwargs)
  kmeans.fit(features)

  x = np.transpose(features)[0]
  y = np.transpose(features)[1]

  # Assign the mini-burst cluster as label 1
  if np.nanmean(x[kmeans.labels_==0]) > np.nanmean(x[kmeans.labels_==1]):
    cluster_x_1 = x[kmeans.labels_==0]
    cluster_x_2 = x[kmeans.labels_==1]
    cluster_y_1 = y[kmeans.labels_==0]
    cluster_y_2 = y[kmeans.labels_==1]
  else:
    cluster_x_1 = x[kmeans.labels_==1]
    cluster_x_2 = x[kmeans.labels_==0]
    cluster_y_1 = y[kmeans.labels_==1]
    cluster_y_2 = y[kmeans.labels_==0]
  #sns.scatterplot(x=cluster_x_1, y=cluster_y_1, color='green')
  #sns.scatterplot(x=cluster_x_2, y=cluster_y_2, color='red')

  # Check what the percent of overlap is
  num_of_bins = int(len(fr)/2)
  # Overlapping x
  rng_x = min(cluster_x_1.min(), cluster_x_2.min()), max(cluster_x_1.max(), cluster_x_2.max())
  n1_x, bins1_x = np.histogram(cluster_x_1, bins=num_of_bins, range=rng_x)
  n2_x, bins2_x = np.histogram(cluster_x_2, bins=num_of_bins, range=rng_x)
  intersection_x = np.minimum(n1_x,n2_x)

  overlapping_bins_x = np.nonzero(intersection_x)
  if len(overlapping_bins_x) > 0:
      overlapping_bursts_x = np.sum(n1_x[overlapping_bins_x] + n2_x[overlapping_bins_x])
      percent_overlapping_x = overlapping_bursts_x/len(x)
  else:
      percent_overlapping_x = 0

  if (percent_overlapping_x > 0.2) | (sum(kmeans.labels_==1)<sum(kmeans.labels_==0)):
    print(f"The well is probably not reverberating strongly if at all")
    return False, x, y, kmeans.labels_
  else:
    return True, x, y, kmeans.labels_


isReverb, cluster_x, cluster_y, labels = k_means(ibpi, fr)
```


```python
sns.scatterplot(x=cluster_x, y=cluster_y, hue=labels)
plt.xlabel("Inter-burst-peak-interval (s)")
plt.ylabel("Firing rate (Hz)")
plt.title(f"Reverberation status: {isReverb}")
sns.despine()
```


    
<img width="75%" alt="image" src="../Reverberating_Super_Burst_Detection_files/Reverberating_Super_Burst_Detection_56_0.png">
    


### Inter-burst-peak-interval skewness


```python
def burst_skewness(ibpi):
  if skew(ibpi) > 0:
    return True
  else:
    return False

ibpi_skewness = burst_skewness(burst_peak_times)

sns.histplot(data=ibpi, bins=20)
plt.xlabel("Inter-burst-peak-interval (s)")
sns.despine()
```


    
<img width="75%" alt="image" src="../Reverberating_Super_Burst_Detection_files/Reverberating_Super_Burst_Detection_58_0.png">
    


## Rmax calculation (Crawling the histogram)

There are a number of different approaches to find the local minimum between a distribution. I initially attempted using Hartigan's Dip Test (via Unidip package) but found that depending on the frequency of network events, it would struggle to detect the local minimum. The approach that I found first that worked best was trying to replicate how my visual detection worked. In essence, I would plot histograms with varying bin sizes/ number of bins and see if there was a point where there was a 0 count bin between the two clear peaks. That approach worked most of the time. The issue with it is that it goes with the first bin that has a 0 count, which may result in certain RSBs being missed. This did a good enough job to be the main approach for Pradeepan et al., 2023.


```python
def hist_find_rmax(ibpi, labels, steps=2):
  initiation_burst = ibpi[labels==0]
  max_num_bins = int(len(ibpi/1.5))
  rmax = 0
  r_tmp = []
  s_tmp = []
  if (np.median(ibpi) < np.median(initiation_burst)):
    for size in (np.arange(max_num_bins, 5, -steps)):
      counts, binEdges = np.histogram(ibpi, bins=size)
      if len(np.where(counts==0)[0]) > 0:
        first_min = np.where(counts==0)[0][0]
        rmax_tmp = binEdges[first_min]
        r_tmp.append(rmax_tmp)
        s_tmp.append(size)
        if (rmax_tmp > rmax) & (rmax_tmp < (mode(list(map(int, initiation_burst)))[0]*0.9)):
          print(f"New Rmax found of {rmax_tmp} using {size} bin histogram")
          rmax = rmax_tmp
  return rmax

hist_rmax = hist_find_rmax(ibpi, labels, steps=2)
print(f"The calculated Rmax using histogram crawling was {hist_rmax} seconds.")
```

    New Rmax found of 0.5439549999999964 using 144 bin histogram
    New Rmax found of 0.5486146478873204 using 142 bin histogram
    New Rmax found of 0.553407428571425 using 140 bin histogram
    New Rmax found of 0.558339130434779 using 138 bin histogram
    New Rmax found of 0.5634158823529376 using 136 bin histogram
    New Rmax found of 0.5686441791044741 using 134 bin histogram
    New Rmax found of 0.5740309090909055 using 132 bin histogram
    New Rmax found of 0.5795833846153811 using 130 bin histogram
    New Rmax found of 0.5853093749999965 using 128 bin histogram
    New Rmax found of 0.5912171428571393 using 126 bin histogram
    New Rmax found of 0.5973154838709642 using 124 bin histogram
    New Rmax found of 0.6036137704917997 using 122 bin histogram
    New Rmax found of 0.6101219999999965 using 120 bin histogram
    New Rmax found of 0.6168508474576235 using 118 bin histogram
    New Rmax found of 1.3082979310344791 using 116 bin histogram
    New Rmax found of 1.3275115789473648 using 114 bin histogram
    New Rmax found of 1.347411428571425 using 112 bin histogram
    New Rmax found of 1.3680349090909054 using 110 bin histogram
    New Rmax found of 1.3894222222222186 using 108 bin histogram
    New Rmax found of 1.4041259999999962 using 80 bin histogram
    New Rmax found of 1.4346646153846117 using 78 bin histogram
    New Rmax found of 1.4668105263157858 using 76 bin histogram
    New Rmax found of 1.4937716129032221 using 62 bin histogram
    New Rmax found of 1.5364599999999964 using 48 bin histogram
    New Rmax found of 1.5939965217391268 using 46 bin histogram
    New Rmax found of 1.6143035294117611 using 34 bin histogram
    New Rmax found of 1.7018774999999964 using 32 bin histogram
    New Rmax found of 2.595131999999996 using 20 bin histogram
    New Rmax found of 2.8597999999999963 using 18 bin histogram
    New Rmax found of 3.1906349999999963 using 16 bin histogram
    The calculated Rmax using histogram crawling was 3.1906349999999963 seconds.


As we can see here, the detected Rmax or the maximum IBPI between mini-bursts was found to be 3.19 seconds. Referring back to the histogram and raster plots seen above, it appears this is a pretty good estimate about the maximum mini-burst IBPI.

## Rmax calculation (Simple overlap)


```python
def overlap_find_rmax(ibpi, labels):
  initiation_bursts = ibpi[labels==0]
  mini_bursts = ibpi[labels==1]

  overlap_start = max(min(initiation_bursts), min(mini_bursts))
  overlap_end = min(max(initiation_bursts), max(mini_bursts))

  return overlap_start

overlap_rmax = overlap_find_rmax(ibpi, labels)
print(f"The calculated Rmax using min overlap was {overlap_rmax} seconds.")
```

    The calculated Rmax using min overlap was 2.4002399999999966 seconds.


This is an alternative approach that is computationally cheaper and works as well.

You can use these functions to also calculate Amax, or the maximum amplitude to be considered a mini-bursts.

# Burst Detection Loop
This part of the algorithm calculates the ordinary burst boundaries (regardless of RSB status). These burst boundaries will be merged in the next step.

Recall, this is our spike density function. We are going to detect where the bursts start and end. Generally these line up when the first derivative peaks (i.e., begins to decline).


```python
plt.figure(figsize=(10,5))
plt.rcParams.update({'font.size': 12})
plt.rc('axes', linewidth=1)
t = np.arange(0,duration,1/fs)
plt.eventplot(raster, color='black', linelengths=0.5, linewidths=0.75, alpha=0.35);
plt.plot(t,weighted_network_sdf, color='black')
plt.xlim(146, 157)
plt.ylabel("Firing Rate (Hz)")
plt.xlabel("Time (s)")
```




    Text(0.5, 0, 'Time (s)')




    
<img width="100%" alt="image" src="../Reverberating_Super_Burst_Detection_files/Reverberating_Super_Burst_Detection_68_1.png">
    



```python
plt.plot(t[1:], np.diff(weighted_network_sdf), color='red')
plt.xlim(146, 149)
```




    (146.0, 149.0)




    
<img width="100%" alt="image" src="../Reverberating_Super_Burst_Detection_files/Reverberating_Super_Burst_Detection_69_1.png">
    


## Resample the signal

However, as you can tell, because of the sampling frequency of the original signal (12.5kHz), the signal is too noisy. This will make detecting any local maxima or minima difficult.

To address, we are going to downsample the original SDF. In hindsight, you could do this much earlier one when first generating your spike density function. It should be more appropriately placed in the "Signal Conditioning" section.


```python
RESAMPLE_FACTOR = 150
ds_weighted_network_sdf = resample(weighted_network_sdf, int(len(weighted_network_sdf)/RESAMPLE_FACTOR))
```

Sanity check, but a resample factor of 150 seems to be good for us. Preserves enough.


```python
# Scale the arrays between 0 and 1 so that we can plot the SDF and the first
# derivative together and see them both
min_max_scaler = MinMaxScaler()
scaled_ds_sdf = min_max_scaler.fit_transform(ds_weighted_network_sdf.reshape(-1,1))
t = np.arange(0,duration,RESAMPLE_FACTOR/fs)
first_deriv = np.diff(ds_weighted_network_sdf)

scaled_first_deriv = min_max_scaler.fit_transform(first_deriv.reshape(-1,1))

plt.figure(figsize=(10,5))
plt.plot(t, scaled_ds_sdf.reshape(1,-1)[0], color='black')
plt.plot(t[1:], scaled_first_deriv.reshape(1,-1)[0], color='red')
plt.xlim(146, 157)
```




    (146.0, 157.0)




    
<img width="100%" alt="image" src="../Reverberating_Super_Burst_Detection_files/Reverberating_Super_Burst_Detection_74_1.png">
    


Signal looks a lot smoother. Peaks of first derivative align with where burst starts and troughs align where burst generally end.

## Detect the beginning and end based on the peaks and troughs of the first derivative


```python
# Resample spike density function
ds_weighted_network_sdf = resample(weighted_network_sdf, int(len(weighted_network_sdf)/RESAMPLE_FACTOR))
# Generate a new time array
t = np.arange(0,duration,RESAMPLE_FACTOR/fs)

# Detect positive peaks (representing when the rate of increase is just about to go negative)
burst_start_ind,_ = find_peaks(np.diff(ds_weighted_network_sdf), prominence=0.1)
# Detect positive peaks of negative SDF (representing when the rate of increase is just about to go positive)
burst_end_ind,_ = find_peaks(-np.diff(ds_weighted_network_sdf), prominence=0.1)

# Assign detected peaks a time
burst_start = t[burst_start_ind]
burst_end = t[burst_end_ind]

# Check if the array length is the same. In some cases, it may not be and you
# can either drop the extra value where the start and end do not alternate or you
# can put a bit of work and find the value.
print(f"Length of burst_start is {len(burst_start)} and burst_end is {len(burst_end)}")
```

    Length of burst_start is 177 and burst_end is 177


### Visualize the above work


```python
plt.figure(figsize=(10,5))
first_deriv = np.diff(ds_weighted_network_sdf)
plt.plot(t[1:],first_deriv, color='black')
plt.plot(burst_start, first_deriv[burst_start_ind], 'v', color='green')
plt.plot(burst_end, first_deriv[burst_end_ind], '^', color='red')
plt.xlim(146, 157)
```




    (146.0, 157.0)




    
<img width="100%" alt="image" src="../Reverberating_Super_Burst_Detection_files/Reverberating_Super_Burst_Detection_79_1.png">
    


### Wrap the above into a function as we don't need to be doing this type of EDA (exploratory data analysis) every time.


```python
def detect_burst_boundaries(weighted_network_sdf, RESAMPLE_FACTOR=150):
  ds_weighted_network_sdf = resample(weighted_network_sdf, int(len(weighted_network_sdf)/RESAMPLE_FACTOR))
  t = np.arange(0,duration,RESAMPLE_FACTOR/fs)
  burst_start_ind,_ = find_peaks(np.diff(ds_weighted_network_sdf), prominence=0.1)
  burst_end_ind,_ = find_peaks(-np.diff(ds_weighted_network_sdf), prominence=0.1)
  burst_start = t[burst_start_ind]
  burst_end = t[burst_end_ind]
  print(f"Length of burst_start is {len(burst_start)} and burst_end is {len(burst_end)}")
  return burst_start, burst_end

burst_start, burst_end = detect_burst_boundaries(weighted_network_sdf)
```

    Length of burst_start is 177 and burst_end is 177


## Assign burst boundaries to the peaks that were detected (if possible)


```python
burst_borders = []
for burst_peak in burst_peak_times:
  closest_start_to_peak = burst_start[burst_start < burst_peak].max()
  closest_end_to_peak = burst_end[burst_end > burst_peak].min()
  burst_borders.append((closest_start_to_peak, closest_end_to_peak))
```


```python
plt.figure(figsize=(10,5))
plt.rcParams.update({'font.size': 12})
plt.rc('axes', linewidth=1)
t = np.arange(0,duration,1/fs)
plt.eventplot(raster, color='black', linelengths=0.5, linewidths=0.75, alpha=0.35);
plt.plot(t,weighted_network_sdf, color='black')
plt.xlim(146, 157)
for b in burst_borders:
  plt.axvline(b[0], color='green')
  plt.axvline(b[1], color='red')
plt.ylabel("Firing Rate (Hz)")
plt.xlabel("Time (s)")
```




    Text(0.5, 0, 'Time (s)')




    
<img width="100%" alt="image" src="../Reverberating_Super_Burst_Detection_files/Reverberating_Super_Burst_Detection_84_1.png">
    


As you can see, the initiation burst isn't getting called as nicely as we want to. An additional step you can do is to check what the firing rate is at the end of the burst and make sure it has a maximum value. For example: Check if the firing rate of the end of the burst is below 10 Hz, if not, make the end the closest value between the burst peak and the start of the next burst that is at 10 Hz.

# Reverberating Super Burst Reconstruction Loop
Now that we have our Rmax (and if necessary, Amax) as well as our preliminary burst boundaries, we can now reconstructing the network events into reverberating super bursts if they meet the appropriate criteria. Do this only if "isReverb" is True.


```python
burst_peaks = burst_peak_times[1:] # Because we ignored the first burst above

initiation_bursts = burst_peaks[labels==0]
mini_bursts = burst_peaks[labels==1]

t = np.arange(0,duration,1/12500)
plt.plot(t,weighted_network_sdf, color='black')
plt.plot(initiation_bursts, weighted_network_sdf[(initiation_bursts*fs).astype(int)], 'v', color='green')
plt.plot(mini_bursts, weighted_network_sdf[(mini_bursts*fs).astype(int)], 'v', color='red')
plt.xlim(146, 157)
plt.ylabel("Firing Rate (Hz)")
plt.xlabel("Time (s)")
```




    Text(0.5, 0, 'Time (s)')




    
<img width="100%" alt="image" src="../Reverberating_Super_Burst_Detection_files/Reverberating_Super_Burst_Detection_87_1.png">
    


## Merge the burst boundaries of all possible bursts into larger, complex network events if they meet the criteria


```python
def merge_bursts_into_RSB(rmax, burst_borders, prime_burst_peak_times, burst_peak_times):
  '''
  Rmax: either hist_rmax or overlap_rmax
  Burst borders: start and end tuple for each burst peak that we just recently calculated
  Prime_burst_peak_times: a variable we calculated close to the top where the only peaks
  would meet the criteria of a high prominence were detected
  Burst_peak_times: all possible burst peaks
  '''
  ne_start = []
  ne_end = []
  nReverbs = []
  r = 0
  in_super_burst = False
  initial_burst = False

  for i in range(0, len(burst_borders)-1):
    if (burst_peak_times[i] in prime_burst_peak_times) & (initial_burst == True):
      initial_burst = False
      ne_end.append(burst_borders[i-1][1])
      nReverbs.append(r)
      r = 0
    if (burst_peak_times[i] in prime_burst_peak_times) & (initial_burst == False):
      initial_burst = True
      ne_start.append(burst_borders[i][0])
    if ((burst_borders[i+1][0] - burst_borders[i][1]) <= rmax) & (initial_burst == True):
      if in_super_burst:
        r += 1
      in_super_burst = True
    elif ((burst_borders[i + 1][0] - burst_borders[i][1]) > rmax) & (initial_burst == True):
      if in_super_burst:
        r += 1
        in_super_burst = False
        initial_burst = False
        ne_end.append(burst_borders[i][1])
        nReverbs.append(r)
        r = 0
  if (burst_peak_times[len(burst_borders) - 1] in prime_burst_peak_times):
   ne_start.append(burst_borders[i][0])
   ne_end.append(burst_borders[i][1])
  else:
   ne_end.append(burst_borders[i][1])
   nReverbs.append(r)
  return ne_start, ne_end, nReverbs

ne_start, ne_end, nReverbs = merge_bursts_into_RSB(overlap_rmax, burst_borders, prime_burst_peak_times, burst_peak_times)
```

### Visualize if the merging worked appropriately


```python
plt.figure(figsize=(10,5))
plt.rcParams.update({'font.size': 12})
plt.rc('axes', linewidth=1)
t = np.arange(0,duration,1/fs)
plt.eventplot(raster, color='black', linelengths=0.5, linewidths=0.75, alpha=0.35);
plt.plot(t,weighted_network_sdf, color='black')
for b in ne_start:
  plt.axvline(b, color='green')
for b in ne_end:
  plt.axvline(b, color='red')
plt.ylabel("Firing Rate (Hz)")
plt.xlabel("Time (s)")
```




    Text(0.5, 0, 'Time (s)')




    
<img width="100%" alt="image" src="../Reverberating_Super_Burst_Detection_files/Reverberating_Super_Burst_Detection_91_1.png">
    



```python
plt.figure(figsize=(10,5))
plt.rcParams.update({'font.size': 12})
plt.rc('axes', linewidth=1)
t = np.arange(0,duration,1/fs)
plt.eventplot(raster, color='black', linelengths=0.5, linewidths=0.75, alpha=0.35);
plt.plot(t,weighted_network_sdf, color='black')
plt.xlim(136, 167)
for b in ne_start:
  plt.axvline(b, color='green')
for b in ne_end:
  plt.axvline(b, color='red')
plt.ylabel("Firing Rate (Hz)")
plt.xlabel("Time (s)")
```




    Text(0.5, 0, 'Time (s)')




    
<img width="100%" alt="image" src="../Reverberating_Super_Burst_Detection_files/Reverberating_Super_Burst_Detection_92_1.png">
    


## and there you have it! 🎉🎉🎉 Detection of reverberating super bursts. Now that you have all the important information (start and end of network events as well as the number of mini-bursts included in each), you can start generating features.

.

.

# Feature Calculation

### Number of mini-bursts per reverberating super burst


```python
sns.histplot(data=nReverbs, binrange=(0,10), bins=10)
plt.xlabel("Number of mini-bursts per RSB")
plt.axvline(np.mean(nReverbs), color='black')
plt.axvline(np.median(nReverbs), color='red')
plt.legend([f"Mean ({np.mean(nReverbs)})",f"Median ({np.median(nReverbs)})"])
sns.despine()
```


    
<img width="75%" alt="image" src="../Reverberating_Super_Burst_Detection_files/Reverberating_Super_Burst_Detection_98_0.png">
    


This network had a median number of 4 mini-bursts per reverberating super bursts. Quite the active one! In this case, not a single non-reverberating network event was seen.

### Network Event Duration


```python
ne_duration = [ne_end[i]-ne_start[i] for i in range(len(ne_start))]

sns.histplot(data=ne_duration, bins=10)
plt.xlabel("Network Event Duration (s)")
plt.axvline(np.mean(ne_duration), color='black')
plt.axvline(np.median(ne_duration), color='red')
plt.legend([f"Mean ({np.mean(ne_duration)})",f"Median ({np.median(ne_duration)})"])
sns.despine()
```


    
<img width="75%" alt="image" src="../Reverberating_Super_Burst_Detection_files/Reverberating_Super_Burst_Detection_101_0.png">
    


On average, this network had a network event duration of approximately 2 seconds. Based on this duration, and the previously calculated number of mini-bursts, what is the mini-burst frequency per reverberating super burst?

### Mini-burst Frequency


```python
mb_frequency = [nReverbs[b]/ne_duration[b] for b in range(len(ne_duration))]

sns.histplot(data=mb_frequency, bins=10)
plt.xlabel("Mini-burst frequency (Hz)")
plt.axvline(np.mean(mb_frequency), color='black')
plt.axvline(np.median(mb_frequency), color='red')
plt.legend([f"Mean ({np.mean(mb_frequency)})",f"Median ({np.median(mb_frequency)})"])
sns.despine()
```


    
<img width="75%" alt="image" src="../Reverberating_Super_Burst_Detection_files/Reverberating_Super_Burst_Detection_104_0.png">
    


The vast majority of reverberating super bursts had a mini-burst frequency between 1.5-2.1 Hz. Fairly consistent frequency.

### Inter-Network-Event-Interval


```python
i_ne_i = [ne_start[i+1]-ne_end[i] for i in range(0,len(ne_start)-1)]

sns.histplot(data=i_ne_i, bins=10)
plt.xlabel("Inter-Network-Event-Interval (s)")
sns.despine()
```


    
<img width="75%" alt="image" src="../Reverberating_Super_Burst_Detection_files/Reverberating_Super_Burst_Detection_107_0.png">
    

