---
title: Deep Learning for the Diagnosis and Classification of Rett Syndrome
description: Classification of stem cell derived Rett syndrome neuronal networks.
date: 2022-09-26
layout: page
img: 
category: neuroscience
related_publications: 
url: https://github.com/KartikP/BIOPHYS9709B-FCNN/blob/main/Pradeepan_Kartik_Final_Code.ipynb
---
[Project link](https://github.com/KartikP/BIOPHYS9709B-FCNN/blob/main/Pradeepan_Kartik_Final_Code.ipynb)

<h3>INTRODUCTION</h3>
Rett Syndrome (RTT) is a rare neurodevelopmental disorder that is caused by a single heterozygous loss-of-function mutation in the gene methyl-CpG-binding protein 2 (MECP2). Recent human cell work has shown that iPSC-derived cortical neurons harboring the MECP2 mutation had smaller cell bodies, shorter dendrites with less branching, as well as a decrease in frequency of spontaneous excitatory post-synaptic currents, suggesting these neurons are not receiving as many inputs. Taken together, neurons that had the MECP2 mutations had morphological and functional hypoconnectivity at a single neuron level. Based on this, I hypothesized that this apparent hypoconnectivity between MECP2-mutant neurons will result in altered network development and function, and these changes will be detectable at early stages of network development. Additionally, based on these functional network-level electrophysiological properties, I am interested in whether these excitatory neuronal cultures can be classified as either wildtype (WT) or Rett syndrome (RTT), and if it is possible identify the developmental stage.

<h3>METHODS</h3>
<strong>Format and source:</strong> Excitatory neuronal cultures were grown on Axion Biosystem’s 12-well Maestro multielectrode array (MEA) system. Within each plate of 12 isolated wells, a grid of 64 electrodes (8 by 8) measured the extracellular activity. Every week for 6 weeks of development, the neuronal cultures were recorded for 5-minutes. The format of the dataset was initially stored as .raw files that contained the 5-minute raw voltage signals from each channel within the well. Additionally, my dataset contained 10 total plates as biological replicates. In total, the initial dataset contained the raw voltage signal from 46, 080 channels (10 plates x 12 wells x 6 weekly recordings x 64 channels). The dataset was produced by my collaborators in the Dr. James Ellis Molecular Genetics Lab at Sick Kids Hospital (in affiliation with University of Toronto). While the .raw voltage signals can be useful, their massive size (~300 Gb) makes any analysis very slow. Using a spike detection algorithm, I detected multi-unit action potentials (channel x spike times) and stored the data in a tabular format (.csv file). This decreased the dataset size from 300 Gb to only 1.5 Gb. While the waveforms were lost in the process, with respect to the research question, they were not as crucial as the spike times. Because I was interested in the network properties of my neuronal cultures, I calculated 53 feature statistics (e.g. Number of active electrodes, Mean ISI within bursts, ISI Coefficient of variation, etc) that reflected the aggregate population-level activity within each isolated well. This further reduced my dataset size from 1.5 Gb to a mere 419 Kb.


<strong>Data Curation and Representation:</strong> No data curation from multiple sources was performed as the dataset was sufficiently large to use in the neural network. Furthermore, this type of data for Rett Syndrome is rare as only recently have researchers begun using multielectrode array (MEA) technology to investigate neurodevelopmental disorders. Due to this, a major drawback of this dataset is the lack of representation. The cell cultures within the dataset were produced only from two individuals, which provided the stem cell lines to generate 60 replicates for both RTT and wildtype (WT). As more data is collected, I hope to generate cell lines from numerous individuals, each with varying background genomes, to make the model more generalizable.

<strong>Data Augmentation:</strong> No data augmentation was performed because the dataset is quite sensitive to variability, because the electrophysiological differences between WT and RTT are already similar across most of the features, except for only a handful (notably network burst frequency, burst duration, and number of active electrodes).

<strong>Allocation:</strong> The dataset was divided into 25% testing, 7.5% validation, 67.5% training sets. These ratios were chosen based on trial-and-error on what produced the greatest accuracy across multiple runs of the data.

<strong>Activation functions, loss functions, gradient descent, and hyper-parameters:</strong> For most of the neural network, the <strong>rectified linear activation function (ReLU)</strong> was used. This activation function was appropriate because my inputs were standardized to a mean of 0 and STD of 1. Surprisingly, normalizing the inputs to [0 1] resulted in dramatically slower learning. For my output layer, a <strong>softmax activation function</strong> was used because of the multi-class nature of my classification problem. The loss function I used was <strong>categorical cross-entrop</strong>. This loss was chosen because the network was trying to solve a multi-class problem. This was chosen over a sparse categorical cross-entropy loss function because I did not have many classes to choose from (i.e. not in the thousands), which sparse categorical cross-entropy excels in. I used <strong>mini batch gradient descent</strong> because it is a good compromise between batch and stochastic gradient descent. My dataset is not large enough to warrant stochastic gradient descent and when I used batch gradient descent, training took more than 2000 epochs to get similar performance to mini batch gradient descent with only 250 epochs. The learning rate and batch size were chosen based on trial-and-error, producing networks that had good enough accuracy before over-fitting, and performed relatively quickly.

<strong>Regularization:</strong> Additionally, I employed both <strong>drop-out</strong> and <strong>early stopping regularization</strong> because occasionally the neural network tended to overfit the data, and plateau at approximately 70% accuracy. Employing drop-out after each layer was able to improve accuracy by 20%, capturing more randomness. However, despite this improvement in accuracy, after ~250 epochs, most runs of the model would continue to over-fit the training data, represented by an increase in the validation loss function. To deal with this, I used early stopping with a patience hyper-parameter of 40 which describes the number of epochs without improvement in the validation loss before the model is stopped. This not only solved the over-fitting problem, but also shortened the run time of the model.

<img width="100%" alt="image" src="https://user-images.githubusercontent.com/2040394/192342359-aaf91951-9406-45ab-83d2-bcd44dfe7bf2.png">

<h3>RESULTS</h3>
According to the calculated F1 score, the network performs quite well in classifying RTT vs WT as well as developmental stage (Table 1). I opted to use the F1 error metric because it considers the precision and recall for each class. Based on the confusion matrix, for both WT and RTT, the model sometimes confused the middle and late stages with each other, however had good performance distinguishing between WT and RTT as well as the early stages of development (Figure 2).

<img width="100%" alt="image" src="https://user-images.githubusercontent.com/2040394/192342216-7216e36b-58b8-4965-932b-e49101ac19e6.png">

<img width="100%" alt="image" src="https://user-images.githubusercontent.com/2040394/192342266-a53b418b-bc03-4cf0-9a25-490cff83b5b1.png">


<h3>DISCUSSION</h3>
The study objective was to classify electrophysiological features derived from developing neuronal cultures into wildtype or Rett Syndrome and then classify into developmental stages. The fully- connected neural network performed incredibly well based on the limited data I have. One of the reasons why I chose to use three developmental stages (i.e. early, middle, and late) is because the initial culturing of the neurons into the MEA wells are based on morphological maturity and the first time point in one plate may not necessarily equate to the first time point on another plate. By aggregating two weeks of data into one developmental stage, we’re able to take this into consideration.

One of the biggest strengths of this model is that because of the nature of the data and the neural network, training and then testing the model does not take very long. I was able to modulate a variety of hyperparameters, without having to subsample the population to expedite the process. However, while the model is highly accurate and quick, one of the biggest limitations is the dependency on feature statistic generation to create the dataset. This step, although not talked about earlier in the report, is in fact incredibly time consuming. To generate the feature statistics for the 10 plates in the current dataset took over 2 weeks to run. While this step only needs to be done once, it is a bottleneck. To possibly address this limitation of a pre-processing bottleneck, a Long Short Term Memory (LSTM) Recurrent Neural Network may be used on the raw voltage time series data from each of the electrodes contained in a well or even an aggregate/ weighted average signal. A recent paper by Aghasfari et al developed a LSTM architecture to classify and translate observations from cardiac activity of in vitro iPSC- cardiomyocytes to predict corresponding adult ventricular myocyte cardiac activity.

While RTT is diagnosable by identification of mutations in the gene, MECP2, Autism Spectrum Disorders (ASD), a related disorder with many overlapping behavioural symptoms, is not as easily diagnosable. This RTT neural network is serving as a preliminary study/ proof-of-concept showing that it is feasible to potentially diagnose SHANK2- related ASD by using electrophysiological features of neurons derived from easily accessibly adult cells (through induced pluripotent stem cells) of a patient. The significance of this study is that it is the first to classify iPSC-derived neuronal cultures based on extracellular electrophysiological properties measured via high-throughput multielectrode arrays.