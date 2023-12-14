---
title: "Detection of Reverberating Super Bursts in Neuronal Multielectrode Data"
excerpt: "Production implementation of burst reverberation toolbox. Packages a data pipeline (that applies unsupervised learning, regression techniques, and feature generators) into a desktop GUI for use by non-coding research scientists."
collection: projects
url: https://github.com/KartikP/Burst-Reverberation-Toolbox
date: 2023-07-21
---

[Project Link](https://github.com/KartikP/Burst-Reverberation-Toolbox)

The Burst Reverberation Toolbox (BRT) is a project that introduces a powerful electrophysiological tool designed to detect and analyze nested patterned bursts. These bursts, which arise from neuronal networks, represent a complex form of spontaneous activity characterized by a rapid succession of action potentials above a baseline firing rate. In the context of network development, synchronized bursts among neurons indicate the wiring and connectivity of the network.

In the field of stem cell derived disease models, accurate and consistent detection of bursts is of utmost importance. However, currently available "off-the-shelf" burst detection algorithms provided by in vitro multielectrode array (MEA) systems are overly simplistic and lack the necessary control to effectively detect diverse patterning, particularly nested (oscillatory) bursts. Consequently, this limitation can have significant implications on the analysis of phenotypic endpoints for rescue experiments.

The primary objective of the BRT project is to address these concerns by offering a flexible and transparent burst detection solution specifically tailored for temporally complex MEA data. The BRT will provide researchers with enhanced control over burst detection algorithms, allowing for improved accuracy and adaptability to diverse burst patterns. By overcoming the limitations of existing approaches, the BRT will empower researchers to conduct more comprehensive analyses, leading to a deeper understanding of neural network behavior in stem cell derived disease models.

GUI made with CustomTkinter/Tkinter and packaged with Pyinstaller.


<img width="1312" alt="image" src="https://github.com/KartikP/Burst-Reverberation-Toolbox/assets/2040394/a347bc99-d9be-411e-9f25-dbe08f8fb147">

>Simple and straight forward user interface

<img width="1312" alt="image" src="https://github.com/KartikP/Burst-Reverberation-Toolbox/assets/2040394/3e810021-dac3-436f-83c9-33caaf05ec6b">


>Batch processing of all wells within a plate


<img width="1312" alt="image" src="https://github.com/KartikP/Burst-Reverberation-Toolbox/assets/2040394/c3931d53-bf26-472d-8e8c-cee6b653bc1f">

>Interactive plots

<img width="562" alt="image" src="https://github.com/KartikP/Burst-Reverberation-Toolbox/assets/2040394/779433ed-174e-4815-a130-f6c7f6964dea">

>Settings to give users flexibility in burst detection


# Algorithm Workflow
![BRT_Block_Diagram](https://github.com/KartikP/Burst-Reverberation-Toolbox/assets/2040394/2497e0aa-1116-4275-868b-de457a65531c)
