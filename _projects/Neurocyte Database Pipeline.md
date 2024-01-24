---
layout: page
title: Neurocyte 
description: A cost-effective serverless AWS ETL pipeline
img: 
category: data
related_publications: 
url: 
date: 2024-01-24
---

Over the next couple of months, I'll be working on a *greenfield* data project. This kind of project is built from scratch, from the ground up, without any constraints from prior work. Here, I will provide the motivation and a high-level overview of how this project will be accomplished and updates along the way.

In the field of biology, there is a growing trend of larger and larger datasets. Over the last decade, this has been dominated by large-scale, high-throughput single cell RNA-sequencing (sc-RNAseq), capable of producing terabytes of data with relative ease. In neuroscience, there is another type of data that is equally large, and in my opinion, more computationally demanding - that is electrophysiology. Electrophysiology is the study of the electrical signals and properties of biological cells. In neuroscience, we call it neuroelectrophysiology. Recently, there is has been this attraction to higher-throughput microelectrode arrays, capable of recording from thousands of electrodes at once. Even at a conservative 20kHz sampling frequency, a 5 minute recording, from 1000 channels will produce a data structure that is at least a 12+ GB (depending on the data type). Considering most researchers either record for much longer than 5 minutes or have multiple replicates, the total size of these files can quickly get out of hand.

The goal of this project will be develop a relatively simple front-end that can accept file(s) of a specific format, process it, and provide *incyte* to the user without needing to develop complex spatiotemporal analyses. The long term goal will be to produce a *drag-and-drop* user interface that allows researchers to more easily interact and develop their own pipelines. 

In this project page, I will not go into too much detail how I did everything and provide the exact code. However, I do want to use this space to demonstrate my data engineering chops.

# Picking a cloud service provider

At the time of writing this, Amazon Web Services (AWS) currently holds [majority market share at 32%](https://www.srgresearch.com/articles/q1-cloud-spending-grows-by-over-10-billion-from-2022-the-big-three-account-for-65-of-the-total). As such, I decided to develop a completely AWS-based pipeline.


According to [AWS Architecture Blog](https://aws.amazon.com/blogs/architecture/how-to-select-a-region-for-your-workload-based-on-sustainability-goals/), there are four considerations for evaluating and shortlisting AWS region for your services.
1. Latency
1. Cost
1. Services and features
1. Compliance

### Latency

[AWS Speed Test](https://awsspeedtest.com/latency) allows you to ping AWS datacenters around the world from your IP address. It's probably not the best way to go about doing this, but it tells me that ```us-east-1``` i.e., US East (N. Virginia) has the lowest latency from my location (Figure 1). However, it is important to recognize that the latency between your location and the datacenter is not the only one that matters. The latency between your users and the datacenter are just as important. For this product, my collaborators are scattered throughout North America. [CloudPing](https://www.cloudping.co/grid) can give you some useful insight into which inter-region communication/transactions have the least/most latency. Based on this, I think ```us-east-1``` is an adequate region for my use. As more users around the world use this, it may be important to go regions scattered around where they are.

<img width="100%" alt="image" src="../../assets/img/Neurocyte Data Pipeline/AWS-Region-Latency.png">

### Cost

Here it's important to design your architecture so that you can run an estimated cost calculator based on your projected needs. I did that very loosely and did not happen to save the document. Aligned with the lowest latency, ```us-east-1``` wins again here.

For now, you can rely on [ConcurrencyLabs](https://www.concurrencylabs.com/blog/choose-your-aws-region-wisely/) analysis on this.

### Services and features

Based on the previous two considerations, ```us-east-1``` also has all the services and features we need.

### Compliance

The data I will be using does not contain sensitive or private information, compliance is not currently a major consideration right now.

### Other: Sustainability

Something I believe more people should consider is the environmental impact of the data centers they use. Thankfully, AWS being a very mature company, they have already made quite significant strides in making their datacenters run on renewable energy. Based on [Amazon's Sustainability page](https://sustainability.aboutamazon.com/products-services/the-cloud?energyType=true), AWS is on track to be powered by 100% renewable energy by 2025, as well as net-zero carbon emissions by 2040. The page also provides a list of AWS regions that are already powered by 100% renewable energy. Once again, ```us-east-1``` can be found on the list! What's quite nice is that they also provide a map of all their renewable energy projects and the total energy production.


# The pre-processing pipeline

<img width="100%" alt="image" src="../../assets/img/Neurocyte Data Pipeline/Neurocyte-preprocess.png">

# To be continued.