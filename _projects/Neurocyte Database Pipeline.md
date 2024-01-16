---
layout: page
title: NDD-ePhys-dB development pipeline
description: Not available - How to create your own Allen Institute-style open access database
img: 
category: data
related_publications: 
url: 
date: 2025-01-13
---

Over the next couple of months, I'll be working on a *greenfield* data project. This kind of project is built from scratch, from the ground up, without any constraints from prior work. Here, I will provide the motivation and a high-level overview of how this project will be accomplished.

In neuroscience, there's a growing trend of producing large datasets. Up to now, the focus has primarily been on transcriptomics, which has flourished thanks to high-throughput transcriptomic assays. Transcriptomics, a study of RNA transcripts, reveals gene expression profiles in neurons, offering insights into their potential functions and states. However, it represents just one aspect of neurobiology.

Another crucial, yet complex, type of data comes from electrophysiology — the study of electrical properties in cells. While it often centers on neurons, other cell types in the brain also exhibit interesting electrophysiological characteristics worth exploring. Electrophysiology is vital for understanding how neurons function: it looks at how they communicate, process information, and contribute to complex behaviors and brain activities. Unlike transcriptomics, which provides a static, broader view of what neurons might be capable of, electrophysiology offers a dynamic, real-time look at how neurons actually behave and interact.

The two primary tools used in electrophysiological research are patch-clamp and microelectrode arrays, each complementing the other in their application. Patch-clamp involves attaching a single electrode directly inside a cell and then applying current to see how the neuron reacts to these external stimuli. This technique has various forms and can be manipulated in different ways to explore distinct mechanisms and states of neurons.

In contrast, microelectrode arrays focus on the population-level activity of neuronal networks. This approach uses up to several thousand electrodes to record the activity of a group of neurons simultaneously, helping researchers understand how neurons interact within a network. Unlike the patch-clamp method, which ultimately kill the cell, microelectrode arrays are non-destructive. This non-destructive nature allows for the long-term observation of the same group of neurons, which is particularly valuable for studying their development over time. Due to this advantage, microelectrode arrays have become increasingly favored by scientists researching neurodevelopmental disorders.

The following project aims to facilitate data sharing and collaboration among various university/research labs investigating the electrophysiological properties of various neurodevelopmental disorders. The project was inspired by the Allen Institute for Brain Science Atlas, which provides the world's most comprehensive collection of electrophysiology-transcriptomic-morphology recordings/reconstructions of rodent, non-human primate, and human neurons.

# The data

Patch-clamp single neuron recordings are fairly straightforward. They are either in .abf or .nwb file formats and are generally only a few megabytes per experiment (containing various stimulation protocols, and the resulting output from the neuron). My current average file size here is approximately 1.3 megabytes. If I have 1000 neurons, the total file size will be 1300 megabytes. Not much at all.

Microelectrode array recordings are considerably more complex. These recordings vary based on the number of recording electrodes and the sampling frequency. For example, Axion Biosystem's Maestro MEA contains 12 wells, each containing 64 channels. It records at a sampling frequency of 12.5 kHz, meaning every second, there are 12500 data points per channel. Every second, across the entire plate (which contains 12 wells x 64 channels), there are 9,600,000 data points. Each data point, if I recall correctly, is 2-bytes. Therefore, each second, this produces 19,200,000 bytes of data or 18.3105 MB. A 5 minute recording (300 seconds), will produce a file size of 5.49315 GB. If you record multiple time points over 6 weeks (lets say twice a week), the total dataset size for one plate is approximately 65.9 GB. If you do this 10 times (as biological replicates), the full dataset will be 659 GB. This is considerably larger than the patch-clamp data.

# The pipeline
I want to build a pipeline that will accept both of these file types in their raw format (extract), process the data (transform), and then store the harmonized data into various tables (load). I decided to go completely with Amazon Web Services (AWS) for this.

<img width="100%" alt="image" src="../../assets/img/Neurocyte%20Data%20Pipeline/Neurocyte%20Data%20Pipeline.png">


### Data Ingestion and Initial Storage

1. **Data Sources to AWS S3 via AWS Direct Connect and Internet**:
   - Internal data sources are transferred to AWS S3 using AWS Direct Connect. This dedicated network connection ensures a more reliable and consistent network experience compared to standard internet-based connections.
   - External data sources upload data directly onto S3 via the internet. This setup provides a straightforward and accessible method for external data ingestion.

### Data Processing

2. **Data Processing with AWS Glue and Spark**:
   - Once in S3, the data triggers AWS Glue jobs. AWS Glue, a serverless data integration service, is configured to run Spark operations on this data.
   - Spark operations in Glue are used to transform and process the raw data. This step will clean, aggregate, and process the data.

### Processed Data Storage

3. **Storing Processed Data on S3**:
   - After processing, the transformed data is stored back in AWS S3. This separation of raw and processed data helps maintain an organized and efficient data storage strategy.

### Data Cataloging and Querying

4. **Cataloging Data with Glue Crawlers**:
   - AWS Glue Crawlers are then used to scan the processed data stored in S3. These crawlers automatically catalog the data, making it searchable and queryable. They classify the data and extract schema and metadata, which are then stored in the AWS Glue Data Catalog.

5. **Querying Data with Athena**:
   - The cataloged data is queried using AWS Athena, an interactive query service. Athena allows SQL queries directly on data stored in S3, providing a flexible and powerful way to analyze large datasets without the need for traditional database servers. Compared to RedShift, Athena charges per query versus the total runtime. For my purpose, this is currently advisable however may change on usage.

### API Integration and Data Retrieval

6. **API Gateway and Lambda for Data Retrieval**:
   - An AWS API Gateway is set up to manage and route requests. This acts as a front door to your data processing backend.
   - AWS Lambda functions are triggered by this API Gateway to handle specific tasks such as querying Athena for data visualization or directly retrieving files from S3 for user download.
   - This setup ensures that data retrieval and processing are efficiently managed, scalable, and secure.

### Frontend Visualization and Download (not shown)

7. **Frontend Data Visualization and Download**:
   - On the frontend, user requests to view or download data are handled through interactions with the website.
   - For data visualization, Plotly, a powerful graphing library, is used. It renders interactive visualizations in the user's browser based on the JSON-generated data retrieved through the API Gateway and Lambda functions.
   - For data downloads, the Lambda function fetches the requested files directly from S3 and facilitates the download process for the user.


# Estimated Cost
Now it's great to design a full-fledged pipeline and while it's most likely functional and get the job done, it's really important to consider the the costs of all the products that I will be using. To do this, I will provide a hypothetical but realistic scenario.

## Description of Files

Microelectrode array: 20 plates, with 12 time points. Axion Biosystems RAW file format. 5.5GB per file. Total size is 1320GB.
- CSV pre-processed: 20 plates, with 12 time points. Comma separated value file format. 300MB per 12 time points. Total size is 6GB.

Patch clamp: 100 recordings. ABF file format. 1.3MB per file. Total size is  1.3GB.

Total size of all raw files: 1327.3GB rounded up to 1328GB

## Storage Pricing
Amazon S3 Pricing: https://aws.amazon.com/s3/pricing/

### Transfer In
Transferring in is $0.00 per GB

### Storage
S3 Intelligent - Tiering - General purpose storage for any type of data, typically used for frequently accessed data.

Frequent Access Tier, First 50TB/Month: $0.023 per GB
Total for **just** storage: $30.544

### Requests
As shown in the pipeline, once the initial data is stored on S3, AWS Glue via PySpark will retrieve the data to perform a variety of transformations and processing.

Theoretically, each file should be only requested once by AWS Glue.

The total number of GET requests will be approximately 580 (for the total number of files). GET/SELECT requests are $0.0004 per 1000 requests. Therefore, this one time cost during initialization will be $0.0004.

**TO BE COMPLETED**