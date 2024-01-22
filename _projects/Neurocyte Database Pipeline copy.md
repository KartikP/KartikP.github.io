---
layout: page
title: Frontend -> Amazon S3
description: How to create a file(s) uploader
img: 
category: data
related_publications: 
url: 
date: 2025-01-13
---

# Prerequisite
You will need to install [Node.js and NPM (Node Package Manager)](https://nodejs.org/en/download/) and an AWS account.

# 1. Create an S3 Bucket

Amazon Simple Storage Service (S3) is a fantastic objective storage service that is highly scalability, availability, with low latency. For whatever your purpose is (outside of this project), you can probably find it within S3.

S3 stores data as objects within a resource called a "bucket". These are the fundamental storage containers that act as the *top-level* folder for your storage needs. 

# 2. Create a Presigned URL

The presigned URL will allow you to initiate direct uploads to S3 without exposing your AWS credentials. 