# DALR: Denoising Alignment with Large Language Model for Recommendation

This is the PyTorch implementation for DALR proposed **Denoising Alignment with Large Language Model for Recommendation** (under review).

## introduction

In this paper, we propose a Denoising Alignment framework with LLMs for GNN-based Recommenders (DALR), which aims to align structural representation with textual representation and mitigate the effects of noise. Specifically, We propose a modeling framework that integrates the representation of graph structure with textual information from LLMs to capture intricate user-item interactions. We also suggest an alignment paradigm to enhance representation performance by aligning semantic signals from LLMs and structural features from GNN models. Additionally, we introduce a contrastive learning scheme to relieve the impact of noise and improve model performance.

## Requirement

+ python=3.9
+ torch==1.13.1

## Usage

The hyper-parameter search range and optimal settings have been clearly stated in the codes.

`python encoder/train_encoder.py`

## Datasets

We provide two processed datasets: Amazon-Book and Steam.


Our code was implemented based on RLMRec
