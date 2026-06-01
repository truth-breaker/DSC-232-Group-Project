# DSC 232R Group Project: Spotify Track Popularity Prediction

**Team:** Heta Joshi, Dennise Arenas, Zytal Lenus

**Dataset:** [Spotify Metadata by Anna's Archive (~190GB)](https://www.kaggle.com/datasets/lordpatil/spotify-metadata-by-annas-archive)

## Introduction

## Notebooks

## Cluster Configuration

## Milestone 1: Project Setup and Data Understanding

## Milestone 2: Data Exploration and Preprocessing Plan

## Milestone 3: Preprocessing and Random Forest


## Written Report


### Introduction
Since the early 2000s, music streaming has become the primary method through which songs become popular, and specifically in recent years with recommendation algorithms advancing rapidly, artists rely on trends to ensure their music is pushed to the public as much as possible. On Spotify, a track’s popularity score on a scale of 0-100 represents its listening activity relative to all other tracks on the platform, and is crucial to influencing the algorithms that control recommendations, top charts, and artist revenue. This popularity score is relevant not only to the artist, but also their brand label, radio stations, and other artists using their competitors to study trends and experiment with what generates a “hit”. 
We chose this project because we were interested in exploring how popularity - typically associated with the subjective nature of music taste - can be explained by objective, measurable factors through big data analysis. We found a dataset of the complete Spotify archive with over 256 million tracks and their metadata such as artists, albums, duration, album art, explicit label, and more. Our goal was to build a model that can predict popularity by analyzing the patterns among existing popular songs.
A good predictive model matters because understanding which features impact popularity can reveal structural biases: for example, whether algorithmic popularity pushes existing fame and revenue (rich artists get richer) or whether up and coming artists are given a fair chance to be discovered. On a broader scale, this question applies to any industry where predicting consumer engagement through big data is valuable, from video streaming platforms like YouTube and TikTok deciding which content to recommend, to social media algorithms such as Instagram and LinkedIn determining which posts show up on users’ feed. The underlying question is the same in all of these cases: what objective factors predict whether something becomes popular, and who do they benefit? As recommendation algorithms grow more powerful and more embedded in day to day life, ensuring predictive models can be scrutinized for bias becomes increasingly important, not just for artists and businesses, but for the fairness and diversity of the users who these platforms are supposed to be for.

This problem requires big data and distributed computing because our dataset consists of 189 GB of the entire Spotify library data. The tracks table alone contains 256 million rows, and we have 8 tables total. Without distributed computing, even loading the full dataset of this size at once would be impossible on a personal machine due to memory limitations. We would have to take a very small sample of the data, which would not help us generalize the results obtained to properly answer the question we are examining. This was confirmed in Part 3 where we tested this using Amdahl’s Law and saw that running our model with 1 executor as opposed to 8 was 4x slower. Specific operations that required distributed computing include: 4-way joins across hundreds of millions of rows, window-based artist deduplication across 15 million rows, parallel Random Forest tree training, and distributed PCA covariance computation.

### Figures

