# DSC-232 Group Project: Spotify Popularity Prediction

***

# Milestone 1: Project Setup and Data Understanding

## GitHub and Repository Setup

* Each team member created and configured a personal GitHub account
* A shared GitHub repository was created for collaboration and version control
* All group members were added to the repository to track progress and share work

***

## Dataset

* Dataset source:  
  <https://www.kaggle.com/datasets/lordpatil/spotify-metadata-by-annas-archive>

* The dataset contains Spotify metadata including tracks, artists, albums, and relationships between them

* The dataset size is approximately 190GB

***

## Expanse Environment Setup

* Expanse accounts were set up and synced across all group members
* A shared workspace was created to allow access to the dataset and notebooks
* Jupyter notebooks on Expanse were used to run Spark jobs

Data ingestion:

* Dataset was downloaded using Kaggle API
* Files were unzipped and converted into Parquet format for distributed processing

***

## Cluster Configuration

| Resource        | Value    |
| --------------- | -------- |
| Number of Cores | 8        |
| Total Memory    | 128 GB   |
| Dataset Size    | \~190 GB |

***

## Executor Configuration

| Parameter          | Value   |
| ------------------ | ------- |
| Executor Instances | 7       |
| Executor Memory    | \~18 GB |

Calculation:
Executor Instances = 8 - 1 = 7  
Executor Memory = (128 - 2) / 7 ≈ 18 GB

***

## Data Size and Structure

* Total observations: \~792,244,463 rows
* Data stored in `spotify_clean_parquet` format

***

## Dataset Structure

The dataset includes the following tables:

| Table              | Description                                        |
| ------------------ | -------------------------------------------------- |
| Artists            | Artist metadata including popularity and followers |
| Artist\_genre      | Mapping between artists and genres                 |
| Artist\_albums     | Relationship between artists and albums            |
| Available\_markets | Country availability                               |
| Albums             | Album metadata (label, release, popularity)        |
| Tracks             | Track-level data (duration, popularity)            |
| Track\_artist      | Mapping between tracks and artists                 |

***

## Missing Data Analysis

Missing values were found in:

| Table          | Missing Columns                     |
| -------------- | ----------------------------------- |
| Tracks         | preview\_url, external\_id\_isrc    |
| Artist\_albums | index\_in\_album                    |
| Albums         | external\_id\_upc, copyright fields |

Observations:

* Missing values are very small (<0.01%)
* Not critical for the prediction task
* Safe to proceed with the dataset

***

# Milestone 2: Data Visualization and Preprocessing

## Data Visualization

Visualizations were created using Spark aggregations and matplotlib/seaborn.

### Key Visualizations

Track Popularity Distribution

* Most tracks have very low popularity
* A small number of tracks have high popularity

Top 10 Artists by Popularity

* Shows the most popular artists in the dataset

Top Artists by Genre

* Demonstrates that top artists span multiple genres

Duration vs Popularity

* Most popular tracks fall between 2–5 minutes

Full code and visualizations are available in:
[Spotify_Popularity_Explorer.ipynb](./Spotify_Popularity_Explorer.ipynb) 

***

## Preprocessing

### Data Cleaning

* Joined tracks, artists, albums, and audio feature tables using Spark joins
* Sampled the dataset early to reduce shuffle size
* Aggregated data to one row per `track_id` to remove duplicates

***

### Handling Missing Values

* Rows missing critical fields (`track_id`, `popularity`) were removed
* Missing numerical values were filled with 0
* Missing audio features were interpreted as absence of data

***

### Feature Engineering

The following features were created:

* `log_followers`: reduces skew from large follower counts
* `energy_dance`: interaction between energy and danceability

***

### Feature Construction

* `VectorAssembler` used to combine all numeric features into a single vector
* `Normalizer` applied to scale feature vectors

***

### Spark Operations Used

* dropDuplicates()
* filter()
* fillna() / na.drop()
* groupBy().agg()
* withColumn()
* VectorAssembler
* Normalizer

***

# Milestone 3: Modeling and Evaluation

## Model Setup

A Random Forest Regressor was used to predict track popularity.

***

## Models Trained

| Model   | Trees | Depth | RMSE (Test) | R² (Test) |
| ------- | ----- | ----- | ----------- | --------- |
| Model 1 | 50    | 8     | 2.722       | 0.708     |
| Model 2 | 100   | 12    | 2.560       | 0.742     |

***

## Model Comparison

Model 1:

* Balanced performance across train, validation, and test sets
* Minimal overfitting

Model 2:

* Improved performance with lower RMSE and higher R²
* Slight overfitting, but still generalizes well


# Fitting Analysis
## Model Fitting:
For milestone 3, we tested 2 two random forest models with varying hyperparameters; the varying parameters for each model are numTrees and maxDepth. In the first iteration we set numTrees to 50 and maxDepth to 8 and in the second iteration numTrees to 100 and maxDepth to 12. Both the models showed balance on the fitting graph and only had a little overfitting but there was still a difference in results between the two models. The parameters directly affected the learning abilities of the random forest model.

See full code and visualizations in [Part3_Random_Forest.ipynb](./Part3_Random_Forest.ipynb) 

## Model 1 Overview:
For our first iteration, we had a relatively stable performance for the training, validation, and test dataset. When testing the training data, the predictions were typically off by 2.709 points from the actual value which is not a severe difference. When tested against the test dataset, the predictions were off by 2.722 points which was quite similar to the initial training dataset. This meant the model was overall good for generalization and was only slightly overfitting the results. The R2 value of 0.71 shows that the model was able to account for roughly 70% of the variance in Spotify track popularity. The model is able to capture relationships between artist popularity, album popularity, and follower count. Though 70% of variance in the model was captured, it’s important to note the 30% that wasn’t explained; the model struggled to predict popularity for extremely popular songs, it predicted their values inaccurately by predicting their value to be closer to average popularity range established by the previous values.  

## Model 2 Overview:
For the second iteration, the number of trees and depth increased to try and approve model error. There was an improvement in training data predictions in comparison to our first iteration as the RMSE was 2.504; showing that the model became slightly more accurate. Similarly, the test RMSE and R2 value showed general improvement, with the test error value being 2.560 and R2 of 0.74. The variation in the hyperparameters showed overall improvements in results for model 2. Model 2 has better predictive performance as it captures more nonlinear relationships of the Spotify dataset than model 1. By doubling the number of trees, the model was about to learn more about the nonlinear relationship between different features while the increase in the number of trees improved stability and reduced variance.

# Conclusion Section
## Model Conclusions:
Moving forward, since model 2 showed the most amount of accuracy and had the lowest amount of error, it would be the most ideal to use.  Having more trees and increasing the depth allows the model to learn more about the nonlinear relationships that are present in the data set.

## Future Improvements of the Model:
The model struggles with extremely popular songs so moving forward we would want to run a model that could handle improving the accuracy for these certain cases. One possible approach involves using the gradient boosting model since this type of model performs well on structured data. XGBoost models are more effective at understanding relationships that are nonlinear and could improve predictions for extremely popular songs. 

## Distributed Computation Benefits:
Distributed computing is necessary due to the Spotify dataset being extremely large; the amount of rows of data is well into the millions. With such a large dataset and a computer's limited memory, it would take significantly longer to perform any assessments on the data. By using Spark, we are able to distribute computation across multiple executors which helps reduce runtime during training. Distributed computing helps expand a system's capacity since more nodes are available for computation. If a node fails, other nodes are able to continue the operation since the workload is divided amongst many processors, allowing for parallel efficient processing. 

## Speedup Analysis
1. Baseline Measurement: 1 executor ~19 minutes and 45 seconds to run model training
2. Scaled Measurement: 8 executors  ~4 minutes and 30 seconds to run model training
3. Calculate Metrics: Speedup
4. Analyze:

| Executors | Time (sec) | Speedup | Efficiency |
|---|---|---|---|
| 1 | 1185 | 1x | 100% |
| 8 | 270 | 4.39x | 54.9% |
