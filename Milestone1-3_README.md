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
Spotify\_Popularity\_Explorer.ipynb

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

Best-performing model: Model 2

***

## Feature Importance

Top contributing features:

* album\_popularity (\~62%)
* total\_tracks (\~21%)
* track\_duration (\~6%)
* followers\_total and artist\_popularity (\~8% combined)

Most audio features contributed very little to the model.

### Interpretation

Track popularity appears to be primarily influenced by metadata such as album popularity and artist-related features, rather than intrinsic audio characteristics like tempo or danceability.

***

## Model Behavior

The models show good generalization:

* Model 1: well-balanced, no overfitting
* Model 2: slightly overfit but performs better overall

***

## Conclusion

The Random Forest model successfully learned patterns in the dataset, achieving strong predictive performance. Increasing model complexity improved performance, though it introduced slight overfitting.

Future improvements may include:

* Using Gradient Boosted Trees
* Additional feature engineering
* Further hyperparameter tuning

***

## Distributed Computing Impact

Apache Spark enabled:

* Processing of large-scale data (\~190GB)
* Efficient joins across multiple tables
* Parallel feature engineering
* Scalable model training

Without distributed computing, this task would not be computationally feasible.





