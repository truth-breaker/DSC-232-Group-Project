# DSC 232R Group Project: Spotify Track Popularity Prediction

**Team:** Heta Joshi, Dennise Arenas, Zytal Lenus

**Dataset:** [Spotify Metadata by Anna's Archive (~190GB)](https://www.kaggle.com/datasets/lordpatil/spotify-metadata-by-annas-archive)

## Introduction

## Notebooks

## Cluster Configuration

| Resource | Value |
|---|---|
| Number of Cores | 8 |
| Total Memory | 128 GB |
| Dataset Size | ~190 GB |

| Parameter          | Value   |
| ------------------ | ------- |
| Executor Instances | 7       |
| Executor Memory    | \~18 GB |

Calculation:
Executor Instances = 8 - 1 = 7  
Executor Memory = (128 - 2) / 7 ≈ 18 GB

## Milestone 1: Project Setup and Data Understanding

We set up individual SDSC Expanse accounts and created a shared GitHub repository. The dataset was downloaded using the Kaggle API, unzipped, and converted to Parquet format for distributed processing. The dataset contains approximately 792 million rows across 8 tables:

| Table | Description |
|---|---|
| Tracks (256M rows) | Track name, duration, popularity, explicit flag |
| Artists (15M rows) | Artist name, followers, popularity |
| Albums (58M rows) | Album name, label, release date, popularity |
| Track_artist | Maps tracks to their artists |
| Artist_genre | Maps artists to genres |
| Artist_albums | Maps artists to albums |
| Available_markets | Country availability per track/album |
| Audio_features | Audio characteristics per track (tempo, energy, danceability, etc.) |

**Missing data:** Only three tables had any missing values: tracks (preview_url, external_id_isrc), artist_albums (index_in_album), and albums (external_id_upc, copyright fields). All missing values were under 0.01% and none were critical for prediction.

## Milestone 2: Data Exploration and Preprocessing Plan

All exploration was performed using Spark DataFrames using `.count()`, `df.describe().show()`, `df.printSchema()`, `groupBy().agg()`, and `select().distinct().count()`. Key findings:

- Track popularity is heavily right-skewed: most tracks cluster near 0, with only a small fraction reaching high popularity
- Most popular tracks fall between 2–5 minutes in duration
- Each artist has exactly 3 images stored (for different screen resolutions)
- Album and artist image dimensions vary widely, stored in transformed units rather than raw pixels

**Preprocessing plan:**
- Missing values under 10%: drop rows with missing critical fields, fill others with 0
- Scaling: StandardScaler/Normalizer for numerical features
- Encoding: StringIndexer + OneHotEncoder for categorical features
- Feature engineering: derived features from existing columns
- Spark operations to use: `dropDuplicates()`, `filter()`, `fillna()`, `groupBy().agg()`, `withColumn()`, `VectorAssembler`, `StringIndexer`, `Normalizer`

## Milestone 3: Preprocessing and Random Forest

Full preprocessing and Model 1 code: [Part3_Random_Forest.ipynb](https://github.com/truth-breaker/DSC-232-Group-Project/blob/Milestone-3/Part3_Random_Forest.ipynb)

### Preprocessing

To prevent shuffle disk spill on Expanse executor nodes, tracks were sampled at 0.5% before any join operations (~1.25M rows). The full preprocessing pipeline:

- Audio cleaning: renamed `duration_ms` to avoid column collision, filtered null API responses, cast all audio columns to double
- Artist deduplication: window function to keep the highest-follower row per artist name
- 4-way left join: tracks × audio × track_artists × artists × albums
- Track deduplication: `groupBy("track_id").agg()` to collapse multi-artist tracks to one row
- Missing values: dropped rows missing `track_id` or `popularity`; filled all other numeric nulls with 0
- Feature engineering: `log_followers` (log1p of follower count), `energy_dance` (energy × danceability)
- Assembly: `VectorAssembler` combining 21 features; `Normalizer` (L2) for scaling
- Split: 70% train / 15% validation / 15% test

Final preprocessed dataset: ~870,000 unique tracks.

### Model 1 Results

| Model | numTrees | maxDepth | RMSE Train | RMSE Val | RMSE Test | R² Test |
|---|---|---|---|---|---|---|
| 1a | 50 | 8 | 2.709 | 2.757 | 2.722 | 0.708 |
| 1b | 100 | 12 | 2.504 | 2.587 | 2.560 | 0.742 |

Model 1b is the stronger model — lower error and higher R² with only a small train/val gap indicating minimal overfitting. The increase in trees and depth allowed the model to capture more nonlinear relationships.

### Speedup Analysis

| Executors | Time (sec) | Speedup | Efficiency |
|---|---|---|---|
| 1 | 1185 | 1.00x | 100% |
| 8 | 270 | 4.39x | 54.9% |

Applying Amdahl's Law (`S(n) = 1 / ((1−p) + p/n)` with n=8, S=4.39): approximately 88% of the computation is parallelizable (p=0.8825). The remaining ~12% is sequential; driver coordination, result collection, and cache materialization, limiting the achieved speedup to 4.39x rather than the theoretical 8x maximum.

## Milestone 4: Dimensionality Reduction and XGBoost

## Written Report


### Introduction
Since the early 2000s, music streaming has become the primary method through which songs become popular, and specifically in recent years with recommendation algorithms advancing rapidly, artists rely on trends to ensure their music is pushed to the public as much as possible. On Spotify, a track’s popularity score on a scale of 0-100 represents its listening activity relative to all other tracks on the platform, and is crucial to influencing the algorithms that control recommendations, top charts, and artist revenue. This popularity score is relevant not only to the artist, but also their brand label, radio stations, and other artists using their competitors to study trends and experiment with what generates a “hit”. 

We chose this project because we were interested in exploring how popularity - typically associated with the subjective nature of music taste - can be explained by objective, measurable factors through big data analysis. We found a dataset of the complete Spotify archive with over 256 million tracks and their metadata such as artists, albums, duration, album art, explicit label, and more. Our goal was to build a model that can predict popularity by analyzing the patterns among existing popular songs.

A good predictive model matters because understanding which features impact popularity can reveal structural biases: for example, whether algorithmic popularity pushes existing fame and revenue (rich artists get richer) or whether up and coming artists are given a fair chance to be discovered. On a broader scale, this question applies to any industry where predicting consumer engagement through big data is valuable, from video streaming platforms like YouTube and TikTok deciding which content to recommend, to social media algorithms such as Instagram and LinkedIn determining which posts show up on users’ feed. The underlying question is the same in all of these cases: what objective factors predict whether something becomes popular, and who do they benefit? As recommendation algorithms grow more powerful and more embedded in day to day life, ensuring predictive models can be scrutinized for bias becomes increasingly important, not just for artists and businesses, but for the fairness and diversity of the users who these platforms are supposed to be for.

This problem requires big data and distributed computing because our dataset consists of 189 GB of the entire Spotify library data. The tracks table alone contains 256 million rows, and we have 8 tables total. Without distributed computing, even loading the full dataset of this size at once would be impossible on a personal machine due to memory limitations. We would have to take a very small sample of the data, which would not help us generalize the results obtained to properly answer the question we are examining. This was confirmed in Part 3 where we tested this using Amdahl’s Law and saw that running our model with 1 executor as opposed to 8 was 4x slower. Specific operations that required distributed computing include: 4-way joins across hundreds of millions of rows, window-based artist deduplication across 15 million rows, parallel Random Forest tree training, and distributed PCA covariance computation.

### Figures

**Figure 1: Track Popularity Distribution**
<img width="2100" height="750" alt="fig1_popularity_dist" src="https://github.com/user-attachments/assets/07f96a38-6c75-4454-927b-637ab54ee837" />

As this histogram shows, track popularity is heavily right-skewed; over 1.1 million tracks in the sample have popularity near 0, while very few exceed 40. This confirms that on Spotify, only a small fraction of the catalog drives the majority of listener engagement. Because a model that always predicts 0 would be accurate, RMSE and R² are more informative metrics than accuracy for this problem.


**Figure 2: Top 10 Artists by Max Track Popularity**
<img width="1500" height="900" alt="fig2_top_artists" src="https://github.com/user-attachments/assets/cb37f53f-ae14-47df-842e-a4e18001cb18" />

Bar graph showing the top 10 artists by maximum track popularity present in the 0.5% sample highlights artists including Chappell Roan, Morgan Wallen, Bruno Mars, and Avicii, which confirms the sample is representative of the dataset and captures globally reknown artists across multiple genres.

**Figure 3: Track Duration vs Popularity**
<img width="1500" height="750" alt="fig3_duration_vs_popularity" src="https://github.com/user-attachments/assets/70c70f7c-5417-46a0-b15c-847c008883bd" />
This scatter plot shows track duration vs popularity. The highest-popularity tracks cluster between 2–5 minutes, consistent with streaming platform preferences for shorter, more replayable content. Beyond this, popularity drops significantly for tracks longer than 6 minutes.

**Figure 4: Random Forest Feature Importances (Model 1)**
<img width="1500" height="1200" alt="fig4_feature_importance" src="https://github.com/user-attachments/assets/7f5f4250-bcc0-4ea6-8698-20fac3a63a79" />

Bar graph showing feature importances from Model 1a: `album_popularity` (62%) and `total_tracks` (21%) dominate predictions, while all 13 audio features contribute 0 importance. This suggests the model learned album-level patterns rather than track-level audio patterns.




### Methods

#### Data Exploration

We loaded 8 Parquet tables from `/expanse/lustre/projects/uci157/darenas/shared/spotify_clean_parquet` into Spark DataFrames. Row counts were computed using `.count()` and null counts using `count(when(col(c).isNull(), c))`. Visualizations were generated using Spark aggregations collected to the driver and rendered with matplotlib.

#### Preprocessing

```python
# Sample tracks at 0.5% BEFORE joins to prevent shuffle disk spill
tracks_clean = tracks_clean.sample(False, 0.005, seed=42)

# Audio cleaning
audio = audio.withColumnRenamed("duration_ms", "audio_duration_ms")
audio = audio.filter(F.col("null_response").isNull())
for c in audio_cols:
    audio = audio.withColumn(c, F.col(c).cast("double"))

# Artist deduplication using window function
w = Window.partitionBy("name").orderBy(F.desc("followers_total"), F.desc("artist_popularity"))
artists_clean = artists_clean.withColumn("rank", F.row_number().over(w))\
    .filter(F.col("rank") == 1).drop("rank")

# 4-way left join
main_df = (
    tracks_clean
    .join(audio, tracks_clean.track_id == audio.track_id, "left")
    .join(track_artists, tracks_clean.track_rowid == track_artists.track_rowid, "left")
    .join(artists_clean, track_artists.artist_rowid == artists_clean.artist_rowid, "left")
    .join(albums_clean, tracks_clean.album_rowid == albums_clean.album_rowid, "left")
)

# Deduplicate to one row per track
df_clean = main_df.groupBy("track_id").agg(
    F.first("track_name").alias("track_name"),
    F.max("followers_total").alias("followers_total"),
    ...
)

# Feature engineering
df_clean = df_clean.withColumn("log_followers", F.log1p(F.col("followers_total")))
df_clean = df_clean.withColumn("energy_dance", F.col("energy") * F.col("danceability"))

# Assembly and scaling
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw", handleInvalid="keep")
normalizer = Normalizer(inputCol="features_raw", outputCol="features", p=2)
```

Split: 70% train / 15% validation / 15% test (`seed=42`)

#### Model 1: Distributed Random Forest Regressor

```python
# Model 1a: baseline
rf = RandomForestRegressor(labelCol="popularity", featuresCol="features",
                           numTrees=50, maxDepth=8, featureSubsetStrategy="auto", seed=42)

# Model 1b: increased capacity
rf2 = RandomForestRegressor(labelCol="popularity", featuresCol="features",
                            numTrees=100, maxDepth=12, featureSubsetStrategy="auto", seed=42)
```

Both models evaluated using `RegressionEvaluator` with RMSE and R² on train, validation, and test sets. Feature importances extracted from `rf_model.featureImportances`.

#### Model 2: PCA + XGBoost

**Model 2a: PCA + XGBoost (Ray distributed):**
```python
# StandardScaler before PCA
scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled",
                        withMean=False, withStd=True)

# PCA dimensionality reduction
pca = PCA(k=21, inputCol="features_scaled", outputCol="pca_features")
pca_model = pca.fit(df_scaled)
# Explained variance showed k=1 exceeds 90% threshold, selected k=1

# Ray-distributed XGBoost on PCA features
from xgboost.spark import SparkXGBRegressor
```

**Model 2b: XGBoost without PCA (improved):**
```python
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

pdf = df_no_pca.select("track_id", "track_name", "artist_name",
                        "popularity", "features_raw").limit(200000).toPandas()
X = np.vstack(pdf["features_raw"].apply(lambda v: v.toArray()))
y = pdf["popularity"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
```

### Results

#### Data Exploration
- Tracks: 256,039,007 rows, Artists: 15,430,442 rows, Albums: 58,590,982 rows
- Missing values <0.01% in all critical columns
- Popularity heavily right-skewed: majority of tracks have popularity near 0
- Most popular tracks between 2–5 minutes duration

#### Preprocessing
After joining and deduplication on the 0.5% sample, final dataset: ~870,000 unique tracks. Left joins preserved all sampled tracks regardless of missing audio or artist data.

#### Model 1 Results

| Model | numTrees | maxDepth | RMSE Train | RMSE Val | RMSE Test | R² Test |
|---|---|---|---|---|---|---|
| 1a | 50 | 8 | 2.709 | 2.757 | 2.722 | 0.708 |
| 1b | 100 | 12 | 2.504 | 2.587 | 2.560 | 0.742 |

**Feature Importances (Model 1a):**

| Feature | Importance |
|---|---|
| album_popularity | 62.0% |
| total_tracks | 21.4% |
| track_duration | 6.2% |
| followers_total | 4.8% |
| artist_popularity | 2.9% |
| log_followers | 2.3% |
| explicit | 0.4% |
| All 13 audio features | 0.0% |

**Sample Predictions (Model 1b, test set):**

| Track | Artist | Actual | Predicted |
|---|---|---|---|
| Grenade | Bruno Mars | 82 | 17.3 |
| Running Wild | Jin | 78 | 22.5 |
| MIENTRAS ME CURO DEL CORA | KAROL G | 78 | 16.9 |
| River | Leon Bridges | 77 | 24.0 |
| Radio/Video | System Of A Down | 71 | 23.0 |


#### Model 2 Results

**PCA Explained Variance:**

| Component | Explained Variance |
|---|---|
| PC1 | ~97% |
| PC2 | ~2% |
| PC3+ | ~0% each |

**Model 2a: PCA + XGBoost:**

| Split | RMSE | R² |
|---|---|---|
| Validation | 4.86 | — |
| Test | 4.87 | 0.07 |

Confusion matrix (threshold ≥ 70 = popular): 191,735 true negatives, 8 false negatives, 0 true positives, 0 false positives. The model predicted every track as "not popular."

**Model 2b: XGBoost without PCA:**

| Split | RMSE | R² |
|---|---|---|
| Train | 1.27 | 0.936 |
| Test | 2.16 | 0.810 |

**Sample Predictions (Model 2b, test set):**

| Track | Artist | Actual | Predicted |
|---|---|---|---|
| MIENTRAS ME CURO DEL CORA | KAROL G | 78 | 67.2 |
| Sol solecito caliéntame un poquito | NULL | 66 | 66.2 |
| Fall Fast in Love | Rod Wave | 64 | 51.5 |
| Dj Waley Babu (feat. Aastha Gill) | Badshah | 63 | 41.0 |
| Alone With You | Arz | 63 | 65.9 |


**All Models Comparison:**

| Model | Features | RMSE Test | R² Test |
|---|---|---|---|
| RF 1a (50 trees, depth 8) | 21 full features | 2.722 | 0.708 |
| RF 1b (100 trees, depth 12) | 21 full features | 2.560 | 0.742 |
| PCA + XGBoost (k=1) | 1 component | 4.870 | 0.070 |
| XGBoost (no PCA) | 21 full features | 2.160 | 0.810 |

#### Speedup Analysis

| Executors | Time (sec) | Speedup | Efficiency |
|---|---|---|---|
| 1 | 1185 | 1.00x | 100% |
| 8 | 270 | 4.39x | 54.9% |


### Discussion

#### Data Exploration
The strong right skew we saw in the popularity field makes sense and seems consistent with what we know about streaming platforms and engagement. Only a small number of tracks actually become a global big hit while the majority of tracks have comparatively low engagement. This skew tells us that a model that frequently predicts a low popularity would actually have a high accuracy rate, making RMSE and R² more informative metrics for evaluation than accuracy alone. This shaped our decision to frame the problem as regression rather than binary classification.

#### Preprocessing
Due to the Expanse executor disk filling up when we tried to perform join operations on the full 256M rows dataset, we had to train on a 0.5% sample instead. This left us with 1.25M rows which is still a significant size to use for the model and is expected to be statistically diverse enough (although ideally the full dataset would be used). The decision to sample before joins rather than after was critical because sampling after joins still requires shuffling 256M rows, which causes a disk spill once again.

#### Model 1

#### Model 2


### Conclusion

#### Model 1

#### Model 2

#### What We Learned About Big Data Processing
We learned that while distributed computing makes big data processing tasks much more manageable, they come with their own limitations as well. During part 3, we kept running into a disk spill error and learned that we had to be very intentional about all of our code and even the order in which we execute it. The order of shuffle operations, sampling, and joins made a crucial difference to whether our code was finally able to run at all, and Amdahl's Law confirmed that even with 8 executors we achieved only 4.39x speedup rather than the theoretical 8x, because ~12% of computation is inherently sequential.

#### How Distributed Computing Changed Our Approach
If we were attempting this project on a single machine we would have been forced to use a much smaller sample and lose the opportunity to gain insights from the 256M tracks, 15M artists, and 58M albums. Distributed computing changed our approach by giving us a variety of options and try different joins, deduplication, feature engineering, scaling, and model training. This was possible because Spark allowed the processes to run across multiple executors.

#### What We Would Explore With More Time


### Statement of Collaboration
