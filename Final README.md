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

## Milestone 4: Dimensionality Reduction and XGBoost

## Written Report


### Introduction
Since the early 2000s, music streaming has become the primary method through which songs become popular, and specifically in recent years with recommendation algorithms advancing rapidly, artists rely on trends to ensure their music is pushed to the public as much as possible. On Spotify, a track’s popularity score on a scale of 0-100 represents its listening activity relative to all other tracks on the platform, and is crucial to influencing the algorithms that control recommendations, top charts, and artist revenue. This popularity score is relevant not only to the artist, but also their brand label, radio stations, and other artists using their competitors to study trends and experiment with what generates a “hit”. 
We chose this project because we were interested in exploring how popularity - typically associated with the subjective nature of music taste - can be explained by objective, measurable factors through big data analysis. We found a dataset of the complete Spotify archive with over 256 million tracks and their metadata such as artists, albums, duration, album art, explicit label, and more. Our goal was to build a model that can predict popularity by analyzing the patterns among existing popular songs.
A good predictive model matters because understanding which features impact popularity can reveal structural biases: for example, whether algorithmic popularity pushes existing fame and revenue (rich artists get richer) or whether up and coming artists are given a fair chance to be discovered. On a broader scale, this question applies to any industry where predicting consumer engagement through big data is valuable, from video streaming platforms like YouTube and TikTok deciding which content to recommend, to social media algorithms such as Instagram and LinkedIn determining which posts show up on users’ feed. The underlying question is the same in all of these cases: what objective factors predict whether something becomes popular, and who do they benefit? As recommendation algorithms grow more powerful and more embedded in day to day life, ensuring predictive models can be scrutinized for bias becomes increasingly important, not just for artists and businesses, but for the fairness and diversity of the users who these platforms are supposed to be for.

This problem requires big data and distributed computing because our dataset consists of 189 GB of the entire Spotify library data. The tracks table alone contains 256 million rows, and we have 8 tables total. Without distributed computing, even loading the full dataset of this size at once would be impossible on a personal machine due to memory limitations. We would have to take a very small sample of the data, which would not help us generalize the results obtained to properly answer the question we are examining. This was confirmed in Part 3 where we tested this using Amdahl’s Law and saw that running our model with 1 executor as opposed to 8 was 4x slower. Specific operations that required distributed computing include: 4-way joins across hundreds of millions of rows, window-based artist deduplication across 15 million rows, parallel Random Forest tree training, and distributed PCA covariance computation.

### Figures


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

### Discussion

### Conclusion

### Statement of Collaboration
