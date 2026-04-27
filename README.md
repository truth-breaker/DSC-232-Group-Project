
## DSC-232-Group-Project

# Question 1:
A. Github ID set up for each team member  
B. Github repository set for the team  
C. https://www.kaggle.com/datasets/lordpatil/spotify-metadata-by-annas-archive

# Question 2:
A. Set up and synced our access and expanse accounts. Shared folder added that is accessible to every group member. Group members added each other's workspace so we can access each others progress. Within the hosted jupyter notebook, dataset was uploaded using Kaggle API and then unzipped; dataset was then explored to answer following questions.  

B.Number of Cores: 8
Total Memory: 128GB 
Our dataset is about 190GB in size so using the executor instances formula this justifies the amount of memory needed to run our notebooks.  

C. Total memory = 189.94 GB - Driver Memory/ Executor Instances 
8-1 = 7 Executor Instances
(128-2)/7=18  

D. 
<img width="512" height="303" alt="unnamed" src="https://github.com/user-attachments/assets/44f30d44-c16b-4bf7-b8d5-c7535d6b1e7c" />

# Question 3:
A. The total amount of observations is 792,244,463  

B. The data is broken up into 7 subsets tables: artists, artist_genre, artist_images, artist_albums, available markets, albums, album_images, tracks, and track_artist. Each table contains columns defining the metadata for each topic.   
Artists -  has columns related to artist name, Spotify ID, popularity, and follower count; this table contains both categorical and continuous variables.  
Artist_genre - columns linking artist name and Spotify ID to an associated genre; this table contains both categorical and continuous variables.  
Artist_albums - columns define whether an artist is the primary album artist or is a featured guest artist; this table contains only continuous variables.  
Available markets - columns include country codes where music content is available; this table contains only  categorical variables.  
Albums - contains metadata identifying an album name, type, release date, label, popularity, UPC, and available markets; this table contains both categorical and continuous variables.  
Tracks - columns show track name, Spotify ID, ISRC, duration, popularity, explicit tracker, track number, and the album it originates from; this table contains both categorical and continuous variables.  
Track_artist - columns associate an artist or artists to a track; this table contains only categorical data.  
Our target columns relate to data that captures an artists popularity,country codes and time metrics for eachv track/album.  

C. The only tables that are missing/null values are the tracks, artist albums, and albums; there are no duplicate values within the table. The following list goes over which columns per table is missing data:
Tracks - missing data in the preview_url  and external_id_isrc
Artist_albums - missing data in the index_in_album column
Albums - missing data in the external_id_upc, copyright_c, copyright_p, and external_id_amgid
The majority of columns with missing data details sourcing/ownership of certain tracks. The lack of this qualitative data isn’t concerning since it isn’t relevant to our research. The lack of external_id_isrc values can lead to issues when linking data from table to table but it seems to only apply to unique values. The amount of null values for this column is less than 0.01% of the data so this is not too concerning for our study.  

# Question 4:
## 4. Data Plots (4 points)

a. Created visualizations using Spark aggregations and matplotlib/seaborn.  
b. Plotted data with bar charts (top artists by popularity, colored by genre), histograms (track popularity distribution), and scatter plots (duration vs popularity).  
c. Each plot is explained below with insights:
   - **Track Popularity Distribution:** Shows most tracks have low popularity, with only a few hits.
   - **Top 10 Artists by Popularity:** Highlights the most popular artists in the dataset.
   - **Top 10 Artists by Popularity (Colored by Genre):**  
     This bar chart displays the top 10 artists by popularity, with each bar colored according to the artist’s primary genre. The legend on the right shows the genre categories.  
     **Insight:** This visualization highlights both the most popular artists and the diversity of genres represented among the top performers, showing that top artists span a range of genres, not just one musical style.
   - **Duration vs Popularity:** Demonstrates that most popular tracks are 2–5 minutes long.
d. For image data, plotted the distribution of album image sizes and included sample album covers.

**See full code and visualizations in [Spotify_Popularity_Explorer.ipynb](./Spotify_Popularity_Explorer.ipynb).**
# Question 5:

## Preprocessing Plan

### Handling Missing Values

### Handling Data Imbalance
The dataset may be skewed to more popular artists and tracks which wouldn’t represent the data accurately. To handle this, we will closely examine the popularity distribution and use techniques such as sampling or weights to ensure the model does not overfit to  only the most popular ones.

### Transformations (Scaling, Encoding, Feature Engineering)
- Scaling: for numerical variables such as artist followers, popularity, and track duration in order to keep a consistent range across all rows and prevent the extremely large values from taking over the model.
- Encoding: for categorical variables such as genre, album type, and available markets in order to convert them to a numerical form via processes such as one hot encoding so that we can perform analysis.
- Feature engineering: creating features such as year, number of artists on a track, and number of markets available from the variables provided will help us break down and better understand the data.

### Spark Operations for Preprocessing

- `dropDuplicates()` to remove duplicate rows.
- `filter()` to select or remove rows based on conditions.
- `fillna()` or `na.drop()` to handle missing values.
- `groupBy().agg()` for aggregation and combining duplicates.
- `withColumn()` to create new features or transform existing columns.
- Spark MLlib transformers (e.g., `StringIndexer`, `OneHotEncoder`, `MinMaxScaler`) for encoding and scaling.

