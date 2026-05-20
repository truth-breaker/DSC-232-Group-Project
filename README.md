
## DSC-232-Group-Project

# Question 1:
A. Github ID set up for each team member  
B. Github repository set for the team  
C. https://www.kaggle.com/datasets/lordpatil/spotify-metadata-by-annas-archive

# Question 2:
A. Set up and synced our access and expanse accounts. Shared folder added that is accessible to every group member. Group members added each other's workspace so we can access each others progress. Within the hosted jupyter notebook, dataset was uploaded using Kaggle API and then unzipped; dataset was then explored to answer following questions.  

B. Number of Cores: 8  
Total Memory: 128GB  
Our dataset is about 190GB in size so using the executor instances formula this justifies the amount of memory needed to run our notebooks.  

C. Executor Instances = 8-1=7  
Executor Memory = (128-2)/7 =18  

D. 
<img width="512" height="303" alt="unnamed" src="https://github.com/user-attachments/assets/44f30d44-c16b-4bf7-b8d5-c7535d6b1e7c" />

# Question 3:
A. The total amount of observations is 792,244,463. This data was aggregated into the spotify_clean_parquet folder.  

B. The data is broken up into 7 subsets tables: artists, artist_genre, artist_images, artist_albums, available markets, albums, album_images, tracks, and track_artist. Each table contains columns defining the metadata for each topic.   
Artists -  has columns related to artist name, Spotify ID, popularity, and follower count; this table contains both nominal and continuous variables.  
Artist_genre - columns linking artist name and Spotify ID to an associated genre; this table contains both nominal and continuous/discrete variables.  
Artist_albums - columns define whether an artist is the primary album artist or is a featured guest artist; this table contains only discrete variables.  
Available markets - columns include country codes where music content is available; this table contains only nominal variables.  
Albums - contains metadata identifying an album name, type, release date, label, popularity, UPC, and available markets; this table contains both nominal and discrete/continuous variables.  
Tracks - columns show track name, Spotify ID, ISRC, duration, popularity, explicit tracker, track number, and the album it originates from; this table contains both nominal and discrete/continuous variables.  
Track_artist - columns associate an artist or artists to a track; this table contains only nominal data.  
Our target columns are quantitative types of data, such as artists popularity, country codes and time metrics for each track/album. Additionally, we can potentially integrate text-based analysis using columns like artist name, track title, and album title.  

C. The only tables that are missing/null values are the tracks, artist albums, and albums; there are no duplicate values within the table. The following list goes over which columns per table is missing data:  
Tracks - missing data in the preview_url  and external_id_isrc  
Artist_albums - missing data in the index_in_album column  
Albums - missing data in the external_id_upc, copyright_c, copyright_p, and external_id_amgid  
The missing data details sourcing/ownership of certain tracks. The lack of this qualitative data isn’t concerning since it isn’t relevant to our research. The lack of external_id_isrc values can lead to issues when linking data from table to table but it seems to only apply to unique values. The amount of null values for this column is less than 0.01% of the total data; with such a tiny percentage of data missing we feel comfortable moving forward with this dataset.  

# Question 4:
## Data Plots

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
Missing data will be handled based on the importance of each feature for predicting track popularity. For key fields such as track name, album name, and artist, rows with missing values will be removed since they are essential for analysis. For less critical fields such as image url, missing values can be left as-is because the other available features still allow for analysis. For numerical features, missing values can be filled with a value that will not change the values that are present. For example, missing duration values can be replaced with the mean or median track duration in the album, and missing popularity values can be filled with the mean or median of the album. For categorical features, missing values such as genre can be labeled as "unknown".

### Handling Data Imbalance
The dataset may be skewed to more popular artists and tracks which wouldn’t represent the data accurately. To handle this, we will closely examine the popularity distribution and use techniques such as sampling or weights to ensure the model does not overfit to  only the most popular ones.

### Transformations (Scaling, Encoding, Feature Engineering)
- **Scaling:** for numerical variables such as artist followers, popularity, and track duration in order to keep a consistent range across all rows and prevent the extremely large values from taking over the model.
- **Encoding:** for categorical variables such as genre, album type, and available markets in order to convert them to a numerical form via processes such as one hot encoding so that we can perform analysis.
- **Feature engineering:** creating features such as year, number of artists on a track, and number of markets available from the variables provided will help us break down and better understand the data. This will allow us to view the patterns on a broader scale and understand what factors influence popularity.
- **Image handling:** standardizing the resolutions by choosing a fixed size available, and examining the popularity across that particular size only.


### Spark Operations for Preprocessing

- `dropDuplicates()` to remove duplicate rows.
- `filter()` to select or remove rows based on conditions.
- `fillna()` or `na.drop()` to handle missing values.
- `groupBy().agg()` for aggregation and combining duplicates.
- `withColumn()` to create new features or transform existing columns.
- Spark MLlib transformers (e.g., `StringIndexer`, `OneHotEncoder`, `MinMaxScaler`) for encoding and scaling.

# Project 3: Fitting Analysis
## Model Fitting:
For milestone 3, we tested 2 two random forest models with varying hyperparameters; the varying parameters for each model are numTrees and maxDepth. In the first iteration we set numTrees to 50 and maxDepth to 8 and in the second iteration numTrees to 100 and maxDepth to 12. Both the models showed balance on the fitting graph and only had a little overfitting but there was still a difference in results between the two models. The parameters directly affected the learning abilities of the random forest model.

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
