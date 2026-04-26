
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

D. <img width="1196" height="328" alt="Screenshot 2026-04-22 at 9 05 41 PM" src="https://github.com/user-attachments/assets/b6f633f6-1f32-4810-82f5-4c969bcbd033" />  

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

# Question 5:
