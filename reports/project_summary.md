# Project summary

## Aim

The aim of this project was to explore what makes songs popular on Spotify using global Top 50 chart data and audio features. The project looked at title patterns, seasonal and regional audio trends, regression modelling for popularity, and classification modelling for long-term chart presence.

## Dataset

The project uses the Kaggle dataset **Top Spotify Songs in 73 Countries**. The dataset contains daily Spotify chart entries and audio features for songs appearing in Top 50 playlists across many countries.

The dataset is not included in the repository because it is large and updated over time. Users should download it directly from Kaggle and place it in the `data/raw/` folder.

## Main methods

The raw daily data was cleaned and aggregated into a monthly song-level dataset. This reduced repeated daily chart entries and created clearer modelling features such as average monthly popularity, best monthly rank, worst monthly rank and chart days.

Regression models were used to predict average monthly popularity. A classification model was used to predict whether a song stayed in the charts for more than two months.

The GitHub version improves the original coursework by using a song-level split where needed. This helps avoid testing on the same song that the model has already seen during training.

## Main findings from the original coursework

The original analysis found that Random Forest performed best among the tested regression models. However, audio features alone could only explain part of song popularity, suggesting that external factors such as artist popularity, genre, marketing and social media trends are important.

The original classification model achieved around 90% accuracy for predicting long-term chart presence. The public GitHub version keeps this task but uses a more careful validation approach.

Seasonal patterns were also found, with more energetic and positive-sounding songs becoming more common during summer months.

## Limitations

The project only uses songs that already reached Spotify Top 50 charts. This means the results should not be treated as a general prediction model for all new songs.

The dataset also lacks important external variables such as artist popularity, playlist placement, genre, marketing budget and social media activity.

## Personal contribution

I worked across the project, especially on data preparation, feature analysis, modelling, interpretation and preparing the final project structure. I had lighter involvement in the text analysis section compared with the modelling and data preparation parts.
