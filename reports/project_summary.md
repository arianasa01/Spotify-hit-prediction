# Spotify Hit Prediction — Project Summary

## Aim

The aim of this project was to explore what makes a song popular on Spotify and whether audio features can help predict chart performance.

The project used a Kaggle dataset of Spotify Top 50 songs across more than 70 countries. The dataset includes daily chart rankings, popularity scores, country information and audio features such as danceability, energy, loudness, acousticness, liveness, valence and tempo.

## Main questions

The project focused on four questions:

1. What words and sentiments appear most often in popular song titles?
2. How do Spotify audio features vary across countries and seasons?
3. Can audio features explain average song popularity?
4. Can audio features predict whether a song stays in the charts for more than two months?

## Data preparation

The raw dataset contains daily chart records. To reduce repeated daily entries for the same song, the data was converted into a monthly song-level format.

For each song and month, the workflow calculates:

- average popularity
- maximum popularity
- minimum popularity
- average chart rank
- best chart rank
- worst chart rank

The final processed dataset is created automatically from the raw Kaggle file.

## Modelling

Two modelling tasks were used.

The first task was regression. Several models were compared to predict average monthly popularity from audio and technical features.

The second task was classification. A Gradient Boosting model was used to predict whether a song was a short-term hit or a long-term hit.

In the public GitHub version, the classification workflow uses a song-level split. This is more careful because it avoids putting the same song into both the training and test sets.

## Main findings

The original project found that Random Forest performed best among the regression models tested. However, audio features alone could only explain part of the variation in song popularity. This suggests that external factors, such as artist popularity, genre, marketing and social media trends, are likely very important.

The original classification model performed strongly when predicting long-term chart presence, but the GitHub version improves the validation approach so the result is less likely to be over-optimistic.

The exploratory analysis also showed seasonal patterns. For example, energetic and positive songs appeared more strongly in summer months. Song title analysis showed repeated themes such as Christmas, love and romance across English-speaking countries.

## Limitations

The project has some important limitations. The dataset only includes songs that already appeared in Spotify Top 50 charts, so it does not represent all released songs. The dataset also does not include artist popularity, playlist placement, marketing activity, social media trends or genre labels.

Because of these missing features, the model should be understood as an exploratory project rather than a complete commercial prediction system.

## Skills shown

This project demonstrates practical data science skills including data cleaning, feature engineering, exploratory analysis, regression, classification, model evaluation and clear project documentation.
