# Project summary

This project explores what makes songs successful on Spotify using global Top 50 chart data from more than 70 countries.

The work includes data preparation, feature analysis, text and sentiment analysis, regression modelling, and classification modelling. The original university version was later adapted into a cleaner GitHub portfolio project.

## Main question

What makes a song popular on Spotify?

## Dataset

The project uses a Kaggle dataset containing daily Spotify Top 50 chart entries across many countries. The original coursework used a snapshot downloaded in November 2024.

## Approach

The raw daily data is converted into a monthly song-level dataset. This reduces repeated daily chart rows and makes the data easier to use for modelling.

The project then compares several regression models to explain average monthly song popularity. It also trains a classification model to predict whether a song is likely to stay in the charts for more than two months.

## Original findings

The original analysis suggested that audio features alone cannot fully explain Spotify popularity. Random Forest gave the strongest regression performance in the original model comparison, but many external factors are missing from the dataset, such as artist popularity, genre, marketing, playlist placement and social media trends.

The original classification model reached around 90% accuracy, but the GitHub version uses a more careful song-level validation approach to reduce over-optimistic results.

## Personal contribution

I contributed across the project, especially to data preparation, feature analysis, modelling, interpretation of results and adapting the work into a public GitHub portfolio project. I had lighter involvement in the text analysis section compared with the modelling and data preparation work.


## Visual outputs

Selected plots from the original notebook are included in `images/key_charts/`. They include word clouds, sentiment plots, geographic audio-feature maps, monthly trend plots, model comparison charts and classification model outputs.
