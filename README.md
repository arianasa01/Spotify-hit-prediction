# Spotify Hit Prediction

Predicting short-term and long-term Spotify chart success using global Top 50 song data across more than 70 countries.

This project started as a group data science project for my MSc. I have adapted it into a public GitHub portfolio project with a cleaner structure, clearer setup steps, and a more reproducible modelling workflow.

## Project overview

The main question behind this project is:

**What makes a song popular on Spotify?**

To explore this, the project uses Spotify Top 50 chart data and audio features such as danceability, energy, loudness, acousticness, liveness, valence and tempo.

The project covers four main areas:

1. Song title and sentiment analysis
2. Regional and seasonal patterns in Spotify audio features
3. Regression modelling to explain average song popularity
4. Classification modelling to predict whether a song is likely to stay in the charts for more than two months

## Dataset

The dataset is not included in this repository because it is too large.

Please download it from Kaggle:

https://www.kaggle.com/datasets/asaniczka/top-spotify-songs-in-73-countries-daily-updated

After downloading the CSV file, place it here:

```text
data/raw/universal_top_spotify_songs.csv
```

The original coursework used a snapshot downloaded in November 2024. Because this Kaggle dataset is updated over time, results may be slightly different if a newer version is downloaded.

## Project structure

```text
spotify-hit-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── universal_top_spotify_songs.csv      # not included
│   ├── processed/
│   │   └── spotify_monthly_2024.csv             # created after running the code
│   └── README.md
│
├── src/
│   ├── data_preparation.py
│   ├── eda.py
│   ├── modelling.py
│   ├── text_analysis.py
│   └── main.py
│
├── reports/
│   └── project_summary.md
│
├── notebooks/
│   └── README.md
│
├── outputs/
│   └── generated model results and charts
│
└── images/
    └── optional images for README/report
```

## How to run the project

### 1. Clone the repository

```bash
git clone https://github.com/arianasa01/Spotify-hit-prediction.git
cd Spotify-hit-prediction
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On Mac or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the requirements

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the dataset from Kaggle and place the CSV file in:

```text
data/raw/universal_top_spotify_songs.csv
```

### 5. Run the full workflow

```bash
python src/main.py
```

This will:

- check that the raw dataset exists
- create a cleaned monthly modelling dataset
- run regression models for average popularity
- run a song-level classification model for long-term chart presence
- save results in the `outputs/` folder

## Methods

### Data preparation

The raw dataset contains daily chart entries. Since the same song can appear in the chart for many days, the project aggregates the data by `spotify_id` and month. This creates one row per song per month, rather than one row per daily chart entry.

The processed dataset includes:

- average monthly popularity
- maximum and minimum monthly popularity
- average monthly rank
- best and worst monthly rank
- audio features
- song duration in minutes
- explicit content flag
- key and mode

### Regression modelling

The regression task predicts average monthly popularity using audio and technical song features.

Models compared include:

- Linear Regression
- Decision Tree Regressor
- K Nearest Neighbours Regressor
- Support Vector Regressor
- Stochastic Gradient Descent Regressor
- Gradient Boosting Regressor
- Random Forest Regressor
- Dummy baseline models

The project reports RMSE, MAE and R² so the model is compared against simple baselines.

### Classification modelling

The classification task predicts whether a song is a:

- **short-term hit**: appears in the chart for two months or less
- **long-term hit**: appears in the chart for more than two months

The improved GitHub version uses a song-level dataset for this task. This avoids placing the same song in both training and test data, which would make the model look better than it really is.

The model uses Gradient Boosting with selected interaction features such as:

- energy × loudness
- danceability × energy
- acousticness × instrumentalness

## Main findings from the original project

The original analysis found that:

- Spotify song popularity is difficult to explain using audio features alone.
- Random Forest gave the strongest regression performance in the original model comparison.
- Audio features explained part of the variation in popularity, but external factors such as artist popularity, genre, marketing and social media trends are likely very important.
- The original classification model reached around 90% accuracy for predicting whether a song stayed in the charts for more than two months, although the GitHub version uses a more careful validation approach.
- Seasonal patterns appeared in the data, with more energetic and positive songs becoming more common in summer months.
- Song title analysis showed recurring terms linked to themes such as Christmas, love and romance across English-speaking countries.

## My contribution

I contributed across the project, especially on data preparation, feature analysis, modelling, interpretation of results and preparing the final project structure. I had lighter involvement in the text analysis part compared with the modelling and data preparation work.

## Limitations

This project only uses songs that already entered Spotify Top 50 charts. This means the model is not trained on all released songs, so it should not be treated as a general model for predicting every new song.

The dataset also does not include some important external factors, such as:

- artist popularity
- playlist placement
- marketing spend
- TikTok or social media trends
- genre labels
- listener demographics

These factors are likely to play a major role in real-world song popularity.

## Future improvements

Useful next steps would include:

- adding artist-level features
- adding genre information
- comparing country-specific models
- testing models on a later time period
- using time-based validation
- adding a simple dashboard for exploring chart trends

## Skills demonstrated

- Python
- pandas
- NumPy
- scikit-learn
- data cleaning
- feature engineering
- exploratory analysis
- regression modelling
- classification modelling
- model evaluation
- data visualisation
- GitHub project organisation
