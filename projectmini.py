
# ==========================================================
# PROJECT: NETFLIX MOVIES AND TV SHOWS ANALYSIS
# Libraries: Pandas, NumPy, Matplotlib, Seaborn
# ==========================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. PROJECT SETTINGS
sns.set_theme(style="whitegrid", palette="Set2")
OUTPUT_FOLDER = "project_outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("=" * 60)
print("       NETFLIX MOVIES AND TV SHOWS ANALYSIS")
print("=" * 60)

# 2. LOAD DATASET
df = pd.read_csv("netflix_titles.csv")

print("\nFIRST 5 ROWS:")
print(df.head().to_string())

print("\nORIGINAL DATASET SHAPE:", df.shape)
print("\nCOLUMN INFORMATION:")
df.info()

# 3. DATA QUALITY CHECK
print("\nMISSING VALUES BEFORE CLEANING:")
print(df.isnull().sum())

print("\nDUPLICATE ROWS:", df.duplicated().sum())

# 4. DATA CLEANING
df = df.drop_duplicates().copy()

# Remove extra spaces from text columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].apply(
        lambda value: value.strip()
        if isinstance(value, str) else value
    )

# Convert date_added to date format
df["date_added"] = pd.to_datetime(
    df["date_added"], errors="coerce"
)

# Convert numeric columns safely
df["release_year"] = pd.to_numeric(
    df["release_year"], errors="coerce"
)

# Fill missing text fields with Unknown
text_columns = [
    "director", "cast", "country",
    "rating", "duration", "listed_in", "description"
]

for col in text_columns:
    df[col] = df[col].fillna("Unknown")

print("\nCLEANED DATASET SHAPE:", df.shape)
print("\nMISSING VALUES AFTER CLEANING:")
print(df.isnull().sum())

# 5. BASIC STATISTICS USING NUMPY
years = df["release_year"].dropna().to_numpy()

print("\nNUMERICAL ANALYSIS")
print("-" * 40)
print("Mean release year:", round(np.mean(years), 2))
print("Median release year:", np.median(years))
print("Standard deviation:", round(np.std(years), 2))
print("Earliest release year:", int(np.min(years)))
print("Latest release year:", int(np.max(years)))

# 6. MOVIES VS TV SHOWS
print("\nCONTENT TYPE DISTRIBUTION:")
content_counts = df["type"].value_counts()
print(content_counts)

plt.figure(figsize=(8, 5))
ax = sns.barplot(
    x=content_counts.index,
    y=content_counts.values
)
ax.bar_label(ax.containers[0], fmt="%.0f", padding=3)
plt.title("Netflix: Movies vs TV Shows")
plt.xlabel("Content Type")
plt.ylabel("Number of Titles")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/01_content_type.png", dpi=300)
plt.show()

# 7. RELEASE YEAR TRENDS
year_counts = df["release_year"].value_counts().sort_index()

plt.figure(figsize=(12, 5))
plt.plot(year_counts.index, year_counts.values)
plt.title("Netflix Titles by Release Year")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/02_release_trends.png", dpi=300)
plt.show()

# 8. TOP 10 GENRES
genres = (
    df.loc[df["listed_in"] != "Unknown", "listed_in"]
    .str.split(", ")
    .explode()
)

top_genres = genres.value_counts().head(10)
print("\nTOP 10 GENRES:")
print(top_genres)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_genres.values, y=top_genres.index)
plt.title("Top 10 Netflix Genres")
plt.xlabel("Number of Titles")
plt.ylabel("Genre")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/03_top_genres.png", dpi=300)
plt.show()

# 9. CONTENT RATINGS
rating_counts = (
    df.loc[df["rating"] != "Unknown", "rating"]
    .value_counts().head(10)
)

print("\nTOP CONTENT RATINGS:")
print(rating_counts)

plt.figure(figsize=(10, 5))
sns.barplot(x=rating_counts.index, y=rating_counts.values)
plt.title("Netflix Content Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/04_ratings.png", dpi=300)
plt.show()

# 10. TOP COUNTRIES
countries = (
    df.loc[df["country"] != "Unknown", "country"]
    .str.split(", ")
    .explode()
)

top_countries = countries.value_counts().head(10)
print("\nTOP 10 COUNTRIES:")
print(top_countries)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title("Top 10 Countries by Netflix Titles")
plt.xlabel("Number of Titles")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/05_countries.png", dpi=300)
plt.show()

# 11. MOVIE DURATION ANALYSIS
movies = df[df["type"] == "Movie"].copy()

movies["duration_minutes"] = pd.to_numeric(
    movies["duration"].str.extract(r"(\d+)")[0],
    errors="coerce"
)

valid_durations = movies["duration_minutes"].dropna()

print("\nMOVIE DURATION STATISTICS (MINUTES):")
print(valid_durations.describe())

plt.figure(figsize=(10, 5))
sns.histplot(valid_durations, bins=30, kde=True)
plt.title("Distribution of Movie Durations")
plt.xlabel("Duration (minutes)")
plt.ylabel("Number of Movies")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/06_movie_durations.png", dpi=300)
plt.show()

# 12. TV SHOW SEASON ANALYSIS
shows = df[df["type"] == "TV Show"].copy()

shows["seasons"] = pd.to_numeric(
    shows["duration"].str.extract(r"(\d+)")[0],
    errors="coerce"
)

season_counts = shows["seasons"].dropna().value_counts().sort_index()

print("\nTV SHOWS BY NUMBER OF SEASONS:")
print(season_counts)

plt.figure(figsize=(10, 5))
sns.barplot(
    x=season_counts.index.astype(int),
    y=season_counts.values
)
plt.title("TV Shows by Number of Seasons")
plt.xlabel("Number of Seasons")
plt.ylabel("Number of Shows")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/07_tv_seasons.png", dpi=300)
plt.show()

# 13. CONTENT ADDED TO NETFLIX BY YEAR
added = df.dropna(subset=["date_added"]).copy()
added["year_added"] = added["date_added"].dt.year

added_counts = added["year_added"].value_counts().sort_index()

print("\nTITLES ADDED TO NETFLIX BY YEAR:")
print(added_counts)

plt.figure(figsize=(10, 5))
sns.lineplot(x=added_counts.index, y=added_counts.values, marker="o")
plt.title("Netflix Titles Added by Year")
plt.xlabel("Year Added")
plt.ylabel("Number of Titles")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/08_added_trends.png", dpi=300)
plt.show()

# 14. CORRELATION ANALYSIS
correlation_data = df[
    ["release_year"]
].copy()

correlation_data["year_added"] = df["date_added"].dt.year

correlation_matrix = correlation_data.corr()

print("\nCORRELATION MATRIX:")
print(correlation_matrix)

plt.figure(figsize=(6, 4))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    fmt=".2f"
)
plt.title("Release Year and Year Added Correlation")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/09_correlation.png", dpi=300)
plt.show()

# 15. AUTOMATIC FINDINGS
print("\n" + "=" * 60)
print("PROJECT FINDINGS")
print("=" * 60)

print("1. Most common content type:", content_counts.idxmax())
print("2. Number of movies:", int(content_counts.get("Movie", 0)))
print("3. Number of TV shows:", int(content_counts.get("TV Show", 0)))

if not top_genres.empty:
    print("4. Most common genre:", top_genres.index[0])

if not top_countries.empty:
    print("5. Country with the most listed titles:", top_countries.index[0])

if not valid_durations.empty:
    print("6. Average movie duration:",
          round(valid_durations.mean(), 2), "minutes")

if not season_counts.empty:
    print("7. Most common TV show season count:",
          int(season_counts.idxmax()))

print("\nAll graphs saved in:", OUTPUT_FOLDER)
print("\nPROJECT COMPLETED SUCCESSFULLY!")
