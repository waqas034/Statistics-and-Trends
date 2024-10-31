# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the extracted CSV file
csv_file_path = 'netflix_movies.csv'
netflix_data = pd.read_csv(csv_file_path)

# Filter data for release years between 2000 and 2020
filtered_data = netflix_data[(netflix_data['release_year'] >= 2000) & (netflix_data['release_year'] <= 2020)]

# Statistical Analysis
# Summary statistics
summary_statistics = netflix_data.describe(include='all')

# Correlation
correlation_matrix = netflix_data[['release_year']].corr()

# Kurtosis and Skewness for numeric features
release_year_kurtosis = netflix_data['release_year'].kurtosis()
release_year_skewness = netflix_data['release_year'].skew()

def plot_release_year_trend(data):
    """Plot the trend of content added by release year."""
    plt.figure(figsize=(12, 6))
    release_year_count_filtered = data['release_year'].value_counts().sort_index()
    plt.plot(
        release_year_count_filtered.index,
        release_year_count_filtered.values,
        marker='o',
        linestyle='-',
        color='b'
    )
    plt.title('Trend of Content Added by Release Year (2000-2020)', fontsize=16, fontweight='bold')
    plt.xlabel('Release Year', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Releases', fontsize=14, fontweight='bold')
    for x, y in zip(release_year_count_filtered.index, release_year_count_filtered.values):
        plt.text(x, y, str(y), fontsize=10, ha='center', va='bottom', fontweight='bold')
    plt.grid(visible=True, linestyle='--', linewidth=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_content_type_proportion(data):
    """Plot the proportion of content types (Movies vs TV Shows)."""
    plt.figure(figsize=(10, 10))
    content_type_count = data['type'].value_counts()
    colors = sns.color_palette('pastel', len(content_type_count))
    explode = [0.05, 0.05]  # Exploding both segments slightly for better visualization

    plt.pie(
        content_type_count,
        labels=content_type_count.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5},
        explode=explode,
        shadow=True,
        textprops={'fontsize': 14, 'fontweight': 'bold'}
    )
    plt.title('Proportion of Content Types (Movies vs TV Shows)', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_movie_duration_boxplot(data):
    """Plot the box plot of movie durations."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(y='duration_minutes', data=data, palette='viridis')
    plt.title('Box Plot of Movie Durations', fontsize=16, fontweight='bold')
    plt.ylabel('Duration (Minutes)', fontsize=14, fontweight='bold')
    plt.grid(visible=True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(data):
    """Plot the correlation heatmap for numeric features."""
    plt.figure(figsize=(8, 6))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', linewidths=0.5, fmt='.2f')
    plt.title('Correlation Heatmap for Numeric Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Extract numeric values from duration (in minutes)
movies_data = netflix_data[netflix_data['type'] == 'Movie'].copy()
movies_data['duration_minutes'] = movies_data['duration'].str.extract(r'(\d+)').astype(float)

# Plotting using the defined functions
plot_release_year_trend(filtered_data)
plot_content_type_proportion(netflix_data)
plot_movie_duration_boxplot(movies_data)
plot_correlation_heatmap(movies_data[['release_year', 'duration_minutes']])

# Print Summary Statistics, Correlation, Kurtosis, and Skewness
print("Summary Statistics:\n", summary_statistics)
print("\nCorrelation Matrix:\n", correlation_matrix)
print("\nKurtosis of Release Year:", release_year_kurtosis)
print("Skewness of Release Year:", release_year_skewness)