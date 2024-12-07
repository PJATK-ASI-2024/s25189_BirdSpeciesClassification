import pandas as pd
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv


    
def plot_class_distribution(class_counts):
    # Split the class counts into three parts
    class_counts_sorted = class_counts.sort_values(ascending=False)
    num_classes = len(class_counts_sorted)
    split_size = num_classes // 3

    # Plot the top third
    plt.figure(figsize=(22, 11))
    class_counts_sorted[:split_size].plot(kind='bar', width=0.8)
    plt.title("Top third bird species distribution")
    plt.xlabel("Bird species")
    plt.ylabel("Count")
    plt.xticks(rotation=90, ha='right', fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.show()

    # Plot the middle third
    plt.figure(figsize=(22, 11))
    class_counts_sorted[split_size:2*split_size].plot(kind='bar', width=0.8)
    plt.title("Middle third bird species distribution")
    plt.xlabel("Bird species")
    plt.ylabel("Count")
    plt.xticks(rotation=90, ha='right', fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.show()

    # Plot the bottom third
    plt.figure(figsize=(22, 11))
    class_counts_sorted[2*split_size:].plot(kind='bar', width=0.8)
    plt.title("Bottom third bird species distribution")
    plt.xlabel("Bird species")
    plt.ylabel("Count")
    plt.xticks(rotation=90, ha='right', fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.show()
    
    # Save the top third plot
    plt.figure(figsize=(22, 11))
    class_counts_sorted[:split_size].plot(kind='bar', width=0.8)
    plt.title("Top third bird species distribution")
    plt.xlabel("Bird species")
    plt.ylabel("Count")
    plt.xticks(rotation=90, ha='right', fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.savefig('/Users/hubertsienicki/Projects/s25189_BirdSpeciesClassification/reports/figures/top_third_distribution.png')
    plt.close()

    # Save the middle third plot
    plt.figure(figsize=(22, 11))
    class_counts_sorted[split_size:2*split_size].plot(kind='bar', width=0.8)
    plt.title("Middle third bird species distribution")
    plt.xlabel("Bird species")
    plt.ylabel("Count")
    plt.xticks(rotation=90, ha='right', fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.savefig('/Users/hubertsienicki/Projects/s25189_BirdSpeciesClassification/reports/figures/middle_third_distribution.png')
    plt.close()

    # Save the bottom third plot
    plt.figure(figsize=(22, 11))
    class_counts_sorted[2*split_size:].plot(kind='bar', width=0.8)
    plt.title("Bottom third bird species distribution")
    plt.xlabel("Bird species")
    plt.ylabel("Count")
    plt.xticks(rotation=90, ha='right', fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.savefig('/Users/hubertsienicki/Projects/s25189_BirdSpeciesClassification/reports/figures/bottom_third_distribution.png')
    plt.close()

def plot_image_width_distribution(df):
    df['img_width_px'].hist(bins=30)
    plt.title("Distribution of image width in pixels")
    plt.savefig('/Users/hubertsienicki/Projects/s25189_BirdSpeciesClassification/reports/figures/image_width_distribution.png')
    plt.show()

def plot_rgb_distribution(df):
    df['mean_r'].hist(bins=30, alpha=0.5, label='mean_r')
    df['mean_g'].hist(bins=30, alpha=0.5, label='mean_g')
    df['mean_b'].hist(bins=30, alpha=0.5, label='mean_b')
    plt.title("Distribution of mean RGB values")
    plt.legend()
    plt.savefig('/Users/hubertsienicki/Projects/s25189_BirdSpeciesClassification/reports/figures/mean_rgb_distribution.png')
    plt.show()

def show_outliers(df, numeric_cols):
    plt.figure(figsize=(12,6))
    sns.boxplot(data=df[numeric_cols])
    plt.title("Boxplot of numeric columns")
    plt.xticks(rotation=45)
    plt.savefig('/Users/hubertsienicki/Projects/s25189_BirdSpeciesClassification/reports/figures/boxplot.png')
    plt.show()

def plot_correration_matrix(df, numeric_cols):
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation matrix of numeric columns")
    plt.savefig('/Users/hubertsienicki/Projects/s25189_BirdSpeciesClassification/reports/figures/correlation_matrix.png')
    plt.show()

def generate_report(df):
    report = sv.analyze(df)
    report.show_html('CUB_200_2011_sweetviz_report.html')

def main():
    # Create a logger
    logger = logging.getLogger('EDA_Logger')
    logger.setLevel(logging.DEBUG)

    # Create a file handler
    log_filename = f"/Users/hubertsienicki/Projects/s25189_BirdSpeciesClassification/logs/eda/eda_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)

    logger.info('EDA logger initialized successfully.')
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(console_handler)

    df = pd.read_csv('data/train/metadata.csv')
    logger.info('Metadata loaded successfully.')
    logger.info('Displaying the first 5 rows of the dataset.')
    print(df.head())
    logger.info('Displaying datastructure of the dataset.')
    print(df.info())
    logger.info('Displaying basic statistics of the dataset.')
    print(df.describe())

    logger.info('Checking for missing values in the dataset.')

    print(df.isnull().sum())

    logger.info('Showing the distribution of the target variable.')
    class_counts = df['classes_name'].value_counts()
    print(class_counts)
    # plot_class_distribution(class_counts)

    logger.info('Showing the distribution of the numeric columns.')
    numeric_cols = ['mean_r', 'mean_g', 'mean_b', 'img_width_px', 'img_height_px', 'num_pixels', 'width', 'height', 'x', 'y']
    print(df[numeric_cols].describe())

    logger.info('Showing histograms of the numeric columns.')

    #plot_rgb_distribution(df)

    logger.info('Showing the distribution of image width in pixels.')
    #plot_image_width_distribution(df)

    logger.info('Showing outliers in the dataset.')
    #show_outliers(df, numeric_cols)

    logger.info('Showing the correlation matrix of the numeric columns.')
    #plot_correration_matrix(df, numeric_cols)

    logger.info('Generating EDA report.')
    generate_report(df)

    logger.info('EDA completed successfully.')

if __name__ == '__main__':
    main()