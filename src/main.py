"""
Crisis Detection and Analysis Pipeline

This module serves as the main entry point for the crisis detection and analysis pipeline.
It orchestrates the data extraction, preprocessing, analysis, and visualization of crisis-related
data from social media platforms (Twitter and Reddit).

The pipeline consists of the following steps:
1. Data extraction from social media platforms
2. Text cleaning and preprocessing
3. Sentiment analysis and risk classification
4. Location extraction and geocoding
5. Visualization of results through interactive maps

Dependencies:
- tweepy: For Twitter API access
- praw: For Reddit API access
- pandas: For data manipulation
- dotenv: For environment variable management
- Custom modules from data_extraction, preprocessing, analysis, and visualization packages
"""

import argparse
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Import custom modules
from data_extraction.twitter_api import TwitterDataExtractor
from data_extraction.reddit_api import RedditDataExtractor
from preprocessing.text_cleaning import TextCleaner
from analysis.sentiment_analysis import CrisisAnalyzer
from visualization.geocoding import LocationExtractor
from visualization.mapping import CrisisMapVisualizer

def setup_directories():
    """
    Create necessary directories for data storage and output.
    
    Creates the following directory structure if it doesn't exist:
    - data/raw: For storing raw data from social media
    - data/processed: For storing processed and cleaned data
    - output/maps: For storing generated map visualizations
    - output/reports: For storing analysis reports
    """
    directories = [
        'data/raw',
        'data/processed',
        'output/maps',
        'output/reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_twitter_analysis(args):
    """
    Run the complete analysis pipeline for Twitter data.
    
    Args:
        args: Command line arguments containing configuration options
        
    Returns:
        pd.DataFrame: Processed and analyzed Twitter data with geocoded locations
        
    The pipeline includes:
    1. Data extraction from Twitter
    2. Text cleaning and preprocessing
    3. Sentiment analysis and risk classification
    4. Location extraction and geocoding
    5. Creation of interactive visualizations
    """
    print("Starting Twitter data extraction...")
    twitter_extractor = TwitterDataExtractor()
    tweets_df = twitter_extractor.run_extraction('data/raw/tweets.csv')
    
    print("\nCleaning Twitter data...")
    cleaner = TextCleaner()
    cleaned_tweets = cleaner.process_dataframe(tweets_df, ['text'])
    cleaner.save_cleaned_data(cleaned_tweets, 'data/processed/cleaned_tweets.csv')
    
    print("\nAnalyzing sentiment and risk levels...")
    analyzer = CrisisAnalyzer()
    analyzed_tweets = analyzer.process_dataframe(cleaned_tweets, 'text_cleaned')
    analyzer.save_results(analyzed_tweets, 'data/processed/analyzed_tweets.csv')
    
    print("\nExtracting and geocoding locations...")
    location_extractor = LocationExtractor()
    geocoded_tweets = location_extractor.process_dataframe(analyzed_tweets, 'text_cleaned')
    location_extractor.save_geocoded_data(geocoded_tweets, 'data/processed/geocoded_tweets.csv')
    
    print("\nCreating visualizations...")
    visualizer = CrisisMapVisualizer()
    
    # Create heatmap
    visualizer.create_base_map()
    visualizer.add_heatmap(geocoded_tweets)
    visualizer.save_map('output/maps/twitter_heatmap.html')
    
    # Create risk level map
    visualizer.create_base_map()
    visualizer.add_risk_level_layer(geocoded_tweets)
    visualizer.save_map('output/maps/twitter_risk_map.html')
    
    # Create top locations map
    visualizer.show_top_locations(geocoded_tweets)
    visualizer.save_map('output/maps/twitter_top_locations.html')
    
    return geocoded_tweets

def run_reddit_analysis(args):
    """
    Run the complete analysis pipeline for Reddit data.
    
    Args:
        args: Command line arguments containing configuration options
        
    Returns:
        pd.DataFrame: Processed and analyzed Reddit data with geocoded locations
        
    The pipeline includes:
    1. Data extraction from Reddit
    2. Text cleaning and preprocessing
    3. Sentiment analysis and risk classification
    4. Location extraction and geocoding
    5. Creation of interactive visualizations
    """
    print("Starting Reddit data extraction...")
    reddit_extractor = RedditDataExtractor()
    posts_df = reddit_extractor.run_extraction('data/raw/reddit_posts.csv')
    
    print("\nCleaning Reddit data...")
    cleaner = TextCleaner()
    cleaned_posts = cleaner.process_dataframe(posts_df, ['text'])
    cleaner.save_cleaned_data(cleaned_posts, 'data/processed/cleaned_reddit_posts.csv')
    
    print("\nAnalyzing sentiment and risk levels...")
    analyzer = CrisisAnalyzer()
    analyzed_posts = analyzer.process_dataframe(cleaned_posts, 'text_cleaned')
    analyzer.save_results(analyzed_posts, 'data/processed/analyzed_reddit_posts.csv')
    
    print("\nExtracting and geocoding locations...")
    location_extractor = LocationExtractor()
    geocoded_posts = location_extractor.process_dataframe(analyzed_posts, 'text_cleaned')
    location_extractor.save_geocoded_data(geocoded_posts, 'data/processed/geocoded_reddit_posts.csv')
    
    print("\nCreating visualizations...")
    visualizer = CrisisMapVisualizer()
    
    # Create heatmap
    visualizer.create_base_map()
    visualizer.add_heatmap(geocoded_posts)
    visualizer.save_map('output/maps/reddit_heatmap.html')
    
    # Create risk level map
    visualizer.create_base_map()
    visualizer.add_risk_level_layer(geocoded_posts)
    visualizer.save_map('output/maps/reddit_risk_map.html')
    
    # Create top locations map
    visualizer.show_top_locations(geocoded_posts)
    visualizer.save_map('output/maps/reddit_top_locations.html')
    
    return geocoded_posts

def generate_report(df, platform):
    """
    Generate a summary report of the analysis results.
    
    Args:
        df (pd.DataFrame): Processed and analyzed data
        platform (str): Name of the social media platform (e.g., 'Twitter', 'Reddit')
        
    Returns:
        str: Formatted report containing:
        - Total number of posts analyzed
        - Number of posts with location data
        - Distribution of risk levels
        - Summary statistics of sentiment scores
        - Top 5 locations mentioned
    """
    report = f"""
    Crisis Analysis Report - {platform}
    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    Total Posts Analyzed: {len(df)}
    Posts with Location Data: {df['latitude'].notna().sum()}
    
    Risk Level Distribution:
    {df['risk_level'].value_counts()}
    
    Sentiment Analysis Summary:
    {df[['sentiment_negative', 'sentiment_neutral', 'sentiment_positive', 'sentiment_compound']].describe()}
    
    Top 5 Locations:
    {df['extracted_location'].value_counts().head()}
    """
    
    return report

def main():
    """
    Main entry point for the crisis detection and analysis pipeline.
    
    Parses command line arguments and runs the analysis pipeline for the specified
    social media platform(s). Supports analysis of Twitter, Reddit, or both platforms.
    
    Command Line Arguments:
        --platform: Social media platform to analyze ('twitter', 'reddit', or 'all')
        --output-dir: Directory to save output files (default: 'output')
    """
    parser = argparse.ArgumentParser(description='Crisis Detection and Analysis Pipeline')
    parser.add_argument('--platform', choices=['twitter', 'reddit', 'all'], default='all',
                      help='Social media platform to analyze')
    parser.add_argument('--output-dir', default='output',
                      help='Directory to save output files')
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Setup directories
    setup_directories()
    
    results = {}
    
    if args.platform in ['twitter', 'all']:
        print("\n=== Analyzing Twitter Data ===")
        twitter_results = run_twitter_analysis(args)
        results['twitter'] = twitter_results
        
        # Generate report
        twitter_report = generate_report(twitter_results, 'Twitter')
        with open('output/reports/twitter_report.txt', 'w') as f:
            f.write(twitter_report)
    
    if args.platform in ['reddit', 'all']:
        print("\n=== Analyzing Reddit Data ===")
        reddit_results = run_reddit_analysis(args)
        results['reddit'] = reddit_results
        
        # Generate report
        reddit_report = generate_report(reddit_results, 'Reddit')
        with open('output/reports/reddit_report.txt', 'w') as f:
            f.write(reddit_report)
    
    print("\nAnalysis complete! Check the output directory for results.")

if __name__ == "__main__":
    main() 