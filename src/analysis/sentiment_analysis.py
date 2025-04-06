"""
Sentiment Analysis and Risk Classification Module

This module provides functionality for:
1. Sentiment analysis using VADER
2. Risk level classification based on content
3. Visualization of sentiment and risk distributions
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import os

class CrisisAnalyzer:
    """
    Analyzes social media posts for sentiment and risk levels.
    
    This class provides methods for:
    - Sentiment analysis using VADER
    - Risk level classification
    - Distribution visualization
    """
    
    def __init__(self):
        """Initialize the analyzer with VADER sentiment analyzer."""
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Define risk level keywords
        self.risk_keywords = {
            'high': [
                'suicidal', 'kill myself', 'end it all', 'want to die',
                'can\'t go on', 'no way out', 'hopeless', 'worthless'
            ],
            'moderate': [
                'depressed', 'anxious', 'overwhelmed', 'struggling',
                'need help', 'can\'t cope', 'lost', 'alone'
            ],
            'low': [
                'mental health', 'therapy', 'counseling', 'support',
                'self care', 'wellness', 'mindfulness'
            ]
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze the sentiment of a text using VADER.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            Dict[str, float]: Sentiment scores (negative, neutral, positive, compound)
        """
        # Handle NaN values and convert to string
        if pd.isna(text):
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
        
        # Ensure text is string type
        text = str(text)
        
        return self.analyzer.polarity_scores(text)
    
    def classify_risk_level(self, text: str) -> str:
        """
        Classify the risk level of a post based on its content.
        
        Args:
            text (str): The text to classify
            
        Returns:
            str: Risk level ('high', 'moderate', 'low')
        """
        # Handle NaN values
        if pd.isna(text):
            return 'unknown'
            
        # Ensure text is string type
        text_lower = str(text).lower()
        
        # Check for high-risk keywords
        if any(keyword in text_lower for keyword in self.risk_keywords['high']):
            return 'high'
            
        # Check for moderate-risk keywords
        if any(keyword in text_lower for keyword in self.risk_keywords['moderate']):
            return 'moderate'
            
        # Check for low-risk keywords
        if any(keyword in text_lower for keyword in self.risk_keywords['low']):
            return 'low'
        
        return 'unknown'
    
    def analyze_posts(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Analyze a DataFrame of posts for sentiment and risk levels.
        
        Args:
            df (pd.DataFrame): DataFrame containing posts
            text_column (str): Name of the column containing post text
            
        Returns:
            pd.DataFrame: DataFrame with added sentiment and risk columns
        """
        # Add sentiment analysis columns
        sentiment_scores = df[text_column].apply(self.analyze_sentiment)
        df['sentiment_neg'] = sentiment_scores.apply(lambda x: x['neg'])
        df['sentiment_neu'] = sentiment_scores.apply(lambda x: x['neu'])
        df['sentiment_pos'] = sentiment_scores.apply(lambda x: x['pos'])
        df['sentiment_compound'] = sentiment_scores.apply(lambda x: x['compound'])
        
        # Add risk level column
        df['risk_level'] = df[text_column].apply(self.classify_risk_level)
        
        return df
    
    def plot_distributions(self, df: pd.DataFrame, output_dir: str = 'output/reports'):
        """
        Create and save plots showing the distribution of posts by sentiment and risk level.
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment and risk analysis
            output_dir (str): Directory to save the plots
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sentiment and Risk Level Distributions', fontsize=16)
        
        # Plot 1: Sentiment Compound Score Distribution
        sns.histplot(data=df, x='sentiment_compound', bins=30, ax=axes[0, 0])
        axes[0, 0].set_title('Sentiment Compound Score Distribution')
        axes[0, 0].set_xlabel('Compound Score')
        axes[0, 0].set_ylabel('Count')
        
        # Plot 2: Risk Level Distribution
        risk_counts = df['risk_level'].value_counts()
        sns.barplot(x=risk_counts.index, y=risk_counts.values, ax=axes[0, 1])
        axes[0, 1].set_title('Risk Level Distribution')
        axes[0, 1].set_xlabel('Risk Level')
        axes[0, 1].set_ylabel('Count')
        
        # Plot 3: Sentiment by Risk Level
        sns.boxplot(data=df, x='risk_level', y='sentiment_compound', ax=axes[1, 0])
        axes[1, 0].set_title('Sentiment by Risk Level')
        axes[1, 0].set_xlabel('Risk Level')
        axes[1, 0].set_ylabel('Compound Score')
        
        # Plot 4: Risk Level by Platform
        if 'platform' in df.columns:
            sns.countplot(data=df, x='platform', hue='risk_level', ax=axes[1, 1])
            axes[1, 1].set_title('Risk Level by Platform')
            axes[1, 1].set_xlabel('Platform')
            axes[1, 1].set_ylabel('Count')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sentiment_risk_distributions.png'))
        plt.close()
        
        # Save statistics to text file
        with open(os.path.join(output_dir, 'sentiment_risk_stats.txt'), 'w') as f:
            f.write("Sentiment and Risk Level Statistics\n")
            f.write("=================================\n\n")
            
            f.write("Risk Level Distribution:\n")
            f.write(str(risk_counts) + "\n\n")
            
            f.write("Sentiment Statistics by Risk Level:\n")
            for risk_level in df['risk_level'].unique():
                f.write(f"\n{risk_level.upper()} RISK:\n")
                stats = df[df['risk_level'] == risk_level]['sentiment_compound'].describe()
                f.write(str(stats) + "\n")

def main():
    """Main function to run the analysis."""
    # Create analyzer instance
    analyzer = CrisisAnalyzer()
    
    # Load data
    try:
        # Try to load Reddit data
        reddit_df = pd.read_csv('data/raw/reddit_posts.csv')
        reddit_df['platform'] = 'reddit'
        df = reddit_df
    except FileNotFoundError:
        print("Reddit data not found. Please run the data extraction script first.")
        return
    
    # Analyze posts
    analyzed_df = analyzer.analyze_posts(df, text_column='text')
    
    # Save analyzed data
    os.makedirs('data/processed', exist_ok=True)
    analyzed_df.to_csv('data/processed/analyzed_posts.csv', index=False)
    
    # Create and save visualizations
    analyzer.plot_distributions(analyzed_df)
    
    print("Analysis complete! Check the output directory for results.")

if __name__ == "__main__":
    main() 