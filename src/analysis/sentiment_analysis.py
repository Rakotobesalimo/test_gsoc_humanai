"""
Sentiment Analysis and Risk Classification Module

This module provides functionality for analyzing sentiment and detecting risk levels
in crisis-related text data. It uses VADER sentiment analysis and pattern matching
to identify high-risk content.

Key Features:
- Sentiment analysis using VADER
- Risk level classification
- Pattern matching for crisis-related phrases
- Batch processing of text data
- Statistical analysis of results

Dependencies:
- vaderSentiment: For sentiment analysis
- pandas: For data manipulation
- numpy: For numerical operations
- sklearn: For text vectorization
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class CrisisAnalyzer:
    """
    A class for analyzing sentiment and detecting risk levels in crisis-related text.
    
    This class combines sentiment analysis with pattern matching to identify
    potentially high-risk content. It provides methods for both individual text
    analysis and batch processing of datasets.
    
    Attributes:
        sentiment_analyzer (SentimentIntensityAnalyzer): VADER sentiment analyzer
        high_risk_phrases (list): List of high-risk phrases to detect
        moderate_risk_phrases (list): List of moderate-risk phrases to detect
        high_risk_patterns (list): Compiled regex patterns for high-risk phrases
        moderate_risk_patterns (list): Compiled regex patterns for moderate-risk phrases
    """
    
    def __init__(self):
        """
        Initialize the CrisisAnalyzer with sentiment analyzer and risk patterns.
        
        Sets up the VADER sentiment analyzer and compiles regex patterns for
        detecting high-risk and moderate-risk phrases in text.
        """
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Define high-risk keywords and phrases
        self.high_risk_phrases = [
            r"don't want to live",
            r"want to die",
            r"end it all",
            r"kill myself",
            r"suicide",
            r"no way out",
            r"can't go on",
            r"give up",
            r"hopeless",
            r"worthless"
        ]
        
        self.moderate_risk_phrases = [
            r"need help",
            r"struggling",
            r"overwhelmed",
            r"can't cope",
            r"depressed",
            r"anxious",
            r"lost",
            r"alone",
            r"scared",
            r"worried"
        ]
        
        # Compile regex patterns
        self.high_risk_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.high_risk_phrases]
        self.moderate_risk_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.moderate_risk_phrases]
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment using VADER.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            dict: Sentiment scores containing:
                - neg: Negative sentiment score
                - neu: Neutral sentiment score
                - pos: Positive sentiment score
                - compound: Overall sentiment score
                
        Returns zero scores for non-string input.
        """
        if not isinstance(text, str):
            return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
        
        return self.sentiment_analyzer.polarity_scores(text)
    
    def detect_risk_level(self, text):
        """
        Detect risk level based on keyword patterns.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            str: Risk level ('high', 'moderate', or 'low')
            
        The risk level is determined by:
        1. Checking for high-risk phrases
        2. Checking for moderate-risk phrases
        3. Defaulting to 'low' if no risk phrases are found
        """
        if not isinstance(text, str):
            return 'low'
        
        text = text.lower()
        
        # Check for high-risk phrases
        for pattern in self.high_risk_patterns:
            if pattern.search(text):
                return 'high'
        
        # Check for moderate-risk phrases
        for pattern in self.moderate_risk_patterns:
            if pattern.search(text):
                return 'moderate'
        
        return 'low'
    
    def process_text(self, text):
        """
        Process a single text entry.
        
        Args:
            text (str): Input text to process
            
        Returns:
            dict: Analysis results containing:
                - sentiment_negative: Negative sentiment score
                - sentiment_neutral: Neutral sentiment score
                - sentiment_positive: Positive sentiment score
                - sentiment_compound: Overall sentiment score
                - risk_level: Detected risk level
        """
        sentiment = self.analyze_sentiment(text)
        risk_level = self.detect_risk_level(text)
        
        return {
            'sentiment_negative': sentiment['neg'],
            'sentiment_neutral': sentiment['neu'],
            'sentiment_positive': sentiment['pos'],
            'sentiment_compound': sentiment['compound'],
            'risk_level': risk_level
        }
    
    def process_dataframe(self, df, text_column):
        """
        Process a DataFrame containing text data.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Name of the column containing text to analyze
            
        Returns:
            pd.DataFrame: Original DataFrame with analysis results added
            
        Creates new columns for sentiment scores and risk level.
        """
        results = []
        
        for text in df[text_column]:
            result = self.process_text(text)
            results.append(result)
        
        # Create DataFrame from results
        results_df = pd.DataFrame(results)
        
        # Combine with original DataFrame
        return pd.concat([df, results_df], axis=1)
    
    def save_results(self, df, output_file):
        """
        Save analysis results to CSV file.
        
        Args:
            df (pd.DataFrame): DataFrame containing analysis results
            output_file (str): Path to save the CSV file
        """
        df.to_csv(output_file, index=False)
        print(f"Analysis results saved to {output_file}")
    
    def get_risk_distribution(self, df):
        """
        Get distribution of risk levels.
        
        Args:
            df (pd.DataFrame): DataFrame containing risk level data
            
        Returns:
            pd.Series: Count of posts by risk level
        """
        return df['risk_level'].value_counts()
    
    def get_sentiment_summary(self, df):
        """
        Get summary statistics of sentiment scores.
        
        Args:
            df (pd.DataFrame): DataFrame containing sentiment scores
            
        Returns:
            pd.DataFrame: Summary statistics for each sentiment score
        """
        return df[['sentiment_negative', 'sentiment_neutral', 'sentiment_positive', 'sentiment_compound']].describe()

if __name__ == "__main__":
    # Example usage
    analyzer = CrisisAnalyzer()
    
    # Sample data
    sample_data = pd.DataFrame({
        'text': [
            "I don't want to live anymore, everything is hopeless",
            "Feeling a bit overwhelmed with work lately",
            "Had a great day today! Feeling positive about the future",
            "Struggling with anxiety but trying to stay strong",
            "Need help with my mental health"
        ]
    })
    
    # Process the data
    results = analyzer.process_dataframe(sample_data, 'text')
    print("\nProcessed Data:")
    print(results)
    
    print("\nRisk Level Distribution:")
    print(analyzer.get_risk_distribution(results))
    
    print("\nSentiment Summary:")
    print(analyzer.get_sentiment_summary(results)) 