"""
Text Cleaning and Preprocessing Module

This module provides functionality for cleaning and preprocessing text data from social media.
It includes features for removing emojis, URLs, special characters, and stopwords.

Key Features:
- Emoji removal
- URL removal
- Special character cleaning
- Stopword removal
- Text normalization
- Batch processing of DataFrame columns

Dependencies:
- nltk: For text processing and stopwords
- emoji: For emoji handling
- pandas: For data manipulation
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import emoji
import pandas as pd

class TextCleaner:
    """
    A class for cleaning and preprocessing text data.
    
    This class provides methods for cleaning social media text by removing
    emojis, URLs, special characters, and stopwords. It supports both
    individual text cleaning and batch processing of DataFrame columns.
    
    Attributes:
        stop_words (set): Set of English stopwords with custom additions
    """
    
    def __init__(self):
        """
        Initialize the TextCleaner with required NLTK data and stopwords.
        
        Downloads necessary NLTK data and sets up a custom stopword list
        that includes common social media terms.
        """
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        # Add custom stopwords specific to social media
        self.stop_words.update(['rt', 'http', 'https', 'www', 'com'])
    
    def remove_emojis(self, text):
        """
        Remove emojis from text.
        
        Args:
            text (str): Input text containing emojis
            
        Returns:
            str: Text with emojis removed
            
        Uses the emoji library to identify and remove emoji characters.
        """
        return emoji.replace_emoji(text, '')
    
    def remove_urls(self, text):
        """
        Remove URLs from text.
        
        Args:
            text (str): Input text containing URLs
            
        Returns:
            str: Text with URLs removed
            
        Uses regular expressions to identify and remove both http(s) and www URLs.
        """
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)
    
    def remove_special_chars(self, text):
        """
        Remove special characters and numbers from text.
        
        Args:
            text (str): Input text containing special characters
            
        Returns:
            str: Text with special characters removed
            
        Keeps only alphanumeric characters and basic punctuation marks.
        """
        # Keep only alphanumeric characters and basic punctuation
        return re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    
    def remove_stopwords(self, text):
        """
        Remove stopwords from text.
        
        Args:
            text (str): Input text containing stopwords
            
        Returns:
            str: Text with stopwords removed
            
        Uses NLTK's word tokenizer and stopword list, plus custom social media stopwords.
        """
        words = word_tokenize(text.lower())
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def clean_text(self, text):
        """
        Apply all cleaning steps to text.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
            
        The cleaning process includes:
        1. Converting to lowercase
        2. Removing emojis
        3. Removing URLs
        4. Removing special characters
        5. Removing stopwords
        6. Normalizing whitespace
        """
        if not isinstance(text, str):
            return ""
            
        text = text.lower()
        text = self.remove_emojis(text)
        text = self.remove_urls(text)
        text = self.remove_special_chars(text)
        text = self.remove_stopwords(text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def process_dataframe(self, df, text_columns):
        """
        Process text columns in a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_columns (list): List of column names containing text to clean
            
        Returns:
            pd.DataFrame: DataFrame with cleaned text columns
            
        Creates new columns with '_cleaned' suffix for each processed text column.
        """
        df_clean = df.copy()
        
        for column in text_columns:
            if column in df.columns:
                df_clean[f'{column}_cleaned'] = df[column].apply(self.clean_text)
        
        return df_clean
    
    def save_cleaned_data(self, df, output_file):
        """
        Save cleaned DataFrame to CSV file.
        
        Args:
            df (pd.DataFrame): Cleaned DataFrame to save
            output_file (str): Path to save the CSV file
            
        Creates the output directory if it doesn't exist.
        """
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")

# if __name__ == "__main__":
#     # Example usage
#     cleaner = TextCleaner()
    
#     # Load Reddit posts data
#     reddit_posts = pd.read_csv("data/raw/reddit_posts.csv")
    
#     # Clean the text columns
#     text_columns = ['title', 'selftext'] 
#     cleaned_data = cleaner.process_dataframe(reddit_posts, text_columns)
    
#     # Save cleaned data
#     cleaner.save_cleaned_data(cleaned_data, "data/processed/reddit_posts_cleaned.csv")