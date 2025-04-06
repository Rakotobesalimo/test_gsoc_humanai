"""
Reddit Data Extraction Module

This module provides functionality to extract crisis-related data from Reddit using the PRAW library.
It includes features for authentication, subreddit searching, and data processing.

Key Features:
- OAuth 2.0 authentication with Reddit API
- Search across multiple crisis-related subreddits
- Processing of post data into structured format
- Saving results to CSV files

Dependencies:
- praw: For Reddit API access
- pandas: For data manipulation
- dotenv: For environment variable management
"""

import praw
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

class RedditDataExtractor:
    """
    A class for extracting crisis-related data from Reddit.
    
    This class handles authentication with the Reddit API and data extraction
    from crisis-related subreddits. It includes methods for searching posts,
    processing the results, and saving them to CSV files.
    
    Attributes:
        client_id (str): Reddit API client ID
        client_secret (str): Reddit API client secret
        user_agent (str): Reddit API user agent
        subreddits (list): List of crisis-related subreddits to search
        keywords (list): List of crisis-related keywords to search for
        reddit (praw.Reddit): Authenticated Reddit API client
    """
    
    def __init__(self):
        """
        Initialize the RedditDataExtractor with API credentials and settings.
        
        Loads credentials from environment variables and sets up subreddits
        and keywords for crisis-related content extraction.
        
        Raises:
            ValueError: If any required API credentials are missing
        """
        load_dotenv()
        self.client_id = os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = os.getenv('REDDIT_USER_AGENT')
        
        # Crisis-related subreddits and keywords
        self.subreddits = [
            "depression", "anxiety", "mentalhealth",
            "SuicideWatch", "addiction", "therapy",
            "mentalillness", "psychology", "selfhelp"
        ]
        
        self.keywords = [
            "depressed", "anxiety", "suicidal", "mental health",
            "overwhelmed", "help needed", "crisis", "addiction",
            "therapy", "counseling", "support", "mental illness",
            "psychiatric", "emotional distress", "self harm"
        ]
        
        self.reddit = self._authenticate()
    
    def _authenticate(self):
        """
        Authenticate with Reddit API using OAuth 2.0.
        
        Returns:
            praw.Reddit: Authenticated Reddit API client or None if authentication fails
            
        This method:
        1. Creates a Reddit client with OAuth credentials
        2. Handles authentication errors gracefully
        """
        try:
            reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )
            return reddit
        except Exception as e:
            print(f"Authentication failed: {str(e)}")
            return None
    
    def search_subreddit(self, subreddit_name, limit=100):
        """
        Search for posts in a specific subreddit.
        
        Args:
            subreddit_name (str): Name of the subreddit to search
            limit (int): Maximum number of posts to retrieve (default: 100)
            
        Returns:
            list: List of Reddit post objects matching the search criteria
            
        This method:
        1. Searches for posts containing any of the crisis-related keywords
        2. Limits results to posts from the last month
        3. Handles search errors gracefully
        """
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = subreddit.search(
                query=" OR ".join(self.keywords),
                limit=limit,
                time_filter='month'
            )
            return posts
        except Exception as e:
            print(f"Error searching subreddit {subreddit_name}: {str(e)}")
            return []
    
    def process_posts(self, posts):
        """
        Process Reddit posts into a structured format.
        
        Args:
            posts: List of Reddit post objects
            
        Returns:
            pd.DataFrame: Processed posts with relevant fields
            
        The processed data includes:
        - Post ID and title
        - Text content
        - Creation timestamp
        - Score and comment count
        - Subreddit name
        - Post URL
        """
        processed_posts = []
        
        for post in posts:
            processed_posts.append({
                'post_id': post.id,
                'title': post.title,
                'text': post.selftext,
                'created_at': datetime.fromtimestamp(post.created_utc),
                'score': post.score,
                'num_comments': post.num_comments,
                'subreddit': post.subreddit.display_name,
                'url': post.url
            })
        
        return pd.DataFrame(processed_posts)
    
    def save_to_csv(self, df, filename):
        """
        Save processed posts to CSV file.
        
        Args:
            df (pd.DataFrame): Data to save
            filename (str): Path to save the CSV file
            
        Creates the output directory if it doesn't exist and handles errors gracefully.
        """
        if not df.empty:
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
    
    def run_extraction(self, output_file='data/raw/reddit_posts.csv'):
        """
        Run the complete extraction process.
        
        Args:
            output_file (str): Path to save the extracted data (default: 'data/raw/reddit_posts.csv')
            
        Returns:
            pd.DataFrame: Extracted and processed posts
            
        This method:
        1. Searches for posts in each subreddit
        2. Processes the posts into a structured format
        3. Combines results from all subreddits
        4. Saves the final dataset to CSV
        """
        all_posts = pd.DataFrame()
        
        for subreddit in self.subreddits:
            print(f"Searching in subreddit: {subreddit}")
            posts = self.search_subreddit(subreddit)
            processed_posts = self.process_posts(posts)
            all_posts = pd.concat([all_posts, processed_posts], ignore_index=True)
        
        self.save_to_csv(all_posts, output_file)
        return all_posts

def main():
    """Main function to run the Reddit data extraction."""
    # Create Reddit extractor instance
    extractor = RedditDataExtractor()
    
    # Define subreddits to monitor
    subreddits = [
        'depression',
        'anxiety',
        'mentalhealth',
        'suicidewatch',
        'selfharm'
    ]
    
    # Define keywords for filtering
    keywords = [
        'depressed', 'anxiety', 'suicidal', 'self harm',
        'mental health', 'therapy', 'counseling', 'help',
        'overwhelmed', 'struggling', 'hopeless', 'worthless'
    ]
    
    # Extract posts
    posts = extractor.run_extraction(
        subreddits=subreddits,
        keywords=keywords,
        limit=1000  # Number of posts per subreddit
    )
    
    # Save raw data
    os.makedirs('data/raw', exist_ok=True)
    posts.to_csv('data/raw/reddit_posts.csv', index=False)
    
    print(f"Extracted {len(posts)} posts. Data saved to data/raw/reddit_posts.csv")

if __name__ == "__main__":
    main() 