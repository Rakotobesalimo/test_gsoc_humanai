"""
Twitter Data Extraction Module

This module provides functionality to extract crisis-related data from Twitter using the Twitter API v2.
It includes features for authentication, rate limiting, and data processing.

Key Features:
- OAuth 2.0 authentication with Twitter API
- Rate limiting with exponential backoff
- Search for crisis-related tweets
- Processing of tweet data into structured format
- Saving results to CSV files

Dependencies:
- tweepy: For Twitter API access
- pandas: For data manipulation
- dotenv: For environment variable management
"""

import tweepy
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import time
from typing import Optional, Dict, Any

class TwitterDataExtractor:
    """
    A class for extracting crisis-related data from Twitter.
    
    This class handles authentication with the Twitter API, rate limiting,
    and data extraction for crisis-related content. It includes methods for
    searching tweets, processing the results, and saving them to CSV files.
    
    Attributes:
        api_key (str): Twitter API key
        api_secret (str): Twitter API secret
        access_token (str): Twitter access token
        access_token_secret (str): Twitter access token secret
        bearer_token (str): Twitter bearer token for v2 API
        rate_limit_window (int): Time window for rate limiting in seconds
        requests_per_window (int): Maximum requests allowed per time window
        request_timestamps (list): Timestamps of recent requests
        last_request_time (float): Timestamp of the last request
        min_request_interval (int): Minimum seconds between requests
        keywords (list): List of crisis-related keywords to search for
        client (tweepy.Client): Authenticated Twitter API client
    """
    
    def __init__(self):
        """
        Initialize the TwitterDataExtractor with API credentials and settings.
        
        Loads credentials from environment variables and sets up rate limiting parameters.
        Verifies that all required credentials are present and authenticates with Twitter API.
        
        Raises:
            ValueError: If any required API credentials are missing
            Exception: If authentication with Twitter API fails
        """
        load_dotenv()
        self.api_key = os.getenv('TWITTER_API_KEY')
        self.api_secret = os.getenv('TWITTER_API_SECRET')
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        
        # More conservative rate limiting settings
        self.rate_limit_window = 15 * 60  # 15 minutes in seconds
        self.requests_per_window = 100  # Reduced from 180 to be more conservative
        self.request_timestamps = []
        self.last_request_time = 0
        self.min_request_interval = 3  # Minimum seconds between requests
        
        # Verify credentials are loaded
        if not all([self.api_key, self.api_secret, self.access_token, self.access_token_secret, self.bearer_token]):
            raise ValueError("Missing Twitter API credentials. Please check your .env file.")
        
        # Crisis-related keywords
        self.keywords = [
            "depressed", "anxiety", "suicidal", "mental health",
            "overwhelmed", "help needed", "crisis", "addiction",
            "therapy", "counseling", "support", "mental illness",
            "psychiatric", "emotional distress", "self harm"
        ]
        
        self.client = self._authenticate()
        if not self.client:
            raise Exception("Failed to authenticate with Twitter API")
    
    def _wait_for_rate_limit(self):
        """
        Implement more conservative rate limiting logic.
        
        This method ensures that API requests are made within Twitter's rate limits
        by tracking request timestamps and implementing delays when necessary.
        
        The rate limiting strategy includes:
        - Tracking requests within a 15-minute window
        - Limiting to 100 requests per window
        - Enforcing a minimum 3-second interval between requests
        - Implementing exponential backoff for rate limit errors
        """
        current_time = time.time()
        
        # Remove timestamps older than the rate limit window
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                 if current_time - ts < self.rate_limit_window]
        
        # Check if we've hit the rate limit
        if len(self.request_timestamps) >= self.requests_per_window:
            oldest_ts = self.request_timestamps[0]
            wait_time = self.rate_limit_window - (current_time - oldest_ts)
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                return self._wait_for_rate_limit()
        
        # Ensure minimum time between requests
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last_request
            print(f"Waiting {wait_time:.1f} seconds between requests...")
            time.sleep(wait_time)
        
        # Update timestamps
        self.request_timestamps.append(current_time)
        self.last_request_time = current_time
    
    def _handle_rate_limit_error(self, e: tweepy.TweepyException) -> bool:
        """
        Handle rate limit errors with exponential backoff.
        
        Args:
            e (tweepy.TweepyException): The rate limit error exception
            
        Returns:
            bool: True if the error was handled, False otherwise
            
        This method implements a sophisticated rate limit handling strategy:
        1. First attempts to use Twitter's provided reset time
        2. Falls back to exponential backoff if reset time is not available
        3. Caps maximum wait time at 5 minutes
        """
        if e.response is not None and e.response.status_code == 429:
            try:
                reset_time = int(e.response.headers.get('x-rate-limit-reset', 0))
                wait_time = max(reset_time - time.time(), 0)
                
                if wait_time > 0:
                    print(f"Rate limit exceeded. Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    return True
            except (ValueError, TypeError):
                # If we can't parse the reset time, use exponential backoff
                wait_time = min(60 * (2 ** len(self.request_timestamps)), 300)  # Max 5 minutes
                print(f"Rate limit exceeded. Waiting {wait_time:.1f} seconds (exponential backoff)...")
                time.sleep(wait_time)
                return True
        return False
    
    def _authenticate(self):
        """
        Authenticate with Twitter API using OAuth 2.0.
        
        Returns:
            tweepy.Client: Authenticated Twitter API client or None if authentication fails
            
        This method:
        1. Creates a client with both bearer token and OAuth credentials
        2. Verifies authentication by making a test request
        3. Handles authentication errors gracefully
        """
        try:
            print("Attempting to authenticate with Twitter API...")
            
            # Create the client with bearer token for v2 API
            client = tweepy.Client(
                bearer_token=self.bearer_token,
                consumer_key=self.api_key,
                consumer_secret=self.api_secret,
                access_token=self.access_token,
                access_token_secret=self.access_token_secret
            )
            
            # Verify authentication by making a test request
            try:
                client.get_me()
                print("Authentication successful!")
                return client
            except tweepy.TweepyException as e:
                print(f"Authentication verification failed: {str(e)}")
                return None
                
        except Exception as e:
            print(f"Authentication failed: {str(e)}")
            return None
    
    def search_tweets(self, query: str, max_results: int = 100) -> Optional[Dict[str, Any]]:
        """
        Search for tweets containing the specified query.
        
        Args:
            query (str): Search query string
            max_results (int): Maximum number of results to return (default: 100)
            
        Returns:
            Optional[Dict[str, Any]]: Search results or None if search fails
            
        This method:
        1. Implements rate limiting
        2. Uses the search_recent_tweets endpoint
        3. Includes tweet metadata and user information
        4. Handles errors and retries with exponential backoff
        """
        if not self.client:
            print("No authenticated client available")
            return None
            
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Implement rate limiting
                self._wait_for_rate_limit()
                
                # Use the search_recent_tweets endpoint with proper authentication
                tweets = self.client.search_recent_tweets(
                    query=query,
                    max_results=max_results,
                    tweet_fields=['created_at', 'public_metrics', 'geo', 'lang'],
                    expansions=['author_id'],
                    user_fields=['username', 'location']
                )
                
                if not tweets:
                    print(f"No tweets found for query: {query}")
                    return None
                    
                return tweets
                
            except tweepy.TweepyException as e:
                if self._handle_rate_limit_error(e):
                    retry_count += 1
                    continue
                else:
                    print(f"Error searching tweets: {str(e)}")
                    return None
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                return None
        
        print(f"Max retries ({max_retries}) exceeded for query: {query}")
        return None
    
    def process_tweets(self, tweets):
        """
        Process tweets into a structured format.
        
        Args:
            tweets: Raw tweet data from Twitter API
            
        Returns:
            pd.DataFrame: Processed tweets with relevant fields
            
        The processed data includes:
        - Tweet ID and text
        - Creation timestamp
        - Engagement metrics (likes, retweets, replies, quotes)
        - Language
        - User information (username, location)
        """
        if not tweets or not hasattr(tweets, 'data'):
            return pd.DataFrame()
        
        processed_tweets = []
        for tweet in tweets.data:
            try:
                processed_tweet = {
                    'tweet_id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'likes': tweet.public_metrics.get('like_count', 0),
                    'retweets': tweet.public_metrics.get('retweet_count', 0),
                    'replies': tweet.public_metrics.get('reply_count', 0),
                    'quotes': tweet.public_metrics.get('quote_count', 0),
                    'language': getattr(tweet, 'lang', None)
                }
                
                # Add user information if available
                if hasattr(tweets, 'includes') and 'users' in tweets.includes:
                    user = next((u for u in tweets.includes['users'] if u.id == tweet.author_id), None)
                    if user:
                        processed_tweet['username'] = user.username
                        processed_tweet['user_location'] = user.location
                
                processed_tweets.append(processed_tweet)
            except Exception as e:
                print(f"Error processing tweet {tweet.id}: {str(e)}")
                continue
        
        return pd.DataFrame(processed_tweets)
    
    def save_to_csv(self, df, filename):
        """
        Save processed tweets to CSV file.
        
        Args:
            df (pd.DataFrame): Data to save
            filename (str): Path to save the CSV file
            
        Creates the output directory if it doesn't exist and handles errors gracefully.
        """
        if df is None or df.empty:
            print("No data to save")
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
        except Exception as e:
            print(f"Error saving data to {filename}: {str(e)}")
    
    def run_extraction(self, output_file: str = 'data/raw/tweets.csv') -> pd.DataFrame:
        """
        Run the complete extraction process.
        
        Args:
            output_file (str): Path to save the extracted data (default: 'data/raw/tweets.csv')
            
        Returns:
            pd.DataFrame: Extracted and processed tweets
            
        This method:
        1. Searches for tweets using each keyword
        2. Processes the tweets into a structured format
        3. Combines results from all keywords
        4. Saves the final dataset to CSV
        5. Includes delays between keyword searches to avoid rate limits
        """
        if not self.client:
            print("Cannot run extraction: No authenticated client available")
            return pd.DataFrame()
            
        all_tweets = pd.DataFrame()
        
        for keyword in self.keywords:
            print(f"\nSearching for tweets containing: {keyword}")
            tweets = self.search_tweets(keyword)
            if tweets:
                processed_tweets = self.process_tweets(tweets)
                if not processed_tweets.empty:
                    all_tweets = pd.concat([all_tweets, processed_tweets], ignore_index=True)
                    print(f"Found {len(processed_tweets)} tweets for keyword: {keyword}")
                else:
                    print(f"No tweets found for keyword: {keyword}")
            else:
                print(f"Failed to search tweets for keyword: {keyword}")
            
            # Add a longer delay between keywords
            time.sleep(10)  # Increased to 10 seconds between keywords
        
        if not all_tweets.empty:
            self.save_to_csv(all_tweets, output_file)
        else:
            print("No tweets were collected")
            
        return all_tweets

if __name__ == "__main__":
    try:
        extractor = TwitterDataExtractor()
        extractor.run_extraction()
    except Exception as e:
        print(f"Error running extraction: {str(e)}") 