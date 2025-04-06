"""
Geocoding and Location Extraction Module

This module provides functionality for extracting and geocoding locations from text data.
It uses the Nominatim geocoding service to convert location names to coordinates.

Key Features:
- Location extraction from text using regex patterns
- Geocoding of extracted locations
- Caching of geocoding results
- Batch processing of DataFrame columns
- Top locations analysis

Dependencies:
- geopy: For geocoding services
- pandas: For data manipulation
- typing: For type hints
"""

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import pandas as pd
import re
import time
from typing import Dict, List, Optional, Tuple

class LocationExtractor:
    """
    A class for extracting and geocoding locations from text.
    
    This class provides methods for identifying location mentions in text,
    converting them to coordinates, and processing location data in bulk.
    It includes caching to improve performance and reduce API calls.
    
    Attributes:
        geolocator (Nominatim): Geocoding service client
        location_cache (dict): Cache for geocoding results
    """
    
    def __init__(self):
        """
        Initialize the LocationExtractor with geocoding service.
        
        Sets up the Nominatim geocoder with a custom user agent and
        initializes the location cache.
        """
        self.geolocator = Nominatim(user_agent="crisis_analysis_app")
        self.location_cache = {}  # Cache for geocoding results
    
    def extract_location_from_text(self, text: str) -> Optional[str]:
        """
        Extract potential location mentions from text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Optional[str]: Extracted location name or None if no location found
            
        Uses regex patterns to identify location mentions in the format:
        - "in [Location]"
        - "from [Location]"
        - "at [Location]"
        - "[Location]"
        
        Only returns locations with 3 or fewer words to avoid false positives.
        """
        if not isinstance(text, str):
            return None
        
        # Common location patterns
        patterns = [
            r'in\s+([A-Za-z\s]+(?:City|Town|Village|County|State|Country))',
            r'from\s+([A-Za-z\s]+(?:City|Town|Village|County|State|Country))',
            r'at\s+([A-Za-z\s]+(?:City|Town|Village|County|State|Country))',
            r'([A-Za-z\s]+(?:City|Town|Village|County|State|Country))',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                # Basic validation
                if len(location.split()) <= 3:  # Avoid long strings that are likely not locations
                    return location
        
        return None
    
    def geocode_location(self, location: str) -> Optional[Tuple[float, float]]:
        """
        Convert location string to coordinates.
        
        Args:
            location (str): Location name to geocode
            
        Returns:
            Optional[Tuple[float, float]]: (latitude, longitude) tuple or None if geocoding fails
            
        This method:
        1. Checks the cache for existing results
        2. Makes API calls with rate limiting
        3. Caches successful results
        4. Handles geocoding errors gracefully
        """
        if not location:
            return None
        
        # Check cache first
        if location in self.location_cache:
            return self.location_cache[location]
        
        try:
            # Add delay to avoid rate limiting
            time.sleep(1)
            location_data = self.geolocator.geocode(location)
            
            if location_data:
                coords = (location_data.latitude, location_data.longitude)
                self.location_cache[location] = coords
                return coords
        except GeocoderTimedOut:
            print(f"Geocoding timed out for location: {location}")
        except Exception as e:
            print(f"Error geocoding location {location}: {str(e)}")
        
        return None
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Process DataFrame to extract and geocode locations.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Name of the column containing text to analyze
            
        Returns:
            pd.DataFrame: Original DataFrame with location data added
            
        Creates new columns:
        - extracted_location: Name of the extracted location
        - coordinates: Tuple of (latitude, longitude)
        - latitude: Latitude value
        - longitude: Longitude value
        """
        df_processed = df.copy()
        
        # Extract locations
        df_processed['extracted_location'] = df_processed[text_column].apply(
            self.extract_location_from_text
        )
        
        # Geocode locations
        df_processed['coordinates'] = df_processed['extracted_location'].apply(
            self.geocode_location
        )
        
        # Split coordinates into latitude and longitude
        df_processed['latitude'] = df_processed['coordinates'].apply(
            lambda x: x[0] if x else None
        )
        df_processed['longitude'] = df_processed['coordinates'].apply(
            lambda x: x[1] if x else None
        )
        
        return df_processed
    
    def get_top_locations(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """
        Get top n locations by post count.
        
        Args:
            df (pd.DataFrame): DataFrame containing location data
            n (int): Number of top locations to return (default: 5)
            
        Returns:
            pd.DataFrame: DataFrame with location names and post counts
        """
        location_counts = df['extracted_location'].value_counts().head(n)
        return pd.DataFrame({
            'location': location_counts.index,
            'post_count': location_counts.values
        })
    
    def save_geocoded_data(self, df: pd.DataFrame, output_file: str):
        """
        Save geocoded data to CSV file.
        
        Args:
            df (pd.DataFrame): DataFrame containing geocoded data
            output_file (str): Path to save the CSV file
        """
        df.to_csv(output_file, index=False)
        print(f"Geocoded data saved to {output_file}")

# if __name__ == "__main__":
#     # Example usage
#     extractor = LocationExtractor()
    
#     # Sample data
#     sample_data = pd.DataFrame({
#         'text': [
#             "Feeling overwhelmed in New York City",
#             "Need help in Los Angeles",
#             "Struggling with anxiety in Chicago",
#             "Depressed in London",
#             "Looking for support in Toronto"
#         ]
#     })
    
#     # Process the data
#     processed_data = extractor.process_dataframe(sample_data, 'text')
#     print("\nProcessed Data:")
#     print(processed_data[['text', 'extracted_location', 'latitude', 'longitude']])
    
#     print("\nTop Locations:")
#     print(extractor.get_top_locations(processed_data)) 