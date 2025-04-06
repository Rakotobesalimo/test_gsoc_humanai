"""
Crisis Map Visualization Module

This module provides tools for creating interactive maps to visualize crisis-related data
from social media platforms. It uses Folium to create HTML-based maps with various
visualization layers including heatmaps, markers, and risk level indicators.

Key Features:
- Heatmap visualization of crisis-related posts
- Location markers with customizable popups
- Risk level visualization with color-coded markers
- Top locations visualization
- Interactive HTML maps that can be opened in web browsers

Dependencies:
- folium: For creating interactive maps
- pandas: For data handling
- webbrowser: For opening maps in default browser

Example Usage:
    # Create a visualizer instance
    visualizer = CrisisMapVisualizer()
    
    # Load your data (e.g., from Reddit analysis)
    df = pd.read_csv('data/processed/analyzed_reddit.csv')
    
    # Create a heatmap
    visualizer.create_base_map()
    visualizer.add_heatmap(df)
    visualizer.save_map('reddit_heatmap.html')
    
    # Create a risk level map
    visualizer.create_base_map()
    visualizer.add_risk_level_layer(df)
    visualizer.save_map('reddit_risk_map.html')
    
    # Show top locations
    visualizer.show_top_locations(df, n=10)
    visualizer.save_map('reddit_top_locations.html')
"""

import folium
from folium.plugins import HeatMap
import pandas as pd
from typing import List, Optional
import webbrowser
import os
import numpy as np

class CrisisMapVisualizer:
    """
    A class for creating interactive maps to visualize crisis-related data.
    
    This class provides methods to create various types of visualizations:
    - Base maps with customizable center and zoom
    - Heatmaps showing density of crisis-related posts
    - Markers with popup information
    - Risk level visualizations with color-coded markers
    - Top locations visualization
    
    Attributes:
        map (folium.Map): The current map instance
        default_location (List[float]): Default center coordinates [lat, lon]
        default_zoom (int): Default zoom level
    """
    
    def __init__(self):
        """
        Initialize the CrisisMapVisualizer with default settings.
        
        Sets up default map center and zoom level for new map instances.
        """
        self.map = None
        self.default_location = [20, 0]  # Default center of the map
        self.default_zoom = 2
    
    def create_base_map(self, location: Optional[List[float]] = None, zoom: Optional[int] = None):
        """
        Create a base map centered at the specified location.
        
        Args:
            location (Optional[List[float]]): Center coordinates [lat, lon]. 
                Defaults to [20, 0] if not specified.
            zoom (Optional[int]): Initial zoom level. Defaults to 2 if not specified.
            
        Returns:
            folium.Map: The created map instance
        """
        if location is None:
            location = self.default_location
        if zoom is None:
            zoom = self.default_zoom
            
        self.map = folium.Map(location=location, zoom_start=zoom)
        return self.map
    
    def _is_valid_location(self, lat: float, lon: float) -> bool:
        """
        Check if location coordinates are valid.
        
        Args:
            lat (float): Latitude value
            lon (float): Longitude value
            
        Returns:
            bool: True if coordinates are valid, False otherwise
            
        A location is considered valid if:
        - Both latitude and longitude are not NaN
        - Both are numeric values
        - Latitude is between -90 and 90
        - Longitude is between -180 and 180
        """
        return (
            not pd.isna(lat) and not pd.isna(lon) and
            isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and
            -90 <= lat <= 90 and -180 <= lon <= 180
        )
    
    def add_heatmap(self, df: pd.DataFrame, latitude_col: str = 'latitude', 
                   longitude_col: str = 'longitude', weight_col: Optional[str] = None):
        """
        Add a heatmap layer to the map showing the density of crisis-related posts.
        
        Args:
            df (pd.DataFrame): DataFrame containing location data
            latitude_col (str): Column name for latitude values
            longitude_col (str): Column name for longitude values
            weight_col (Optional[str]): Column name for weight values (e.g., risk level)
            
        Returns:
            folium.Map: The map with heatmap layer added
            
        Note:
            - Invalid coordinates are automatically filtered out
            - If weight_col is provided, the heatmap intensity will be weighted
            - Prints a warning if no valid coordinates are found
        """
        if self.map is None:
            self.create_base_map()
        
        # Filter out rows with invalid coordinates
        valid_coords = df[
            df.apply(lambda row: self._is_valid_location(row[latitude_col], row[longitude_col]), axis=1)
        ]
        
        if weight_col:
            # Create weighted heatmap data
            heat_data = [[row[latitude_col], row[longitude_col], row[weight_col]] 
                        for _, row in valid_coords.iterrows()]
        else:
            # Create unweighted heatmap data
            heat_data = [[row[latitude_col], row[longitude_col]] 
                        for _, row in valid_coords.iterrows()]
        
        if heat_data:  # Only add heatmap if there's valid data
            HeatMap(heat_data).add_to(self.map)
        else:
            print("Warning: No valid coordinates found for heatmap")
        
        return self.map
    
    def add_markers(self, df: pd.DataFrame, latitude_col: str = 'latitude', 
                   longitude_col: str = 'longitude', popup_col: Optional[str] = None):
        """
        Add markers to the map with optional popup text.
        
        Args:
            df (pd.DataFrame): DataFrame containing location data
            latitude_col (str): Column name for latitude values
            longitude_col (str): Column name for longitude values
            popup_col (Optional[str]): Column name for popup text
            
        Returns:
            folium.Map: The map with markers added
            
        Note:
            - Invalid coordinates are automatically filtered out
            - Markers are red circles with radius 5
            - Popup text is shown when clicking on markers
        """
        if self.map is None:
            self.create_base_map()
        
        # Filter out rows with invalid coordinates
        valid_coords = df[
            df.apply(lambda row: self._is_valid_location(row[latitude_col], row[longitude_col]), axis=1)
        ]
        
        for _, row in valid_coords.iterrows():
            popup_text = str(row[popup_col]) if popup_col and not pd.isna(row[popup_col]) else None
            
            folium.CircleMarker(
                location=[row[latitude_col], row[longitude_col]],
                radius=5,
                popup=popup_text,
                color='red',
                fill=True,
                fill_color='red'
            ).add_to(self.map)
        
        return self.map
    
    def add_risk_level_layer(self, df: pd.DataFrame, latitude_col: str = 'latitude',
                           longitude_col: str = 'longitude', risk_col: str = 'risk_level'):
        """
        Add markers colored by risk level.
        
        Args:
            df (pd.DataFrame): DataFrame containing location and risk data
            latitude_col (str): Column name for latitude values
            longitude_col (str): Column name for longitude values
            risk_col (str): Column name for risk level values
            
        Returns:
            folium.Map: The map with risk level markers added
            
        Note:
            - Risk levels are color-coded:
                - high: red
                - moderate: orange
                - low: green
                - unknown: gray
            - Invalid coordinates are automatically filtered out
            - Markers show risk level in popup
        """
        if self.map is None:
            self.create_base_map()
        
        # Define colors for different risk levels
        risk_colors = {
            'high': 'red',
            'moderate': 'orange',
            'low': 'green'
        }
        
        # Filter out rows with invalid coordinates
        valid_coords = df[
            df.apply(lambda row: self._is_valid_location(row[latitude_col], row[longitude_col]), axis=1)
        ]
        
        for _, row in valid_coords.iterrows():
            risk_level = row[risk_col] if not pd.isna(row[risk_col]) else 'unknown'
            color = risk_colors.get(risk_level, 'gray')
            
            folium.CircleMarker(
                location=[row[latitude_col], row[longitude_col]],
                radius=5,
                popup=f"Risk Level: {risk_level}",
                color=color,
                fill=True,
                fill_color=color
            ).add_to(self.map)
        
        return self.map
    
    def save_map(self, output_file: str = 'crisis_heatmap.html'):
        """
        Save the current map to an HTML file.
        
        Args:
            output_file (str): Path to save the HTML file
            
        Creates the output directory if it doesn't exist and opens the map
        in the default web browser.
        """
        if self.map is None:
            print("No map to save. Create a map first using create_base_map().")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the map
        self.map.save(output_file)
        print(f"Map saved to {output_file}")
        
        # Open in default browser
        webbrowser.open('file://' + os.path.abspath(output_file))
    
    def show_top_locations(self, df: pd.DataFrame, n: int = 5,
                         latitude_col: str = 'latitude', longitude_col: str = 'longitude',
                         location_col: str = 'extracted_location'):
        """
        Create a map showing the top n locations by post count.
        
        Args:
            df (pd.DataFrame): DataFrame containing location data
            n (int): Number of top locations to show (default: 5)
            latitude_col (str): Column name for latitude values
            longitude_col (str): Column name for longitude values
            location_col (str): Column name for location names
            
        Returns:
            folium.Map: The map with top location markers
            
        Note:
            - Locations are sorted by post count
            - Markers show location name and post count in popup
            - Invalid coordinates are automatically filtered out
        """
        if self.map is None:
            self.create_base_map()
        
        # Get top locations by post count
        top_locations = df[location_col].value_counts().head(n)
        
        # Filter out rows with invalid coordinates
        valid_coords = df[
            df.apply(lambda row: self._is_valid_location(row[latitude_col], row[longitude_col]), axis=1)
        ]
        
        # Add markers for top locations
        for location in top_locations.index:
            location_data = valid_coords[valid_coords[location_col] == location]
            if not location_data.empty:
                # Use the first occurrence's coordinates
                row = location_data.iloc[0]
                count = top_locations[location]
                
                folium.CircleMarker(
                    location=[row[latitude_col], row[longitude_col]],
                    radius=5,
                    popup=f"{location}<br>Posts: {count}",
                    color='blue',
                    fill=True,
                    fill_color='blue'
                ).add_to(self.map)
        
        return self.map

# if __name__ == "__main__":
#     # Example usage with sample data
#     visualizer = CrisisMapVisualizer()
    
#     # Sample data
#     sample_data = pd.DataFrame({
#         'latitude': [40.7128, 34.0522, 41.8781, 51.5074, 43.6532],
#         'longitude': [-74.0060, -118.2437, -87.6298, -0.1278, -79.3832],
#         'extracted_location': ['New York', 'Los Angeles', 'Chicago', 'London', 'Toronto'],
#         'risk_level': ['high', 'moderate', 'low', 'high', 'moderate']
#     })
    
#     # Create and display different types of maps
#     print("Creating heatmap...")
#     visualizer.create_base_map()
#     visualizer.add_heatmap(sample_data)
#     visualizer.save_map('heatmap.html')
    
#     print("\nCreating risk level map...")
#     visualizer.create_base_map()
#     visualizer.add_risk_level_layer(sample_data)
#     visualizer.save_map('risk_map.html')
    
#     print("\nCreating top locations map...")
#     visualizer.show_top_locations(sample_data)
#     visualizer.save_map('top_locations.html') 