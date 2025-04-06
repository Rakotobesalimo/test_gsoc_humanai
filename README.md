# GSoC Candidate Assessment: Crisis Detection and Analysis

## Project Overview
This project focuses on developing a system to analyze social media content for crisis-related discussions, with a particular emphasis on mental health distress, substance use, and suicidality. 
The system will extract, process, and analyze social media data, apply sentiment analysis and NLP techniques, and visualize crisis-related trends on an interactive map.

## Project Structure
```
test_gsoc_humanai/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_extraction/
│   │   └── reddit_api.py
│   ├── preprocessing/
│   │   ├── text_cleaning.py
│   │   └── data_formatter.py
│   ├── analysis/
│   │   ├── sentiment_analysis.py
│   │   └── risk_classification.py
│   └── visualization/
│       ├── geocoding.py
│       └── mapping.py
└── notebooks/
    ├── data_exploration.ipynb
    └── visualization.ipynb
```

## Tasks

### Task 1: Social Media Data Extraction & Preprocessing
- Extract posts  Reddit API
- Filter posts using predefined keywords
- Store data in structured format (CSV)
- Clean text data (remove stopwords, emojis, special characters)

### Task 2: Sentiment & Crisis Risk Classification
- Apply VADER or TextBlob for sentiment analysis
- Use TF-IDF or Word Embeddings for risk detection
- Categorize posts into risk levels (High, Moderate, Low)

### Task 3: Crisis Geolocation & Mapping
- Extract location metadata
- Implement geocoding
- Generate heatmap visualization
- Display top crisis locations

## Requirements
- Python 3.8+
- Reddit API credentials 
- Required Python packages (see requirements.txt)

## Deliverables
1. Data extraction and preprocessing scripts
2. Sentiment analysis and risk classification system
3. Geospatial visualization of crisis patterns
4. Documentation and analysis reports 