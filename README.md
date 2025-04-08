# GSoC Candidate Assessment: Crisis Detection and Analysis

## Project Overview
This project is a GSoC candidate assessment focused on developing a system to analyze social media content for crisis-related discussions, with a particular emphasis on mental health distress, substance use, and suicidality. The system extracts, processes, and analyzes social media data, applies sentiment analysis and NLP techniques, and visualizes crisis-related trends on an interactive map.

## Assessment Objectives
✅ Extract, process, and analyze crisis-related discussions from social media
✅ Apply sentiment analysis and NLP techniques to assess high-risk content
✅ Geocode and visualize crisis-related trends on a basic interactive map

## Project Structure
```
test_gsoc_humanai/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                    # Raw social media data
│   └── processed/              # Processed and analyzed data
├── src/
│   ├── data_extraction/        # Twitter and Reddit API integration
│   │   ├── twitter_api.py      # Twitter data extraction
│   │   └── reddit_api.py       # Reddit data extraction
│   ├── preprocessing/          # Text cleaning and formatting
│   │   ├── text_cleaning.py    # Text preprocessing utilities
│   │   └── data_formatter.py   # Data formatting utilities
│   ├── analysis/               # Sentiment and risk analysis
│   │   ├── sentiment_analysis.py  # Sentiment classification
│   │   └── risk_classification.py # Risk level assessment
│   └── visualization/          # Mapping and visualization
│       ├── geocoding.py        # Location extraction and geocoding
│       └── mapping.py          # Interactive map generation
└── notebooks/
    ├── data_exploration.ipynb  # Data analysis notebook
    └── visualization.ipynb     # Visualization notebook
```

## Tasks and Implementation

### Task 1: Social Media Data Extraction & Preprocessing
- Extract posts from Twitter/X API or Reddit API
- Filter posts using predefined keywords (e.g., "depressed," "addiction help," "overwhelmed," "suicidal")
- Store Post ID, Timestamp, Content, and Engagement Metrics in structured format
- Clean text data (remove stopwords, emojis, special characters)

### Task 2: Sentiment & Crisis Risk Classification
- Apply VADER (for Twitter) or TextBlob for sentiment classification
- Use TF-IDF or Word Embeddings for risk detection
- Categorize posts into Risk Levels:
  - High-Risk: Direct crisis language
  - Moderate Concern: Seeking help, discussing struggles
  - Low Concern: General mental health discussions

### Task 3: Crisis Geolocation & Mapping
- Extract location metadata from:
  - Geotagged posts
  - NLP-based place recognition
- Generate heatmap of crisis-related posts
- Display top 5 locations with highest crisis discussions

## Setup and Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd test_gsoc_humanai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with your API credentials:
```env
# Twitter API credentials
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret
TWITTER_BEARER_TOKEN=your_bearer_token

# Reddit API credentials
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent
```
## Running the Code using jupyter notebook 
```
run on the all.ipynb
```

## Running the Python Code

### Data Extraction and Preprocessing
```bash
# Run Twitter data extraction
python src/data_extraction/twitter_api.py

# Run Reddit data extraction
python src/data_extraction/reddit_api.py

# Preprocess the data
python src/preprocessing/text_cleaning.py
```

### Sentiment and Risk Analysis
```bash
# Run sentiment analysis and risk classification
python src/analysis/sentiment_analysis.py

```

### Visualization
```bash
# Generate geocoded data
python src/visualization/geocoding.py

# Create interactive maps
python src/visualization/mapping.py
```

### Jupyter Notebooks
```bash
# Launch Jupyter
jupyter notebook

# Open and run:
# - notebooks/data_exploration.ipynb
# - notebooks/visualization.ipynb
```

## Output Files
- Raw data: `data/raw/*.csv`
- Processed data: `data/processed/*.csv`
- Analysis results: `data/processed/analyzed_*.csv`
- Visualizations: `output/maps/*.html`
- Reports: `output/reports/*.txt`

## Requirements
- Python 3.8+
- Twitter API credentials
- Reddit API credentials 
- Required Python packages (see requirements.txt)

## Deliverables
1. Data extraction and preprocessing scripts
2. Sentiment analysis and risk classification system
3. Geospatial visualization of crisis patterns
4. Documentation and analysis reports
