# Steam Game Recommendation System

This is a **Steam Game Recommendation System** built with **Streamlit** and **scikit-learn**. The system provides personalized game recommendations based on user input, using content-based filtering (TF-IDF and cosine similarity). The application fetches game details from the Steam API to display additional information such as price, description, and reviews.

## Features

- **Game Search**: Type the name of a game, and get game recommendations based on the input.
- **Steam Details**: Fetches additional information from the Steam API (e.g., price, description, genres).
- **Recommendations**: Displays a list of recommended games based on similar tags, genres, and categories.
- **Responsive UI**: Built with Streamlit for an easy-to-use and interactive interface.

## Technologies Used

- **Streamlit**: For building the user interface.
- **Pandas**: For handling the dataset and data processing.
- **scikit-learn**: For machine learning and similarity calculations (TF-IDF vectorization and cosine similarity).
- **Requests**: For making API calls to the Steam API to fetch game details.

## Setup and Installation

### Requirements

- Python 3.7 or higher
- Streamlit
- pandas
- scikit-learn
- requests

### Installation

1. Clone this repository:

```bash
   git clone https://github.com/yourusername/steam-game-recommendation.git
```
   Dataset: https://www.kaggle.com/datasets/fronkongames/steam-games-dataset
   
2. Move to the project direcotry

```bash
cd steam-game-recommendation
```
3. Install the requirements
```bash
pip install -r requirements.txt
```
4. Run the app
```bash
streamlit run app.py
```

## Credits

- Background Video: ["Steam Delivery Girl Winter Sale 2024"](https://moewalls.com/others/steam-delivery-girl-winter-sale-2024-live-wallpaper/) by [X @nemupanart](https://x.com/nemupanart)
