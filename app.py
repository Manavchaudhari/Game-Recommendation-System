import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Set the page configuration (MUST be the first Streamlit command)
st.set_page_config(page_title="Game Recommendation System", layout="wide")

# === BACKGROUND IMAGE INJECTION ===
video_html = '''
    <style>
    #myVideo {
      position: fixed;
      right: 0;
      bottom: 0;
      min-width: 100%; 
      min-height: 100%;
      z-index: -1;
      object-fit: cover;
    }

    .stApp {
      background: transparent;
    }
    </style>	

    <video autoplay muted loop id="myVideo">
      <source src="https://static.moewalls.com/videos/preview/2024/steam-delivery-girl-winter-sale-2024-preview.webm" type="video/webm">
      Your browser does not support HTML5 video.
    </video>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Inject custom styles
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("games_cleaned.csv")
    df['combined_features'] = (
        df['Tags'].fillna('') + ' ' +
        df['Genres'].fillna('') + ' ' +
        df['Categories'].fillna('')
    )
    return df

df_filtered = load_data()

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_filtered['combined_features'])
indices = pd.Series(df_filtered.index, index=df_filtered['Name'].str.lower()).drop_duplicates()

# Steam API fetch (ensure proper error handling)
@st.cache_data(show_spinner=False)
def get_steam_details(appid):
    try:
        url = f"https://store.steampowered.com/api/appdetails?appids={appid}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json().get(str(appid), {}).get("data", {})
            return {
                "name": data.get("name"),
                "short_description": data.get("short_description"),
                "header_image": data.get("header_image"),
                "genres": [g['description'] for g in data.get("genres", [])],
                "positive": data.get("recommendations", {}).get("total"),
                "price": data.get("price_overview", {}).get("final_formatted") if data.get("price_overview") else "Free"
            }
    except:
        return None
    return None

# Recommendation engine
def recommend(name, num=5):
    name = name.lower()
    if name not in indices:
        return None, pd.DataFrame()
    idx = indices[name]
    cosine_similarities = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[-(num + 1):-1][::-1]
    return name, df_filtered.iloc[similar_indices]

# === UI ===
st.title("🎮 Steam Game Recommendation System")

game_list = df_filtered['Name'].dropna().sort_values().tolist()
game_list.insert(0, "")
selected_game = st.selectbox("🎮 Type in game name:", game_list)

if selected_game:
    name, recs = recommend(selected_game)
    if recs.empty:
        st.error("No recommendations found.")
    else:
        st.subheader(f"🎯 Recommendations for '{name}':")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**Select a recommendation:**")
            for _, row in recs.iterrows():
                if st.button(row['Name'], key=f"select_{row['AppID']}"):
                    st.session_state['selected_game'] = row['Name']

        if 'selected_game' in st.session_state:
            selected_row = recs[recs['Name'] == st.session_state['selected_game']]
            if not selected_row.empty:
                game_row = selected_row.iloc[0]
                steam_info = get_steam_details(game_row['AppID'])

                with col2:
                    st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
                    st.markdown(f"## 🎮 {game_row['Name']}")
                    if steam_info:
                        if steam_info.get("header_image"):
                            st.image(steam_info["header_image"], use_container_width=True)
                        if steam_info.get("price"):
                            st.markdown(f"💵 **Price**: {steam_info['price']}")
                        if steam_info.get("positive") is not None:
                            st.markdown(f"👍 **Positive Reviews**: {steam_info['positive']}")
                        if steam_info.get("short_description"):
                            st.markdown(f"📝 {steam_info['short_description']}")
                        if steam_info.get("genres"):
                            st.markdown("🏷️ **Tags:** " + ", ".join(steam_info["genres"][:8]))
                    if isinstance(game_row['Steam URL'], str) and game_row['Steam URL'].startswith("http"):
                        st.markdown(
                            f'<a href="{game_row["Steam URL"]}" target="_blank">'
                            f'<button class="steam-button">Steam Page</button></a>',
                            unsafe_allow_html=True
                        )
                    st.markdown("</div>", unsafe_allow_html=True)
