from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer


st.set_page_config(
    page_title="Leaf | Book recommendations",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
BOOKS_FILE = ROOT / "Books (1).csv"
RATINGS_FILE = ROOT / "Ratings (1).csv"
USERS_FILE = ROOT / "Users (2).csv"


def clean_text(value: object) -> str:
    value = html.unescape(str(value or "unknown")).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", value)).strip()


def author_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "unknown").lower())


@st.cache_data(show_spinner="Preparing the book collection and ratings…")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    for path in (BOOKS_FILE, RATINGS_FILE, USERS_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Could not find {path.name} next to app.py")

    books = pd.read_csv(BOOKS_FILE, encoding="latin-1", low_memory=False)
    ratings = pd.read_csv(RATINGS_FILE, encoding="latin-1", low_memory=False)
    books.columns = [str(c).strip().lower().replace("-", "_") for c in books.columns]
    ratings.columns = [str(c).strip().lower().replace("-", "_") for c in ratings.columns]
    books = books.rename(columns={"isbn": "isbn", "book_title": "title", "book_author": "author", "year_of_publication": "year", "publisher": "publisher"})
    ratings = ratings.rename(columns={"user_id": "user_id", "isbn": "isbn", "book_rating": "rating"})

    # Keep image URLs from the source file for the book cards.
    img_col = next((c for c in ("image_url_m", "image_url_l", "image_url_s") if c in books.columns), None)
    keep = [c for c in ("isbn", "title", "author", "year", "publisher", img_col) if c and c in books.columns]
    books = books[keep].copy()
    if img_col:
        books = books.rename(columns={img_col: "cover"})
    else:
        books["cover"] = ""
    books["isbn"] = books["isbn"].astype(str).str.strip()
    books["title"] = books["title"].fillna("Untitled").astype(str).map(html.unescape)
    books["author"] = books["author"].fillna("Unknown").astype(str).map(html.unescape)
    books["publisher"] = books["publisher"].fillna("Unknown").astype(str).map(html.unescape)
    books["year"] = pd.to_numeric(books.get("year"), errors="coerce")
    books.loc[~books["year"].between(1000, 2026), "year"] = np.nan
    books["year"] = books["year"].astype("Int64")
    books = books.drop_duplicates("isbn").reset_index(drop=True)

    ratings = ratings[["user_id", "isbn", "rating"]].copy()
    ratings["isbn"] = ratings["isbn"].astype(str).str.strip()
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["user_id", "rating"])
    ratings["user_id"] = pd.to_numeric(ratings["user_id"], errors="coerce")
    ratings = ratings.dropna(subset=["user_id"])
    ratings["user_id"] = ratings["user_id"].astype("int64")
    ratings = ratings[(ratings["rating"] > 0) & ratings["isbn"].isin(books["isbn"])]

    # Match the notebook's repeated minimum-activity filtering.
    model_ratings = ratings
    for _ in range(10):
        user_counts = model_ratings["user_id"].value_counts()
        model_ratings = model_ratings[model_ratings["user_id"].isin(user_counts[user_counts >= 6].index)]
        book_counts = model_ratings["isbn"].value_counts()
        model_ratings = model_ratings[model_ratings["isbn"].isin(book_counts[book_counts >= 8].index)]
    model_ratings = model_ratings.reset_index(drop=True)
    catalog = books[books["isbn"].isin(model_ratings["isbn"].unique())].copy().reset_index(drop=True)

    stats = model_ratings.groupby("isbn")["rating"].agg(rating_count="count", average_rating="mean").reset_index()
    c = float(model_ratings["rating"].mean())
    m = float(stats["rating_count"].quantile(0.90))
    stats["weighted_rating"] = (stats["rating_count"] / (stats["rating_count"] + m) * stats["average_rating"]
                                 + m / (stats["rating_count"] + m) * c)
    catalog = catalog.merge(stats, on="isbn", how="left")
    catalog["rating_count"] = catalog["rating_count"].fillna(0).astype(int)
    catalog["average_rating"] = catalog["average_rating"].fillna(c)
    catalog["weighted_rating"] = catalog["weighted_rating"].fillna(c)

    author = catalog["author"].map(author_token)
    publisher = catalog["publisher"].map(author_token)
    catalog["tags"] = author + " " + author + " " + publisher + " " + catalog["title"].map(clean_text)
    return catalog, model_ratings, int(books.shape[0]), int(ratings.shape[0])


@st.cache_resource(show_spinner="Learning book similarities…")
def build_index(tags: tuple[str, ...]):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=40000, min_df=1)
    matrix = vectorizer.fit_transform(tags)
    return vectorizer, matrix


def recommend_for_user(user_id: int, catalog: pd.DataFrame, ratings: pd.DataFrame, matrix, count: int = 12):
    history = ratings[ratings["user_id"] == user_id]
    idx = catalog.reset_index(drop=True)
    isbn_to_idx = pd.Series(idx.index, index=idx["isbn"]).to_dict()
    history = history[history["isbn"].isin(isbn_to_idx)]
    seen = set(history["isbn"])
    if history.empty:
        return pd.DataFrame(), 0
    user_mean = float(history["rating"].mean())
    seed_rows = np.array([isbn_to_idx[x] for x in history["isbn"]])
    weights = np.maximum(history["rating"].to_numpy(dtype=float) - user_mean, 0)
    if not np.any(weights):
        weights = np.maximum(history["rating"].to_numpy(dtype=float) - 5.0, 0.1)
    profile = matrix[seed_rows].multiply(weights[:, None]).sum(axis=0)
    profile = np.asarray(profile).ravel()
    scores = np.asarray(matrix @ profile).ravel()
    out = idx.copy()
    out["match_score"] = scores
    out = out[~out["isbn"].isin(seen)]
    out = out.sort_values(["match_score", "weighted_rating"], ascending=False)
    out = out.drop_duplicates("title").head(count)
    return out, len(history)


def recommend_from_titles(titles: list[str], catalog: pd.DataFrame, matrix, count: int = 12):
    idx = catalog.reset_index(drop=True)
    title_to_idx = {title: i for i, title in enumerate(idx["title"])}
    rows = [title_to_idx[t] for t in titles if t in title_to_idx]
    if not rows:
        return pd.DataFrame()
    profile = matrix[rows].mean(axis=0)
    scores = np.asarray(matrix @ np.asarray(profile).ravel()).ravel()
    out = idx.copy()
    out["match_score"] = scores
    out = out[~out["title"].isin(titles)].sort_values(["match_score", "weighted_rating"], ascending=False)
    return out.drop_duplicates("title").head(count)


def render_book(book: pd.Series, score_label: str = ""):
    with st.container(border=True):
        cover, info = st.columns([1, 3])
        image = str(book.get("cover", ""))
        if image.startswith("http"):
            cover.image(image, use_container_width=True)
        else:
            cover.markdown("<div class='cover-placeholder'>📖</div>", unsafe_allow_html=True)
        info.markdown(f"**{book['title']}**")
        info.caption(f"{book['author']} · {book['year'] if pd.notna(book['year']) else 'Year unknown'}")
        rating = float(book.get("average_rating", 0))
        count = int(book.get("rating_count", 0))
        info.markdown(f"⭐ **{rating:.1f}** <span class='muted'>from {count:,} ratings</span>", unsafe_allow_html=True)
        if score_label:
            info.caption(score_label)


def render_books(data: pd.DataFrame, score_label: str = ""):
    if data.empty:
        st.info("No books to show for this selection.")
        return
    for start in range(0, len(data), 3):
        cols = st.columns(3)
        for col, (_, book) in zip(cols, data.iloc[start:start + 3].iterrows()):
            with col:
                render_book(book, score_label)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
.stApp {background: #f7f5ef; color: #20251f;}
[data-testid="stSidebar"] {background: #e9eee5;}
h1, h2, h3 {font-family: 'Playfair Display', Georgia, serif !important; color: #223d32;}
.hero {padding: 2.2rem 2.5rem; border-radius: 22px; background: linear-gradient(115deg,#1e4035,#38634e); color: #f7f5ef; margin: .4rem 0 1.5rem;}
.hero h1 {color: #fff; font-size: 3rem; margin: 0;}
.hero p {font-size: 1.08rem; color: #e0eadf; margin: .6rem 0 0;}
.eyebrow {letter-spacing: .16em; text-transform: uppercase; font-size: .72rem; color: #c9d9c3; font-weight: 700;}
.cover-placeholder {height: 135px; background: #e9eee5; border-radius: 8px; display:flex; align-items:center; justify-content:center; font-size: 2.3rem;}
.muted {color: #788078; font-size: .82rem;}
[data-testid="stMetric"] {background:white; padding:1rem; border-radius:14px; border:1px solid #e6e7df;}
div[data-testid="stVerticalBlockBorderWrapper"] {background: #fffefa; border-color:#e6e7df;}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 📚 Leaf")
page = st.sidebar.radio("Explore", ["For you", "Similar books", "About the project"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("A book discovery demo built from the Book Recommendation System project.")

try:
    catalog, ratings, total_books, total_explicit = load_data()
except Exception as exc:
    st.error(f"The app could not prepare the project data: {exc}")
    st.stop()

vectorizer, matrix = build_index(tuple(catalog["tags"].fillna("").astype(str)))

if page == "For you":
    st.markdown("<div class='hero'><div class='eyebrow'>Find your next favorite</div><h1>Stories worth getting lost in.</h1><p>Recommendations shaped by the books readers rate highly.</p></div>", unsafe_allow_html=True)
    st.subheader("Your reading shelf")
    mode = st.radio("Choose how to personalize", ["Use a reader ID", "Pick books I like"], horizontal=True)
    if mode == "Use a reader ID":
        active_users = ratings["user_id"].value_counts()
        examples = active_users.head(5).index.tolist()
        default_id = int(examples[0]) if examples else 0
        user_id = st.number_input("Enter a reader ID from the project dataset", min_value=1, value=default_id, step=1, help="The notebook uses readers who have at least six explicit ratings.")
        recommendations, history_count = recommend_for_user(int(user_id), catalog, ratings, matrix)
        if recommendations.empty:
            st.info("That reader has no usable history in the trained recommendation cohort. Try a sample reader below or choose books you like.")
            st.caption("Sample reader IDs: " + " · ".join(map(str, examples)))
            popular = catalog.sort_values(["weighted_rating", "rating_count"], ascending=False).drop_duplicates("title").head(12)
            st.markdown("### Popular with readers")
            render_books(popular)
        else:
            st.caption(f"Personalized from {history_count} explicit ratings in this dataset.")
            st.markdown("### Recommended for you")
            render_books(recommendations, "Content match · based on your higher rated books")
    else:
        choices = catalog.sort_values("weighted_rating", ascending=False).drop_duplicates("title")["title"].tolist()
        selected = st.multiselect("Choose a few books you already enjoy", choices, max_selections=8, placeholder="Search titles, for example Harry Potter")
        if len(selected) >= 1:
            recommendations = recommend_from_titles(selected, catalog, matrix)
            st.markdown("### Because you like those books")
            render_books(recommendations, "Content match · author, publisher and title")
        else:
            st.markdown("### Popular with readers")
            popular = catalog.sort_values(["weighted_rating", "rating_count"], ascending=False).drop_duplicates("title").head(12)
            render_books(popular)

elif page == "Similar books":
    st.markdown("<div class='hero'><div class='eyebrow'>Explore a title</div><h1>Find your next read.</h1><p>Choose a book and discover titles with similar authors, publishers, and words in their titles.</p></div>", unsafe_allow_html=True)
    choices = catalog.sort_values("weighted_rating", ascending=False).drop_duplicates("title")["title"].tolist()
    selected = st.selectbox("Choose a book", choices, index=None, placeholder="Search for a title…")
    if selected:
        matches = recommend_from_titles([selected], catalog, matrix, count=9)
        st.markdown(f"### Similar to *{selected}*")
        render_books(matches, "Content similarity")

else:
    st.markdown("<div class='hero'><div class='eyebrow'>Project 708</div><h1>How Leaf recommends.</h1><p>A book recommender built from the Book-Crossing ratings dataset.</p></div>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Books in source data", f"{total_books:,}")
    b.metric("Explicit ratings", f"{total_explicit:,}")
    c.metric("Books in recommender", f"{len(catalog):,}")
    st.markdown("### Recommendation approach")
    st.write("The main recommender uses content-based filtering. It represents each book with TF-IDF features from its author, publisher, and title. For a known reader, it builds a profile from books they rated above their own average, then ranks unseen books by similarity to that profile.")
    st.write("Readers without a matching rating history can build a taste profile by selecting books they already like. If there is no history or selection, the app shows popular books using a weighted rating that accounts for both rating average and number of ratings.")
    st.markdown("### Notebook evaluation")
    results = pd.DataFrame({"Model": ["Content-Based", "Item-Based CF", "User-Based CF", "SVD", "Popularity-Based"], "RMSE": [1.568, 1.588, 1.589, 1.602, 1.742], "Hit Rate@10": [0.147, 0.063, 0.068, 0.061, 0.048]})
    st.dataframe(results, hide_index=True, use_container_width=True)
    st.caption("These figures come from the notebook's held-out evaluation. A predicted content match is a ranking signal, not a promised rating.")
    st.markdown("### Data notes")
    st.write("Zero ratings are treated as implicit interactions and excluded from rating-based recommendations. To reduce sparsity, the notebook's model cohort retains readers with at least six explicit ratings and books with at least eight.")
