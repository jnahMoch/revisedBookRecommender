import html
import re
from pathlib import Path

import gradio as gr
import pandas as pd


APP_DIR = Path(__file__).parent
DATA_FILES = [
    "books_with_categories_genre_and_emotion.csv",
    "books_with_categories_and_genre.csv",
    "cleaned_books_dataset_binary_category.csv",
    "cleaned_books_dataset.csv",
]
EMOTION_COLUMNS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


def clean_text(value, fallback="Unknown"):
    if pd.isna(value):
        return fallback
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return fallback
    return text


def load_dataset():
    for file_name in DATA_FILES:
        file_path = APP_DIR / file_name
        if file_path.exists():
            df = pd.read_csv(file_path)
            df.attrs["source_file"] = file_name
            return df
    empty = pd.DataFrame()
    empty.attrs["source_file"] = "No CSV found"
    return empty


BOOKS_DF = load_dataset()
SOURCE_FILE = BOOKS_DF.attrs.get("source_file", "No CSV found")


def build_search_index(df):
    if df.empty:
        return pd.Series(dtype=str)

    fields = [
        "title",
        "authors",
        "categories",
        "genre",
        "description",
        "cleaned_description",
        "simple_categories_binary",
    ]
    available = [field for field in fields if field in df.columns]
    if not available:
        return pd.Series([""] * len(df), index=df.index)

    return (
        df[available]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )


SEARCH_INDEX = build_search_index(BOOKS_DF)


def dropdown_values(column):
    if BOOKS_DF.empty or column not in BOOKS_DF.columns:
        return ["All"]
    values = sorted(v for v in BOOKS_DF[column].dropna().astype(str).unique() if v.strip())
    return ["All"] + values


COLLECTIONS = dropdown_values("simple_categories_binary")
GENRES = dropdown_values("genre")
MOODS = ["All"] + [column.title() for column in EMOTION_COLUMNS if column in BOOKS_DF.columns]
EMOTION_ALIASES = {
    "angry": "anger",
    "anger": "anger",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "scary": "fear",
    "afraid": "fear",
    "joy": "joy",
    "happy": "joy",
    "happiness": "joy",
    "neutral": "neutral",
    "sad": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "surprising": "surprise",
}


def truncate(value, limit=280):
    text = clean_text(value, "")
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def book_cover_html(row):
    title = html.escape(clean_text(row.get("title"), "Untitled"))
    author = html.escape(clean_text(row.get("authors")))
    thumb = clean_text(row.get("thumbnail"), "")
    if thumb.startswith("http"):
        return f'<img src="{html.escape(thumb)}" alt="{title} book cover" loading="lazy">'

    initials = "".join(part[:1] for part in re.split(r"\s+", title)[:2]).upper() or "BI"
    return f"""
    <div class="cover-fallback" aria-label="{title}">
      <span>{html.escape(initials)}</span>
      <small>{author}</small>
    </div>
    """


def dominant_emotion_key(row):
    available = [column for column in EMOTION_COLUMNS if column in row.index]
    if not available:
        return ""

    scores = pd.to_numeric(row[available], errors="coerce").fillna(0)
    if scores.empty or scores.max() <= 0:
        return ""
    return scores.idxmax()


def dominant_emotion_series(df):
    available = [column for column in EMOTION_COLUMNS if column in df.columns]
    if not available or df.empty:
        return pd.Series("", index=df.index)

    scores = df[available].apply(pd.to_numeric, errors="coerce").fillna(0)
    return scores.idxmax(axis=1).where(scores.max(axis=1) > 0, "")


def dominant_emotion(row):
    emotion = dominant_emotion_key(row)
    return emotion.title() if emotion else "Unknown"


def render_cards(rows):
    if rows.empty:
        return '<div class="empty">No matching books found. Try a broader search.</div>'

    cards = []
    for _, row in rows.iterrows():
        title = html.escape(clean_text(row.get("title"), "Untitled"))
        authors = html.escape(clean_text(row.get("authors")))
        genre = html.escape(clean_text(row.get("genre", row.get("categories")), "Uncategorized"))
        collection = html.escape(clean_text(row.get("simple_categories_binary"), "All"))
        rating = html.escape(clean_text(row.get("average_rating"), "Not rated"))
        pages = html.escape(clean_text(row.get("num_pages"), ""))
        emotion = html.escape(dominant_emotion(row))
        description = html.escape(truncate(row.get("description", row.get("cleaned_description")), 320))
        pages_text = f"<span>{pages} pages</span>" if pages and pages != "Unknown" else ""
        emotion_text = f'<span class="emotion-badge">{emotion}</span>' if emotion != "Unknown" else ""
        cards.append(
            f"""
            <article class="book-card">
              <div class="book-cover">{book_cover_html(row)}</div>
              <div class="book-body">
                <div class="book-kicker">{collection} / {genre}</div>
                <h3>{title}</h3>
                <p class="author">{authors}</p>
                <p class="description">{description}</p>
                <div class="book-meta">
                  <span>Rating {rating}</span>
                  {pages_text}
                  {emotion_text}
                </div>
              </div>
            </article>
            """
        )
    return '<div class="results-grid">' + "\n".join(cards) + "</div>"


def rank_books(query, collection, genre, mood, minimum_rating):
    if BOOKS_DF.empty:
        return BOOKS_DF

    df = BOOKS_DF.copy()
    score = pd.Series(0.0, index=df.index)

    if collection != "All" and "simple_categories_binary" in df.columns:
        matches = df["simple_categories_binary"].fillna("").astype(str).str.lower() == collection.lower()
        df = df[matches]
        score = score.loc[df.index]

    if genre != "All" and "genre" in df.columns:
        matches = df["genre"].fillna("").astype(str).str.lower() == genre.lower()
        df = df[matches]
        score = score.loc[df.index]

    if minimum_rating and "average_rating" in df.columns:
        ratings = pd.to_numeric(df["average_rating"], errors="coerce").fillna(0)
        df = df[ratings >= float(minimum_rating)]
        score = score.loc[df.index]

    terms = [term.lower() for term in re.findall(r"[\w']+", query or "") if len(term) > 1]
    query_emotions = [EMOTION_ALIASES[term] for term in terms if term in EMOTION_ALIASES and EMOTION_ALIASES[term] in df.columns]
    text_terms = [term for term in terms if term not in EMOTION_ALIASES]
    requested_emotions = list(dict.fromkeys(query_emotions))
    if mood != "All" and mood.lower() in df.columns:
        requested_emotions.append(mood.lower())

    if requested_emotions and not df.empty:
        dominant = dominant_emotion_series(df)
        emotion_matches = dominant.isin(requested_emotions)
        if emotion_matches.any():
            df = df[emotion_matches]
            score = score.loc[df.index]

    if text_terms and not df.empty:
        haystack = SEARCH_INDEX.loc[df.index]
        for term in text_terms:
            score.loc[df.index] += haystack.str.contains(re.escape(term), regex=True).astype(float)
            if "title" in df.columns:
                score.loc[df.index] += (
                    df["title"].fillna("").astype(str).str.lower().str.contains(re.escape(term), regex=True).astype(float) * 2
                )
            if "authors" in df.columns:
                score.loc[df.index] += (
                    df["authors"].fillna("").astype(str).str.lower().str.contains(re.escape(term), regex=True).astype(float) * 1.25
                )

    for emotion_column in requested_emotions:
        if emotion_column not in df.columns:
            continue
        emotion_scores = pd.to_numeric(df[emotion_column], errors="coerce").fillna(0)
        score.loc[df.index] += emotion_scores.rank(pct=True).fillna(0) * 4

    if "average_rating" in df.columns and not df.empty:
        ratings = pd.to_numeric(df["average_rating"], errors="coerce").fillna(0)
        score.loc[df.index] += ratings.rank(pct=True).fillna(0) * 0.75

    ranked = df.assign(_score=score.loc[df.index]).sort_values(
        by=["_score", "average_rating"] if "average_rating" in df.columns else ["_score"],
        ascending=False,
    )

    if text_terms:
        ranked = ranked[ranked["_score"] > 0]
    return ranked.drop(columns=["_score"], errors="ignore")


def search_books(query, collection, genre, mood):
    ranked = rank_books(query, collection, genre, mood, minimum_rating=0)
    return render_cards(ranked.head(6))


def featured_books():
    ranked = rank_books("", "All", "All", "All", 0)
    return render_cards(ranked.head(6))


CUSTOM_CSS = """
:root {
  --ink:#f7efe5;
  --paper:#090b0f;
  --muted:#b9aa98;
  --line:rgba(255,255,255,.14);
  --panel:#151a20;
  --panel-2:#101318;
  --accent:#d8a24b;
  --accent-2:#78b8a5;
  --accent-3:#c86f54;
  --field:#0d1117;
  --shadow:0 22px 70px rgba(0, 0, 0, .42);
}

html, body, #root, .gradio-container, .main, .app {
  background:var(--paper) !important;
  color:var(--ink) !important;
  font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

body {
  min-height:100vh;
  background:
    radial-gradient(circle at 18% 0%, rgba(216,162,75,.16), transparent 34%),
    radial-gradient(circle at 88% 12%, rgba(120,184,165,.12), transparent 32%),
    linear-gradient(180deg, #090b0f 0%, #11151b 58%, #090b0f 100%) !important;
}

.gradio-container {
  max-width:1360px !important;
  margin:0 auto !important;
  padding:28px 28px 56px !important;
  min-height:100vh !important;
  background:transparent !important;
}

.prose, .markdown, .gr-form, .gr-box, .block, .input-container, .form, .panel, .gap {
  border-color:var(--line) !important;
  background:transparent !important;
}

.wrap, .contain, .form {
  background:transparent !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .dropdown-container,
.gradio-container .wrap,
.gradio-container .container {
  background:var(--field) !important;
  color:var(--ink) !important;
  border-color:rgba(255,255,255,.11) !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
  color:rgba(247,239,229,.48) !important;
}

.hero {
  min-height:390px;
  display:grid;
  align-items:center;
  padding:46px;
  border:1px solid rgba(255,255,255,.16);
  border-radius:8px;
  background:
    linear-gradient(90deg, rgba(9,11,15,.94), rgba(9,11,15,.76) 46%, rgba(9,11,15,.26)),
    url("https://images.unsplash.com/photo-1521587760476-6c12a4b040da?auto=format&fit=crop&w=1600&q=80");
  background-size:cover;
  background-position:center;
  box-shadow:var(--shadow);
  overflow:hidden;
}

.eyebrow {
  margin:0 0 12px;
  color:var(--accent);
  font-size:12px;
  font-weight:800;
  letter-spacing:1.8px;
  text-transform:uppercase;
}

.hero h1 {
  max-width:760px;
  margin:0;
  color:var(--ink);
  font-family:Georgia, "Times New Roman", serif;
  font-size:clamp(38px, 6vw, 74px);
  line-height:.98;
  letter-spacing:0;
}

#search-panel {
  margin-top:22px;
  padding:22px;
  border:1px solid rgba(255,255,255,.14);
  border-radius:8px;
  background:linear-gradient(180deg, rgba(21,26,32,.96), rgba(15,18,23,.96)) !important;
  box-shadow:0 16px 52px rgba(0, 0, 0, .32);
}

#search-panel label, #search-panel .wrap label {
  color:var(--ink) !important;
  font-weight:700 !important;
}

button.primary {
  background:linear-gradient(135deg, #d8a24b, #c86f54) !important;
  border-color:transparent !important;
  color:#101318 !important;
  border-radius:6px !important;
  font-weight:800 !important;
}

button.primary:hover {
  filter:brightness(1.06);
}

.section-title {
  margin:34px 0 14px;
  font-family:Georgia, "Times New Roman", serif;
  font-size:30px;
  line-height:1.1;
  color:#fff4df !important;
}

.results-grid {
  display:grid;
  grid-template-columns:repeat(2, minmax(0, 1fr));
  gap:14px;
}

.results-grid,
.results-grid * {
  color:inherit;
}

.book-card {
  display:grid;
  grid-template-columns:170px minmax(0, 1fr);
  gap:22px;
  min-height:268px;
  padding:18px;
  border:1px solid var(--line);
  border-radius:8px;
  background:var(--panel);
  color:#f7efe5 !important;
  box-shadow:0 14px 34px rgba(0, 0, 0, .24);
}

.book-cover {
  width:170px;
  min-height:238px;
  border-radius:6px;
  overflow:hidden;
  background:#1e242b;
}

.book-cover img {
  width:100%;
  height:100%;
  object-fit:cover;
  display:block;
}

.cover-fallback {
  height:100%;
  min-height:238px;
  display:flex;
  flex-direction:column;
  justify-content:space-between;
  padding:14px;
  background:linear-gradient(145deg, #254f55, #9f5745 82%);
  color:#fffaf2;
}

.cover-fallback span {
  font-family:Georgia, "Times New Roman", serif;
  font-size:32px;
}

.cover-fallback small {
  font-size:10px;
  line-height:1.3;
}

.book-kicker {
  color:var(--accent);
  font-size:11px;
  font-weight:800;
  letter-spacing:1.1px;
  text-transform:uppercase;
}

.book-card h3 {
  margin:7px 0 3px;
  font-family:Georgia, "Times New Roman", serif;
  font-size:24px;
  line-height:1.1;
  letter-spacing:0;
  color:#fff4df !important;
}

.author {
  margin:0 0 10px;
  color:#f0d9bd !important;
  font-size:13px;
}

.description {
  margin:0;
  color:#f7efe5 !important;
  font-size:14px;
  line-height:1.55;
}

.book-meta {
  display:flex;
  flex-wrap:wrap;
  gap:8px;
  margin-top:12px;
}

.book-meta span {
  padding:5px 8px;
  border-radius:999px;
  background:rgba(216,162,75,.15);
  color:#f5d59b;
  font-size:12px;
}

.book-meta .emotion-badge {
  background:rgba(120,184,165,.18);
  color:#a7ead8;
}

.empty {
  padding:28px;
  border:1px dashed var(--line);
  border-radius:8px;
  color:var(--muted);
  background:rgba(255,255,255,.05);
}

@media (max-width:820px) {
  .gradio-container {
    padding:14px !important;
  }

  .hero {
    padding:24px;
    min-height:310px;
    background:
      linear-gradient(180deg, rgba(9,11,15,.94), rgba(9,11,15,.76)),
      url("https://images.unsplash.com/photo-1521587760476-6c12a4b040da?auto=format&fit=crop&w=1000&q=80");
    background-size:cover;
  }

  .results-grid {
    grid-template-columns:1fr;
  }

  .book-card {
    grid-template-columns:124px minmax(0, 1fr);
    gap:16px;
  }

  .book-cover {
    width:124px;
    min-height:176px;
  }

  .cover-fallback {
    min-height:176px;
  }
}
"""


HEADER_HTML = """
<section class="hero">
  <div>
    <p class="eyebrow">The Obsidian Index</p>
    <h1>Find your next favorite book.</h1>
  </div>
</section>
"""


with gr.Blocks(title="The Obsidian Index") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Group(elem_id="search-panel"):
        with gr.Row():
            query = gr.Textbox(
                placeholder="Try: sad mystery, joyful children books, philosophy, Jane Austen...",
                label="Search",
                scale=3,
            )
            search_btn = gr.Button("Search Books", variant="primary", scale=1)

        with gr.Row():
            collection = gr.Dropdown(COLLECTIONS, value="All", label="Collection")
            genre = gr.Dropdown(GENRES, value="All", label="Genre")
            mood = gr.Dropdown(MOODS, value="All", label="Mood / Emotion")

    gr.HTML('<h2 class="section-title">Recommendations</h2>')
    results = gr.HTML(value=featured_books())

    search_inputs = [query, collection, genre, mood]
    search_btn.click(search_books, inputs=search_inputs, outputs=results)
    query.submit(search_books, inputs=search_inputs, outputs=results)


if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS, theme=gr.themes.Soft())
