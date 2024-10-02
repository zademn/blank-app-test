import praw
import streamlit as st
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id=st.secrets["CLIENT_ID"],
    client_secret=st.secrets["CLIENT_SECRET"],
    user_agent=st.secrets["USER_AGENT"],
)

# Initialize BERT model
model = SentenceTransformer("distilbert-base-nli-mean-tokens")


def search_reddit(
    query: str,
    subreddit: str = None,
    limit: int = 100,
    start_date: datetime = None,
    end_date: datetime = None,
) -> List[Dict[str, Any]]:
    if subreddit:
        subreddit = reddit.subreddit(subreddit)
        results = subreddit.search(
            query, sort="new", limit=limit, time_filter="all")
    else:
        results = reddit.subreddit("all").search(
            query, sort="new", limit=limit, time_filter="all"
        )

    comments = []
    for post in results:
        post_date = datetime.fromtimestamp(post.created_utc)
        if (start_date is None or post_date >= start_date) and (
            end_date is None or post_date <= end_date
        ):
            post.comments.replace_more(limit=0)
            for comment in post.comments.list():
                comment_date = datetime.fromtimestamp(comment.created_utc)
                if (start_date is None or comment_date >= start_date) and (
                    end_date is None or comment_date <= end_date
                ):
                    comments.append(
                        {
                            "text": comment.body,
                            "url": f"https://www.reddit.com{comment.permalink}",
                            "score": comment.score,
                            "date": comment_date,
                        }
                    )

    return comments


@st.cache_resource
def get_bert_model():
    return SentenceTransformer("distilbert-base-nli-mean-tokens")


def analyze_comments(
    comments: List[Dict[str, Any]], query: str
) -> List[Dict[str, Any]]:
    if not comments:
        return []

    texts = [comment["text"] for comment in comments]
    texts.append(query)  # Add the query to the texts for embedding

    # Get BERT embeddings
    model = get_bert_model()
    embeddings = model.encode(texts, show_progress_bar=False)

    # Compute cosine similarity
    query_embedding = embeddings[-1]  # The last embedding is the query
    comment_embeddings = embeddings[
        :-1
    ]  # All embeddings except the last one are comments
    similarities = cosine_similarity([query_embedding], comment_embeddings)[0]

    # Add similarity scores to comments
    for i, comment in enumerate(comments):
        comment["similarity"] = similarities[i]

    # Sort comments by similarity score
    comments.sort(key=lambda x: x["similarity"], reverse=True)

    return comments


st.title("Reddit Comment Scraper and BERT Analyzer")

query = st.text_input("Enter your search query:")
subreddit = st.text_input("Enter subreddit (optional):")
limit = st.slider(
    "Number of posts to search:", min_value=10, max_value=500, value=100, step=10
)

# Date range selection
st.subheader("Date Range")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "Start date", value=datetime.now() - timedelta(days=30))
with col2:
    end_date = st.date_input("End date", value=datetime.now())

if start_date > end_date:
    st.error("Error: End date must be after start date.")

if st.button("Search and Analyze"):
    if query:
        with st.spinner("Searching Reddit and analyzing comments..."):
            comments = search_reddit(
                query,
                subreddit,
                limit,
                start_date=datetime.combine(start_date, datetime.min.time()),
                end_date=datetime.combine(end_date, datetime.max.time()),
            )

            if comments:
                analyzed_comments = analyze_comments(comments, query)

                st.success(
                    f"Found and analyzed {len(analyzed_comments)} comments")

                # Display top comments
                st.subheader("Top Comments by Relevance")
                for i, comment in enumerate(analyzed_comments[:10], 1):
                    st.write(
                        f"#{i} Ranked Comment (Similarity: {comment['similarity']:.4f})"
                    )
                    st.write(
                        comment["text"][:200] + "..."
                        if len(comment["text"]) > 200
                        else comment["text"]
                    )
                    st.write(f"Score: {comment['score']}")
                    st.write(
                        f"Date: {comment['date'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"[Link to comment]({comment['url']})")
                    st.write("---")
            else:
                st.warning("No comments found for the given search criteria.")
    else:
        st.warning("Please enter a search query.")

st.sidebar.title("About")
st.sidebar.info(
    "This app scrapes Reddit comments based on your search query and date range, "
    "analyzes them using BERT embeddings, and displays the most relevant comments."
)
