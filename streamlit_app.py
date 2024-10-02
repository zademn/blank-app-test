import praw
import streamlit as st
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id=st.secrets["CLIENT_ID"],
    client_secret=st.secrets["CLIENT_SECRET"],
    user_agent=st.secrets["USER_AGENT"],
)


@st.cache_resource
def get_bert_model():
    return SentenceTransformer("distilbert-base-nli-mean-tokens")


# Initialize BERT model
bert_model = get_bert_model()


def search_reddit(
    query: str,
    subreddits: List[str] = None,
    limit: int = 100,
    start_date: datetime = None,
    end_date: datetime = None,
    search_type: str = "comments",
) -> List[Dict[str, Any]]:
    results = []
    if subreddits:
        subreddit_list = subreddits
    else:
        subreddit_list = ["all"]

    for subreddit_name in subreddit_list:
        subreddit = reddit.subreddit(subreddit_name)
        if search_type == "comments":
            search_results = subreddit.search(
                query, sort="new", limit=limit, time_filter="all"
            )
            for post in search_results:
                post_date = datetime.fromtimestamp(post.created_utc)
                if (start_date is None or post_date >= start_date) and (
                    end_date is None or post_date <= end_date
                ):
                    post.comments.replace_more(limit=0)
                    for comment in post.comments.list():
                        comment_date = datetime.fromtimestamp(
                            comment.created_utc)
                        if (start_date is None or comment_date >= start_date) and (
                            end_date is None or comment_date <= end_date
                        ):
                            results.append(
                                {
                                    "text": comment.body,
                                    "url": f"https://www.reddit.com{comment.permalink}",
                                    "score": comment.score,
                                    "date": comment_date,
                                    "subreddit": comment.subreddit.display_name,
                                }
                            )
        else:  # search_type == 'threads'
            search_results = subreddit.search(
                query, sort="new", limit=limit, time_filter="all"
            )
            for post in search_results:
                post_date = datetime.fromtimestamp(post.created_utc)
                if (start_date is None or post_date >= start_date) and (
                    end_date is None or post_date <= end_date
                ):
                    results.append(
                        {
                            "text": post.title + "\n" + post.selftext,
                            "url": f"https://www.reddit.com{post.permalink}",
                            "score": post.score,
                            "date": post_date,
                            "subreddit": post.subreddit.display_name,
                        }
                    )
    return results


def analyze_texts(texts: List[str], query: str, method: str = "bert") -> np.ndarray:
    print(query)
    for i, text in enumerate(texts):
        print(f"Text: {i}")
        print(text)
    if method == "bert":
        model = get_bert_model()
        embeddings = model.encode(texts + [query], show_progress_bar=False)
        query_embedding = embeddings[-1]
        text_embeddings = embeddings[:-1]
        similarities = cosine_similarity([query_embedding], text_embeddings)[0]
    else:  # method == 'tfidf'
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts + [query])
        query_vector = tfidf_matrix[-1]
        text_vectors = tfidf_matrix[:-1]
        similarities = cosine_similarity(query_vector, text_vectors)[0]
    return similarities


def analyze_results(
    results: List[Dict[str, Any]], query: str, method: str = "bert"
) -> List[Dict[str, Any]]:
    if not results:
        return []

    texts = [result["text"] for result in results]
    similarities = analyze_texts(texts, query, method)

    # Add similarity scores to results
    for i, result in enumerate(results):
        result["similarity"] = similarities[i]

    # Sort results by similarity score
    results.sort(key=lambda x: x["similarity"], reverse=True)

    return results


st.title("Enhanced Reddit Scraper and Analyzer")

col1, col2 = st.columns(2)

with col1:
    st.header("Thread Search")
    thread_query = st.text_input("Enter your thread search query:")
    thread_subreddits = st.text_input(
        "Enter subreddits for thread search (comma-separated, optional):"
    )
    thread_limit = st.slider(
        "Number of threads to search:", min_value=10, max_value=500, value=100, step=10
    )

with col2:
    st.header("Comment Search")
    comment_query = st.text_input("Enter your comment search query:")
    comment_subreddits = st.text_input(
        "Enter subreddits for comment search (comma-separated, optional):"
    )
    comment_limit = st.slider(
        "Number of posts to search for comments:",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
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

# Analysis method selection
analysis_method = st.radio("Select analysis method:", ("BERT", "TF-IDF"))

if st.button("Search and Analyze"):
    method = analysis_method.lower()

    if thread_query or comment_query:
        with st.spinner("Searching Reddit and analyzing results..."):
            col1, col2 = st.columns(2)

            with col1:
                if thread_query:
                    thread_subreddit_list = (
                        [s.strip()
                         for s in thread_subreddits.split(",") if s.strip()]
                        if thread_subreddits
                        else None
                    )
                    thread_results = search_reddit(
                        thread_query,
                        thread_subreddit_list,
                        thread_limit,
                        start_date=datetime.combine(
                            start_date, datetime.min.time()),
                        end_date=datetime.combine(
                            end_date, datetime.max.time()),
                        search_type="threads",
                    )
                    analyzed_threads = analyze_results(
                        thread_results, thread_query, method
                    )

                    st.subheader("Top Thread Results")
                    if analyzed_threads:
                        for i, thread in enumerate(analyzed_threads[:10], 1):
                            st.write(
                                f"#{i} Ranked Thread (Similarity: {thread['similarity']:.4f})"
                            )
                            st.write(f"Subreddit: r/{thread['subreddit']}")
                            st.write(
                                thread["text"][:200] + "..."
                                if len(thread["text"]) > 200
                                else thread["text"]
                            )
                            st.write(f"Score: {thread['score']}")
                            st.write(
                                f"Date: {thread['date'].strftime('%Y-%m-%d %H:%M:%S')}"
                            )
                            st.write(f"[Link to thread]({thread['url']})")
                            st.write("---")
                    else:
                        st.warning(
                            "No threads found for the given search criteria.")

            with col2:
                if comment_query:
                    comment_subreddit_list = (
                        [s.strip()
                         for s in comment_subreddits.split(",") if s.strip()]
                        if comment_subreddits
                        else None
                    )
                    comment_results = search_reddit(
                        comment_query,
                        comment_subreddit_list,
                        comment_limit,
                        start_date=datetime.combine(
                            start_date, datetime.min.time()),
                        end_date=datetime.combine(
                            end_date, datetime.max.time()),
                        search_type="comments",
                    )
                    analyzed_comments = analyze_results(
                        comment_results, comment_query, method
                    )

                    st.subheader("Top Comment Results")
                    if analyzed_comments:
                        for i, comment in enumerate(analyzed_comments[:10], 1):
                            st.write(
                                f"#{i} Ranked Comment (Similarity: {comment['similarity']:.4f})"
                            )
                            st.write(f"Subreddit: r/{comment['subreddit']}")
                            st.write(
                                comment["text"][:200] + "..."
                                if len(comment["text"]) > 200
                                else comment["text"]
                            )
                            st.write(f"Score: {comment['score']}")
                            st.write(
                                f"Date: {comment['date'].strftime('%Y-%m-%d %H:%M:%S')}"
                            )
                            st.write(f"[Link to comment]({comment['url']})")
                            st.write("---")
                    else:
                        st.warning(
                            "No comments found for the given search criteria.")
    else:
        st.warning("Please enter at least one search query (thread or comment).")

st.sidebar.title("About")
st.sidebar.info(
    "This app scrapes Reddit threads and comments based on your search queries, optional subreddits, and date range. "
    "It analyzes the results using either BERT embeddings or TF-IDF, and displays the most relevant threads and comments in separate columns."
)
