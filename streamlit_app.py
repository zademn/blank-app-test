import praw
import streamlit as st
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any
from transformers import pipeline


if "analyzed_threads" not in st.session_state:
    st.session_state.analyzed_threads = None
if "analyzed_comments" not in st.session_state:
    st.session_state.analyzed_comments = None


# Initialize Reddit API client
reddit = praw.Reddit(
    client_id=st.secrets["CLIENT_ID"],
    client_secret=st.secrets["CLIENT_SECRET"],
    user_agent=st.secrets["USER_AGENT"],
)


@st.cache_resource
def get_bert_model():
    return SentenceTransformer("distilbert-base-nli-mean-tokens")


@st.cache_resource()
def get_llm():
    return pipeline(
        "text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=1000
    )


@st.dialog("Generate Response")
def generate_response(context):
    pipe = get_llm()
    # Generate response using the LLM
    query = st.text_input("Query")
    if st.button("Submit"):
        with st.spinner("Generating response"):
            message = [
                {"role": "user", "content": f"{context}\n\n{query}"},
            ]
            response = pipe(message)
            st.write(response[0]["generated_text"][1]["content"])
    # return response["generated_text"]


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
                        comment_date = datetime.fromtimestamp(comment.created_utc)
                        if (start_date is None or comment_date >= start_date) and (
                            end_date is None or comment_date <= end_date
                        ):
                            # Get the comment chain
                            comment_chain = []
                            current_comment = comment
                            while current_comment is not None:
                                comment_chain.append(
                                    {
                                        "body": current_comment.body,
                                        "author": (
                                            current_comment.author.name
                                            if current_comment.author
                                            else "[deleted]"
                                        ),
                                        "score": current_comment.score,
                                    }
                                )
                                current_comment = (
                                    current_comment.parent()
                                    if current_comment.parent_id.startswith("t1_")
                                    else None
                                )
                            comment_chain.reverse()  # Reverse to get chronological order

                            results.append(
                                {
                                    "text": comment.body,
                                    "url": f"https://www.reddit.com{comment.permalink}",
                                    "score": comment.score,
                                    "date": comment_date,
                                    "subreddit": comment.subreddit.display_name,
                                    "comment_chain": comment_chain,
                                    "post_title": post.title,
                                    "post_text": post.selftext,
                                    "post_author": (
                                        post.author.name if post.author else "[deleted]"
                                    ),
                                    "post_score": post.score,
                                    "post_date": post_date,
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
    start_date = st.date_input("Start date", value=datetime.now() - timedelta(days=30))
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
            if thread_query:
                thread_subreddit_list = (
                    [s.strip() for s in thread_subreddits.split(",") if s.strip()]
                    if thread_subreddits
                    else None
                )
                thread_results = search_reddit(
                    thread_query,
                    thread_subreddit_list,
                    thread_limit,
                    start_date=datetime.combine(start_date, datetime.min.time()),
                    end_date=datetime.combine(end_date, datetime.max.time()),
                    search_type="threads",
                )
                st.session_state.analyzed_threads = analyze_results(
                    thread_results, thread_query, method
                )

            if comment_query:
                comment_subreddit_list = (
                    [s.strip() for s in comment_subreddits.split(",") if s.strip()]
                    if comment_subreddits
                    else None
                )
                comment_results = search_reddit(
                    comment_query,
                    comment_subreddit_list,
                    comment_limit,
                    start_date=datetime.combine(start_date, datetime.min.time()),
                    end_date=datetime.combine(end_date, datetime.max.time()),
                    search_type="comments",
                )
                st.session_state.analyzed_comments = analyze_results(
                    comment_results, comment_query, method
                )
    else:
        st.warning("Please enter at least one search query (thread or comment).")

# Display threads (put this outside the if st.button() block)
# Display results in two columns
if st.session_state.analyzed_threads or st.session_state.analyzed_comments:
    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.analyzed_threads:
            st.subheader("Top Thread Results")
            for i, thread in enumerate(st.session_state.analyzed_threads[:10], 1):
                with st.expander(
                    f"#{i} Ranked Thread (Similarity: {thread['similarity']:.4f})"
                ):
                    st.write(f"**Subreddit:** r/{thread['subreddit']}")
                    st.write(
                        thread["text"][:200] + "..."
                        if len(thread["text"]) > 200
                        else thread["text"]
                    )
                    st.write(f"**Score:** {thread['score']}")
                    st.write(
                        f"**Date:** {thread['date'].strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    st.write(f"[Link to thread]({thread['url']})")
        else:
            st.info("No thread results to display.")

    with col2:
        if st.session_state.analyzed_comments:
            st.subheader("Top Comment Results")
            for i, comment in enumerate(st.session_state.analyzed_comments[:10], 1):
                with st.expander(
                    f"#{i} Ranked Comment (Similarity: {comment['similarity']:.4f})"
                ):
                    st.write(f"**Subreddit:** r/{comment['subreddit']}")
                    st.write(f"**Score:** {comment['score']}")
                    st.write(
                        f"**Date:** {comment['date'].strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    st.write(f"[Link to comment]({comment['url']})")

                    # Display context with prettier formatting
                    st.write("### Context")
                    st.write(f"#### Original Post")
                    st.write(f"**Title:** {comment['post_title']}")
                    st.write(
                        f"**Content:** {comment['post_text'][:200]}..."
                        if len(comment["post_text"]) > 200
                        else comment["post_text"]
                    )

                    st.write("#### Comment Chain")
                    for idx, chain_comment in enumerate(comment["comment_chain"]):
                        if idx < len(comment["comment_chain"]) - 1:
                            st.write(f"**u/{chain_comment['author']}:**")
                            st.write(f"> {chain_comment['body']}")
                        else:
                            st.write(
                                f"**Current Comment (u/{chain_comment['author']}):**"
                            )
                            st.write(f"**> {chain_comment['body']}**")

                        st.write("---")

                    # Context for LLM
                    context = f"Post Title: {comment['post_title']}\n\n"
                    context += f"Post Content: {comment['post_text']}\n\n"
                    context += "Comment Chain:\n"
                    for idx, chain_comment in enumerate(comment["comment_chain"]):
                        if idx < len(comment["comment_chain"]) - 1:
                            context += f"u/{chain_comment['author']}: {chain_comment['body']}\n\n"
                        else:
                            context += f"Current Comment (u/{chain_comment['author']}): {chain_comment['body']}\n"
                    if st.button(f"{i}. Generate response"):
                        generate_response(context)
                    st.write("---")

        else:
            st.info("No comment results to display.")

else:
    st.info("Click 'Search and Analyze' to see results.")


st.sidebar.title("About")
st.sidebar.info(
    "This app scrapes Reddit threads and comments based on your search queries, optional subreddits, and date range. "
    "It analyzes the results using either BERT embeddings or TF-IDF, and displays the most relevant threads and comments in separate columns."
)
