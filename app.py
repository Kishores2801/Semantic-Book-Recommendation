
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma    


import gradio as gr

load_dotenv()

books = pd.read_csv("data/processed/book_categorized_with_emotions.csv")
books["large_thumbnail"] = books['thumbnail'] + "&fife=w800"

books['large_thumbnail'] = np.where(
    books['large_thumbnail'].isna(),
    'cover-not-found.jpg',
    books['large_thumbnail']
)

raw_documents = TextLoader("data/processed/tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n")
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())


def retrieve_semantic_recomendation(
        query: str,
        category: str = None,
        tone: str=None,
        initial_top_k: int =50,
        final_top_k: int =16
        
        ) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)



    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category][:final_top_k]
    else:
        book_recs = book_recs.head(final_top_k)


    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Neutral":
        book_recs.sort_values(by="neutral", ascending=False, inplace=True)
    elif tone == "Suprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        
        query: str,
        category: str,
        tone: str
        ) :
    recommendations = retrieve_semantic_recomendation(query, category, tone)
    results =[]

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_desc = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(",")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]
        
        caption = f"{row['title']} by {authors_str}\n\n{truncated_desc}"
        results.append((row["large_thumbnail"], caption))


    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones =["All"] + [ "Happy", "Sad", "Angry", "Neutral", "Suprising", "Suspenseful"]


with gr.Blocks(theme =gr.themes.Glass()) as dashboard:
    gr.Markdown("## Book Recommendation System")
    with gr.Row():
            query_input = gr.Textbox(label="Enter your book preferences or mood", placeholder="e.g., I want a thrilling mystery Novel")
            category_input = gr.Dropdown(label="Select a category", choices=categories, value="All")
            tone_input = gr.Dropdown(label="Select a tone", choices=tones, value="All")
            recommend_button = gr.Button("Recommend Books")
    gr.Markdown("### Recommended Books")
    recommendation_gallery = gr.Gallery(label="Recommended Books", columns=8, rows=2,height="auto")

    recommend_button.click(
        fn=recommend_books,
        inputs=[query_input, category_input, tone_input],
        outputs=recommendation_gallery
    )


    if __name__ == "__main__":
        dashboard.launch(share=True)