import os
import json
import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


logging.basicConfig(level=logging.INFO)


CRAWLED_DATA_PATH = Path(__file__).parent / "crawl" / "employee_retention_articles.json"
VECTOR_DATABASE_PATH = Path(__file__).parent / "chroma_langchain_db"


def load_crawled_data(path: Path | str) -> tuple[list[str], list[str]]:
    """
    Load crawled articles from .json file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Crawled articles '{path}' does not exist. "
            "Please run './crawl/1. webscrape_articles.ipynb'"
        )

    with open(path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    docs, docs_url = [], []
    for a in articles:
        content = a.get("content")
        url = a.get("url")
        if isinstance(content, (dict, list)):
            content = str(content)
        if not content:
            continue
        docs.append(content)
        docs_url.append(url)

    logging.info(f"Loaded {len(docs)} crawled articles.")
    return docs, docs_url


def load_or_rebuild_embeddings(path: Path) -> Chroma:
    """
    Load existing Chroma DB if present, otherwise rebuild it from crawled data.
    """
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    if path.exists() and any(path.iterdir()):
        logging.info(f"Loading existing Chroma DB from {path}")
        vector_store = Chroma(
            collection_name="employee_retention_articles",
            persist_directory=path,
            embedding_function=embeddings,
        )
    else:
        logging.info(f"No existing DB found. Rebuilding new Chroma DB at {path}")
        path.mkdir(parents=True, exist_ok=True)

        docs, docs_url = load_crawled_data(CRAWLED_DATA_PATH)

        vector_store = Chroma(
            collection_name="employee_retention_articles",
            persist_directory=path,
            embedding_function=embeddings,
        )

        documents = [
            Document(page_content=doc, metadata={"source": url}, id=url)
            for doc, url in zip(docs, docs_url)
        ]
        vector_store.add_documents(documents=documents, ids=docs_url)

    return vector_store


db = load_or_rebuild_embeddings(VECTOR_DATABASE_PATH)
logging.info("Chroma DB ready")
