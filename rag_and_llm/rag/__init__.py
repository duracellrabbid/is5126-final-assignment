import json

from langchain.tools import tool

from .database import db


@tool(description="Search the database for best practices and case studies relevant to employee retention.")
def search_database(query: str, limit: int = 10) -> str:
    """Search the employee retention resources database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    results = db.similarity_search(query, k=limit)

    if not results:
        return json.dumps({"message": "No results found."}, ensure_ascii=False, indent=2)

    output = [
        {
            "source": r.metadata.get("source", "unknown"),
            "content": r.page_content[:500] + ("..." if len(r.page_content) > 500 else "")
        }
        for r in results
    ]
    
    return json.dumps(output, ensure_ascii=False, indent=2)