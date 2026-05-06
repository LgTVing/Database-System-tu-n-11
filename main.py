# main.py
# -*- coding: utf-8 -*-

import os
import json
import mysql.connector
from typing import List, Dict, Any
import os
os.environ["LANGCHAIN_ALLOWED_OBJECTS"] = "core"

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
# from langchain_classic.agents.openai_tools import create_openai_tools_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
# from langchain.agents import AgentExecutor, create_openai_tools_agent


# =========================
# CONFIG
# =========================
MYSQL_CONFIG = {
    "host": "127.0.0.1",
    "port": 3309,
    "user": "root",
    "password": "",
    "database": "classicmodels"
}

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333  # <- đổi sang port của qdrant-117
COLLECTION_NAME = "products_collection"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OPENAI_API_KEY = os.getenv("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")


# =========================
# [1] BUILD VECTOR DB
# =========================
def fetch_products_from_mysql() -> List[Dict[str, Any]]:
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = conn.cursor(dictionary=True)

    query = """
    SELECT productCode, productName, productLine, productDescription, buyPrice, MSRP
    FROM products
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    cursor.close()
    conn.close()
    return rows


def build_qdrant_collection_if_not_exists(qdrant: QdrantClient, vector_size: int):
    collections = qdrant.get_collections()
    existing = [c.name for c in collections.collections]

    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"[QDRANT] Created collection: {COLLECTION_NAME}")
    else:
        print(f"[QDRANT] Collection already exists: {COLLECTION_NAME}")


def insert_products_into_qdrant():
    model = SentenceTransformer(EMBED_MODEL_NAME)
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    products = fetch_products_from_mysql()
    print(f"[MYSQL] Loaded {len(products)} products")

    test_vec = model.encode("test")
    build_qdrant_collection_if_not_exists(qdrant, len(test_vec))

    points = []
    for idx, p in enumerate(products):
        text = f"{p['productName']} - {p['productDescription']}"
        vector = model.encode(text).tolist()

        payload = {
            "productCode": p["productCode"],
            "productName": p["productName"],
            "productLine": p["productLine"],
            "buyPrice": float(p["buyPrice"]),
            "MSRP": float(p["MSRP"])
        }

        points.append(
            PointStruct(
                id=idx + 1,
                vector=vector,
                payload=payload
            )
        )

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[QDRANT] Inserted {len(points)} points into {COLLECTION_NAME}")


# =========================
# [2] VECTOR SEARCH + LLM
# =========================
def get_llm():
    if OPENAI_API_KEY:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    else:
        return Ollama(model="llama3")


def vector_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    model = SentenceTransformer(EMBED_MODEL_NAME)
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    query_vec = model.encode(query).tolist()
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k
    )

    hits = []
    for r in results:
        payload = r.payload
        payload["score"] = r.score
        hits.append(payload)

    return hits


def search(query: str, top_k: int = 5) -> str:
    results = vector_search(query, top_k=top_k)
    llm = get_llm()

    prompt = f"""
Bạn là trợ lý tư vấn sản phẩm.
Query: {query}

Danh sách sản phẩm tìm thấy (JSON):
{json.dumps(results, ensure_ascii=False, indent=2)}

Hãy:
- Trả về danh sách sản phẩm (tên, giá, dòng sản phẩm)
- Giải thích ngắn vì sao phù hợp với query.
"""

    response = llm.invoke(prompt)
    return response


# =========================
# [3] LANGCHAIN AGENT + TOOLS
# =========================
@tool
def tool_vector_search(query: str) -> str:
    """Tìm sản phẩm tương đồng bằng vector search (Qdrant)."""
    results = vector_search(query, top_k=5)
    return json.dumps(results, ensure_ascii=False)


@tool
def tool_text_to_sql(query: str) -> str:
    """
    Truy vấn MySQL classicmodels bằng SQL phù hợp.
    """
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = conn.cursor(dictionary=True)

    if "giá cao nhất" in query.lower():
        sql = """
        SELECT productName, buyPrice, MSRP, productLine
        FROM products
        ORDER BY MSRP DESC
        LIMIT 5
        """
    elif "classic cars" in query.lower():
        sql = """
        SELECT productName, buyPrice, MSRP, productLine
        FROM products
        WHERE productLine = 'Classic Cars'
        LIMIT 10
        """
    else:
        sql = "SELECT productName, buyPrice, MSRP, productLine FROM products LIMIT 5"

    cursor.execute(sql)
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return json.dumps(rows, ensure_ascii=False)


def build_agent():
    llm = get_llm()

    tools = [tool_vector_search, tool_text_to_sql]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Bạn là trợ lý dữ liệu. Hãy chọn tool phù hợp để trả lời."),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def test_agent_queries():
    agent = build_agent()

    queries = [
        "Tìm sản phẩm giống xe Ferrari",
        "Sản phẩm nào có giá cao nhất?",
        "Gợi ý sản phẩm phù hợp cho người thích mô hình cổ điển",
        "Danh sách sản phẩm thuộc dòng Classic Cars"
    ]

    for q in queries:
        print("=" * 60)
        print(f"[QUERY] {q}")
        result = agent.invoke({"input": q})
        print("[ANSWER]")
        print(result["output"])
        print("=> Agent đã chọn tool dựa trên tính chất câu hỏi (semantic vs SQL).")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    insert_products_into_qdrant()

    print("\n=== VECTOR SEARCH + LLM ===")
    print(search("xe mô hình cổ điển giá rẻ"))

    print("\n=== LANGCHAIN AGENT TEST ===")
    test_agent_queries()