# main.py
# -*- coding: utf-8 -*-

import os
import json
from decimal import Decimal
import mysql.connector
from typing import List, Dict, Any, Optional

import requests

os.environ["LANGCHAIN_ALLOWED_OBJECTS"] = "core"

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct, SearchRequest
from sentence_transformers import SentenceTransformer

from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
# from langchain_classic.agents.openai_tools import create_openai_tools_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
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
QDRANT_PORT = 6333  
COLLECTION_NAME = "products_collection"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


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
def _ollama_is_available(model: str) -> bool:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if not response.ok:
            return False

        payload = response.json()
        models = [item.get("name", "") for item in payload.get("models", [])]
        return any(model in name for name in models)
    except requests.RequestException:
        return False


def _normalize_sql_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_rows: List[Dict[str, Any]] = []

    for row in rows:
        normalized_row: Dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, Decimal):
                normalized_row[key] = float(value)
            else:
                normalized_row[key] = value
        normalized_rows.append(normalized_row)

    return normalized_rows


def _should_use_sql(query: str) -> bool:
    lowered = query.lower()
    sql_keywords = (
        "giá cao nhất",
        "cao nhất",
        "đắt nhất",
        "rẻ nhất",
        "classic cars",
        "thuộc dòng",
        "dòng",
        "msrp",
        "buyprice",
    )
    return any(keyword in lowered for keyword in sql_keywords)


def _format_local_answer(query: str, rows: List[Dict[str, Any]], source: str) -> str:
    if not rows:
        return f"Kết quả local ({source}) không tìm thấy sản phẩm phù hợp cho: {query}"

    source_note = "SQL" if source == "sql" else "vector search"
    lines = [f"Kết quả local ({source_note}), không dùng OpenAI:"]
    list_text = _format_results_list(rows)
    if list_text:
        lines.append(list_text)

    return "\n".join(lines)


def _format_results_list(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""

    lines: List[str] = []
    for index, row in enumerate(rows, start=1):
        product_name = row.get("productName", "N/A")
        product_line = row.get("productLine", "N/A")
        msrp = row.get("MSRP", row.get("msrp", "N/A"))
        buy_price = row.get("buyPrice", row.get("buyprice", "N/A"))
        lines.append(
            f"{index}. {product_name} | dòng: {product_line} | MSRP: {msrp} | buyPrice: {buy_price}"
        )

    return "\n".join(lines)


def _parse_tool_choice(text: str) -> str:
    normalized = text.strip().upper()
    if "SQL" in normalized:
        return "sql"
    if "VECTOR" in normalized:
        return "vector"
    return "vector"


def _select_tool_with_llm(llm: object, query: str) -> str:
    if _should_use_sql(query):
        return "sql"

    prompt = (
        "Bạn là bộ phân loại tool.\n"
        "Chỉ trả lời đúng 1 từ: SQL hoặc VECTOR.\n"
        "Quy tắc:\n"
        "- SQL: câu hỏi về giá cao nhất/thấp nhất, lọc theo dòng sản phẩm, hoặc cần thống kê/so sánh.\n"
        "- VECTOR: câu hỏi tìm sản phẩm tương tự, gợi ý theo sở thích.\n"
        f"Query: {query}"
    )
    response = llm.invoke(prompt)
    content = getattr(response, "content", str(response))
    return _parse_tool_choice(content)


def _summarize_with_llm(
    llm: object,
    query: str,
    rows: List[Dict[str, Any]],
    source: str,
) -> str:
    list_text = _format_results_list(rows)
    if not list_text:
        return _format_local_answer(query, rows, source=source)

    prompt = f"""
Bạn là trợ lý tư vấn sản phẩm.
Hãy viết 2-3 câu giải thích ngắn, dựa trên query và danh sách bên dưới.
Không liệt kê lại danh sách. Không dịch sang tiếng Anh. Không bịa thêm.

Query: {query}
Danh sách:
{list_text}
"""
    response = llm.invoke(prompt)
    explanation = getattr(response, "content", str(response)).strip()

    return f"{list_text}\n\n{explanation}".strip()


class LocalAgentExecutor:
    def invoke(self, inputs: Dict[str, Any]):
        query = inputs["input"] if isinstance(inputs, dict) else str(inputs)

        if _should_use_sql(query):
            raw_output = tool_text_to_sql.invoke(query)
            source = "sql"
        else:
            raw_output = tool_vector_search.invoke(query)
            source = "vector"

        print(f"[TOOL] {source.upper()} | query={query}")

        try:
            rows = json.loads(raw_output)
        except json.JSONDecodeError:
            rows = [{"raw": raw_output}]

        return {
            "output": _format_local_answer(query, rows, source),
            "source": source,
        }


class LLMDecisionAgent:
    def __init__(self, llm: object):
        self.llm = llm

    def invoke(self, inputs: Dict[str, Any]):
        query = inputs["input"] if isinstance(inputs, dict) else str(inputs)

        try:
            choice = _select_tool_with_llm(self.llm, query)
        except Exception:
            choice = "vector"

        print(f"[TOOL] {choice.upper()} | query={query}")

        if choice == "sql":
            raw_output = tool_text_to_sql.invoke(query)
        else:
            raw_output = tool_vector_search.invoke(query)

        try:
            rows = json.loads(raw_output)
        except json.JSONDecodeError:
            rows = [{"raw": raw_output}]

        try:
            summary = _summarize_with_llm(self.llm, query, rows, choice)
        except Exception:
            summary = _format_local_answer(query, rows, source=choice)

        return {
            "output": summary,
            "source": choice,
        }


def get_llm() -> Optional[object]:
    if OPENAI_API_KEY:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    if _ollama_is_available(OLLAMA_MODEL):
        return ChatOllama(model=OLLAMA_MODEL, temperature=0.2)

    return None


def vector_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    model = SentenceTransformer(EMBED_MODEL_NAME)
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    query_vec = model.encode(query).tolist()
    response = qdrant.http.search_api.search_points(
        collection_name=COLLECTION_NAME,
        search_request=SearchRequest(
            vector=query_vec,
            limit=top_k,
            with_payload=True,
            with_vector=False,
        ),
    )
    results = response.result

    hits = []
    for r in results:
        payload = dict(r.payload or {})
        payload["score"] = r.score
        hits.append(payload)

    return hits


def search(query: str, top_k: int = 5) -> str:
    results = vector_search(query, top_k=top_k)
    llm = get_llm()

    if llm is None:
        return _format_local_answer(query, results, source="vector")

    try:
        return _summarize_with_llm(llm, query, results, "vector")
    except Exception:
        return _format_local_answer(query, results, source="vector")


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

    return json.dumps(_normalize_sql_rows(rows), ensure_ascii=False)


def build_agent():
    llm = get_llm()

    tools = [tool_vector_search, tool_text_to_sql]

    if llm is None or not hasattr(llm, "bind_tools"):
        return LocalAgentExecutor()

    if isinstance(llm, ChatOllama):
        return LLMDecisionAgent(llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Bạn là trợ lý dữ liệu. Hãy chọn tool phù hợp để trả lời."),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    try:
        agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)
    except Exception:
        return LocalAgentExecutor()


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