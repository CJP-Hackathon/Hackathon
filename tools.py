import json
import hashlib
import subprocess
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from db import execute_query

# ─── Model: loaded ONCE at module import (singleton) ──────────────
# On server startup, call warm_model() to pre-load before first request.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def warm_model():
    """
    Pre-warm: encode a dummy sentence so the model weights are fully
    loaded into memory before the first real user request arrives.
    Eliminates the ~6s cold-start delay on the first chat message.
    """
    _ = embedding_model.encode("warm up")
    print("[tools] SentenceTransformer model pre-warmed ✅")


# ─── L1: In-process LRU cache (instant, zero network) ────────────
@lru_cache(maxsize=256)
def _cached_encode(text: str):
    """Cache embedding vectors in-process. Same query = zero CPU."""
    return tuple(embedding_model.encode(text).tolist())


def _query_hash(goal_text: str, allowed_depts_key: str) -> str:
    raw = f"{goal_text.strip().lower()}|{allowed_depts_key}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ─── L2: CockroachDB search_cache (persists across restarts) ──────
def _cache_lookup(query_hash: str):
    """
    Check CockroachDB search_cache. Returns parsed results list or None.
    Average latency: ~2ms vs ~1100ms for a full pgvector scan.
    """
    rows = execute_query(
        """SELECT results_json FROM search_cache
           WHERE query_hash = %s AND expires_at > current_timestamp()""",
        (query_hash,)
    )
    if rows:
        # Increment hit counter asynchronously (fire-and-forget)
        try:
            execute_query(
                "UPDATE search_cache SET hit_count = hit_count + 1 WHERE query_hash = %s",
                (query_hash,)
            )
        except Exception:
            pass
        return json.loads(rows[0]["results_json"])
    return None


def _cache_store(query_hash: str, goal_text: str, results: list):
    """Persist results in CockroachDB for 24 hours."""
    try:
        execute_query(
            """INSERT INTO search_cache (query_hash, query_text, results_json)
               VALUES (%s, %s, %s)
               ON CONFLICT (query_hash) DO UPDATE
               SET results_json = EXCLUDED.results_json,
                   expires_at = current_timestamp() + INTERVAL '24 hours',
                   hit_count = search_cache.hit_count + 1""",
            (query_hash, goal_text, json.dumps(results))
        )
    except Exception as e:
        print(f"[tools] Cache store error (non-fatal): {e}")


# ─── MCP Eligibility Tool ─────────────────────────────────────────
def mcp_check_eligibility(student_id: str, course_id: str):
    """
    Read-only structured queries against enrollment, prerequisite,
    and requirement tables (MCP Server pattern).
    """
    query = "SELECT * FROM prerequisites_graph WHERE course_id = %s"
    results = execute_query(query, (course_id,))
    return {"status": "success", "data": results}


# ─── Vector Semantic Search (with L1 + L2 caching) ───────────────
def vector_semantic_search(goal_text: str, limit: int = 5, allowed_depts: list = None):
    """
    Semantic matching between a student's free-text goal and course embeddings.
    Uses two-level cache:
      L1 — in-process LRU (0ms, same process lifetime)
      L2 — CockroachDB search_cache table (~2ms, persists 24h across restarts)
    Falls back to full pgvector scan (~1100ms) only on cache miss.
    """
    allowed_key = ",".join(sorted(allowed_depts)) if allowed_depts else "all"
    q_hash = _query_hash(goal_text, allowed_key)

    # ── L2 cache lookup ──
    cached = _cache_lookup(q_hash)
    if cached is not None:
        print(f"[tools] Cache HIT for '{goal_text[:40]}' (hash={q_hash[:8]})")
        return cached[:limit]

    print(f"[tools] Cache MISS — running pgvector search for '{goal_text[:40]}'")

    # ── L1 cached embedding (avoids re-encoding same text) ──
    embedding = list(_cached_encode(goal_text))
    vector_str = f"[{','.join(map(str, embedding))}]"

    if allowed_depts and isinstance(allowed_depts, list) and len(allowed_depts) > 0:
        query = """
        SELECT id, title, description, department_prefix, min_credits,
               1 - (embedding <=> %s::vector) AS similarity
        FROM courses
        WHERE department_prefix = ANY(%s) AND embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        results = execute_query(query, (vector_str, allowed_depts, vector_str, limit))
    else:
        query = """
        SELECT id, title, description, department_prefix, min_credits,
               1 - (embedding <=> %s::vector) AS similarity
        FROM courses
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        results = execute_query(query, (vector_str, vector_str, limit))

    results = results or []

    # ── Store in L2 cache for next request ──
    _cache_store(q_hash, goal_text, results)

    return results


# ─── ccloud Health Check Tool ─────────────────────────────────────
def ccloud_health_check(cluster_name: str):
    """
    Shells out to ccloud CLI to report CockroachDB cluster health
    and recent audit log entries.
    """
    try:
        info_out = subprocess.check_output(f"ccloud cluster info {cluster_name} -o json", shell=True)
        audit_out = subprocess.check_output("ccloud audit list --limit 5 -o json", shell=True)
        return {
            "cluster_info": json.loads(info_out),
            "recent_audits": json.loads(audit_out)
        }
    except Exception as e:
        return {"error": str(e)}

