import os
import json
import concurrent.futures
from google import genai
from google.genai import types
from tools import mcp_check_eligibility, vector_semantic_search, ccloud_health_check

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

SINGLE_SHOT_PROMPT = """You are an AI Academic Advisor at Penn State University.

Given the student context and pre-fetched course search results below, respond with EXACTLY this JSON structure:

{
  "agent_answer": "A 2-3 sentence summary explaining your recommendation strategy and how it connects to the student's goal",
  "recommendations": [
    {
      "course_code": "CMPSC 442",
      "title": "Artificial Intelligence",
      "credits": 3,
      "fit": 92,
      "avg_grade": "B+",
      "explanation": "Why this specific course fits the student's goal",
      "benefits": "How the student will benefit — skills gained, career impact, etc.",
      "tags": ["career match 0.91", "prereqs met", "high demand"]
    }
  ]
}

RULES:
- Include exactly 3 recommendations, ranked by fit % (highest first)
- "fit" must be an integer 0-100
- "avg_grade" must be a realistic letter grade (A, A-, B+, B, B-, C+, C)
- "credits" must be an integer
- Each recommendation must have 2-4 short tags
- Do NOT recommend courses the student has already completed
- Respond ONLY with valid JSON, no markdown fences or extra text"""


def _fetch_tool_results_parallel(goal, student_id, allowed_depts):
    results = {}

    def do_vector():
        try:
            return vector_semantic_search(goal_text=goal, limit=6, allowed_depts=allowed_depts)
        except Exception as e:
            return {"error": str(e)}

    def do_mcp():
        try:
            return mcp_check_eligibility(student_id, "")
        except Exception:
            return {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_vec = pool.submit(do_vector)
        f_mcp = pool.submit(do_mcp)
        results["vector"] = f_vec.result(timeout=8)
        results["mcp"] = f_mcp.result(timeout=8)

    return results


def run_agent(goal, student_id, history=None, completed_courses="", allowed_depts=None):
    try:
        tool_results = _fetch_tool_results_parallel(goal, student_id, allowed_depts)
        vector_courses = tool_results.get("vector", [])
        mcp_info = tool_results.get("mcp", {})

        courses_text = ""
        if isinstance(vector_courses, list) and vector_courses:
            course_lines = []
            for c in vector_courses[:6]:
                course_lines.append(
                    "- " + str(c.get("id","?")) + " | " + str(c.get("title","?")) + " | " +
                    str(c.get("min_credits", c.get("credits","?"))) + " cr | " +
                    "Dept: " + str(c.get("department_prefix","?")) + " | " +
                    "Similarity: " + str(round(c.get("similarity", 0), 2)) + " | " +
                    "Desc: " + str(c.get("description",""))[:120]
                )
            courses_text = "\n".join(course_lines)
        else:
            courses_text = "No matching courses found from search."

        depts_str = ", ".join(allowed_depts) if allowed_depts else "All"
        prompt = (
            "Student goal: " + goal + "\n"
            "Student ID: " + student_id + "\n"
            "Completed courses (do NOT recommend these): " + (completed_courses or "None") + "\n"
            "Allowed department prefixes: " + depts_str + "\n\n"
            "Pre-fetched semantically matching courses:\n" + courses_text + "\n\n"
            "Eligibility context: " + json.dumps(mcp_info)[:300] + "\n\n"
            "Now produce the structured JSON response with exactly 3 course recommendations."
        )

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SINGLE_SHOT_PROMPT,
                response_mime_type="application/json",
                temperature=0.3,
                max_output_tokens=1024,
            ),
        )

        result_text = response.text
        parsed = json.loads(result_text)

        if "agent_answer" not in parsed:
            parsed["agent_answer"] = "Here are your recommended courses based on your goal."
        if "recommendations" not in parsed:
            parsed["recommendations"] = []

        return json.dumps(parsed)

    except json.JSONDecodeError:
        raw = result_text if "result_text" in dir() else "Could not parse recommendations."
        return json.dumps({"agent_answer": raw, "recommendations": []})
    except Exception as e:
        print("[agent] Gemini failed: " + str(e) + ". Trying Ollama (local)...")
        return run_agent_ollama(goal, student_id, completed_courses, allowed_depts=allowed_depts)


def _build_prompt(goal, student_id, completed_courses, allowed_depts, search_result):
    """Build the shared single-shot prompt from pre-fetched vector results."""
    if isinstance(search_result, list) and search_result:
        course_lines = []
        for c in search_result[:6]:
            course_lines.append(
                "- " + str(c.get("id","?")) + " | " + str(c.get("title","?")) + " | " +
                str(c.get("min_credits", c.get("credits","?"))) + " cr | " +
                "Dept: " + str(c.get("department_prefix","?")) + " | " +
                "Desc: " + str(c.get("description",""))[:100]
            )
        courses_text = "\n".join(course_lines)
    else:
        courses_text = "No matching courses found."

    depts_str = ", ".join(allowed_depts) if allowed_depts else "All"
    return (
        SINGLE_SHOT_PROMPT + "\n\n"
        "Student goal: " + goal + "\n"
        "Student ID: " + student_id + "\n"
        "Completed courses (skip these): " + (completed_courses or "None") + "\n"
        "Allowed departments: " + depts_str + "\n\n"
        "Pre-fetched matching courses:\n" + courses_text + "\n\n"
        "Respond with ONLY valid JSON — no markdown, no explanation."
    )


def run_agent_ollama(goal, student_id, completed_courses="", allowed_depts=None):
    """
    Fallback 1: Locally hosted Ollama (zero cost, no network required).
    Tries mistral:instruct first (fast, 4.1GB), then llama3.1:8b.
    Uses Ollama's OpenAI-compatible /v1/chat/completions endpoint.
    """
    import requests

    OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
    MODELS = ["mistral:instruct", "llama3.1:8b"]  # in preference order

    try:
        search_result = vector_semantic_search(goal_text=goal, limit=6, allowed_depts=allowed_depts)
    except Exception as e:
        search_result = []
        print("[ollama] Vector search failed: " + str(e))

    prompt = _build_prompt(goal, student_id, completed_courses, allowed_depts, search_result)

    for model in MODELS:
        try:
            print("[ollama] Trying model: " + model)
            resp = requests.post(
                OLLAMA_URL,
                headers={"Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SINGLE_SHOT_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "temperature": 0.3,
                    "max_tokens": 1024,
                },
                timeout=60,
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                # Strip markdown fences if present
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                parsed = json.loads(content.strip())
                if "agent_answer" not in parsed:
                    parsed["agent_answer"] = "Here are your recommended courses (via local AI)."
                if "recommendations" not in parsed:
                    parsed["recommendations"] = []
                print("[ollama] ✅ Success with model: " + model)
                return json.dumps(parsed)
        except json.JSONDecodeError as je:
            print("[ollama] JSON parse error from " + model + ": " + str(je))
        except Exception as e:
            print("[ollama] Model " + model + " failed: " + str(e))

    # Both Ollama models failed — fall through to Mistral API
    print("[agent] Ollama unavailable. Falling back to Mistral API.")
    return run_agent_mistral(goal, student_id, completed_courses, allowed_depts=allowed_depts, search_result=search_result)


def run_agent_mistral(goal, student_id, completed_courses="", allowed_depts=None, search_result=None):
    """
    Fallback 2: Mistral Cloud API (mistral-small-latest).
    Only called when both Gemini and Ollama are unavailable.
    """
    import requests

    mistral_key = os.environ.get("MISTRAL_API_KEY", "")

    if search_result is None:
        try:
            search_result = vector_semantic_search(goal_text=goal, limit=5, allowed_depts=allowed_depts)
        except Exception as e:
            search_result = []
            print("[mistral] Vector search failed: " + str(e))

    prompt = _build_prompt(goal, student_id, completed_courses, allowed_depts, search_result)

    if mistral_key:
        try:
            resp = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": "Bearer " + mistral_key,
                    "Content-Type": "application/json",
                },
                json={
                    "model": "mistral-small-latest",
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 1024,
                },
                timeout=20,
            )
            if resp.status_code == 200:
                result_text = resp.json()["choices"][0]["message"]["content"]
                parsed = json.loads(result_text)
                if "agent_answer" not in parsed:
                    parsed["agent_answer"] = "Here are your recommended courses."
                if "recommendations" not in parsed:
                    parsed["recommendations"] = []
                print("[mistral] ✅ Success via Mistral API.")
                return json.dumps(parsed)
        except Exception as e:
            print("[agent] Mistral API also failed: " + str(e))

    # Fallback 3: Build response directly from vector results, no LLM
    print("[agent] All LLMs failed — returning direct vector results.")
    recommendations = []
    try:
        for i, c in enumerate((search_result or [])[:3]):
            recommendations.append({
                "course_code": c.get("id", ""),
                "title": c.get("title", ""),
                "credits": c.get("min_credits", 3),
                "fit": max(95 - i * 8, 70),
                "avg_grade": ["B+", "B", "B-"][i] if i < 3 else "B",
                "explanation": c.get("description", "Matches your academic interest.")[:200],
                "benefits": "Builds skills relevant to your stated goal.",
                "tags": ["discipline match", c.get("department_prefix", "")]
            })
    except Exception:
        pass

    return json.dumps({
        "agent_answer": "Based on your interest in " + goal + ", here are your top matching courses.",
        "recommendations": recommendations
    })
