import os
import json
import uuid
import threading
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from dotenv import load_dotenv

# Load env variables before other imports
load_dotenv()

from agent import run_agent
from tools import warm_model
from db import (
    execute_query, get_departments, get_courses_by_dept,
    get_all_courses, get_student, get_student_courses,
    save_chat_message, search_courses, enroll_course, drop_course,
    get_courses_by_allowed_depts, get_departments_by_allowed
)
from s3_utils import save_recommendation_report

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # No caching for static files


# ─── Student Major Profiles ─────────────────────────────────────

STUDENT_PROFILES = {
    "student_123": {
        "id": "student_123",
        "name": "Alice Smith",
        "email": "alice@psu.edu",
        "major": "Computer Science",
        "department_category": "Engineering & Technology",
        "allowed_departments": ["CMPSC", "CMPEN", "SWENG", "DS", "EE", "ME", "MATH", "STAT", "AERSP", "IE", "EDSGN", "PHYS", "EMET", "EET"],
        "disallowed_categories": "Architecture, Law, Health",
        "sample_prompts": [
            "Which courses suit someone interested in AI/ML?",
            "What courses help for a software engineering career?",
            "What are high-GPA courses to boost my grades in CS?"
        ]
    },
    "student_456": {
        "id": "student_456",
        "name": "Bob Johnson",
        "email": "bob@psu.edu",
        "major": "Architecture",
        "department_category": "Arts & Architecture",
        "allowed_departments": ["ARCH", "LARCH", "ART", "ARTH", "ARTSA", "DART", "INART", "EDSGN", "AE", "CIVE"],
        "disallowed_categories": "Computer Science, Law, Health",
        "sample_prompts": [
            "Which courses suit someone interested in Architectural Design?",
            "What courses help for a Landscape Architecture career?",
            "What foundation art and drafting courses should I take?"
        ]
    },
    "student_789": {
        "id": "student_789",
        "name": "Carol Williams",
        "email": "carol@psu.edu",
        "major": "Law & Legal Studies",
        "department_category": "Liberal Arts & Pre-Law",
        "allowed_departments": ["BLAW", "CRIM", "CRIMJ", "PLSC", "PHIL", "SOC", "PSYCH", "HIST", "ECON"],
        "disallowed_categories": "Computer Science, Architecture, Health",
        "sample_prompts": [
            "Which courses suit someone interested in Corporate Law?",
            "What courses help for a Criminal Justice & Pre-Law career?",
            "What high-GPA ethics and legal studies courses boost my GPA?"
        ]
    }
}


# ─── Page Routes ────────────────────────────────────────────────

@app.route('/')
def index():
    """Login page."""
    return render_template('index.html')


@app.route('/advisor')
def advisor():
    """Main advisor page — requires login."""
    student_id = request.args.get('student_id')
    if not student_id:
        return redirect(url_for('index'))
    return render_template('advisor.html')


# ─── API: Authentication ────────────────────────────────────────

@app.route('/api/login', methods=['POST'])
def login():
    """Validate student ID against CockroachDB and return student profile."""
    data = request.json
    student_id = data.get('student_id', '').strip()

    if not student_id:
        return jsonify({"error": "Student ID is required"}), 400

    student = get_student(student_id)
    if not student:
        return jsonify({"error": "Student not found. Try: student_123, student_456, or student_789"}), 404

    completed = get_student_courses(student_id)
    profile = STUDENT_PROFILES.get(student_id, {
        "id": student["id"],
        "name": student["name"],
        "email": student.get("email", ""),
        "major": "General Studies",
        "department_category": "All Departments",
        "allowed_departments": [],
        "sample_prompts": [
            "Which courses suit someone interested in AI/ML?",
            "What courses help for a software engineering career?",
            "What are high-GPA courses to boost my grades?"
        ]
    })

    return jsonify({
        "student": {
            "id": student["id"],
            "name": student["name"],
            "email": student.get("email", ""),
            "major": profile.get("major", "General Studies"),
            "department_category": profile.get("department_category", ""),
            "allowed_departments": profile.get("allowed_departments", []),
            "sample_prompts": profile.get("sample_prompts", [])
        },
        "completed_courses": completed or [],
        "upcoming_term": "Spring 2026"
    })


# ─── API: Course Catalog ────────────────────────────────────────

@app.route('/api/departments')
def departments():
    """Return all departments in the university catalog."""
    depts = get_departments()
    return jsonify(depts or [])


@app.route('/api/courses')
def courses():
    """Return courses from the full university catalog."""
    dept = request.args.get('dept', '').strip()
    q = request.args.get('q', '').strip()

    if q:
        results = search_courses(q)
    elif dept:
        results = get_courses_by_dept(dept)
    else:
        results = get_all_courses()

    return jsonify(results or [])


# ─── API: AI Chat ───────────────────────────────────────────────

@app.route('/api/chat', methods=['POST'])
def chat():
    """Optimized chat: parallel pre-fetch, single-shot AI, async post-save."""
    data = request.json
    message = data.get('message', '').strip()
    student_id = data.get('student_id', 'student_123')
    session_id = data.get('session_id')

    if not message:
        return jsonify({"error": "Message is required"}), 400

    profile = STUDENT_PROFILES.get(student_id)
    allowed = profile.get("allowed_departments") if profile else None

    if not session_id:
        session_id = str(uuid.uuid4())

    # ── Parallel: session upsert + user message save + course fetch ──
    import concurrent.futures as _cf

    def _init_session():
        existing = execute_query(
            "SELECT id FROM recommendation_sessions WHERE id = %s", (session_id,)
        )
        if not existing:
            execute_query(
                "INSERT INTO recommendation_sessions (id, student_id, status) VALUES (%s, %s, %s)",
                (session_id, student_id, "in_progress")
            )
        save_chat_message(session_id, "user", message)

    def _get_courses():
        completed = get_student_courses(student_id)
        if completed:
            return ", ".join([c.get("id", "") + " " + c.get("title", "") for c in completed])
        return ""

    with _cf.ThreadPoolExecutor(max_workers=2) as pool:
        f_session = pool.submit(_init_session)
        f_courses = pool.submit(_get_courses)
        f_session.result()  # ensure session is ready before AI starts
        completed_str = f_courses.result()

    # ── Run the optimized single-shot AI agent ──
    recommendation_text = run_agent(message, student_id, completed_courses=completed_str, allowed_depts=allowed)

    try:
        recommendation_data = json.loads(recommendation_text)
    except Exception:
        recommendation_data = {"agent_answer": recommendation_text, "recommendations": []}

    # ── Return response immediately, then save to DB + S3 in background ──
    def _post_save():
        try:
            save_chat_message(session_id, "assistant", recommendation_text)
            execute_query(
                "UPDATE recommendation_sessions SET status = %s, updated_at = current_timestamp() WHERE id = %s",
                ("complete", session_id)
            )
            save_recommendation_report(student_id, session_id, message, recommendation_text)
        except Exception as bg_err:
            print(f"[chat] Background save error: {bg_err}")

    threading.Thread(target=_post_save, daemon=True).start()

    return jsonify({
        "session_id": session_id,
        "recommendation": recommendation_data,
        "s3_path": f"s3://{os.environ.get('AWS_S3_BUCKET_NAME','')}/reports/student_{student_id}/session_{session_id}.json"
    })


# ─── API: Student Profile ───────────────────────────────────────

@app.route('/api/student/<student_id>')
def student_profile(student_id):
    """Return student profile with completed courses."""
    student = get_student(student_id)
    if not student:
        return jsonify({"error": "Student not found"}), 404

    completed = get_student_courses(student_id)
    profile = STUDENT_PROFILES.get(student_id, {})
    return jsonify({
        "student": {
            "id": student["id"],
            "name": student["name"],
            "email": student.get("email", ""),
            "major": profile.get("major", "General Studies"),
            "allowed_departments": profile.get("allowed_departments", [])
        },
        "completed_courses": completed or []
    })


# ─── API: Course Enrollment & Drop ─────────────────────────────

@app.route('/api/enroll', methods=['POST'])
def api_enroll():
    """Enroll a student in a course if permitted by major discipline department rules."""
    data = request.json or {}
    student_id = data.get('student_id')
    course_id = data.get('course_id')
    semester = data.get('semester', 'Spring 2026')

    if not student_id or not course_id:
        return jsonify({"error": "student_id and course_id required"}), 400

    profile = STUDENT_PROFILES.get(student_id)
    if profile and profile.get("allowed_departments"):
        allowed = profile["allowed_departments"]
        # Extract department prefix from course_id (e.g. 'CMPSC 131' -> 'CMPSC')
        dept_prefix = course_id.split()[0] if ' ' in course_id else course_id
        if dept_prefix not in allowed:
            major_name = profile.get("major", "your major")
            disallowed = profile.get("disallowed_categories", "other disciplines")
            return jsonify({
                "error": f"Permission Denied: {course_id} ({dept_prefix}) is restricted to {disallowed} majors. As a {major_name} student, you do not have enrollment permissions for this department."
            }), 403

    updated = enroll_course(student_id, course_id, semester=semester, grade="Enrolled")
    return jsonify({
        "success": True,
        "message": f"Successfully enrolled in {course_id}",
        "completed_courses": updated or []
    })


@app.route('/api/drop', methods=['POST'])
def api_drop():
    """Drop a student from a course (Spring 2026 upcoming term only; past terms locked)."""
    data = request.json or {}
    student_id = data.get('student_id')
    course_id = data.get('course_id')

    if not student_id or not course_id:
        return jsonify({"error": "student_id and course_id required"}), 400

    # Check if course is from a past completed semester (e.g. Fall 2025)
    existing = execute_query(
        "SELECT semester, grade FROM student_courses WHERE student_id = %s AND course_id = %s",
        (student_id, course_id)
    )
    if existing:
        sem = existing[0].get('semester', '')
        if sem and 'Fall' in sem:
            return jsonify({
                "error": f"Cannot drop {course_id}: Courses completed in {sem} are part of your permanent academic record and cannot be dropped."
            }), 403

    updated = drop_course(student_id, course_id)
    return jsonify({
        "success": True,
        "message": f"Successfully dropped {course_id}",
        "completed_courses": updated or []
    })


if __name__ == '__main__':
    # Pre-warm the SentenceTransformer model in a background thread
    # so the first user request is fast (eliminates ~6s cold start).
    threading.Thread(target=warm_model, daemon=True).start()
    app.run(debug=True, host='0.0.0.0', port=8080)
