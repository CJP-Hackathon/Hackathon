import os
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    db_url = os.environ.get("COCKROACHDB_URL")
    if not db_url:
        raise ValueError("COCKROACHDB_URL environment variable is not set")
    return psycopg.connect(db_url, row_factory=dict_row)

def execute_query(query, params=None):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            if cur.description:
                return cur.fetchall()
            conn.commit()
            return None

def get_departments():
    return execute_query("SELECT * FROM departments ORDER BY name")

def get_courses_by_dept(prefix):
    query = "SELECT * FROM courses WHERE department_prefix = %s ORDER BY number"
    return execute_query(query, (prefix,))

def get_all_courses(limit=None):
    if limit:
        query = "SELECT * FROM courses ORDER BY department_prefix, number LIMIT %s"
        return execute_query(query, (limit,))
    query = "SELECT * FROM courses ORDER BY department_prefix, number"
    return execute_query(query)

def get_student(student_id):
    query = "SELECT * FROM students WHERE id = %s"
    res = execute_query(query, (student_id,))
    return res[0] if res else None

def get_student_courses(student_id):
    query = """
        SELECT c.*, sc.grade, sc.semester 
        FROM courses c 
        JOIN student_courses sc ON c.id = sc.course_id 
        WHERE sc.student_id = %s
    """
    return execute_query(query, (student_id,))

def save_chat_message(session_id, role, content):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(MAX(turn), 0) + 1 AS next_turn FROM conversations WHERE session_id = %s", 
                (session_id,)
            )
            row = cur.fetchone()
            next_turn = row['next_turn']
            
            cur.execute("""
                INSERT INTO conversations (session_id, turn, role, content)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (session_id, next_turn, role, content))
            res = cur.fetchone()
            conn.commit()
            return res['id']

def search_courses(query_text):
    query = """
        SELECT * FROM courses 
        WHERE title ILIKE %s OR description ILIKE %s
    """
    search_term = f"%{query_text}%"
    return execute_query(query, (search_term, search_term))

def enroll_course(student_id, course_id, semester="Spring 2026", grade="Enrolled"):
    query = """
        INSERT INTO student_courses (student_id, course_id, semester, grade)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (student_id, course_id) DO UPDATE 
        SET semester = EXCLUDED.semester, grade = EXCLUDED.grade
    """
    execute_query(query, (student_id, course_id, semester, grade))
    return get_student_courses(student_id)

def drop_course(student_id, course_id):
    query = "DELETE FROM student_courses WHERE student_id = %s AND course_id = %s"
    execute_query(query, (student_id, course_id))
    return get_student_courses(student_id)

def get_courses_by_allowed_depts(allowed_depts):
    if not allowed_depts:
        return get_all_courses()
    query = "SELECT * FROM courses WHERE department_prefix = ANY(%s) ORDER BY department_prefix, number"
    return execute_query(query, (allowed_depts,))

def get_departments_by_allowed(allowed_depts):
    if not allowed_depts:
        return get_departments()
    query = "SELECT * FROM departments WHERE prefix = ANY(%s) ORDER BY name"
    return execute_query(query, (allowed_depts,))


