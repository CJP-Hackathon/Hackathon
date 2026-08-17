import csv
import os
import psycopg
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CSV_FILE = "/Users/sainihalkonduti/hackathon/courses.csv"

SCHEMA = """
-- Schema setup
DROP TABLE IF EXISTS conversations CASCADE;
DROP TABLE IF EXISTS recommendation_sessions CASCADE;
DROP TABLE IF EXISTS student_courses CASCADE;
DROP TABLE IF EXISTS courses CASCADE;
DROP TABLE IF EXISTS departments CASCADE;
DROP TABLE IF EXISTS students CASCADE;

CREATE TABLE IF NOT EXISTS departments (
    prefix STRING PRIMARY KEY,
    name STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS courses (
    id STRING PRIMARY KEY,
    title STRING NOT NULL,
    description TEXT,
    department_prefix STRING REFERENCES departments(prefix),
    number INT,
    suffix STRING DEFAULT '',
    min_credits INT,
    max_credits INT,
    prerequisites TEXT,
    embedding VECTOR(384)
);

CREATE TABLE IF NOT EXISTS students (
    id STRING PRIMARY KEY,
    name STRING NOT NULL,
    email STRING,
    created_at TIMESTAMPTZ DEFAULT current_timestamp()
);

CREATE TABLE IF NOT EXISTS student_courses (
    student_id STRING REFERENCES students(id),
    course_id STRING REFERENCES courses(id),
    grade STRING,
    semester STRING,
    PRIMARY KEY (student_id, course_id)
);

CREATE TABLE IF NOT EXISTS recommendation_sessions (
    id STRING PRIMARY KEY,
    student_id STRING REFERENCES students(id),
    status STRING DEFAULT 'in_progress',
    created_at TIMESTAMPTZ DEFAULT current_timestamp(),
    updated_at TIMESTAMPTZ DEFAULT current_timestamp()
);

CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    session_id STRING,
    turn INT,
    role STRING,
    content TEXT,
    created_at TIMESTAMPTZ DEFAULT current_timestamp()
);
"""

DEPT_MAP = {
    'CMPSC': 'Computer Science',
    'ACCTG': 'Accounting',
    'AERSP': 'Aerospace Engineering',
    'MATH': 'Mathematics',
    'PHYS': 'Physics',
    'STAT': 'Statistics',
    'EE': 'Electrical Engineering',
    'ME': 'Mechanical Engineering',
    'CMPEN': 'Computer Engineering',
    'SWENG': 'Software Engineering',
    'DS': 'Data Sciences',
    'ENGL': 'English',
    'HIST': 'History',
    'BIOL': 'Biology',
    'CHEM': 'Chemistry',
    'ECON': 'Economics',
    'PSYCH': 'Psychology',
    'SOC': 'Sociology',
    'ART': 'Art',
    'MUS': 'Music',
    'PHIL': 'Philosophy'
}

PRIORITY_DEPTS = {'CMPSC', 'MATH', 'AERSP', 'EE', 'ME', 'CMPEN', 'SWENG', 'DS', 'PHYS', 'STAT'}

def get_db_connection():
    db_url = os.environ.get("COCKROACHDB_URL")
    if not db_url:
        raise ValueError("COCKROACHDB_URL environment variable is not set")
    return psycopg.connect(db_url)

def setup_schema(conn):
    print("Setting up schema...")
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()
    print("Schema setup complete.")

def parse_int(val):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None

def seed_data():
    try:
        conn = get_db_connection()
    except Exception as e:
        print(f"Failed to connect to db: {e}")
        return

    setup_schema(conn)

    print(f"Loading CSV data from {CSV_FILE}...")
    courses = []
    departments = set()
    
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        
        for row in reader:
            if len(row) < 9:
                continue
            
            prefix = row[2].strip()
            number_str = row[3].strip()
            suffix = row[4].strip()
            title = row[5].strip()
            description = row[6].strip()
            min_credits = parse_int(row[7])
            max_credits = parse_int(row[8])
            
            if not prefix:
                continue
                
            number = parse_int(number_str)
            
            departments.add(prefix)
            course_id = f"{prefix} {number_str}{suffix}".strip()
            
            courses.append({
                'id': course_id,
                'prefix': prefix,
                'number': number,
                'suffix': suffix,
                'title': title,
                'description': description,
                'min_credits': min_credits,
                'max_credits': max_credits
            })

    print(f"Found {len(departments)} departments and {len(courses)} courses.")

    # Seed departments
    print("Seeding departments...")
    with conn.cursor() as cur:
        for prefix in departments:
            name = DEPT_MAP.get(prefix, prefix)
            cur.execute("""
                INSERT INTO departments (prefix, name) 
                VALUES (%s, %s)
                ON CONFLICT (prefix) DO NOTHING
            """, (prefix, name))
    conn.commit()

    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Seeding courses...")
    
    # Priority sorting for embeddings
    def course_priority(c):
        if c['prefix'] in PRIORITY_DEPTS:
            return 0
        return 1
        
    courses.sort(key=course_priority)
    
    embedded_count = 0
    max_embeddings = 500
    
    batch_size = 100
    for i in range(0, len(courses), batch_size):
        batch = courses[i:i+batch_size]
        
        with conn.cursor() as cur:
            for c in batch:
                embedding = None
                if embedded_count < max_embeddings:
                    text_to_encode = f"{c['title']}: {c['description']}"
                    embedding_list = model.encode(text_to_encode).tolist()
                    embedding = f"[{','.join(map(str, embedding_list))}]"
                    embedded_count += 1
                
                cur.execute("""
                    INSERT INTO courses (id, title, description, department_prefix, number, suffix, min_credits, max_credits, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        title = EXCLUDED.title,
                        description = EXCLUDED.description,
                        embedding = EXCLUDED.embedding
                """, (
                    c['id'], c['title'], c['description'], c['prefix'], 
                    c['number'], c['suffix'], c['min_credits'], c['max_credits'], 
                    embedding
                ))
        conn.commit()
        print(f"Processed {min(i+batch_size, len(courses))} / {len(courses)} courses...")

    print("Seeding students and student_courses...")
    demo_students = [
        ('student_123', 'Alice Smith', 'alice@psu.edu', ['CMPSC 131', 'CMPSC 132', 'MATH 140']),
        ('student_456', 'Bob Johnson', 'bob@psu.edu', ['ACCTG 211', 'FIN 301']),
        ('student_789', 'Carol Williams', 'carol@psu.edu', ['PHYS 211', 'PHYS 212', 'MATH 230'])
    ]

    with conn.cursor() as cur:
        for s_id, name, email, completed_courses in demo_students:
            cur.execute("""
                INSERT INTO students (id, name, email)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (s_id, name, email))
            
            for cid in completed_courses:
                # We assume the course exists since we parsed the CSV and demo courses are standard
                cur.execute("""
                    INSERT INTO student_courses (student_id, course_id, grade, semester)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (student_id, course_id) DO NOTHING
                """, (s_id, cid, 'A', 'Fall 2025'))
    conn.commit()
    conn.close()
    
    print("Seeding complete!")
    print(f"Summary:")
    print(f"- Departments: {len(departments)}")
    print(f"- Courses Total: {len(courses)}")
    print(f"- Courses with Embeddings: {embedded_count}")
    print(f"- Students seeded: {len(demo_students)}")

if __name__ == "__main__":
    seed_data()
