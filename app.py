import string
import pandas as pd
import re
from werkzeug.security import generate_password_hash, check_password_hash
import flask
from flask import Flask, request, render_template, send_file, redirect, url_for, session, jsonify,send_from_directory
from flask_mail import Mail, Message
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from werkzeug.utils import secure_filename
import uuid
import torch
import pdfplumber
from docx import Document
import os
from io import BytesIO
from markupsafe import Markup, escape
import flash
import sqlite3
from datetime import datetime
import requests
import json
from resume_chatbot import ResumeAssistant
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'doc', 'docx'}
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB file size limit

# Make sure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.secret_key = 'd4e5f8a2b6c1e9a7f1234f6789abc012'
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
chat_history_ids = None
resume_assistant=ResumeAssistant()
###########################
def extract_text(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def create_users_database():
    if not os.path.exists('users.db'):
        with sqlite3.connect('users.db') as conn:
            conn.execute("""CREATE TABLE USERS
            (ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT NOT NULL,
            Email TEXT NOT NULL UNIQUE,
            Password TEXT NOT NULL UNIQUE)""")

create_users_database()

def init_jobs_db():
    conn = sqlite3.connect('jobs.db')
    create_users_database()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS employers
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 company_name TEXT NOT NULL,
                 email TEXT UNIQUE NOT NULL,
                 password TEXT NOT NULL,
                 logo TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS jobs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 title TEXT NOT NULL,
                 email TEXT NOT NULL,
                 contact_number TEXT,
                 description TEXT NOT NULL,
                 requirements TEXT,
                 location TEXT,  
                 country TEXT,   
                 city TEXT,    
                 job_type TEXT NOT NULL,
                 salary TEXT,
                 salary_min INTEGER,  
                 salary_max INTEGER,  
                 salary_currency TEXT,  
                 salary_period TEXT,    
                 skills TEXT,
                 posted_date TEXT NOT NULL,
                 employer_id INTEGER NOT NULL,
                 category TEXT,
                 experience_level TEXT,
                 FOREIGN KEY (employer_id) REFERENCES employers(id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS consultants (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        specialization TEXT,
        years_experience INTEGER,
        bio TEXT,
        linkedin TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    full_name TEXT NOT NULL,
    email TEXT NOT NULL,
    phone TEXT NOT NULL,
    resume_path TEXT NOT NULL,
    cover_letter TEXT,
    linkedin_url TEXT,
    portfolio_url TEXT,
    application_date TEXT NOT NULL,
    status TEXT DEFAULT 'Pending',
    FOREIGN KEY (job_id) REFERENCES jobs(id),
    FOREIGN KEY (user_id) REFERENCES users(id)
);''')
    c.execute('''CREATE TABLE IF NOT EXISTS blogs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        consultant_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        category TEXT NOT NULL,
        image_url TEXT,
        summary TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(consultant_id) REFERENCES consultants(id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS interview_questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        consultant_id INTEGER NOT NULL,
        job_role TEXT NOT NULL,
        experience_level TEXT NOT NULL,
        questions TEXT NOT NULL,
        answers TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(consultant_id) REFERENCES consultants(id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS session_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        consultant_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        phone TEXT,
        purpose TEXT NOT NULL,
        preferred_date TEXT,
        preferred_time TEXT,
        status TEXT DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(consultant_id) REFERENCES consultants(id),
        FOREIGN KEY(user_id) REFERENCES USERS(id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS mcq_questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        consultant_id INTEGER NOT NULL,
        job_role TEXT NOT NULL,
        experience_level TEXT NOT NULL,
        question_text TEXT NOT NULL,
        option1 TEXT NOT NULL,
        option2 TEXT NOT NULL,
        option3 TEXT NOT NULL,
        option4 TEXT NOT NULL,
        correct_option INTEGER NOT NULL,
        explanation TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(consultant_id) REFERENCES consultants(id)
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS user_quiz_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        question_id INTEGER NOT NULL,
        selected_option INTEGER NOT NULL,
        is_correct BOOLEAN NOT NULL,
        quiz_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES USERS(id),
        FOREIGN KEY(question_id) REFERENCES mcq_questions(id)
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS consultant_chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        consultant_id INTEGER NOT NULL,
        message TEXT NOT NULL,
        sender_type TEXT NOT NULL,  -- 'user' or 'consultant'
        sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES USERS(id),
        FOREIGN KEY(consultant_id) REFERENCES consultants(id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS consultant_chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    consultant_id INTEGER NOT NULL,
    message TEXT NOT NULL,
    sender_type TEXT NOT NULL,  -- 'user' or 'consultant'
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_read BOOLEAN DEFAULT 0,
    FOREIGN KEY(user_id) REFERENCES USERS(id),
    FOREIGN KEY(consultant_id) REFERENCES consultants(id)
    )''')
    conn.commit()
    conn.close()

# Add this after your init_jobs_db() function
def update_db_schema():
    conn = sqlite3.connect('jobs.db')
    c = conn.cursor()
    try:
        # Check if columns exist
        c.execute("PRAGMA table_info(employers)")
        columns = [col[1] for col in c.fetchall()]
        
        if 'profile_views' not in columns:
            c.execute("ALTER TABLE employers ADD COLUMN profile_views INTEGER DEFAULT 0")
        
        if 'applications_count' not in columns:
            c.execute("ALTER TABLE employers ADD COLUMN applications_count INTEGER DEFAULT 0")
        
        if 'has_paid' not in columns:
            c.execute("ALTER TABLE employers ADD COLUMN has_paid BOOLEAN DEFAULT 0")
            
        conn.commit()
    except Exception as e:
        print(f"Error updating schema: {e}")
    finally:
        conn.close()

# Call this function after init_jobs_db()
init_jobs_db()
update_db_schema()

# Helper function to get database connection
def get_jobs_db():
    conn = sqlite3.connect('jobs.db')
    conn.row_factory = sqlite3.Row
    return conn

# Helper function to get database connection
def get_db():
    conn = sqlite3.connect('jobs.db')
    conn.row_factory = sqlite3.Row  # This enables column access by name
    return conn

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        if len(password) < 8 or not re.search(r"[a-zA-Z]", password) or not re.search(r"\d", password):
            return render_template('signup.html', error="Password must be at least 8 characters long and include both letters and numbers.")
        hashed_password = generate_password_hash(password)
        with sqlite3.connect('users.db') as connect:
            pointer = connect.cursor()
            pointer.execute("SELECT * from USERS WHERE email=?", (email,))
            exists = pointer.fetchone()
            if exists:
                return render_template('signup.html', error="ERROR! User already exists.")
            pointer.execute("INSERT INTO USERS (name, email, password) VALUES (?, ?, ?)", 
                          (name, email, hashed_password))
            connect.commit()
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        with sqlite3.connect('users.db') as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM USERS WHERE email = ?", (email,))
            user = cursor.fetchone()
            if user and check_password_hash(user[3], password):
                session['user'] = user[1]
                session['user_id'] = user[0]
                return redirect(url_for('home'))
            else:
                return render_template("login.html", error="Invalid credentials")
    return render_template("login.html", error=None)

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("home.html", user=session['user'])

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/resume')
def resume():
    return render_template('resume.html')

def get_response():
    user_input = request.form.get("user_input")
    
    if not user_input:
        return jsonify({"bot_response": "Please enter a message."})

    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_output = model.generate(new_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(bot_output[:, new_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return jsonify({"bot_response": response})

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        # Handle file upload
        if 'resume' in request.files:
            file = request.files['resume']
            if file.filename != '':
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Analyze resume
                analysis = resume_assistant.analyze_resume(filepath)
                return jsonify({
                    'type': 'analysis',
                    'content': analysis
                })
        
        # Handle text message
        if 'message' in request.form:
            user_message = request.form['message']
            response = resume_assistant.get_response(user_message)
            return jsonify({
                'type': 'message',
                'content': response
            })
    
    return render_template('chatbot.html')

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    data = request.json
    user_message = data.get('message', '')
    resume_text = data.get('resume_text', '')
    
    # Get response from your chatbot
    response = resume_assistant.get_response(user_message, resume_text)
    
    return jsonify({
        'status': 'success',
        'response': response
    })

@app.route('/api/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze the resume
        analysis = resume_assistant.analyze_resume(filepath)
        resume_text = ""
        
        # Extract text for context
        if filepath.endswith('.pdf'):
            resume_text = resume_assistant._extract_text_from_pdf(filepath)
        elif filepath.endswith('.docx'):
            resume_text = resume_assistant._extract_text_from_docx(filepath)
        
        # Clean up - remove the file after processing
        os.remove(filepath)
        
        return jsonify({
            'status': 'success',
            'message': 'Resume uploaded successfully',
            'analysis': analysis,
            'resume_text': resume_text[:1000]  # Send first 1000 chars for context
        })
    
    return jsonify({'status': 'error', 'message': 'File upload failed'})

@app.route('/get_chat_response', methods=['POST'])
def get_chat_response():
    data = request.json
    user_input = data.get('message', '')
    resume_text = data.get('resume_text', '')
    
    response = resume_assistant.get_response(user_input, resume_text)
    return jsonify({'response': response})

@app.route("/upload_cv_chat", methods=["POST"])
def upload_cv_chat():
    if "cv" in request.files:
        cv_file = request.files["cv"]

        if not cv_file.filename.lower().endswith(".pdf"):
            return render_template("chatbot.html", greeting="Please upload a PDF file.")

        try:
            with pdfplumber.open(cv_file) as pdf:
                all_text = ''
                for page in pdf.pages:
                    all_text += page.extract_text() + '\n'
        except Exception as e:
            return render_template("chatbot.html", greeting="Error reading PDF. Please try another file.")

        if not all_text.strip():
            return render_template("chatbot.html", greeting="Sorry, we couldn't extract readable text from this PDF.")

        prompt = f"Here is my resume:\n{all_text[:1000]}\n\nWhat can I improve?"
        suggestion = chatbot_pipeline(prompt, max_new_tokens=80)[0]["generated_text"]

        return render_template("chatbot.html", greeting=suggestion)
    
    return redirect("/chatbot")

@app.route("/manual_resume", methods=["GET", "POST"])
def manual_resume():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("manual_resume.html")
@app.route('/generate_ai_resume', methods=['POST'])
def generate_resume():
    try:
        data = request.json
        template_id = data.get('template_id')
        resume_data = data.get('resume_data')

        # Create a PDF in memory
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        
        # Simple resume generation - customize this based on your templates
        p.drawString(100, 750, f"Resume for {resume_data.get('name', '')}")
        p.drawString(100, 730, f"Email: {resume_data.get('email', '')}")
        p.drawString(100, 710, f"Phone: {resume_data.get('phone', '')}")
        
        # Add education section
        y_position = 680
        p.drawString(100, y_position, "Education:")
        y_position -= 20
        for edu in resume_data.get('education', []):
            p.drawString(120, y_position, f"{edu.get('degree', '')} at {edu.get('institution', '')}")
            y_position -= 20
        
        # Add experience section
        p.drawString(100, y_position, "Experience:")
        y_position -= 20
        for exp in resume_data.get('experience', []):
            p.drawString(120, y_position, f"{exp.get('job_title', '')} at {exp.get('company', '')}")
            y_position -= 20
        
        p.showPage()
        p.save()
        
        # Get the PDF data and send it
        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name='resume.pdf',
            mimetype='application/pdf'
        )
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
        
def format_date(date_str):
    try:
        if date_str.lower() == 'present':
            return date_str
        if '/' in date_str:
            month, year = date_str.split('/')
            return datetime.strptime(f"{month}/01/{year}", "%m/%d/%Y").strftime("%b %Y")
        return date_str
    except:
        return date_str

@app.route('/template_selection', methods=['GET', 'POST'])
def template_selection():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Store the form data in session for later use
        session['resume_data'] = {
            'name': request.form.get('name', '').strip(),
            'email': request.form.get('email', '').strip(),
            'phone': request.form.get('phone', '').strip(),
            'address': request.form.get('address', '').strip(),
            'skills': [s.strip() for s in request.form.get("skills", "").split(",") if s.strip()],
            'certifications': [c.strip() for c in request.form.get("certifications", "").split(",") if c.strip()],
            'education': [],
            'experience': []
        }
        
        # Process education data
        degrees = request.form.getlist('degree[]')
        institutions = request.form.getlist('institution[]')
        education_start_dates = request.form.getlist('education_start_date[]')
        education_end_dates = request.form.getlist('education_end_date[]')
        education_descriptions = request.form.getlist('education_description[]')
        marks_list = request.form.getlist('marks[]')
        
        for i in range(len(degrees)):
            if degrees[i].strip():
                session['resume_data']['education'].append({
                    'degree': degrees[i].strip(),
                    'institution': institutions[i].strip(),
                    'start_date': format_date(education_start_dates[i].strip()),
                    'end_date': format_date(education_end_dates[i].strip()),
                    'description': education_descriptions[i].strip() if i < len(education_descriptions) else '',
                    'marks': marks_list[i].strip() if i < len(marks_list) else ''
                })
        
        # Process work experience data
        job_titles = request.form.getlist('job_title[]')
        companies = request.form.getlist('company[]')
        start_dates = request.form.getlist('start_date[]')
        end_dates = request.form.getlist('end_date[]')
        job_descriptions = request.form.getlist('job_description[]')
        
        for i in range(len(job_titles)):
            if job_titles[i].strip():
                session['resume_data']['experience'].append({
                    'job_title': job_titles[i].strip(),
                    'company': companies[i].strip(),
                    'start_date': format_date(start_dates[i].strip()),
                    'end_date': format_date(end_dates[i].strip()),
                    'description': job_descriptions[i].strip() if i < len(job_descriptions) else ''
                })
    
    return render_template('template_selection.html')

@app.route('/set_template', methods=['POST'])
def set_template():
    if 'user' not in session or 'resume_data' not in session:
        return jsonify({'success': False, 'error': 'Session expired or data missing'})
    
    data = request.get_json()
    template_id = data.get('template_id')
    
    # Store the selected template in session
    session['selected_template'] = template_id
    
    return jsonify({'success': True})

@app.route('/generate_from_template', methods=['GET'])
def generate_from_template():
    if 'user' not in session or 'resume_data' not in session or 'selected_template' not in session:
        return redirect(url_for('login'))
    
    from jinja2 import Environment, FileSystemLoader
    from weasyprint import HTML
    from io import BytesIO
    
    # Prepare user data from session
    user_data = session['resume_data']
    user_data['current_date'] = datetime.now().strftime("%B %Y")
    template_id = session['selected_template']
    
    # Generate PDF with the selected template
    env = Environment(loader=FileSystemLoader('templates'))
    template_file = f'resume_template{template_id}.html'
    template = env.get_template(template_file)
    html_out = template.render(user_data)
    
    pdf_file = BytesIO()
    HTML(string=html_out).write_pdf(pdf_file)
    pdf_file.seek(0)
    
    # Save to database if needed
    try:
        with sqlite3.connect('users.db') as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS RESUMES (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    resume_data TEXT,
                    template_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES USERS(id)
                )
            """)
            cursor.execute("""
                INSERT INTO RESUMES (user_id, resume_data, template_id)
                VALUES (?, ?, ?)
            """, (session['user_id'], str(user_data), template_id))
            conn.commit()
    except Exception as e:
        print(f"Error saving resume to database: {e}")
    
    return send_file(
        pdf_file,
        as_attachment=True,
        download_name=f"{user_data['name'].replace(' ', '_')}_resume.pdf",
        mimetype='application/pdf'
    )

@app.route('/blogs')
def blogs():
    conn = get_jobs_db()
    c = conn.cursor()
    # Get all blogs with author information
    c.execute('''SELECT b.*, c.name as author_name 
               FROM blogs b JOIN consultants c ON b.consultant_id = c.id
               ORDER BY b.created_at DESC''')
    blogs = c.fetchall()
    conn.close()
    # Get all unique categories for filtering
    categories = ['For Job Seekers', 'CV', 'For Employers', 
                 'Case Studies', 'Internship', 'Reports', 'News']
    return render_template('blogs.html', blogs=blogs, categories=categories)
    
@app.route('/employers')
def employers():
    """Landing page for employers"""
    return render_template('employers.html')

@app.route('/employers/register', methods=['GET', 'POST'])
def employer_register():
    if request.method == 'POST':
        company_name = request.form['company_name']
        email = request.form['email']
        password = request.form['password']
        
        conn = get_jobs_db()
        c = conn.cursor()
        
        # Check if email exists
        c.execute("SELECT * FROM employers WHERE email = ?", (email,))
        if c.fetchone():
            conn.close()
            return render_template('employer_register.html', error="Email already exists")
        
        hashed_password = generate_password_hash(password)
        logo_filename = None
        
        # Handle logo upload
        if 'logo' in request.files:
            logo = request.files['logo']
            if logo.filename != '':
                logo_filename = secure_filename(logo.filename)
                logo.save(os.path.join(app.config['UPLOAD_FOLDER'], logo_filename))
        
        # Insert new employer
        c.execute("INSERT INTO employers (company_name, email, password, logo) VALUES (?, ?, ?, ?)",
                 (company_name, email, hashed_password, logo_filename))
        employer_id = c.lastrowid
        
        conn.commit()
        conn.close()
        
        session['employer_id'] = employer_id
        return redirect(url_for('employer_dashboard'))
    
    return render_template('employer_register.html')

@app.route('/employers/login', methods=['GET', 'POST'])
def employer_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = get_jobs_db()
        c = conn.cursor()
        c.execute("SELECT * FROM employers WHERE email = ?", (email,))
        employer = c.fetchone()
        conn.close()
        
        if employer and check_password_hash(employer['password'], password):
            session['employer_id'] = employer['id']
            return redirect(url_for('employer_dashboard'))
        
        return render_template('employer_login.html', error="Invalid credentials")
    
    return render_template('employer_login.html')

@app.route('/jobs')
def jobs():
    # Get filter parameters from request
    search = request.args.get('search', '')
    country = request.args.get('country', '')
    city = request.args.get('city', '')
    job_type = request.args.get('job_type', '')
    category = request.args.get('category', '')
    experience = request.args.get('experience', '')
    salary_range = request.args.get('salary_range', '')

    conn = get_jobs_db()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Get all unique countries and cities for filters
    c.execute("SELECT DISTINCT country FROM jobs WHERE country IS NOT NULL AND country != ''")
    countries = [row['country'] for row in c.fetchall()]
    
    c.execute("SELECT DISTINCT city FROM jobs WHERE city IS NOT NULL AND city != ''")
    cities = [row['city'] for row in c.fetchall()]

    # Base query
    query = '''SELECT j.*, e.company_name, e.logo 
               FROM jobs j JOIN employers e ON j.employer_id = e.id 
               WHERE 1=1'''
    params = []

    # Add filters
    if search:
        query += " AND (j.title LIKE ? OR j.description LIKE ? OR e.company_name LIKE ? OR j.skills LIKE ?)"
        params.extend([f'%{search}%', f'%{search}%', f'%{search}%', f'%{search}%'])
    
    if country:
        query += " AND j.country = ?"
        params.append(country)
    
    if city:
        query += " AND j.city = ?"
        params.append(city)
    
    if job_type:
        query += " AND j.job_type = ?"
        params.append(job_type)
    
    if category:
        query += " AND j.category = ?"
        params.append(category)
    
    if experience:
        query += " AND j.experience_level = ?"
        params.append(experience)
    
    if salary_range:
        if salary_range == '0-30000':
            query += " AND (j.salary_min >= 0 AND j.salary_max <= 30000)"
        elif salary_range == '30000-60000':
            query += " AND (j.salary_min >= 30000 AND j.salary_max <= 60000)"
        elif salary_range == '60000-90000':
            query += " AND (j.salary_min >= 60000 AND j.salary_max <= 90000)"
        elif salary_range == '90000+':
            query += " AND j.salary_min >= 90000"

    query += " ORDER BY j.posted_date DESC"
    c.execute(query, params)
    jobs = []
    
    # Convert each row to a dict and ensure posted_date is a datetime object
    for row in c.fetchall():
        job = dict(row)
        # Handle posted_date conversion
        if isinstance(job['posted_date'], str):
            try:
                job['posted_date'] = datetime.strptime(job['posted_date'], '%Y-%m-%d %H:%M:%S')
            except ValueError:
                job['posted_date'] = datetime.now()
        jobs.append(job)
    
    # Prepare current filters for template
    current_filters = {
        'search': search,
        'country': country,
        'city': city,
        'job_type': job_type,
        'category': category,
        'experience': experience,
        'salary_range': salary_range
    }
    conn.close()
    return render_template('jobs.html', 
                         jobs=jobs, 
                         current_filters=current_filters,
                         countries=countries,
                         cities=cities,
                         salary_ranges=['0-30000', '30000-60000', '60000-90000', '90000+'])

@app.route('/apply/<int:job_id>', methods=['GET', 'POST'])
def apply_job(job_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_jobs_db()
    c = conn.cursor()
    c.execute('''SELECT j.*, e.company_name, e.logo 
               FROM jobs j JOIN employers e ON j.employer_id = e.id 
               WHERE j.id = ?''', (job_id,))
    job = c.fetchone()
    conn.close()
    
    if not job:
        flash('Job not found', 'error')
        return redirect(url_for('jobs'))
    
    if request.method == 'POST':
        # Handle file upload
        if 'resume' not in request.files:
            flash('No resume file uploaded', 'error')
            return redirect(url_for('apply_job', job_id=job_id))
            
        resume = request.files['resume']
        if resume.filename == '':
            flash('No selected resume file', 'error')
            return redirect(url_for('apply_job', job_id=job_id))
            
        if resume and allowed_file(resume.filename):
            # Create a secure filename and save it
            filename = secure_filename(resume.filename)
            # Add user_id to filename to make it unique
            unique_filename = f"{session['user_id']}_{filename}"
            resume_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            try:
                resume.save(resume_path)
                
                # Save application to database
                conn = get_jobs_db()
                c = conn.cursor()
                c.execute('''INSERT INTO applications 
                          (job_id, user_id, full_name, email, phone, resume_path, 
                           cover_letter, linkedin_url, portfolio_url, application_date)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                         (job_id, session['user_id'], 
                          request.form.get('full_name'),
                          request.form.get('email'),
                          request.form.get('phone'),
                          unique_filename,  # Store the unique filename
                          request.form.get('cover_letter', ''),
                          request.form.get('linkedin_url', ''),
                          request.form.get('portfolio_url', ''),
                          datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                
                # Increment application count for the employer
                c.execute('''UPDATE employers 
                           SET applications_count = applications_count + 1 
                           WHERE id = (SELECT employer_id FROM jobs WHERE id = ?)''', (job_id,))
                conn.commit()
                flash('Application submitted successfully!', 'success')
                return redirect(url_for('jobs'))
            except Exception as e:
                conn.rollback()
                # Clean up the file if there was a database error
                if os.path.exists(resume_path):
                    os.remove(resume_path)
                flash(f'Error submitting application: {str(e)}', 'error')
            finally:
                conn.close()
        else:
            flash('Invalid file type for resume (PDF, DOC, DOCX only)', 'error')    
    return render_template('apply_job.html', job=job)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'pdf', 'doc', 'docx'}

@app.route('/apply/confirmation/<int:job_id>')
def application_confirmation(job_id):
    return redirect(url_for('jobs'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Update the employer_dashboard route
@app.route('/employers/dashboard')
def employer_dashboard():
    if 'employer_id' not in session:
        return redirect(url_for('employer_login'))
    
    conn = get_jobs_db()
    c = conn.cursor()
    
    # Get employer info
    c.execute("SELECT * FROM employers WHERE id = ?", (session['employer_id'],))
    employer = c.fetchone()
    
    # Get employer's jobs with view counts
    c.execute("SELECT * FROM jobs WHERE employer_id = ? ORDER BY posted_date DESC", (session['employer_id'],))
    jobs = c.fetchall()
    
    # Get applications count
    c.execute('''SELECT COUNT(*) as app_count 
               FROM applications a JOIN jobs j ON a.job_id = j.id 
               WHERE j.employer_id = ?''', (session['employer_id'],))
    applications_count = c.fetchone()['app_count']
    
    # Get profile views count (you'll need to add this to your employers table)
    c.execute("SELECT profile_views FROM employers WHERE id = ?", (session['employer_id'],))
    profile_views = c.fetchone()['profile_views']
    
    conn.close()
    
    current_date = datetime.now().strftime('%B %d, %Y')
    return render_template('employer_dashboard.html', 
                         employer=employer, 
                         jobs=jobs,
                         applications_count=applications_count,
                         profile_views=profile_views,
                         current_date=current_date)


@app.route('/employers/post_job', methods=['GET', 'POST'])
def post_job():
    if 'employer_id' not in session:
        return redirect(url_for('employer_login'))

    if request.method == 'POST':
        # Get all form data
        title = request.form['title']
        email = request.form['email']
        contact_number = request.form['contact_number']
        description = request.form['description']
        requirements = request.form['requirements']
        country = request.form['country']
        city = request.form['city']
        job_type = request.form['job_type']
        salary_min = request.form['salary_min']
        salary_max = request.form['salary_max']
        salary_currency = request.form['salary_currency']
        salary_period = request.form['salary_period']
        skills = request.form['skills']
        category = request.form['category']
        experience_level = request.form['experience_level']
        
        # Format salary for display
        salary = f"{salary_currency}{salary_min} - {salary_max} {salary_currency} {salary_period}"
        
        posted_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        conn = get_jobs_db()
        c = conn.cursor()
        
        # Check if location column exists for backward compatibility
        c.execute("PRAGMA table_info(jobs)")
        columns = [col[1] for col in c.fetchall()]
        has_location = 'location' in columns
        
        # Build the insert query based on available columns
        if has_location:
            c.execute('''INSERT INTO jobs 
                        (title, email, contact_number, description, requirements, 
                        country, city, location, job_type, salary, salary_min, salary_max, 
                        salary_currency, salary_period, skills, posted_date, employer_id, 
                        category, experience_level)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (title, email, contact_number, description, requirements,
                      country, city, f"{city}, {country}", job_type, salary, salary_min, salary_max,
                      salary_currency, salary_period, skills, posted_date, session['employer_id'],
                      category, experience_level))
        else:
            c.execute('''INSERT INTO jobs 
                        (title, email, contact_number, description, requirements, 
                        country, city, job_type, salary, salary_min, salary_max, 
                        salary_currency, salary_period, skills, posted_date, employer_id, 
                        category, experience_level)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (title, email, contact_number, description, requirements,
                      country, city, job_type, salary, salary_min, salary_max,
                      salary_currency, salary_period, skills, posted_date, session['employer_id'],
                      category, experience_level))
        
        conn.commit()
        conn.close()
        return redirect(url_for('employer_dashboard'))
    
    # For GET request, show form with all options
    job_types = ['Full-Time', 'Part-Time', 'Remote', 'Temporary', 'Internship', 'Contract']
    categories = ['Technology', 'Healthcare', 'Finance', 'Education', 'Retail', 'Manufacturing', 'Other']
    experience_levels = ['Entry Level', 'Mid Level', 'Senior Level', 'Executive']
    salary_currencies = ['USD', 'EUR', 'GBP', 'PKR', 'INR', 'AED']
    salary_periods = ['hourly', 'daily', 'weekly', 'monthly', 'annually']
    
    return render_template('post_job.html', 
                         job_types=job_types,
                         categories=categories,
                         experience_levels=experience_levels,
                         salary_currencies=salary_currencies,
                         salary_periods=salary_periods)

@app.route('/employer/logout')
def employer_logout():
    session.pop('employer_id', None)
    return redirect(url_for('employers'))

@app.route('/consultants')
def consultant_landing():
    print("Consultant landing page reached")
    return render_template("consultants.html")


@app.route('/consultants/register', methods=['GET', 'POST'])
def consultant_register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        specialization = request.form['specialization']
        years_experience = request.form['years_experience']
        bio = request.form['bio']
        linkedin = request.form.get('linkedin', None)

        # Validate passwords match
        if password != confirm_password:
            return render_template("consultant_register.html", error="Passwords do not match")

        # Check if email exists
        with get_jobs_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM consultants WHERE email = ?", (email,))
            if cursor.fetchone():
                return render_template("consultant_register.html", error="Email already registered")

            # Create consultant without photo
            hashed_password = generate_password_hash(password)
            cursor.execute("""
                INSERT INTO consultants 
                (name, email, password, specialization, years_experience, bio, linkedin)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (name, email, hashed_password, specialization, 
                 years_experience, bio, linkedin))
            conn.commit()

        return redirect(url_for('consultant_login'))

    return render_template("consultant_register.html")

@app.route('/consultants/login', methods=['GET', 'POST'])
def consultant_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        with get_jobs_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM consultants WHERE email = ?", (email,))
            consultant = cursor.fetchone()

            if consultant and check_password_hash(consultant['password'], password):
                session['consultant_id'] = consultant['id']
                return redirect(url_for('consultant_dashboard'))

        return render_template('consultant_login.html', error="Invalid credentials")

    return render_template('consultant_login.html')

@app.route('/consultants/dashboard')
def consultant_dashboard():
    if 'consultant_id' not in session:
        return redirect(url_for('consultant_login'))
    
    with get_jobs_db() as conn:
        c = conn.cursor()
        c.execute("ATTACH DATABASE 'users.db' AS usersdb")
        c.execute("SELECT * FROM consultants WHERE id = ?", (session['consultant_id'],))
        consultant = c.fetchone()
        c.execute("SELECT * FROM session_requests WHERE consultant_id = ?", (session['consultant_id'],))
        requests = c.fetchall()
        c.execute('''
            SELECT DISTINCT usersdb.USERS.id as user_id, 
                            usersdb.USERS.name as user_name, 
                            MAX(c.sent_at) as last_message_time
            FROM consultant_chats c
            JOIN usersdb.USERS ON c.user_id = usersdb.USERS.id
            WHERE c.consultant_id = ?
            GROUP BY usersdb.USERS.id, usersdb.USERS.name
            ORDER BY last_message_time DESC
        ''', (session['consultant_id'],))
        active_sessions = c.fetchall()
        c.execute("SELECT * FROM blogs WHERE consultant_id = ?", (session['consultant_id'],))
        blogs = c.fetchall()
        
        # Get MCQ questions
        c.execute("SELECT * FROM mcq_questions WHERE consultant_id = ? ORDER BY created_at DESC", (session['consultant_id'],))
        questions = c.fetchall()
    
    return render_template("consultant_dashboard.html", 
                           consultant=consultant, 
                           requests=requests,
                           active_sessions=active_sessions,
                           blogs=blogs,
                           questions=questions)


@app.route('/consultants/interview_questions', methods=['GET', 'POST'])
def generate_interview_questions():
    if 'consultant_id' not in session:
        return redirect(url_for('consultant_login'))
    return render_template('consultant_interview_questions.html')

@app.route('/consultants/add_blog', methods=['GET', 'POST'],endpoint='add_blog')
def consultant_add_blog():
    if 'consultant_id' not in session:
        return redirect(url_for('consultant_login'))
    
    if request.method == 'POST':
        title = request.form['title']
        category = request.form['category']
        summary = request.form['summary']
        content = request.form['content']
        
        image_url = None
        if 'image' in request.files:
            image = request.files['image']
            if image.filename != '':
                filename = secure_filename(image.filename)
                image_path = os.path.join('static/uploads/blogs', filename)
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                image.save(image_path)
                image_url = f'/static/uploads/blogs/{filename}'
        
        with get_jobs_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""INSERT INTO blogs 
                            (consultant_id, title, category, image_url, summary, content)
                            VALUES (?, ?, ?, ?, ?, ?)""",
                         (session['consultant_id'], title, category, image_url, summary, content))
            conn.commit()
        
        return redirect(url_for('consultant_dashboard'))
    
    return render_template('consultant_add_blog.html')

@app.route('/consultants/add_questions', methods=['GET','POST'],endpoint="add_questions")
def consultant_add_questions():
    if 'consultant_id' not in session:
        return redirect(url_for('consultant_login'))
    
    if request.method == 'POST':
        job_role = request.form['job_role']
        experience_level = request.form['experience_level']
        questions = request.form['questions']
        answers = request.form.get('answers', '')
        
        with get_jobs_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""INSERT INTO interview_questions 
                            (consultant_id, job_role, experience_level, questions, answers)
                            VALUES (?, ?, ?, ?, ?)""",
                         (session['consultant_id'], job_role, experience_level, questions, answers))
            conn.commit()
        
        return redirect(url_for('consultant_dashboard'))
    
    return render_template('consultant_add_questions.html')

@app.route('/consultants/logout')
def consultant_logout():
    session.pop('consultant_id', None)
    return redirect(url_for('consultant_landing'))

# Interview Questions routes
@app.route('/interview_questions')
def all_interview_questions():
    """View all interview questions (accessible to all users)"""
    conn = get_jobs_db()
    c = conn.cursor()
    c.execute('''SELECT q.*, c.name as author_name 
               FROM interview_questions q JOIN consultants c ON q.consultant_id = c.id
               ORDER BY q.created_at DESC''')
    questions = c.fetchall()
    conn.close()
    return render_template('all_interview_questions.html', questions=questions)

@app.route('/interview_question/<int:question_id>')
def view_interview_question(question_id):
    """View single interview question set (accessible to all users)"""
    conn = get_jobs_db()
    c = conn.cursor()
    c.execute('''SELECT q.*, c.name as author_name, c.specialization
               FROM interview_questions q JOIN consultants c ON q.consultant_id = c.id
               WHERE q.id = ?''', (question_id,))
    question = c.fetchone()
    conn.close()
    return render_template('view_interview_question.html', question=question)

@app.route('/consultants/add_blog', methods=['GET', 'POST'])
def consultant_add_blog():
    if 'consultant_id' not in session:
        return redirect(url_for('consultant_login'))
    
    if request.method == 'POST':
        title = request.form['title']
        category = request.form['category']
        summary = request.form['summary']
        content = request.form['content']
        
        image_url = None
        if 'image' in request.files:
            image = request.files['image']
            if image.filename != '':
                filename = secure_filename(image.filename)
                image_path = os.path.join('static/uploads/blogs', filename)
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                image.save(image_path)
                image_url = f'/static/uploads/blogs/{filename}'
        
        with get_jobs_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""INSERT INTO blogs 
                            (consultant_id, title, category, image_url, summary, content)
                            VALUES (?, ?, ?, ?, ?, ?)""",
                         (session['consultant_id'], title, category, image_url, summary, content))
            conn.commit()
        
        return redirect(url_for('consultant_dashboard'))
    
    return redirect(url_for('consultant_dashboard'))

@app.route('/consultants/add_questions', methods=['GET', 'POST'])
def consultant_add_questions():
    if 'consultant_id' not in session:
        return redirect(url_for('consultant_login'))
    
    if request.method == 'POST':
        job_role = request.form['job_role']
        experience_level = request.form['experience_level']
        questions = request.form['questions']
        answers = request.form.get('answers', '')
        
        with get_jobs_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""INSERT INTO interview_questions 
                            (consultant_id, job_role, experience_level, questions, answers)
                            VALUES (?, ?, ?, ?, ?)""",
                         (session['consultant_id'], job_role, experience_level, questions, answers))
            conn.commit()
        
        return redirect(url_for('consultant_dashboard'))
    
    return redirect(url_for('consultant_dashboard'))

@app.route('/blogs')
def all_blogs():
    conn = get_jobs_db()
    c = conn.cursor()
    
    # Get the selected category from query parameters
    selected_category = request.args.get('category', 'all')
    
    # Base query
    query = '''SELECT b.*, c.name as author_name 
               FROM blogs b JOIN consultants c ON b.consultant_id = c.id'''
    
    # Add category filter if not 'all'
    if selected_category != 'all':
        query += " WHERE b.category = ?"
        c.execute(query, (selected_category,))
    else:
        c.execute(query)
    
    blogs = c.fetchall()
    
    # Get all unique categories for the filter dropdown
    c.execute("SELECT DISTINCT category FROM blogs")
    categories = [row['category'] for row in c.fetchall()]
    
    conn.close()
    
    return render_template('blogs.html', 
                         blogs=blogs, 
                         categories=categories,
                         selected_category=selected_category)

@app.route('/blog/<int:blog_id>')
def view_blog(blog_id):
    """View single blog (accessible to all users)"""
    conn = get_jobs_db()
    c = conn.cursor()
    c.execute('''SELECT b.*, c.name as author_name, c.specialization 
               FROM blogs b JOIN consultants c ON b.consultant_id = c.id
               WHERE b.id = ?''', (blog_id,))
    blog = c.fetchone()
    conn.close()    
    return render_template('view_blog.html', blog=blog)

@app.route('/delete_blog/<int:blog_id>', methods=['DELETE'])
def delete_blog(blog_id):
    if 'consultant_id' not in session:
        return jsonify({'success': False, 'error': 'Not authorized'}), 401
    
    with get_jobs_db() as conn:
        cursor = conn.cursor()
        # First check if the blog belongs to this consultant
        cursor.execute("SELECT consultant_id FROM blogs WHERE id = ?", (blog_id,))
        blog = cursor.fetchone()
        
        if not blog:
            return jsonify({'success': False, 'error': 'Blog not found'}), 404
        
        if blog['consultant_id'] != session['consultant_id']:
            return jsonify({'success': False, 'error': 'Not authorized'}), 403
        
        # Delete the blog
        cursor.execute("DELETE FROM blogs WHERE id = ?", (blog_id,))
        conn.commit()
    
    return jsonify({'success': True})

@app.route('/jobs/delete/<int:job_id>', methods=['POST'])
def delete_job(job_id):
    if 'employer_id' not in session:
        return redirect(url_for('employer_login'))
    
    conn = get_jobs_db()
    c = conn.cursor()
    
    # Verify the job belongs to the logged-in employer
    c.execute("SELECT employer_id FROM jobs WHERE id = ?", (job_id,))
    job = c.fetchone()
    
    if job and job['employer_id'] == session['employer_id']:
        try:
            c.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            conn.commit()
            flash('Job deleted successfully', 'success')
        except Exception as e:
            conn.rollback()
            flash(f'Error deleting job: {str(e)}', 'error')
    else:
        flash('You cannot delete this job', 'error')
    
    conn.close()
    return redirect(url_for('employer_dashboard'))

from flask import Flask, render_template, request, redirect, url_for, flash
import sqlite3

# Edit Job - Display Form
@app.route('/jobs/edit/<int:job_id>', methods=['GET', 'POST'])
def edit_job(job_id):
    if 'employer_id' not in session:
        return redirect(url_for('employer_login'))
    
    conn = get_jobs_db()
    cursor = conn.cursor()
    
    # Verify the job belongs to the logged-in employer
    cursor.execute("SELECT * FROM jobs WHERE id = ? AND employer_id = ?", 
                  (job_id, session['employer_id']))
    job = cursor.fetchone()
    
    if not job:
        flash('Job not found or you do not have permission to edit it', 'error')
        return redirect(url_for('employer_dashboard'))
    
    if request.method == 'POST':
        try:
            # Get all form data
            title = request.form['title']
            email = request.form['email']
            contact_number = request.form.get('contact_number', '')
            description = request.form['description']
            requirements = request.form['requirements']
            country = request.form['country']
            city = request.form['city']
            job_type = request.form['job_type']
            salary_min = request.form['salary_min']
            salary_max = request.form['salary_max']
            salary_currency = request.form['salary_currency']
            salary_period = request.form['salary_period']
            skills = request.form['skills']
            category = request.form['category']
            experience_level = request.form['experience_level']
            
            # Format salary for display
            salary = f"{salary_currency}{salary_min} - {salary_max} {salary_period}"
            
            cursor.execute("""
                UPDATE jobs SET 
                    title = ?,
                    email = ?,
                    contact_number = ?,
                    description = ?,
                    requirements = ?,
                    country = ?,
                    city = ?,
                    job_type = ?,
                    salary = ?,
                    salary_min = ?,
                    salary_max = ?,
                    salary_currency = ?,
                    salary_period = ?,
                    skills = ?,
                    category = ?,
                    experience_level = ?
                WHERE id = ?
            """, (
                title, email, contact_number, description, requirements,
                country, city, job_type, salary, salary_min, salary_max,
                salary_currency, salary_period, skills, category,
                experience_level, job_id
            ))
            
            conn.commit()
            flash('Job updated successfully!', 'success')
            return redirect(url_for('employer_dashboard'))
        except Exception as e:
            conn.rollback()
            flash(f'Error updating job: {str(e)}', 'error')
        finally:
            conn.close()
    
    # For GET request, show the form with job data
    columns = [column[0] for column in cursor.description]
    job_dict = dict(zip(columns, job))
    conn.close()
    
    return render_template('edit_job.html', job=job_dict)

# Edit Job - Process Form Submission
@app.route('/jobs/edit/<int:job_id>', methods=['POST'])
def update_job(job_id):
    # Get form data
    title = request.form['title']
    job_type = request.form['job_type']
    category = request.form['category']
    experience_level = request.form['experience_level']
    city = request.form['city']
    country = request.form['country']
    salary_currency = request.form['salary_currency']
    salary_min = request.form['salary_min']
    salary_max = request.form['salary_max']
    salary_period = request.form['salary_period']
    description = request.form['description']
    requirements = request.form['requirements']
    skills = request.form['skills']
    email = request.form['email']
    contact_number = request.form.get('contact_number', '')
    
    conn = sqlite3.connect('jobs.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            UPDATE jobs SET 
                title = ?,
                job_type = ?,
                category = ?,
                experience_level = ?,
                city = ?,
                country = ?,
                salary_currency = ?,
                salary_min = ?,
                salary_max = ?,
                salary_period = ?,
                description = ?,
                requirements = ?,
                skills = ?,
                email = ?,
                contact_number = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (
            title, job_type, category, experience_level, city, country,
            salary_currency, salary_min, salary_max, salary_period,
            description, requirements, skills, email, contact_number,
            job_id
        ))
        
        conn.commit()
        flash('Job updated successfully!', 'success')
    except Exception as e:
        conn.rollback()
        flash(f'Error updating job: {str(e)}', 'error')
    finally:
        conn.close()
    
    return redirect(url_for('employer_dashboard'))
@app.route('/jobs/<int:job_id>')
def job_detail(job_id):
    conn = get_jobs_db()
    c = conn.cursor()
    
    c.execute('''UPDATE employers 
               SET profile_views = profile_views + 1 
               WHERE id = (SELECT employer_id FROM jobs WHERE id = ?)''', (job_id,))
    conn.commit()
    c.execute('''SELECT j.*, e.company_name, e.logo 
               FROM jobs j JOIN employers e ON j.employer_id = e.id 
               WHERE j.id = ?''', (job_id,))
    job = c.fetchone()
    conn.close()
    
    if not job:
        abort(404)
    
    return render_template('job_detail.html', job=job)

# Add this route to view applications
@app.route('/employers/candidates')
def view_candidates():
    if 'employer_id' not in session:
        return redirect(url_for('employer_login'))
    
    conn = get_jobs_db()
    c = conn.cursor()
    
    # Get employer info
    c.execute("SELECT * FROM employers WHERE id = ?", (session['employer_id'],))
    employer = c.fetchone()
    
    # Get applications for this employer's jobs
    query = '''SELECT a.*, j.title as job_title, u.name as user_name 
               FROM applications a 
               JOIN jobs j ON a.job_id = j.id 
               JOIN users u ON a.user_id = u.id
               WHERE j.employer_id = ?'''
    
    # Filter by status if provided
    status = request.args.get('status')
    if status:
        query += " AND a.status = ?"
        c.execute(query, (session['employer_id'], status))
    else:
        c.execute(query, (session['employer_id'],))
    
    applications = c.fetchall()
    conn.close()
    
    return render_template('employer_candidates.html',
                         employer=employer,
                         applications=applications,
                         current_date=datetime.now().strftime('%B %d, %Y'))

@app.route('/update_application_status/<int:application_id>/<status>', methods=['POST'])
def update_application_status(application_id, status):
    if 'employer_id' not in session:
        return redirect(url_for('employer_login'))
    
    conn = get_jobs_db()
    c = conn.cursor()
    
    # Verify the application belongs to one of the employer's jobs
    c.execute('''UPDATE applications SET status = ? 
               WHERE id = ? AND job_id IN 
               (SELECT id FROM jobs WHERE employer_id = ?)''',
               (status, application_id, session['employer_id']))
    
    if c.rowcount == 0:
        conn.close()
        flash('Application not found or you do not have permission', 'error')
    else:
        conn.commit()
        flash(f'Application status updated to {status}', 'success')
    
    conn.close()
    return redirect(url_for('view_applications'))

@app.route('/interview_prep')
def interview_prep():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('interview_prep_dashboard.html')

@app.route('/start_quiz')
def start_quiz():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_jobs_db()
    c = conn.cursor()
    
    # Get 5 random questions
    c.execute("SELECT * FROM mcq_questions ORDER BY RANDOM() LIMIT 5")
    questions = c.fetchall()
    conn.close()
    
    return render_template('quiz.html', questions=questions)

@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    answers = request.form
    
    conn = get_jobs_db()
    c = conn.cursor()
    
    results = []
    correct = 0
    
    for question_id, selected_option in answers.items():
        if question_id.startswith('question_'):
            q_id = int(question_id.replace('question_', ''))
            
            # Get correct answer
            c.execute("SELECT correct_option FROM mcq_questions WHERE id = ?", (q_id,))
            question = c.fetchone()
            is_correct = int(selected_option) == question['correct_option']
            
            if is_correct:
                correct += 1
            
            # Store result
            c.execute("""INSERT INTO user_quiz_results 
                        (user_id, question_id, selected_option, is_correct)
                        VALUES (?, ?, ?, ?)""",
                     (user_id, q_id, selected_option, is_correct))
            
            results.append({
                'question_id': q_id,
                'selected_option': selected_option,
                'is_correct': is_correct
            })
    
    conn.commit()
    conn.close()
    
    score = (correct / len(results)) * 100 if results else 0
    return render_template('quiz_results.html', score=score, total=len(results), correct=correct)

@app.route('/quiz_analytics')
def quiz_analytics():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    # Initialize default analytics structure
    analytics = {
        'overall': {
            'total_quizzes': 0,
            'total_questions': 0,
            'correct_answers': 0,
            'accuracy_percentage': 0.0
        },
        'by_category': [],
        'by_date': [],
        'weakest_categories': [],
        'strongest_categories': [],
        'improvement': {
            'percentage_change': 0.0,
            'trend': 'neutral'
        }
    }
    
    try:
        conn = get_jobs_db()
        c = conn.cursor()
        
        # Overall statistics
        c.execute('''SELECT 
                    COUNT(DISTINCT DATE(quiz_date)) as total_quizzes,
                    COUNT(*) as total_questions,
                    SUM(is_correct) as correct_answers
                    FROM user_quiz_results
                    WHERE user_id = ?''', (user_id,))
        overall_stats = c.fetchone()
        
        if overall_stats:
            accuracy = (overall_stats['correct_answers'] / overall_stats['total_questions'] * 100) if overall_stats['total_questions'] > 0 else 0
            
            analytics['overall'] = {
                'total_quizzes': overall_stats['total_quizzes'] or 0,
                'total_questions': overall_stats['total_questions'] or 0,
                'correct_answers': overall_stats['correct_answers'] or 0,
                'accuracy_percentage': round(accuracy, 1)
            }
        
        # Performance by category
        c.execute('''SELECT 
                    q.job_role,
                    COUNT(*) as total_questions,
                    SUM(r.is_correct) as correct_answers,
                    (SUM(r.is_correct) * 100.0 / COUNT(*)) as accuracy_percentage
                    FROM user_quiz_results r
                    JOIN mcq_questions q ON r.question_id = q.id
                    WHERE r.user_id = ?
                    GROUP BY q.job_role
                    HAVING COUNT(*) > 0
                    ORDER BY accuracy_percentage DESC''', (user_id,))
        
        analytics['by_category'] = [dict(row) for row in c.fetchall()]
        
        # Identify strongest and weakest categories
        if len(analytics['by_category']) > 0:
            analytics['strongest_categories'] = analytics['by_category'][:3]
            analytics['weakest_categories'] = analytics['by_category'][-3:]
        
        # Performance over time (last 7 quizzes)
        c.execute('''SELECT 
                    DATE(quiz_date) as quiz_date,
                    COUNT(*) as total_questions,
                    SUM(is_correct) as correct_answers,
                    (SUM(is_correct) * 100.0 / COUNT(*)) as accuracy_percentage
                    FROM user_quiz_results
                    WHERE user_id = ?
                    GROUP BY DATE(quiz_date)
                    ORDER BY quiz_date DESC
                    LIMIT 7''', (user_id,))
        
        by_date = [dict(row) for row in c.fetchall()]
        analytics['by_date'] = by_date
        
        # Calculate improvement over time
        if len(by_date) > 1:
            first_quiz = by_date[-1]['accuracy_percentage']
            last_quiz = by_date[0]['accuracy_percentage']
            change = last_quiz - first_quiz
            
            analytics['improvement'] = {
                'percentage_change': round(abs(change), 1),
                'trend': 'up' if change > 0 else 'down' if change < 0 else 'neutral'
            }
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        flash('Error retrieving analytics data', 'error')
    finally:
        conn.close()
    
    return render_template('quiz_analytics.html', analytics=analytics)

@app.route('/consultant_chat', methods=['GET', 'POST'])
def consultant_chat():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        consultant_id = request.form['consultant_id']
        message = request.form['message']
        
        conn = get_jobs_db()
        c = conn.cursor()
        c.execute("""INSERT INTO consultant_chats 
                    (user_id, consultant_id, message, sender_type)
                    VALUES (?, ?, ?, 'user')""",
                 (session['user_id'], consultant_id, message))
        conn.commit()
        conn.close()
        
        return redirect(url_for('consultant_chat'))
    
    # GET request - show chat interface
    conn = get_jobs_db()
    c = conn.cursor()
    
    # Get all consultants
    c.execute("SELECT * FROM consultants ORDER BY name")
    consultants = c.fetchall()
    
    # Get chat history if consultant is selected
    consultant_id = request.args.get('consultant_id')
    chats = []
    if consultant_id:
        c.execute('''SELECT * FROM consultant_chats 
                    WHERE (user_id = ? AND consultant_id = ?)
                    ORDER BY sent_at''',
                 (session['user_id'], consultant_id))
        chats = c.fetchall()
    
    conn.close()
    
    return render_template('consultant_chat.html',
                         consultants=consultants,
                         chats=chats,
                         selected_consultant=consultant_id)

# MCQ Questions Routes
@app.route('/consultants/add_mcq_question', methods=['POST'])
def add_mcq_question():
    if 'consultant_id' not in session:
        return redirect(url_for('consultant_login'))
    
    if request.method == 'POST':
        try:
            # Get form data
            job_role = request.form['job_role']
            experience_level = request.form['experience_level']
            question_text = request.form['question']
            correct_option = int(request.form['correct_option'])
            explanation = request.form.get('explanation', '')
            
            # Get all options
            option1 = request.form['option_1']
            option2 = request.form['option_2']
            option3 = request.form['option_3']
            option4 = request.form['option_4']
            
            # Validate correct option
            if correct_option < 1 or correct_option > 4:
                flash('Please select a valid correct option (1-4)', 'error')
                return redirect(url_for('consultant_dashboard'))
            
            # Insert into database
            with get_jobs_db() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO mcq_questions 
                    (consultant_id, job_role, experience_level, question_text,
                     option1, option2, option3, option4, correct_option, explanation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session['consultant_id'], 
                    job_role, 
                    experience_level,
                    question_text,
                    option1,
                    option2,
                    option3,
                    option4,
                    correct_option,
                    explanation
                ))
                conn.commit()
            
            flash('MCQ question added successfully!', 'success')
            return redirect(url_for('consultant_dashboard'))
            
        except Exception as e:
            print(f"Error adding MCQ question: {e}")
            flash('Error adding MCQ question. Please try again.', 'error')
            return redirect(url_for('consultant_dashboard'))
    
    return redirect(url_for('consultant_dashboard'))

@app.route('/get_mcq/<int:question_id>')
def get_mcq(question_id):
    if 'consultant_id' not in session:
        return jsonify({'error': 'Not authorized'}), 401
    
    with get_jobs_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM mcq_questions WHERE id = ?", (question_id,))
        question = cursor.fetchone()
        
        if question:
            # Convert to dict and handle datetime serialization
            question_dict = dict(question)
            question_dict['created_at'] = str(question_dict['created_at'])
            return jsonify(question_dict)
        
    return jsonify({'error': 'Question not found'}), 404

@app.route('/delete_mcq/<int:question_id>', methods=['DELETE'])
def delete_mcq(question_id):
    if 'consultant_id' not in session:
        return jsonify({'success': False, 'error': 'Not authorized'}), 401
    
    with get_jobs_db() as conn:
        cursor = conn.cursor()
        # Verify the question belongs to this consultant
        cursor.execute("SELECT consultant_id FROM mcq_questions WHERE id = ?", (question_id,))
        question = cursor.fetchone()
        
        if not question:
            return jsonify({'success': False, 'error': 'Question not found'}), 404
        
        if question['consultant_id'] != session['consultant_id']:
            return jsonify({'success': False, 'error': 'Not authorized'}), 403
        
        # Delete the question
        cursor.execute("DELETE FROM mcq_questions WHERE id = ?", (question_id,))
        conn.commit()
    
    return jsonify({'success': True})

@app.route('/employers/subscribe', methods=['GET', 'POST'])
def employer_subscribe():
    if 'employer_id' not in session:
        return redirect(url_for('employer_login'))
    
    if request.method == 'POST':
        # Process payment here (you'll need to integrate with a payment gateway)
        # For now, we'll just mark as paid
        conn = get_jobs_db()
        c = conn.cursor()
        c.execute("UPDATE employers SET has_paid = 1 WHERE id = ?", (session['employer_id'],))
        conn.commit()
        conn.close()
        
        flash('Subscription successful! You can now view full candidate profiles.', 'success')
        return redirect(url_for('view_candidates'))
    
    return render_template('employer_subscribe.html')

@app.route('/get_chat_messages')
def get_chat_messages():
    consultant_id = request.args.get('consultant_id')
    last_id = request.args.get('last_id', 0)
    
    conn = get_jobs_db()
    c = conn.cursor()
    c.execute('''SELECT * FROM consultant_chats 
                WHERE consultant_id = ? AND id > ? 
                ORDER BY sent_at''', (consultant_id, last_id))
    messages = [dict(row) for row in c.fetchall()]
    conn.close()
    
    return jsonify(messages)

@app.route('/send_consultant_message', methods=['POST'])
def send_consultant_message():
    if 'consultant_id' not in session:
        return jsonify({'error': 'Not authorized'}), 401
    
    user_id = request.form.get('user_id')
    message = request.form.get('message')
    
    if not user_id or not message:
        return jsonify({'error': 'Missing parameters'}), 400
    
    conn = get_jobs_db()
    c = conn.cursor()
    
    try:
        # Insert new message from consultant
        c.execute('''INSERT INTO consultant_chats 
                    (user_id, consultant_id, message, sender_type)
                    VALUES (?, ?, ?, 'consultant')''',
                 (user_id, session['consultant_id'], message))
        conn.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        conn.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/mark_messages_read', methods=['POST'])
def mark_messages_read():
    data = request.json
    user_id = data.get('user_id')
    consultant_id = data.get('consultant_id')
    
    if not user_id or not consultant_id:
        return jsonify({'error': 'Missing parameters'}), 400
    
    conn = get_jobs_db()
    c = conn.cursor()
    
    try:
        # Mark all unread messages from this user as read
        c.execute('''UPDATE consultant_chats 
                    SET is_read = 1
                    WHERE user_id = ? AND consultant_id = ? AND sender_type = 'user' AND is_read = 0''',
                 (user_id, consultant_id))
        conn.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        conn.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

if __name__ == "__main__":
    app.run(debug=True)