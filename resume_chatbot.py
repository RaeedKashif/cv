import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import PyPDF2
from docx import Document
import re
import json
from pathlib import Path
from typing import List, Dict, Optional

class ResumeAssistant:
    def __init__(self):
        self.knowledge_base = self._initialize_knowledge_base()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self._train_vectorizer()
        self.nn_model = self._train_nearest_neighbors()
        self.chat_history = []
        
        # Resume analysis parameters
        self.resume_sections = {
            'summary': ['summary', 'profile', 'objective'],
            'experience': ['experience', 'work history', 'employment'],
            'education': ['education', 'qualifications'],
            'skills': ['skills', 'competencies', 'technical skills']
        }

    def _initialize_knowledge_base(self) -> pd.DataFrame:
        """Initialize with local resume knowledge base"""
        data = {
            'question': [
                'How to write a good resume summary?',
                'What skills should I include?',
                'How to format work experience?',
                'What education details to include?',
                'How long should my resume be?',
                'Should I include references?',
                'How to handle employment gaps?',
                'What font size to use?',
                'How to list certifications?',
                'Should I include a photo?'
            ],
            'answer': [
                'Keep your summary concise (2-3 sentences) highlighting your key qualifications and career goals.',
                'Include both hard and soft skills relevant to the job, with technical skills first.',
                'Use reverse chronological order with job title, company, dates, and bullet points of achievements.',
                'List degrees, institutions, graduation years, and honors if relevant to the position.',
                '1 page for <10 years experience, 2 pages max for more experienced candidates.',
                'No, just state "References available upon request" if needed.',
                'Be honest but brief, focusing on skills gained during gaps if possible.',
                '11-12pt for body text, 14-16pt for headings for optimal readability.',
                'Create a separate section listing cert name, issuing organization and date.',
                'Generally no, unless specifically requested for the position.'
            ],
            'keywords': [
                'summary, profile, objective, introduction',
                'skills, competencies, technical, soft, abilities',
                'experience, work history, employment, jobs',
                'education, degrees, schools, qualifications',
                'length, pages, size, limit',
                'references, recommendation, contacts',
                'gaps, unemployment, breaks, explanation',
                'font, size, typography, formatting',
                'certifications, certificates, licenses',
                'photo, picture, image, headshot'
            ]
        }
        return pd.DataFrame(data)

    def _train_vectorizer(self):
        """Train TF-IDF vectorizer on knowledge base"""
        all_text = list(self.knowledge_base['question']) + list(self.knowledge_base['keywords'])
        self.vectorizer.fit(all_text)

    def _train_nearest_neighbors(self, n_neighbors=3):
        """Train nearest neighbors model for question matching"""
        questions = self.knowledge_base['question'].tolist()
        keywords = self.knowledge_base['keywords'].tolist()
        
        # Combine questions and keywords for better matching
        combined_text = [f"{q} {k}" for q, k in zip(questions, keywords)]
        X = self.vectorizer.transform(combined_text)
        
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        nn.fit(X)
        return nn

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF resume"""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text

    def _extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from Word document resume"""
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    def analyze_resume(self, file_path: str) -> Dict:
        """Analyze uploaded resume file"""
        try:
            if file_path.endswith('.pdf'):
                text = self._extract_text_from_pdf(file_path)
            elif file_path.endswith('.docx'):
                text = self._extract_text_from_docx(file_path)
            else:
                return {"error": "Unsupported file format"}
            
            # Simple section detection
            analysis = {"sections_found": [], "missing_sections": []}
            
            for section, keywords in self.resume_sections.items():
                if any(re.search(rf'\b{kw}\b', text, re.IGNORECASE) for kw in keywords):
                    analysis["sections_found"].append(section)
                else:
                    analysis["missing_sections"].append(section)
            
            # Basic metrics
            word_count = len(text.split())
            analysis["word_count"] = word_count
            analysis["summary"] = "Resume appears complete" if len(analysis["missing_sections"]) == 0 \
                else f"Consider adding: {', '.join(analysis['missing_sections'])}"
            
            return analysis
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def get_response(self, user_input: str, resume_text: Optional[str] = None) -> str:
        """Generate response to user query"""
        # Add context if resume text is provided
        context = ""
        if resume_text:
            context = f"User's resume contains: {resume_text[:500]}... "
        
        # Find most similar questions
        query = context + user_input
        query_vec = self.vectorizer.transform([query])
        distances, indices = self.nn_model.kneighbors(query_vec)
        
        # Get best matching answers
        responses = []
        for i, idx in enumerate(indices[0]):
            responses.append(self.knowledge_base.iloc[idx]['answer'])
        
        # Combine responses
        if len(responses) == 1:
            return responses[0]
        else:
            return "Here are some tips:\n- " + "\n- ".join(responses)

    def save_chat_history(self, user_id: str):
        """Save chat history to local JSON file"""
        history_dir = Path("chat_histories")
        history_dir.mkdir(exist_ok=True)
        
        file_path = history_dir / f"{user_id}.json"
        with open(file_path, 'w') as f:
            json.dump(self.chat_history, f)

    def load_chat_history(self, user_id: str):
        """Load chat history from local JSON file"""
        file_path = Path("chat_histories") / f"{user_id}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                self.chat_history = json.load(f)