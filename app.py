import streamlit as st
import fitz
from PIL import Image
import os
import pathlib
import time
import google.generativeai as genai
import openai
from github import Github
import pandas as pd
import tempfile

def validate_github_token(token):
    """Validate GitHub token."""
    try:
        g = Github(token)
        g.get_user().login
        return True
    except Exception as e:
        return False

def validate_gemini_key(key):
    """Validate Gemini API key."""
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        # Simple test generation
        response = model.generate_content("Test")
        return True
    except Exception as e:
        return False

def validate_openai_key(key):
    """Validate OpenAI API key."""
    try:
        openai.api_key = key
        openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        return True
    except Exception as e:
        return False

def check_secrets():
    """Validate all required secrets are present and valid."""
    missing_secrets = []
    invalid_secrets = []
    
    required_secrets = [
        'github_token',
        'github_repo',
        'gemini_api_key',
        'openai_api_key'
    ]
    
    # Check for missing secrets
    for secret in required_secrets:
        if secret not in st.secrets:
            missing_secrets.append(secret)
    
    if missing_secrets:
        st.error(f"Missing required secrets: {', '.join(missing_secrets)}")
        st.info("Please add the missing secrets to your .streamlit/secrets.toml file or Streamlit Cloud settings.")
        return False
    
    # Validate secrets
    if not validate_github_token(st.secrets["github_token"]):
        invalid_secrets.append("github_token")
    
    if not validate_gemini_key(st.secrets["gemini_api_key"]):
        invalid_secrets.append("gemini_api_key")
    
    if not validate_openai_key(st.secrets["openai_api_key"]):
        invalid_secrets.append("openai_api_key")
    
    if invalid_secrets:
        st.error(f"Invalid API keys detected: {', '.join(invalid_secrets)}")
        st.info("Please check your API keys and ensure they are valid and active.")
        return False
    
    return True

class PDFProcessor:
    @staticmethod
    def pdf_to_images(pdf_file, start_page=0, end_page=None):
        """Convert PDF pages to images."""
        images = []
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file.seek(0)
            
            try:
                pdf_document = fitz.open(tmp_file.name)
                end_page = end_page or len(pdf_document)
                
                for page_num in range(start_page, end_page):
                    page = pdf_document.load_page(page_num)
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images.append(img)
                
                pdf_document.close()
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
            finally:
                os.unlink(tmp_file.name)
                
        return images

class GithubManager:
    def __init__(self):
        token = st.secrets["github_token"]
        repo_name = st.secrets["github_repo"]
        self.g = Github(token)
        self.repo = self.g.get_user().get_repo(repo_name)

    def save_to_github(self, content, file_path):
        try:
            # Check if file exists
            try:
                file = self.repo.get_contents(file_path)
                self.repo.update_file(
                    file_path,
                    f"Update {file_path}",
                    content,
                    file.sha
                )
            except Exception:
                self.repo.create_file(
                    file_path,
                    f"Create {file_path}",
                    content
                )
            return True
        except Exception as e:
            st.error(f"Error saving to GitHub: {str(e)}")
            return False

class AIAnalyzer:
    def __init__(self):
        # Initialize API keys from Streamlit secrets
        self.gemini_key = st.secrets["gemini_api_key"]
        self.openai_key = st.secrets["openai_api_key"]
        
        # Configure AI services
        genai.configure(api_key=self.gemini_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
        openai.api_key = self.openai_key

    def analyze_question_paper(self, image):
        """Analyze question paper image using Gemini."""
        try:
            response = self.model.generate_content([
                """From the given image, give me the topics and concepts that the questions depend upon.
                   Also make sure that we constrain the chapter names to JEE syllabus only.
                   Also just give me topics and sub-topics without any description of that sub-topic.
                   Make sure the output is in sequence; don't jumble up question numbers.""",
                image
            ])
            return response.text
        except Exception as e:
            st.error(f"Error analyzing question paper: {str(e)}")
            return None

    def analyze_solutions(self, image):
        """Extract solutions from image using Gemini."""
        try:
            response = self.model.generate_content([
                """From the given image give me the question number and their corresponding answer in the output according to the image.
                   Just read through the image and give me output don't change the answers.
                   Don't jumble up the question and their corresponding answer""",
                image
            ])
            return response.text
        except Exception as e:
            st.error(f"Error analyzing solutions: {str(e)}")
            return None

    def generate_soca(self, student_data, output_data):
        """Generate SOCA analysis using OpenAI."""
        try:
            context = f"""
            You are an intelligent agent designed to support students in their JEE exam preparation.
            Your role is to provide a detailed SOCA (Strengths, Opportunities, Challenges, and Action Plan) analysis
            based on their performance in specific topics within the exam.
            
            paper: {output_data}
            answer_state: {student_data}
            """

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": context},
                    {"role": "user", "content": "Based on the data provided give SOCA for the student"}
                ]
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            st.error(f"Error generating SOCA: {str(e)}")
            return None

def main():
    st.title("JEE Analysis Tool")
    
    # Validate secrets before proceeding
    if not check_secrets():
        st.stop()
    
    try:
        # Initialize managers
        github_manager = GithubManager()
        ai_analyzer = AIAnalyzer()
    except Exception as e:
        st.error(f"Error initializing services: {str(e)}")
        st.info("Please check your configuration and try again.")
        return

    # File upload section
    st.header("Upload Files")
    col1, col2 = st.columns(2)
    
    with col1:
        question_paper = st.file_uploader("Upload Question Paper (PDF)", type=['pdf'])
    
    with col2:
        solution_paper = st.file_uploader("Upload Solution Paper (PDF)", type=['pdf'])

    if question_paper and solution_paper:
        if st.button("Process Papers"):
            with st.spinner("Processing papers..."):
                # Process question paper
                question_images = PDFProcessor.pdf_to_images(question_paper)
                
                # Process solution paper
                solution_images = PDFProcessor.pdf_to_images(solution_paper)

                # Analyze papers
                question_analysis = []
                for img in question_images:
                    analysis = ai_analyzer.analyze_question_paper(img)
                    if analysis:
                        question_analysis.append(analysis)

                solution_analysis = []
                for img in solution_images:
                    analysis = ai_analyzer.analyze_solutions(img)
                    if analysis:
                        solution_analysis.append(analysis)

                # Save analyses to GitHub
                if question_analysis and solution_analysis:
                    # Convert analyses to DataFrame or appropriate format
                    combined_data = {
                        'question_analysis': question_analysis,
                        'solution_analysis': solution_analysis
                    }
                    
                    # Save to GitHub
                    github_manager.save_to_github(
                        str(combined_data),
                        'data/analysis_results.txt'
                    )

                    st.session_state.processed_data = combined_data
                    st.success("Analysis completed and saved to GitHub!")

    # Display results section
    if st.session_state.processed_data:
        st.header("Analysis Results")
        
        # Display question paper analysis
        st.subheader("Question Paper Analysis")
        for idx, analysis in enumerate(st.session_state.processed_data['question_analysis']):
            st.write(f"Page {idx + 1}:")
            st.write(analysis)
        
        # Display solution analysis
        st.subheader("Solution Analysis")
        for idx, analysis in enumerate(st.session_state.processed_data['solution_analysis']):
            st.write(f"Page {idx + 1}:")
            st.write(analysis)

        # SOCA Analysis section
        if st.button("Generate SOCA Analysis"):
            with st.spinner("Generating SOCA analysis..."):
                soca_analysis = ai_analyzer.generate_soca(
                    st.session_state.processed_data['solution_analysis'],
                    st.session_state.processed_data['question_analysis']
                )
                if soca_analysis:
                    st.subheader("SOCA Analysis")
                    st.write(soca_analysis)

if __name__ == "__main__":
    main()
