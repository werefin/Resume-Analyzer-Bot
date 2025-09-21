import streamlit as st
import openai
import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
import PyPDF2
import base64

# Set the page layout to a wider layout
st.set_page_config(layout="wide", page_title="AI recruitment analysis")

# Ask for OpenAI API key
openai_api_key = st.text_input("Enter your OpenAI API key:")
if not openai_api_key:
    st.warning("Please enter your OpenAI API key")
    st.stop()

# Set OpenAI API key
openai.api_key = openai_api_key

def analyze_resume(job_desc_text, resume_text, options):
    # Check if job_desc_text is uploaded
    if job_desc_text is not None:
        try:
            # Attempt to decode the job description content as UTF-8
            job_desc = job_desc_text.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            # If decoding as UTF-8 fails, assume it's a PDF file and try to extract text
            try:
                job_desc = extract_text_from_pdf(job_desc_text)
            except Exception as e:
                st.warning(f"Unable to extract text from job description PDF: {str(e)}")
                return
    else:
        st.warning("Please upload a job description")
        return

    # Check if resume_text is uploaded
    if resume_text is not None:
        try:
            # Attempt to decode the resume content as UTF-8
            resume = resume_text.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            # If decoding as UTF-8 fails, assume it's a PDF file and try to extract text
            try:
                resume = extract_text_from_pdf(resume_text)
            except Exception as e:
                st.warning(f"Unable to extract text from resume PDF: {str(e)}")
                return
    else:
        st.warning("Please upload a resume")
        return

    df = analyze_str(resume, options)
    df_string = df.applymap(lambda x: ', '.join(x) if isinstance(x, list) else x).to_string(index=False)
    st.write("Analyzing with AI...")
    summary_question = f"Job requirements: {{{job_desc}}}" + f"Resume summary: {{{df_string}}}" + "Please return a summary of the candidate's suitability for this position'"
    summary = ask_openAI(summary_question)
    df.loc[len(df)] = ['Summary', summary]
    extra_info = "Scoring criteria: top 10 domestic universities +3 points, 985 universities +2 points, 211 universities +1 point, leading company experience +2 points, well-known company +1 point, overseas background +3 points, foreign company background +1 point."
    score_question = f"Job requirements: {{{job_desc}}}" + f"Resume summary: {{{df.to_string(index=False)}}}" + "Please return a matching score (0-100) for the candidate for this job, please score accurately to facilitate comparison with other candidates, '" + extra_info
    score = ask_openAI(score_question)
    df.loc[len(df)] = ['Match score', score]

    return df

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

def ask_openAI(question):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        temperature=0.7,
        messages=[{"role": "user", "content": question}],
        stop=None,
    )
    return response['choices'][0]['message']['content'].strip()

def analyze_str(resume, options):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(resume)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    df_data = [{'option': option, 'value': []} for option in options]
    st.write("Fetching information...")

    # Create a progress bar and an empty element
    progress_bar = st.progress(0)
    option_status = st.empty()

    for i, option in tqdm(enumerate(options), desc="Fetching information...", unit="option", ncols=100):
        question = f"What is this candidate's {option}? Please return the answer in a concise manner, no more than 250 words. If not found, return 'Not provided'. Please always keep in mind the job description while analyzing."
        docs = knowledge_base.similarity_search(question)
        llm = OpenAI(openai_api_key=openai_api_key, temperature=0.3, model_name="gpt-4o")
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=question )
        df_data[i]['value'] = response
        option_status.text(f"Looking for information: {option}")

        # Update the progress bar
        progress = (i + 1) / len(options)
        progress_bar.progress(progress)

    df = pd.DataFrame(df_data)
    st.success("Resume elements retrieved")
    return df

# Set the page title
st.title("AI recruitment analysis")

# Set default job description and resume information
default_jd = "Business data analyst JD: duties: ..."
default_resume = "Resume: personal information: ..."

# Upload job description
jd_text = st.file_uploader("Upload job description (text or PDF)", type=["txt", "pdf"])

# Upload resume
resume_text = st.file_uploader("Upload candidate resume (text or PDF)", type=["txt", "pdf"])

# Parameter input
options = ["Name", "Contact number", "Gender", "Age", "Years of work experience (number)", "Highest education", "Undergraduate school name", "Master's school name", "Employment status", "Current position", "List of past employers", "Technical skills", "Experience level", "Management skills", "Strenghts", "Weaknesses"]
selected_options = st.multiselect("Please select options", options, default=options)

# Initialize df
df = None

def get_binary_file_downloader_html(bin_str, file_label='File', file_name='file.txt'):
    """Generate a download link for a binary file"""
    bin_str = bin_str.encode()
    b64 = base64.b64encode(bin_str).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{file_name}">{file_label}</a>'
    return href

# Analyze button
if st.button("Start analysis"):
    df = analyze_resume(jd_text, resume_text, selected_options)
    st.subheader("Overall match score: "+ df.loc[df['option'] == 'Match score', 'value'].values[0])
    st.subheader("Detailed display:")
    st.table(df)

    # Download results as TXT
    txt_result = df.to_csv(sep='\t', index=False, header=None)
    st.markdown(get_binary_file_downloader_html(txt_result, file_label="Download results", file_name="recruitment_results.txt"), unsafe_allow_html=True)
