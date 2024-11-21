import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from llama_parse import LlamaParse
import pandas as pd
import os
import nest_asyncio
import re
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import partial

def clean_and_deduplicate_cpt_codes(result):
    codes = re.findall(r'97\d{3}', result)
    unique_codes = []
    for code in codes:
        if code not in unique_codes:
            unique_codes.append(code)
    return ', '.join(unique_codes)

def validate_cpt_codes(extracted_codes, allowed_df, denied_df):
    results = []
    codes = [code.strip() for code in extracted_codes.split(',')]
    
    allowed_codes = set(allowed_df[0].astype(str).str.strip())
    denied_codes = set(denied_df[0].astype(str).str.strip())
    
    for code in codes:
        if code in allowed_codes:
            results.append(f"Valid CPT code - {code}")
        elif code in denied_codes:
            results.append(f"Invalid CPT code - {code}")
        else:
            results.append(f"Unknown CPT code - {code}")
            
    return results

def process_pdf(pdf_file, allowed_excel, denied_excel):
    try:
        progress_container = st.empty()
        
        with progress_container:
            progress_bar = st.progress(0)
            st.write("Initializing parser...")
        
        nest_asyncio.apply()
        
        pdf_file.seek(0)
        pdf_bytes = pdf_file.read()
        llama_parser = LlamaParse(api_key=st.secrets["LLAMA_PARSE_API_KEY"], result_type="text")
        parsed_content = llama_parser.load_data(
            pdf_bytes,
            extra_info={"file_name": pdf_file.name}
        )
        progress_bar.progress(20)
        
        with progress_container:
            st.write("Processing PDF...")
        text = parsed_content.text if hasattr(parsed_content, 'text') else str(parsed_content)
        progress_bar.progress(40)
        
        with progress_container:
            st.write("Extracting text...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=256,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )
        texts = text_splitter.split_text(text)
        progress_bar.progress(60)
        
        with progress_container:
            st.write("Extracting CPT codes...")
        embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
        docsearch = FAISS.from_texts(texts, embeddings)
        chain = load_qa_chain(OpenAI(temperature=0, max_tokens=150, api_key=st.secrets["OPENAI_API_KEY"]), chain_type="stuff")
        
        query = """Follow these extraction rules precisely:

1. Look for the exact structure:
   [Start Marker] "CPT Codes"
   [Your target codes will be here]
   [End Marker] "CPT Code Notes" or "Signature"

2. From ONLY this section:
   - Extract 5-digit codes starting with "97"
   - Ignore descriptions after codes
   - Ignore any other sections containing codes

3. Return format: comma-separated numbers only (e.g., 97530, 97535)

Do not extract from notes sections. Only extract codes listed under the exact heading "CPT Codes"."""
        
        docs = docsearch.similarity_search(query)
        result = chain.invoke({"input_documents": docs, "question": query})
        progress_bar.progress(80)
        
        with progress_container:
            st.write("Validating codes...")
        extracted_codes = clean_and_deduplicate_cpt_codes(result['output_text'])
        allowed_df = pd.read_excel(allowed_excel, header=None)
        denied_df = pd.read_excel(denied_excel, header=None)
        validation_results = validate_cpt_codes(extracted_codes, allowed_df, denied_df)
        progress_bar.progress(100)
        
        progress_container.empty()
        
        return extracted_codes, validation_results
        
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None, None

def main():
    st.set_page_config(
        page_title="CPT Code Validator",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        .main { padding: 2rem; }
        .stProgress > div > div > div > div { background-color: #1f77b4; }
        .validation-result {
            padding: 0.5rem;
            margin: 0.25rem 0;
            border-radius: 0.25rem;
            color: #000000;
            font-weight: 500;
        }
        .valid-code {
            background-color: #4CAF50;
            border-left: 4px solid #2E7D32;
            color: white;
        }
        .invalid-code {
            background-color: #F44336;
            border-left: 4px solid #C62828;
            color: white;
        }
        .unknown-code {
            background-color: #FFC107;
            border-left: 4px solid #FFA000;
            color: black;
        }
        .stCode {
            background-color: #2b303b !important;
            color: #c5c8c6 !important;
        }
        .stMarkdown h2 { color: white !important; }
        </style>
    """, unsafe_allow_html=True)

    st.title("CPT Code Validator")
    
    col1, col2, col3 = st.columns([1,1,1])
    
    with col1:
        pdf_file = st.file_uploader("Upload PDF Document", type=['pdf'])
    with col2:
        allowed_excel = st.file_uploader("Upload Allowed CPT Codes (Excel)", type=['xlsx'])
    with col3:
        denied_excel = st.file_uploader("Upload Denied CPT Codes (Excel)", type=['xlsx'])

    if st.button("Process Document", type="primary"):
        if pdf_file and allowed_excel and denied_excel:
            with st.spinner("Processing document..."):
                try:
                    results_container = st.container()
                    extracted_codes, validation_results = process_pdf(
                        pdf_file, allowed_excel, denied_excel
                    )
                    
                    if extracted_codes and validation_results:
                        with results_container:
                            col1, col2 = st.columns([1,2])
                            
                            with col1:
                                st.subheader("Extracted CPT Codes")
                                codes_list = extracted_codes.split(", ")
                                for code in codes_list:
                                    st.code(code, language="python")
                            
                            with col2:
                                st.subheader("Validation Results")
                                for result in validation_results:
                                    if "Valid" in result:
                                        st.markdown(
                                            f'<div class="validation-result valid-code">{result}</div>',
                                            unsafe_allow_html=True
                                        )
                                    elif "Invalid" in result:
                                        st.markdown(
                                            f'<div class="validation-result invalid-code">{result}</div>',
                                            unsafe_allow_html=True
                                        )
                                    else:
                                        st.markdown(
                                            f'<div class="validation-result unknown-code">{result}</div>',
                                            unsafe_allow_html=True
                                        )
                            
                            results_df = pd.DataFrame({
                                'CPT Codes': codes_list,
                                'Validation': validation_results
                            })
                            
                            st.download_button(
                                label="Download Results as CSV",
                                data=results_df.to_csv(index=False),
                                file_name="cpt_validation_results.csv",
                                mime="text/csv"
                            )
                            
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
        else:
            st.warning("Please upload all required files.")

if __name__ == "__main__":
    main()