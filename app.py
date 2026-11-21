import os
import gc
import tempfile
import uuid
import pandas as pd
from typing import Optional, Dict, Any
import logging

from llama_index.core.response_synthesizers import get_response_synthesizer
from gitingest import ingest
from llama_index.core import Settings, PromptTemplate, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser
import streamlit as st
from dotenv import load_dotenv
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Constants
MAX_REPO_SIZE = 100 * 1024 * 1024  # 100MB
SUPPORTED_REPO_TYPES = ['.py', '.md', '.ipynb', '.js', '.ts', '.json']

class GitHubRAGError(Exception):
    """Custom exception for GitHub RAG application errors"""
    pass

def validate_github_url(url: str) -> bool:
    """Validate GitHub repository URL"""
    return url.startswith(('https://github.com/', 'http://github.com/'))

def get_repo_name(url: str) -> str:
    """Extract repository name from URL"""
    try:
        return url.split('/')[-1].replace('.git', '')
    except Exception as e:
        raise GitHubRAGError(f"Invalid repository URL: {str(e)}")

def reset_chat():
    """Reset chat session and clean up resources"""
    try:
        st.session_state.messages = []
        st.session_state.context = None
        gc.collect()
        logger.info("Chat session reset successfully")
    except Exception as e:
        logger.error(f"Error resetting chat: {str(e)}")
        raise GitHubRAGError("Failed to reset chat session")

def process_with_gitingets(github_url: str) -> tuple:
    """Process GitHub repository using gitingest"""
    try:
        summary, tree, content = ingest(github_url)
        if not all([summary, tree, content]):
            raise GitHubRAGError("Failed to process repository: Missing data")
        return summary, tree, content
    except Exception as e:
        logger.error(f"Error processing repository: {str(e)}")
        raise GitHubRAGError(f"Failed to process repository: {str(e)}")

def create_query_engine(content_path: str, repo_name: str) -> Any:
    """Create query engine using local Ollama models."""

    try:
        # 1. Initialize local embedding model
        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434"
        )

        # 2. Initialize local chat model
        llm = Ollama(
            model="llama3.2",
            base_url="http://localhost:11434",
            request_timeout=300.0
        )

        # 3. Configure LlamaIndex BEFORE creating the index
        Settings.embed_model = embed_model
        Settings.llm = llm

        logger.info("Ollama embedding and LLM configured successfully")

        # 4. Load repository documents
        loader = SimpleDirectoryReader(input_dir=content_path)
        docs = loader.load_data()

        if not docs:
            raise GitHubRAGError("No documents found in repository.")

        # 5. Parse Markdown documents
        node_parser = MarkdownNodeParser()

        # 6. Create vector index explicitly using Ollama embeddings
        index = VectorStoreIndex.from_documents(
            documents=docs,
            transformations=[node_parser],
            embed_model=embed_model,
            show_progress=True
        )

        # 7. Custom prompt
        qa_prompt_tmpl_str = """
            You are an expert AI Software Engineer and GitHub
        Repository Analyst.

        Your job is to understand the provided repository
        and answer questions using the retrieved context.

        INSTRUCTIONS:

        1. Understand the user's question before answering.
        2. Analyze the relevant code and repository context.
        3. Break complex technical problems into logical steps.
        4. Explain code with practical examples.
        5. For debugging, identify:
        - Root cause
        - Why the error occurs
        - Exact solution
        - Corrected code
        6. Never invent files, functions, APIs or code.
        7. If information is missing, clearly say so.
        8. Keep answers technically accurate and structured.

        REPOSITORY CONTEXT:
        ---------------------
        {context_str}
        ---------------------

        USER QUESTION:
        {query_str}

        Provide a clear, accurate and useful answer.
        Answer:
        """

        # Insert repository name into the prompt
        qa_prompt_tmpl_str = qa_prompt_tmpl_str.replace(
            "{repo_name}", repo_name
        )

        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

        # 8. Create query engine explicitly using Ollama LLM
        query_engine = index.as_query_engine(
            llm=llm,
            streaming=True,
            similarity_top_k=3
        )

        # 9. Update response synthesis prompt
        query_engine.update_prompts({
            "response_synthesizer:text_qa_template": qa_prompt_tmpl
        })

        logger.info("Query engine created successfully using Ollama")

        return query_engine

    except Exception as e:
        logger.exception("Error creating query engine")
        raise GitHubRAGError(
            f"Failed to create query engine: {str(e)}"
        )

# Initialize session state
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.messages = []

session_id = st.session_state.id

# Sidebar
with st.sidebar:
    st.header("Add your GitHub repository!")
    
    github_url = st.text_input(
        "Enter GitHub repository URL",
        placeholder="https://github.com/username/repo",
        help="Enter a valid GitHub repository URL"
    )
    
    load_repo = st.button("Load Repository", type="primary")

    if github_url and load_repo:
        try:
            # Validate URL
            if not validate_github_url(github_url):
                st.error("Please enter a valid GitHub repository URL")
                st.stop()

            repo_name = get_repo_name(github_url)
            file_key = f"{session_id}-{repo_name}"
            
            if file_key not in st.session_state.file_cache:
                with st.spinner("Processing your repository..."):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        try:
                            summary, tree, content = process_with_gitingets(github_url)
                            
                            # Write content to temporary file
                            content_path = os.path.join(temp_dir, f"{repo_name}_content.md")
                            with open(content_path, "w", encoding="utf-8") as f:
                                f.write(content)
                            
                            # Create and cache query engine
                            query_engine = create_query_engine(temp_dir, repo_name)
                            st.session_state.file_cache[file_key] = query_engine
                            
                            st.success("Repository loaded successfully! Ready to chat.")
                            logger.info(f"Successfully processed repository: {repo_name}")
                            
                        except GitHubRAGError as e:
                            st.error(str(e))
                            logger.error(f"Error processing repository {repo_name}: {str(e)}")
                            st.stop()
                        except Exception as e:
                            st.error("An unexpected error occurred while processing the repository")
                            logger.error(f"Unexpected error: {str(e)}")
                            st.stop()
            else:
                st.info("Repository already loaded. Ready to chat!")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Error in repository loading process: {str(e)}")
            st.stop()

# Main content
col1, col2 = st.columns([6, 1])

with col1:
    st.header("Chat with GitHub using RAG </>")

with col2:
    st.button("Clear Chat ↺", on_click=reset_chat, help="Clear chat history and reset session")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What's up?"):
    try:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                repo_name = get_repo_name(github_url)
                file_key = f"{session_id}-{repo_name}"
                query_engine = st.session_state.file_cache.get(file_key)
                
                if query_engine is None:
                    raise GitHubRAGError("Please load a repository first!")
                
                response = query_engine.query(prompt)
                
                if hasattr(response, 'response_gen'):
                    for chunk in response.response_gen:
                        if isinstance(chunk, str):
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                else:
                    full_response = str(response)
                    message_placeholder.markdown(full_response)
                    
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except GitHubRAGError as e:
                st.error(str(e))
                logger.error(f"Error in chat processing: {str(e)}")
            except Exception as e:
                st.error("An unexpected error occurred while processing your query")
                logger.error(f"Unexpected error in chat: {str(e)}")
                
    except Exception as e:
        st.error("An error occurred in the chat system")
        logger.error(f"Chat system error: {str(e)}")