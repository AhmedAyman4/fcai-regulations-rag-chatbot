import os
import time
import warnings
import re  # Added for language detection
import uvicorn
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
# Removed arabic_reshaper and bidi imports as they corrupt text for LLMs

# Suppress warnings
warnings.filterwarnings("ignore")

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Configuration ---
pdf_path = "./data//USC Faculty of Computer and Artificial Intelligence Internal Regulations (October 2019).pdf"
persist_directory = "./chroma_fcai_regulations_db"
google_api_key = os.environ.get("GOOGLE_API_KEY")
groq_api_key = os.environ.get("GROQ_API_KEY")
qa_chain = None

# --- Helper Functions ---
def detect_language_and_instruction(text):
    """
    Detects if the text contains Arabic characters.
    Returns a tuple: (detected_language_code, instruction_suffix)
    """
    # Regex range for Arabic characters
    if bool(re.search('[\u0600-\u06FF]', text)):
        return "ar", " (أجب باللغة العربية بناءً على السياق المقدم)"
    return "en", " (Answer strictly in English based on the provided context)"

# --- RAG Setup Function (to be called once) ---
def initialize_rag_chain():
    """
    Initializes all components of the RAG chain (vector store, LLM, retriever)
    and returns the complete qa_chain.
    """
    global qa_chain
    
    print("Initializing RAG chain...")

    # 1. Load the document
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return None
        
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Removed the text cleaning loop. 
    # Passing raw UTF-8 to the LLM is better for semantic understanding and modern rendering.

    # 2. Configure and split the text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=1000,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)

    # 3. Initialize multilingual embedding model
    model_name = "intfloat/multilingual-e5-base"
    model_kwargs = {'device': 'cpu'} # Use 'cuda' if GPU is available
    encode_kwargs = {'normalize_embeddings': True}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # 4. Create or load the Chroma vector store
    if os.path.exists(persist_directory):
        print(f"Loading existing vector store from {persist_directory}")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=hf
        )
    else:
        print(f"Creating new vector store at {persist_directory}")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=hf,
            persist_directory=persist_directory
        )
        vectorstore.persist()
        print("Chroma vector store created and saved successfully!")

    # 5. Initialize Gorq LLM free api
    if not groq_api_key:
        print("Error: GROQ_API_KEY environment variable not set.")
        return None
    
    try:
        groq_llm = ChatGroq(model="llama-3.1-8b-instant",
                            temperature=0.0,
                            max_tokens=None,
                            timeout=None,
                            max_retries=2,
                            api_key=groq_api_key)
    except Exception as e:
        print(f"Error initializing Groq LLM: {e}")
        return None

    
    # if not google_api_key:
    #     print("Error: GOOGLE_API_KEY environment variable not set.")
    #     return None

    # try:
    #     google_llm = ChatGoogleGenerativeAI(
    #         model="gemini-2.5-flash",
    #         temperature=0.0,
    #         max_tokens=None,
    #         timeout=None,
    #         max_retries=2,
    #         convert_system_message_to_human=True,
    #         google_api_key=google_api_key
    #     )
    # except Exception as e:
    #     print(f"Error initializing Google LLM: {e}")
    #     return None

    # 6. Define the prompt template
    prompt_template = """
    أنت مساعد ذكي ومتعمق في التحليل، متخصص في الإجابة على الأسئلة المتعلقة باللائحة الداخلية لكلية الحاسبات والذكاء الاصطناعي.
    You are an intelligent and thoughtful assistant specialized in answering questions about the Faculty of Computer and Artificial Intelligence Internal Regulations.
    **Instructions:**
    1. Analyze the provided context carefully and extract key information.
    2. If the answer is not stated explicitly, infer it logically **based on the meaning and structure** of the context.
    3. Combine information from multiple parts of the context when necessary to form a coherent, detailed answer.
    4. Use reasoning and general understanding of academic or regulatory structures to fill in small gaps **only when it makes sense**.
    5. Always clarify if a part of your answer is inferred or partially uncertain.
    6. Be slightly verbose — provide clear explanations or examples when helpful.
    7. Maintain a professional and informative tone.
    8. **CRITICAL:** Follow the language instruction appended to the question strictly.
    9. Avoid adding any external information unrelated to the provided context.
    **Context from Regulations:**
    {context}
    **Question:** {question}
    **Answer (reasoned and detailed):**
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 7. Create retriever interface
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 25,
            "lambda_mult": 0.7
        }
    )

    # 8. Create a RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=groq_llm, # FIX: Changed from undefined google_llm
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        chain_type="stuff",
        return_source_documents=True,
        input_key="question"
    )
    
    print("RAG chain initialized successfully.")
    return chain

# --- API Endpoints ---

@app.get("/")
def health_check():
    """Health check endpoint to confirm the API is running."""
    return JSONResponse(content={"status": "API is running and healthy"}, status_code=200)

@app.post("/api/chat")
def api_chat(data: dict = Body(default={})):
    """
    The main chat endpoint.
    Expects JSON: {"query": "Your question..."}
    Returns JSON: {"answer": "The answer..."} or {"error": "..."}
    """
    global qa_chain
    
    # Check if API key is set
    if not groq_api_key: # FIX: Changed to groq_api_key to match the active LLM
        return JSONResponse(content={"error": "Groq API key is not configured on the server."}, status_code=500)
        
    # Check if chain is initialized
    if qa_chain is None:
        return JSONResponse(content={"error": "RAG chain is not initialized. Check server logs."}, status_code=500)

    # Get query from request
    query = data.get("query")
    if not query:
        return JSONResponse(content={"error": "No 'query' provided in the request body."}, status_code=400)

    try:
        # Detect language and get specific instruction
        lang_code, lang_instruction = detect_language_and_instruction(query)
        
        # Modify the query passed to the LLM (internal only) to enforce language
        # We append the instruction to the user's question.
        modified_query_for_llm = f"{query} \n\n[System Note: {lang_instruction}]"

        # Use the RAG chain to get a result
        result = qa_chain.invoke({"question": modified_query_for_llm})
        answer = result['result']
        
        # Format sources - KEPT EXACTLY AS REQUESTED (Original Code)
        sources = "\n\n📚 المصادر:\n" + "\n".join([
            f"  • {doc.metadata.get('source', 'Unknown')} | صفحة {doc.metadata.get('page', 'N/A')}"
            for doc in result["source_documents"]
        ])
        
        # Combine answer and sources
        response_text = answer + sources
        
        return JSONResponse(content={"answer": response_text})
        
    except Exception as e:
        print(f"Error during chain invocation: {e}")
        return JSONResponse(content={"error": f"An error occurred while processing your request: {e}"}, status_code=500)

# --- Main Application Runner ---
if __name__ == "__main__":
    # Initialize the RAG chain on startup
    qa_chain = initialize_rag_chain()
    
    # Get port from environment or default to 7860 (for Hugging Face)
    port = int(os.environ.get("PORT", 7860))
    
    # Run the app
    uvicorn.run(app, host="0.0.0.0", port=port)
else:
    # This block runs when uvicorn/gunicorn imports the file
    qa_chain = initialize_rag_chain()