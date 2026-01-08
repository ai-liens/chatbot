import os
from dotenv import load_dotenv
from groq import Groq
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import tempfile

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Initialize OpenAI embeddings for RAG
openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key:
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
else:
    embeddings = None

st.set_page_config(page_title="Groq Chatbot", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
        body {
            background: linear-gradient(120deg, #d4fc79, #96e6a1);
            font-family: 'Segoe UI', sans-serif;
        }
        .stChatMessage {
            border-radius: 20px;
            padding: 15px;
            margin-bottom: 10px;
        }
        .user {
            background-color: #d1ecf1;
            text-align: right;
        }
        .bot {
            background-color: #f8d7da;
            text-align: left;
        }
        .stTextInput>div>div>input {
            font-size: 16px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712039.png", width=100)
st.sidebar.title("Groq AI Chatbot 🤖")
st.sidebar.markdown("### 🧠 Chat Memory")
memory_enabled = st.sidebar.toggle("Enable Chat Memory", value=True)
if memory_enabled:
    st.sidebar.markdown("Chat memory is enabled. Your conversation history will be saved.")

# RAG Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 RAG (Knowledge Base)")

if embeddings:
    if st.session_state.sample_loaded:
        st.sidebar.success("✅ Sample AI/ML knowledge base loaded!")
        st.sidebar.info("📖 Ask questions about AI, Machine Learning, Deep Learning, NLP, Computer Vision, and more!")
    
    st.session_state.rag_enabled = st.sidebar.toggle("Enable RAG", value=st.session_state.rag_enabled)
    
    if st.session_state.rag_enabled:
        st.sidebar.markdown("**Optional:** Upload your own documents:")
        uploaded_file = st.sidebar.file_uploader("Upload a text file", type=["txt", "pdf"], key="file_uploader")
        
        if uploaded_file:
            with st.spinner("Processing document..."):
                # Read the file content
                if uploaded_file.type == "text/plain":
                    text = uploaded_file.read().decode("utf-8")
                elif uploaded_file.type == "application/pdf":
                    # For PDF files, we'll use pypdf
                    import pypdf
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name
                    
                    pdf_reader = pypdf.PdfReader(tmp_file_path)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    os.unlink(tmp_file_path)
                
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text)
                
                # Create documents
                documents = [Document(page_content=chunk) for chunk in chunks]
                
                # Create vector store (this will replace the sample knowledge)
                st.session_state.vector_store = FAISS.from_documents(documents, embeddings)
                st.sidebar.success(f"✅ Custom document processed! {len(chunks)} chunks created.")
        
        if st.session_state.vector_store:
            st.sidebar.markdown("💡 **Try asking:**")
            st.sidebar.markdown("- What is RAG?")
            st.sidebar.markdown("- Explain deep learning")
            st.sidebar.markdown("- What are LLMs?")
else:
    st.sidebar.warning("⚠️ Please set OPENAI_API_KEY in your .env file to use RAG")

st.sidebar.markdown("---")
st.sidebar.markdown("Built using **llama-3.3-70b-versatile** via **Groq API**")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False

# Auto-load sample knowledge base
if "sample_loaded" not in st.session_state:
    st.session_state.sample_loaded = False

def load_sample_knowledge():
    """Load the sample knowledge base file into the vector store"""
    if embeddings and not st.session_state.sample_loaded:
        try:
            sample_file_path = os.path.join(os.path.dirname(__file__), "sample_knowledge.txt")
            if os.path.exists(sample_file_path):
                with open(sample_file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text)
                
                # Create documents
                documents = [Document(page_content=chunk) for chunk in chunks]
                
                # Create vector store
                st.session_state.vector_store = FAISS.from_documents(documents, embeddings)
                st.session_state.sample_loaded = True
                st.session_state.rag_enabled = True
                return True
        except Exception as e:
            st.error(f"Error loading sample knowledge: {str(e)}")
    return False

# Load sample knowledge on startup
if embeddings and not st.session_state.sample_loaded:
    load_sample_knowledge()


st.title("💬 AI Assistant")
st.caption("Ask anything — your AI assistant is here to help!")

if st.session_state.chat_history:
    chat_text = "\n\n".join(
        [f"User: {msg['content']}" if msg["role"] == "user" else f"Assistant: {msg['content']}" for msg in st.session_state.chat_history]
    )

    st.download_button(
        label="💾 Download Chat History",
        data=chat_text,
        file_name="chat_history.txt",
        mime="text/plain",
    )


for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"<div class='stChatMessage user'>🧑‍💻: {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='stChatMessage bot'>🤖: {msg['content']}</div>", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message:", key="input", placeholder="Ask me anything...")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # RAG: Retrieve relevant context if enabled
    context = ""
    if st.session_state.rag_enabled and st.session_state.vector_store:
        try:
            # Retrieve relevant documents
            relevant_docs = st.session_state.vector_store.similarity_search(user_input, k=3)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
        except Exception as e:
            st.error(f"Error retrieving context: {str(e)}")
    
    # Prepare system message with context
    if context:
        system_content = f"""You are an AI assistant. Use the following context from the knowledge base to answer the user's question. 
If the answer is not in the context, say so and provide a general response.

Context from knowledge base:
{context}"""
    else:
        system_content = "You are an AI assistant(LLM)."
 
    if memory_enabled:
        messages = [{"role": "system", "content": system_content}]
        messages += st.session_state.chat_history
        messages.append({"role": "user", "content": user_input})
    else:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_input}
        ]

    response = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
    )

    bot_reply = response.choices[0].message.content

    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

    st.rerun()
