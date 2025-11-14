import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# Initialize OpenAI with GPT-5
@st.cache_resource
def initialize_llm():
    return ChatOpenAI(
        model="gpt-5",
        temperature=0.7,
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

# Initialize embeddings and vector store
@st.cache_resource
def initialize_vector_store():
    # Load documents from knowledge base
    loader = DirectoryLoader(
        'knowledge_base/',
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    return vectorstore

# Initialize RAG chain
@st.cache_resource
def initialize_rag_chain():
    llm = initialize_llm()
    vectorstore = initialize_vector_store()
    
    # Create custom prompt for internal queries
    system_template = """You are an intelligent internal company assistant designed to help employees with their queries about company policies, benefits, technical support, and other internal matters.

Use the following context from our internal knowledge base to answer the question. If you cannot find the answer in the provided context, politely say so and suggest contacting the appropriate department (HR, IT, or Management).

Be professional, helpful, and concise in your responses. Always prioritize accurate information from the knowledge base over general knowledge.

Context: {context}

Chat History: {chat_history}

Question: {question}

Helpful Answer:"""
    
    PROMPT = PromptTemplate(
        template=system_template,
        input_variables=["context", "chat_history", "question"]
    )
    
    # Create conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Create retrieval chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

# Streamlit UI Configuration
st.set_page_config(page_title="Internal AI Assistant", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
        body {
            background: linear-gradient(120deg, #667eea, #764ba2);
            font-family: 'Segoe UI', sans-serif;
        }
        .stChatMessage {
            border-radius: 20px;
            padding: 15px;
            margin-bottom: 10px;
        }
        .user {
            background-color: #e3f2fd;
            text-align: right;
        }
        .bot {
            background-color: #f3e5f5;
            text-align: left;
        }
        .stTextInput>div>div>input {
            font-size: 16px;
            padding: 10px;
        }
        .source-doc {
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 10px;
            margin-top: 10px;
            border-radius: 5px;
            font-size: 12px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712039.png", width=100)
st.sidebar.title("Internal AI Assistant 🤖")
st.sidebar.markdown("### 🧠 Features")
st.sidebar.markdown("✅ RAG-powered responses")
st.sidebar.markdown("✅ Internal knowledge base")
st.sidebar.markdown("✅ Chat memory enabled")
st.sidebar.markdown("✅ Source document references")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** OpenAI GPT-5")
st.sidebar.markdown("**Framework:** LangChain")

# Show source documents toggle
show_sources = st.sidebar.toggle("Show Source Documents", value=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_chain" not in st.session_state:
    with st.spinner("🔄 Initializing AI Assistant and loading knowledge base..."):
        st.session_state.rag_chain = initialize_rag_chain()

st.title("💼 Internal Company AI Assistant")
st.caption("Ask questions about company policies, benefits, IT support, and more!")

# Download chat history
if st.session_state.chat_history:
    chat_text = "\n\n".join(
        [f"User: {msg['content']}" if msg["role"] == "user" else f"Assistant: {msg['content']}" 
         for msg in st.session_state.chat_history]
    )
    
    st.download_button(
        label="💾 Download Chat History",
        data=chat_text,
        file_name="chat_history.txt",
        mime="text/plain",
    )

# Display chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"<div class='stChatMessage user'>🧑‍💻: {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='stChatMessage bot'>🤖: {msg['content']}</div>", unsafe_allow_html=True)
        
        # Show sources if available
        if show_sources and "sources" in msg:
            with st.expander("📚 View Source Documents"):
                for i, source in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source {i}:** {source['file']}")
                    st.markdown(f"```\n{source['content'][:300]}...\n```")

# Chat input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message:", key="input", placeholder="Ask about policies, benefits, IT support...")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Get response from RAG chain
    with st.spinner("🤔 Thinking..."):
        try:
            result = st.session_state.rag_chain({"question": user_input})
            bot_reply = result["answer"]
            source_docs = result.get("source_documents", [])
            
            # Format sources
            sources = []
            for doc in source_docs:
                sources.append({
                    "file": os.path.basename(doc.metadata.get("source", "Unknown")),
                    "content": doc.page_content
                })
            
            # Add bot message to history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": bot_reply,
                "sources": sources
            })
            
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {str(e)}. Please try again or contact IT support."
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_msg
            })
    
    st.rerun()
