import streamlit as st
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from faq_manager import init_db, log_faq, get_top_faqs
import configparser
from dotenv import load_dotenv
import os
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

config = configparser.ConfigParser()
# Load the config file
config.read("./app/utils/.config")

# Access values
logo_icon = config["LOGO"]["logo_path"]

# Set the page configuration to wide layout and add a title
st.set_page_config(
    page_title="NetvirE SmartAssist",
    page_icon=logo_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize the FAQ database
faq_db_name = config["DB"]["db_name"]
init_db(faq_db_name)

# Predefined FAQs
predefined_faqs = [
    "How do I generate a weekly temperature report?",
]


# Initialize embeddings and vector store
# Using HuggingFace embeddings instead of Ollama to avoid connection issues
model_name = "nomic-ai/nomic-embed-text-v1"
model_kwargs = {
    'device': 'cpu',
    'trust_remote_code': True
}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
db_name = "./app/vectorstores/smart_iot_vector_store"
vector_store = FAISS.load_local(db_name, embeddings=embeddings, allow_dangerous_deserialization=True)

# Set up retriever
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 100, "lambda_mult": 1})

# Initialize ChatOllama model
# model = ChatOllama(
#     model="llama3.2:1b",
#     base_url="http://localhost:11434",
#     streaming=True,  # Enable streaming
# )
# model = ChatOllama(
#     model="phi4:latest", 
#     base_url="http://192.168.0.49:2255",
#     streaming=True,  # Enable streaming
# )


model = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            streaming=True, 
        )

# Define the prompt
prompt_template = """
You are a chatbot assistant for Smart Business IoT, designed to provide precise and accurate answers strictly based on the provided context. 
You have access to ingested document related to Smart Business IoT, an application that allows the user to view and manage the data of
various sensors installed on different sites. The user can monitor data focusing on light
intensity, temperature, humidity, and carbon dioxide levels through installed devices.
The application connects multiple sites and acts as a single data point. The user can
add events, set event rules, generate reports based on the collected data, and manage
administration through the application.
### Guidelines:
1. Provide answers **strictly** based on the given context.  
- If the answer is **not available**, respond with: "I'm sorry, I don't have that information."
2. Ensure responses are **clear, concise, and directly relevant** to the question.
3. **Do not** answer questions outside the scope of the provided context.

Question: {question}  
Context: {context}  
Answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ""


# Streamlit app title
st.title("NetvirE SmartAssist")
# Sidebar
with st.sidebar:
    # You can replace the URL below with your own logo URL or local image path
    st.image(logo_icon, use_column_width=True)
    # st.markdown("### 📚 NetvirE AI-Powered Assistant")
    st.markdown("---")
    
    # Navigation Menu
    menu = ["🏠 Home", "🤖 Chatbot", "📧 Contact"]
    choice = st.selectbox("Navigate", menu)

# Home Page
if choice == "🏠 Home": 
    st.markdown("""  
### **Your AI-Powered Assistant for Smart Retail Spaces & Beyond**  

---  

### **What is NetVire SmartAssist?**  
NetVire SmartAssist is an intelligent AI chatbot designed to help you navigate and utilize the full potential of the NetVire Smart Business IoT application.
Whether you're configuring sensors, setting event rules, or analyzing environmental data, SmartAssist delivers fast, reliable, and contextual answers.

---  

### **Key Features**  
- 🔹 **Instant Answers** –  Quickly resolve queries related to sensors, site setup, data interpretation, and event rules. 
- 🔹 **Smart Search** – Pull specific details from setup documentation, admin guides, and sensor manuals.
- 🔹 **Conversational AI** – Ask natural-language questions and receive accurate, actionable responses tailored to your context.

---  

### **How It Works**  
NetVire SmartAssist taps into internal documentation, including:  
✅ **Smart Business IoT Setup Guide**  
✅ **Sensor Calibration and Troubleshooting Manuals**  
✅ **Site and Device Management Docs**  
✅ **Environmental Data Best Practices**  

Just **type a question**, and SmartAssist fetches answers directly from your knowledge base.

---

### **Why Use NetVire SmartAssist?**  
✔️ **Boost Efficiency** – No more manual digging—find insights across multiple sites and devices in seconds.  
✔️ **Accurate & Contextual** – Built on **Phi4** and **FAISS**, ensuring answers are precise and relevant to your IoT ecosystem.  
✔️ **User-Friendly** – Navigate the application’s capabilities with ease and confidence.  
✔️ **Secure & Scalable** – Enterprise-grade data handling and multi-site support keep your operations smooth and protected.  

---

### **Examples of What You Can Ask**  
💡 *"How can I set a CO₂ alert threshold for Site A?"*  
💡 *"Why is the humidity sensor not reporting data?"*  
💡 *"How do I generate a weekly temperature report?"*  
💡 *"Can I set different rules for light intensity across locations?"*

---  

### **Powered by Industry-Leading Technology**  
🚀 **Phi4** – Offers powerful, context-sensitive responses for your IoT queries.
🚀 **Nomic-Embed-Text** – Powers deep document understanding and vector search.
🚀 **FAISS** – Enables ultra-fast search and retrieval across large knowledge sources.

---  

### **About NetVire SmartAssist**  
SmartAssist is part of **NetVire’s mission to simplify smart business operations**.
With easy access to technical knowledge and actionable insights, SmartAssist helps teams make the most of their IoT data—across sites, sensors, and systems. 

Let **SmartAssist** guide you through everything the **Smart Business IoT** platform has to offer. Start optimizing your smart retail spaces today! 😊
""")


elif choice == "🤖 Chatbot":   

    # Predefined FAQs Section
    st.sidebar.subheader("Predefined Questions")
    for question in predefined_faqs:
        if st.sidebar.button(question, key=f"predefined_{question}"):
            st.session_state['user_input'] = question  # Prepopulate input field

    # Dynamic FAQs Section
    st.sidebar.subheader("Popular Questions")
    top_faqs = get_top_faqs(5, faq_db_name)
    if top_faqs:
        for question in top_faqs:
            if st.sidebar.button(question, key=f"popular_{question}"):
                st.session_state['user_input'] = question  # Prepopulate input field
    else:
        st.sidebar.write("No popular questions yet.")



    # Display all previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input box for user query
    user_input = st.chat_input("Ask me anything about NetvirE's Smart Business IoT...")
    if st.session_state['user_input']:
        user_input = st.session_state['user_input']
    # Input box for user query
    if user_input:
        # Append user message to session state
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(user_input)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Format the prompt
        formatted_query = prompt.format_messages(question=user_input, context=context)

        # Stream assistant's response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("⌛ Thinking...")  # Show a loading indicator

            response = ""
            response_stream = model.stream([HumanMessage(content=formatted_query[0].content)])
            for chunk in response_stream:
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                response += token
                response_placeholder.markdown(response)

            # Append assistant response to session state
            st.session_state.messages.append({"role": "assistant", "content": response})
        # Log the question to the FAQ database
        log_faq(user_input, faq_db_name)
        # Clear the input field for the next interaction
        st.session_state['user_input'] = ""

elif choice == "📧 Contact":
    st.title("Contact Us")
    st.write("📞 Contact NetvirE Support \nFor assistance with NetvirE services, troubleshooting, or general inquiries, please reach out to our support team.")

# Footer code
footer = """
<style>
footer {
    visibility: hidden;
}

.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background-color: #f1f1f1;
    text-align: center;
    padding: 10px;
    font-size: 14px;
    color: #555;
    z-index: 1000;
}
</style>
<div class="footer">
    © 2025 Thinkpalm Technologies Pvt Ltd | Powered by AIServices
</div>
"""
st.markdown(footer, unsafe_allow_html=True)