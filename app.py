from dataclasses import dataclass
from typing import Literal
import streamlit as st
import streamlit.components.v1 as components
import os
import time
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from dotenv import load_dotenv


@dataclass
class Message:
    """Class for keeping track of a chat message."""
    Origin: Literal["human", "ai"]
    Message: str

load_dotenv()

##load the GROQ and OpenAI API KEY
#os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
#groq_api_key=os.getenv('GROQ_API_KEY')
os.environ['OPENAI_API_KEY']=st.secrets["OPENAI_API_KEY"]
groq_api_key=st.secrets['GROQ_API_KEY']

# Opções de modelos disponíveis
model_options = ["LLama3-8b-8192", "Gemma-7b-It", "Llama3-70b-8192", "Mixtral-8x7b-32768"]

default_prompt_text = """Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
The answer should match the language of the question.
If you don't know the answer, then say "I don't know", but in the same language of the question.
<context>
{context}
<context>
Question:
{input} 
"""

def vector_embedding():

    if "vectors" not in st.session_state:
        st.session_state.embeddings=OpenAIEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("./Files") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200) ## Chunk creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs) 

        # Store the document names alongside the content
        for doc in st.session_state.final_documents:
            doc.metadata["source"] = doc.metadata.get("source", "Unknown")

        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

# Sidebar
with st.sidebar:
    st.header("Configurations")
    if st.button("Documents Embedding"):
        vector_embedding()
        st.write("Vectorstore DB is ready")

    # Seleção do modelo através de um combobox
    selected_model = st.selectbox("Select a model:", model_options)
    # Select LLM according to combobox
    llm = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=selected_model
    )
    prompt_text = st.text_area("Edit the prompt template:", default_prompt_text, height=300)
    prompt = ChatPromptTemplate.from_template(prompt_text)


def load_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=ConversationSummaryMemory(llm=llm),
        )
 
   
def on_click_callback():
    ##with get_openai_callback() as cb:
        human_prompt = st.session_state.human_prompt

        document_chain=create_stuff_documents_chain(llm,prompt)
        retriever=st.session_state.vectors.as_retriever()
        retriever_chain=create_retrieval_chain(retriever, document_chain)
        start=time.process_time()
        response=retriever_chain.invoke({'input':human_prompt})
        print("Response time :",time.process_time()-start)
        
        llm_response=response['answer']

        # With a streamlit expander
        with st.expander("Document similarity search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(f"Document Name: {doc.metadata.get('source', 'Unknown')}")
                st.write(doc.page_content)
                st.write("---------------------------------")

        #llm_response = st.session_state.conversation.run(
        #    human_prompt
        #)

        st.session_state.history.append(
            Message("human", human_prompt)
        )
        st.session_state.history.append(
            Message("ai", llm_response)
        )
    ##    st.session_state.token_count += cb.total_tokens
    
load_css()
initialize_session_state()

 
st.title("SpiritusPedia")
chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")
credit_card_placeholder = st.empty()

with chat_placeholder:
    for chat in st.session_state.history:
        div = f"""
<div class="chat-row 
    {'' if chat.Origin == 'human' else 'row-reverse'}">
    <img class="chat-icon" src="app/static/{
        'robot.png' if chat.Origin == 'ai' 
                    else 'human.png'}"
         width=32 height=32>
    <div class="chat-bubble
    {'ai-bubble' if chat.Origin == 'ai' else 'human-bubble'}">
    &#8203;{chat.Message}</div>
    </div>
        """
        st.markdown(div, unsafe_allow_html=True)        
        #st.markdown(f"**From {chat.Origin}:** {chat.Message}")
        
with prompt_placeholder:
    st.markdown('**Chat** - _press Enter to Submit_')
    cols = st.columns((6,1))
    cols[0].text_input(
        "Chat",
        #value="",
        label_visibility="collapsed",
        key="human_prompt",
    )
    cols[1].form_submit_button(
        "Send",
        type="primary",
        on_click=on_click_callback,
    )


components.html("""
<script>
const streamlitDoc = window.parent.document;
                
const buttons = Array.from(
    streamlitDoc.querySelectorAll('.stButton > button')
);
            
const submitButton = buttons.find(
    el => el.innerText === 'Submit'
);

streamlitDoc.addEventListener('keydown', funcion(e) {
    switch (e.key) {
        case 'Enter':
            submitButton.click();
            break;
    }
});
                
</script>
""",
    height=0,
    width=0,
)