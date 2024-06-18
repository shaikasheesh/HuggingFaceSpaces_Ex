import warnings
warnings.filterwarnings("ignore") 
import os 
#pdf loader
from PyPDF2 import PdfReader
#textsplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Embeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
#storing vector embeddings
from langchain_community.vectorstores import FAISS
#to connect llm models from huggingface
from langchain import HuggingFaceHub
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space


def get_pdf_text(filename):
    pdf_reader = PdfReader(filename)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    return chunks

def get_vectorstore(textchunks):
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf_embedding = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    db = FAISS.from_texts(textchunks, hf_embedding)
    return db

def get_conversation_chain(db):
    # llm=HuggingFaceHub(repo_id="google/flan-t5-base",
    #                                         model_kwargs={"max_new_tokens": 100,
    #   "temperature": 0.2,

    #                        })

    llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama3-8b-8192",
            temperature = 0
        )
    chain = load_qa_chain(llm, chain_type="stuff") #initialize llm and chain type

    return chain

def main():
    st.set_page_config(page_title="RAG: Internet Expense Invoice PDF 📄")
    st.header("RAG: Internet Expense Invoice PDF 📄")
    user_question = st.text_input("Ask a question about Details of the Invoice:")
    # if user_question:
    #     handle_userinput(user_question)


    with st.sidebar:
        st.markdown('''
        ## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [HuggingFace](https://huggingface.co/google/flan-t5-base) ''')
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload Internet Invoice here and click on 'Process'")
        st.write('Made with ❤️ by Asheesh ')
        if st.button("Process"):
            with st.spinner("Processing"):
                #get pdf text
                raw_text = get_pdf_text(pdf_docs)
                
                #get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)
    #           #create vector store
                st.session_state.vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
                
    if user_question:
        docs = st.session_state.vectorstore.similarity_search(user_question) #perform similarity search in the vector database (db)
        answer = st.session_state.conversation.run(input_documents=docs, question=user_question) #output the answer
        st.write(answer)
        

        

if __name__ == '__main__':
    load_dotenv()
    os.getenv("HUGGINGFACEHUB_API_TOKEN")
    main()