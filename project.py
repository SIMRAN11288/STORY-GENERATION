from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import streamlit as st
from langchain.chains import RetrievalQA
import re
import requests
from langchain_core.documents import Document
load_dotenv()
llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0)
embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#session state Initializations
if "story" not in st.session_state:
    st.session_state.story=[]
if "genre" not in st.session_state:
    st.session_state.genre=[]
if "continuations" not in st.session_state:
    st.session_state.continuations=[]
if "tone" not in st.session_state:
    st.session_state.tone=[]
if "vector_store" not in st.session_state:
    st.session_state.vector_store=None
    
# Vector store Creation

def update_vectorestore(story_text,embedding,vector_store):
    splitter=RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=50)
    docs=splitter.split_documents([Document(page_content=story_text)])
    if not docs:
        st.write("No text generated")
        return vector_store
    if vector_store is None:
        vector_store=FAISS.from_documents(docs,embedding=embedding)
    else:
        vector_store.add_documents(docs)
    return vector_store

def ask_questions(question,vector_store):
    #questiona to be asked
    retriever=vector_store.as_retriever()
    qa=RetrievalQA.from_chain_type(llm=llm,retriever=retriever)
    return qa.invoke(question)

def dict(word):
    url=f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    req=requests.get(url)
    if req.status_code==200:
        data=req.json()
        return data[0]['meanings'][0]['definitions'][0]['definition']
    return None

st.title("STORY GENERATION")

st.session_state.genre=st.text_input('Tell your Genre for story')
st.session_state.tone=st.text_input('Tone should be')

initial_plot=st.text_input('Provide your initial plot of story generation')
if initial_plot and not st.session_state.story:
    st.session_state.story.append(initial_plot)

def generate_continuations():
    prompt1=PromptTemplate(template=
                "You are a creative storyteller.\n"
                "Generate 3 possible {genre} story continuations for the given plot in {tone} tone in not more than 300 words.\n\n"
                "Plot: {story}\n\n"
                "Provide the results as a numbered list:\n"
            ,input_variables=["genre", "story","tone"])
    formatted_prompt=prompt1.format(genre=st.session_state.genre,story=" ".join(st.session_state.story),tone=st.session_state.tone)
        #for _ in range(3):
    response=llm.invoke(formatted_prompt)
    continuations = [
    part.strip() 
    for part in response.content.split("\n")   # 1
    if part.strip() and (part.strip()[0].isdigit())  # 2
        ]
    st.session_state.vector_store=update_vectorestore(" ".join(st.session_state.story),embedding,st.session_state.vector_store)
    st.session_state.continuations=continuations
#calling 1st continuations
if initial_plot and not st.session_state.continuations:
    generate_continuations() 

choice=st.radio("Your preference:",["enter 1/2/3 to choose story continuation option",
        "Choose stop to quit","answer a question from selected story only","get meaning"],key='select1')
if choice=="Choose stop to quit":
    st.write("story ended")
    

elif choice=='get meaning':
    word=st.text_input("Enter the word to get its meaning")
    if st.button('get meaning'):
        meaning=dict(word)
        display=f'{word} menaing: {meaning}'
        st.write(display)
        
elif choice=='answer a question from selected story only':
    if st.session_state.vector_store:
        user_Q=st.text_input("Ask your Q")
        if user_Q:
            answer=ask_questions(user_Q,st.session_state.vector_store)
            display_ans=f'ANSWER: {answer}'
            st.write(display_ans)
            
    else:
        st.error('No answers to print')
        
elif choice=="enter 1/2/3 to choose story continuation option":
    if st.session_state.continuations:
        st.write("### Possible Continuations")
        for i, r in enumerate(st.session_state.continuations, start=1):
            st.write(f"{r}")

        selected_num = st.number_input("Enter 1/2/3 to choose continuation:", min_value=1, max_value=3, step=1)
        if st.button("confirm choice"):
            selected=st.session_state.continuations[selected_num-1]
            #Removing numbering
            selected=selected.lstrip("(1234567890).").strip()
            # Remove headings in bold (like **The Dark Forest**)
            selected = re.sub(r"^\*\*.*?\*\*\s*", "", selected)
            st.session_state.story.append(selected)
            st.session_state.vector_store=update_vectorestore(" ".join(st.session_state.story),embedding,st.session_state.vector_store)
            generate_continuations()

final_story=" ".join(st.session_state.story)
st.write("Final Story")
st.success(final_story)






