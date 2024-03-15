import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from PIL import Image

st.set_page_config(
    page_title = "Chatbot IBK Modelos",
    page_icon = "https://interbank.pe/o/public-zone-dxp-theme/images/favicon.ico"
)

list_nmodels = {
    "Comportamental BPE" : "databpe",
    "ADMISIÓN BANK QUICK WIN" : "dataquickwin",
    "ORIGEN NO BANK BPE" : "datanobankbpe"
}

# Clear the Chat Messages
def clear_chat_history():
    st.session_state.messages = [{"role" : "assistant", "content": msg_chatbot}]

with st.sidebar:
    st.title("Chatbot IBK Modelos")
    image = Image.open('interbank.png')
    st.image(image, caption = 'Interbank')

    nmodel = st.selectbox('Eliga un modelo',('Comportamental BPE', 'ADMISIÓN BANK QUICK WIN', 'ORIGEN NO BANK BPE'), key = "nmodel", on_change = clear_chat_history)

    st.write(list_nmodels[nmodel])

    st.markdown(
        """        
        ### Propósito
        Modificar
        
        ### Fuentes de datos que se han considerado
        - Documento Modelo Comportamental BPE 1
        - DocumentoAdmisionBankQuickWin_comentariosAIS
        - MODELO ORIGEN NO BANK BPE v9
    """
    )

msg_chatbot = """
        Soy un chatbot que te ayudará a conocer información sobre los modelos desarrollados en Interbank: 
        
        ### Preguntas que puedes realizar
        - ¿De qué trata el modelo?
        - ¿Cuáles son las principales fuentes utiizadas, puedes tabularlas?
        - ¿Cuál ha sido el periodo de observación para la construcción del modelo?
        - ¿Cuál es el performance del modelo?
"""

#Store the LLM Generated Reponese
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content" : msg_chatbot}]

# Diplay the chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create a Function to generate
def generate_response(prompt_input, nmodel):

    template = """Responda a la pregunta basada en el siguiente contexto.
    Si no puedes responder a la pregunta, usa la siguiente respuesta "No lo sé disculpa."

    Contexto: 
    {context}
    Pregunta: {question}
    Respuesta: 
    """

    prompt = PromptTemplate(
        input_variables = ["context", "question"],
        template = template
    )

    llm = ChatOpenAI(
        model_name = 'gpt-3.5-turbo-0125',
        temperature = 0.0
    )

    embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")
    vectorstore = FAISS.load_local(list_nmodels[nmodel], embeddings, allow_dangerous_deserialization = True)

    retriever = vectorstore.as_retriever(search_kwargs = {"k": 5})

    rag = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag.invoke(prompt_input + " sobre el modelo " + nmodel)

st.sidebar.button('Limpiar historial de chat', on_click = clear_chat_history)

prompt = st.chat_input("Ingresa tu pregunta")
if prompt:
    st.session_state.messages.append({"role" : "user", "content" : prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generar una nueva respuesta si el último mensaje no es de un assistant, sino un user
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Esperando respuesta, dame unos segundos."):
            response = generate_response(prompt, nmodel)
            placeholder = st.empty()
            placeholder.markdown(response)

    message = {"role" : "assistant", "content" : response}
    st.session_state.messages.append(message)
