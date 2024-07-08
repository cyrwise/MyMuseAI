## STREAMLIT UI

import streamlit as st


def add_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

        body {
            background: url('https://example.com/your_background_image.jpg') no-repeat center center fixed;
            background-size: cover;
            color: white;
            font-family: 'Roboto', sans-serif;
            overflow: hidden;
        }

        .main {
            background-color: rgba(0, 0, 0, 0.6);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.5);
            transform: translateZ(0);
            perspective: 1000px;
        }

        .stTextInput label {
            font-size: 18px;
            color: #fff;
        }

        .stButton button {
            background-color: #4285f4;
            color: white;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            transition: transform 0.3s;
        }

        .stButton button:hover {
            background-color: #357ae8;
            transform: scale(1.1);
        }

        .stMarkdown h1 {
            color: #4285f4;
            font-size: 36px;
            text-align: center;
            animation: fadeIn 2s;
        }

        .stMarkdown h2 {
            color: #4285f4;
            font-size: 24px;
            animation: fadeIn 2s;
        }

        .stMarkdown p {
            color: #fff;
            font-size: 16px;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes moveBg {
            0% { background-position: 0 0; }
            50% { background-position: 100% 100%; }
            100% { background-position: 0 0; }
        }

        body::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('https://example.com/your_overlay_image.png') repeat;
            opacity: 0.1;
            z-index: -1;
            animation: moveBg 20s linear infinite;
        }

        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)


## LLM + RAG

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# Read the content of the text file
with open('social_media_tips.txt', 'r') as file:
    document_content = file.read()

# Split document into chunks
def split_document(doc, chunk_size=200):
    words = doc.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

document_chunks = split_document(document_content)

# print (document_chunks)
# print ('```\n')


# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for each chunk
document_embeddings = model.encode(document_chunks)

documents = [{'id': i, 'content': chunk, 'embedding': embedding}
             for i, (chunk, embedding) in enumerate(zip(document_chunks, document_embeddings))]

embeddings = np.array([doc['embedding'] for doc in documents], dtype='float32')

# Build the FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def retrieve_documents(query, top_k=5):
    query_embedding = model.encode([query]).astype('float32')  # Ensure input is a list and type is float32
    distances, indices = index.search(query_embedding, top_k)
    return [(documents[idx]['content'], distances[0][i]) for i, idx in enumerate(indices[0])]

# Test the retrieval mechanism
# query = "How to get more views on Instagram?"
# retrieved_docs = retrieve_documents(query)
# for content, distance in retrieved_docs:
#     print(f"Content: {content}\nDistance: {distance}\n")

# # Load the generative model
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set API key for Upstage
client = OpenAI(
  api_key="INSERT_API_KEY_HERE",
  base_url="INSERT_BASE_URL_HERE"
)


def generate_response(query):
    retrieved_docs = retrieve_documents(query, top_k=3)
    context = " ".join([doc for doc, _ in retrieved_docs])

    input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
    # print(input_text)

    response = client.chat.completions.create(
        model='solar-1-mini-chat',
        messages=[{"role": "user", "content": input_text}],
        temperature=0, # this is the degree of randomness of the model's output
    )
    # input_ids = tokenizer.encode(input_text, return_tensors='pt')
    # output = gpt_model.generate(input_ids, max_length=1024, num_return_sequences=1, temperature=0.7, top_p=0.9)
    # response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.choices[0].message.content

# Test the complete RAG mechanism
# response = generate_response("Generate a short script for an ad for coffee.")
# print(response)




def get_user_input():
    st.markdown("# MyMuse.AI")
    keyword = st.text_input("Enter a keyword or theme:", placeholder="Suggestions: technology, health, education")
    audience = st.text_input("Enter the target audience:", placeholder="Suggestions: child, teen, adult")
    content_creator = st.text_input("Enter the type of content creator to use as an example:", placeholder="Suggestions: Jenna Marbles, Marques Brownlee, Joe Rogan")
    return keyword, audience, content_creator

def generate_answer(keyword, audience, content_creator):

    content = "You are a scriptwriter for an ad. Your job is to write ad copies for the user. Use all your context to generate the best possible ad copy that can generate 100 million views in 10 days. Input provided in triple backticks is user input. Don't make ads in the form of songs."
    content += f"```Generating content about '{keyword}' for {audience} using {content_creator}```"

    response = generate_response(content)

    return response

def main():
    add_custom_css()

    with st.container():
        keyword, audience, content_creator = get_user_input()

        if st.button("Generate Content"):
            if keyword and audience and content_creator:
                content = generate_answer(keyword, audience, content_creator)
                with st.spinner('Generating content...'):
                    st.success("Content Generated!")
                    st.markdown("## Generated Content:")
                    st.write(content)
            else:
                st.error("Please enter keyword, audience, and content creator.")

if __name__ == "__main__":
    main()
