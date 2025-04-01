import os
import warnings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from huggingface_hub import login
import requests
from bs4 import BeautifulSoup
from config import HUGGINGFACE_API_TOKEN

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Login to Hugging Face
login(token=HUGGINGFACE_API_TOKEN)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Load FAISS index
INDEX_FOLDER = "faiss_index"
vector_db = FAISS.load_local(
    folder_path=INDEX_FOLDER,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)
retriever = vector_db.as_retriever()

# Setup language model with stricter parameters
llm = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct",
    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
    temperature=0.1,  # Very low temperature to prevent creativity
    max_new_tokens=50,
    task="text-generation"
)

# Create custom prompt
custom_prompt = PromptTemplate(
    template="""You are a helpful assistant for UT Dallas students and staff. 
    Answer ONLY using the information provided in the context below.
    If the information is not in the context, say "I don't have information about that in my database."
    DO NOT make up or infer any information.

    Context: {context}

    Question: {question}

    Answer: """,
    input_variables=["context", "question"]
)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

class WebsiteMonitor:
    def __init__(self, urls):
        self.urls = urls

    def scrape_url(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract relevant data from the soup object
            # This is an example; adjust the selectors based on the actual HTML structure
            data = soup.find('div', class_='desired-class').text.strip()
            return data
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None

    def run(self):
        for url in self.urls:
            data = self.scrape_url(url)
            if data:
                print(f"Data from {url}: {data}")
            else:
                print(f"No relevant data found for {url}.")

def main():
    print("UTD Chatbot initialized - Ask me anything about UTD (type 'exit' to quit)")

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            response = qa_chain.invoke({"query": query})
            print("\nChatbot:", response["result"])
            
            print("\nSources:")
            if response["source_documents"]:  # Check if there are any source documents
                for doc in response["source_documents"]:
                    print("Document content:", doc.page_content)  # Debug print
                    if "Source: " in doc.page_content:
                        url = doc.page_content.split("Source: ")[1].split("\n")[0]
                        print(f"- {url}")
                    else:
                        print("Source format is not as expected.")
            else:
                print("No sources found for this query.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
