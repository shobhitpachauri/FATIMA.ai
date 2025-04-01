from flask import Flask, request, jsonify
from chatbot import qa_chain  # Import your existing chatbot logic

app = Flask(__name__)

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    data = request.json
    query = data.get("query")
    response = qa_chain.invoke({"query": query})
    return jsonify({"result": response["result"]})

if __name__ == "__main__":
    app.run(port=8000)  # Run the server on port 8000 