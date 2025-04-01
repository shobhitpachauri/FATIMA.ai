import streamlit as st
import requests

# Set the title of the app
st.title("UT Dallas Chatbot")

# Create a text input for user queries
user_input = st.text_input("Ask me anything about UTD:")

# Button to submit the query
if st.button("Submit"):
    if user_input:
        # Call the chatbot API or function here
        # For now, we'll simulate a response
        # Replace this with your actual chatbot invocation logic
        response = requests.post("http://localhost:8000/chatbot", json={"query": user_input})
        
        if response.status_code == 200:
            chatbot_response = response.json().get("result", "I don't have information about that.")
            st.write("Chatbot:", chatbot_response)
        else:
            st.write("Error: Unable to get a response from the chatbot.")
    else:
        st.write("Please enter a question.")

# Optional: Add a footer
st.markdown("---")
st.write("Powered by Streamlit and your UTD Chatbot")
