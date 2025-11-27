import requests
import streamlit as st

st.title("ChatGPT-like clone")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "llama-3.3-70b-versatile"

if "messages" not in st.session_state:
    # st.session_state.messages = []
    st.session_state.messages = [{"role": "system", "content": "You are expert at Rhyming and poems. you will always answer using a rhyming words or sentences. You will never break the caracter. "}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Maximum allowed messages
max_messages = (
    20  # Counting both user and assistant messages, so 10 iterations of conversation
)

if len(st.session_state.messages) >= max_messages:
    st.info(
        """Notice: The maximum message limit for this demo version has been reached. We value your interest!
        We encourage you to experience further interactions by building your own application with instructions
        from Streamlit's [Build conversational apps](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps)
        tutorial. Thank you for your understanding."""
    )

else:
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                response = requests.post(
                    GROQ_API_URL,
                    headers=HEADERS,
                    json={
                        "model": st.session_state["openai_model"],
                        "messages": [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ]
                    }
                )
                response_json = response.json()
                if "choices" in response_json and response_json["choices"]:
                    full_response = response_json["choices"][0]["message"]["content"]
                    message_placeholder.markdown(full_response)
                else:
                    full_response = "No response received from API."
                    message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"Error: {str(e)}"
                message_placeholder.markdown(full_response)
                
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
