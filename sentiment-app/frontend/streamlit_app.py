import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

def login():
    st.set_page_config(page_title="Login | Sentiment App")
    st.title("Login to Sentiment Analysis App")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        print("Submitting login to FastAPI:")
        print(f"   Username: {username}")
        print(f"   Password: {password}")

        try:
            response = requests.post(
                f"{API_URL}/login",
                json={"username": username, "password": password}
            )

            print("Response status:", response.status_code)
            print("Response body:", response.text)

            if response.status_code == 200:
                st.success("Login successful!")
                st.session_state["user"] = username
                st.session_state["tokens"] = response.json()["tokens"]
                st.switch_page("pages/upload_and_trigger.py")
            else:
                st.error("Login failed. Check your credentials.")
        except Exception as e:
            st.error(f"Request failed: {e}")

if "user" not in st.session_state:
    login()
else:
    st.switch_page("pages/upload_and_trigger.py")
