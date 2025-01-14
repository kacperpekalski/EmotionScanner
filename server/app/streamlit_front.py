import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/v1/upload/" 

def main():
    st.title("Upload JPG for Emotion Detection")
    
    uploaded_file = st.file_uploader("Choose a JPG file", type=["jpg", "jpeg"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded file", use_container_width=True)
        if st.button("Send file"):
            with st.spinner("Sending file to the backend..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "image/jpeg")}
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        st.success("File successfully uploaded!")
                        response_data = response.json()
                        predicted_class = response_data.get("predicted_class", "Unknown")
                        emotion = response_data.get("emotion", "Unknown")
                        
                        st.image(uploaded_file, caption=f"Predicted Class: {predicted_class}", use_container_width=True)
                        st.markdown(f"### Emotion: {emotion}")
                    else:
                        st.error(f"Failed to upload file: {response.status_code}")
                        st.text(response.text)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
