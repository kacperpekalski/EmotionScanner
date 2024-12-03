import streamlit as st
import requests

API_URL = "http://localhost:8000/v1/upload/" 

def main():
    st.title("Upload JPG to FastAPI Backend")
    
    uploaded_file = st.file_uploader("Wybierz plik JPG", type=["jpg", "jpeg"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Przesłany plik", use_container_width=True)
        if st.button("Wyślij plik"):
            with st.spinner("Wysyłanie pliku do backendu..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "image/jpeg")}
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        st.success("Plik został pomyślnie przesłany!")
                        st.json(response.json())
                    else:
                        st.error(f"Nie udało się przesłać pliku: {response.status_code}")
                        st.text(response.text)
                except Exception as e:
                    st.error(f"Wystąpił błąd: {e}")

if __name__ == "__main__":
    main()