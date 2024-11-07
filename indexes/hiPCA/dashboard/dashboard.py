import streamlit as st
import pandas as pd
import io
from .. import hiPCA_calculate

st.title("File Upload Example")

# Upload the file
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # Read a few lines to guess the delimiter
    sample = uploaded_file.read(1024).decode('utf-8')
    uploaded_file.seek(0)  # Reset file pointer

    # Check if tab character is frequent, indicating a TSV
    if sample.count('\t') > sample.count(','):
        delimiter = '\t'
        st.write("Detected a TSV file.")
    else:
        delimiter = ','
        st.write("Detected a CSV file.")

    # Load the data using the detected delimiter
    data = pd.read_csv(uploaded_file, delimiter=delimiter)
    st.write("Preview of the uploaded file:")
    st.write(data)

calculate_hiPCA(uploaded_file)

