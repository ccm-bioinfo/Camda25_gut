import streamlit as st
import pandas as pd
from hiPCA import calculate_hiPCA
from RF_GMHI import get_all_GMHI

st.title("Analysis Tool")

# Create tabs
tab_info, tab1, tab2 = st.tabs(["Information", "hiPCA Calculation", "GMHI Calculation"])

# Tab: Information
with tab_info:
    st.header("About This Tool")
    st.write("""
        This tool provides functionalities to perform high-dimensional PCA (hiPCA) calculations
        and GMHI (Gut Microbiome Health Index) calculations. Users can upload their datasets and 
        obtain analysis results based on selected models.

        ### References

        - Zhu, J., Xie, H., Yang, Z., et al. (2023). **Statistical modeling of gut microbiota for personalized health status monitoring**. *Microbiome, 11*, 184. [https://doi.org/10.1186/s40168-023-01614-x](https://doi.org/10.1186/s40168-023-01614-x)
        - Gupta, V.K., Kim, M., Bakshi, U., et al. (2020). **A predictive index for health status using species-level gut microbiome profiling**. *Nature Communications, 11*, 4635. [https://doi.org/10.1038/s41467-020-18476-8](https://doi.org/10.1038/s41467-020-18476-8)

        ### GitHub Repository
        You can find the source code and documentation on our GitHub repository: 
        [GitHub Link](https://github.com/yourusername/yourrepository)

        ### Usage
        - Upload your dataset in the respective tabs.
        - Select the model for the analysis.
        - Click the "Run" button to perform the calculations.
        - Download the results as CSV files.
    """)
    # st.markdown(
    # """
    # <div style="text-align: center;">
    #     <img src="path/to/footer_image.png" width="150" />
    # </div>
    # """,
    # unsafe_allow_html=True
    # )
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        # st.image(logo)
        st.image("../images/LOGO_CENTRO_DE_CIENCIAS_MATEMATICAS.png")

# Tab 1: hiPCA Calculation
with tab1:
    st.header("hiPCA Calculation")

    # File upload and processing for hiPCA
    uploaded_file = st.file_uploader("Choose a file for hiPCA")

    if uploaded_file is not None:
        # Delimiter detection and file reading
        sample = uploaded_file.read(1024).decode('utf-8')
        uploaded_file.seek(0)

        delimiter = '\t' if sample.count('\t') > sample.count(',') else ','
        data = pd.read_csv(uploaded_file, delimiter=delimiter, index_col=0)
        
        st.write("Preview of the uploaded file:")
        st.write(data)

        models = {'CAMDA MODEL':'camda_all_samples', 'ORIGINAL MODEL (ZHU et al)':'zhu_model'}
        selected_model = st.selectbox("Select a model to use:", list(models.keys()))

        if st.button("Run hiPCA Calculation"):
            results = calculate_hiPCA(f'hiPCA/model_data/{models[selected_model]}', data)
            st.write("Results of hiPCA calculation:")
            st.write(results)

            resultados_csv = results.to_csv(index=False)

            st.download_button(
                label="Download hiPCA results as CSV",
                data=resultados_csv,
                file_name='hipca_results.csv',
                mime='text/csv',
            )

        species = pd.read_csv(f'hiPCA/model_data/{models[selected_model]}/scaling_parameters.csv')
        if len([x for x in list(data.index) if x in list(species['specie'])]) < len(list(species['specie'])) // 2:
            st.warning('Less than half of the model species were found in the samples.')

# Tab 2: GMHI Calculation
with tab2:
    st.header("GMHI Calculation")

    gmhi_file = st.file_uploader("Choose a file for GMHI Calculation")

    if gmhi_file is not None:
        # Delimiter detection and file reading
        gmhi_sample = gmhi_file.read(1024).decode('utf-8')
        gmhi_file.seek(0)

        delimiter = '\t' if gmhi_sample.count('\t') > gmhi_sample.count(',') else ','
        gmhi_data = pd.read_csv(gmhi_file, delimiter=delimiter, index_col=0)
        
        st.write("Preview of the GMHI file:")
        st.write(gmhi_data)

        models = {'CAMDA MODEL':'RF_GMHI/model_data/taxonomy.csv', 'ORIGINAL MODEL':'RF_GMHI/model_data/taxonomy_original.csv'}
        selected_model = st.selectbox("Select a model to use:", list(models.keys()))

        taxonomy = pd.read_csv(f"{models[selected_model]}")
        MH_tax, MN_tax = set(taxonomy['Healthy']), set(taxonomy['Unhealthy'])

        if st.button("Run GMHI Calculation"):
            gmhi_results = get_all_GMHI(gmhi_data, MH_tax, MN_tax)
            st.write("Results of GMHI calculation:")
            st.write(gmhi_results)

            gmhi_results_csv = gmhi_results.to_csv(index=False)

            st.download_button(
                label="Download GMHI results as CSV",
                data=gmhi_results_csv,
                file_name='gmhi_results.csv',
                mime='text/csv',
            )
