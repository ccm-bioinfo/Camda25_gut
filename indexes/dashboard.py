import streamlit as st
import pandas as pd
from hiPCA import calculate_hiPCA
from RF_GMHI import get_all_GMHI
import joblib

st.title("Gut Microbiome Health Calculator")

# Create tabs
tab_info, tab1, tab2, tab3 = st.tabs(["Information", "hiPCA Calculation", "GMHI Calculation", "Ensemble method Calculation"])

# Tab: Information
with tab_info:
    st.header("About This Tool")
    st.write("""
        This tool provides functionalities to perform health index PCA (hiPCA)
        and GMHI (Gut Microbiome Health Index) calculations. Users can upload their datasets and 
        obtain analysis results based on selected models.

        ### References

        - Zhu, J., Xie, H., Yang, Z., et al. (2023). **Statistical modeling of gut microbiota for personalized health status monitoring**. *Microbiome, 11*, 184. [https://doi.org/10.1186/s40168-023-01614-x](https://doi.org/10.1186/s40168-023-01614-x)
        - Gupta, V.K., Kim, M., Bakshi, U., et al. (2020). **A predictive index for health status using species-level gut microbiome profiling**. *Nature Communications, 11*, 4635. [https://doi.org/10.1038/s41467-020-18476-8](https://doi.org/10.1038/s41467-020-18476-8)

        ### GitHub Repository
        You can find the source code and documentation on our GitHub repository: 
        [GitHub Link](https://github.com/ccm-bioinfo/Camda25_gut)

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

    st.markdown('### Development')
    dev = '- Rafael Pérez Estrada (Centro de Ciencias Matemáticas) \n- Juan Francisco Espinosa (Centro de Ciencias Matemáticas)'
    st.markdown(dev)
# st.markdown('- Rafael Pérez Estrada (Amphora Health)')
# st.markdown('- Marco A. Nava Aguilar (Amphora Health)')


    st.markdown('### Acknowledgements')
    acknowledgements = '- Nelly Sélem Mojica (Centro de Ciencias Matemáticas) \n- Shaday Guerrero (Centro de Ciencias Matemáticas)'
    st.markdown(acknowledgements)

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
        st.write(data.head(3))

        models = {'ORIGINAL MODEL (ZHU et al)':'zhu_model', 'CAMDA MODEL 2024':'camda_all_samples', 'CAMDA MODEL 2025':'CAMDA2025_ALL_SAMPLES', 'GASTRO INTESTINAL CAMDA MODEL 2025':'CAMDA2025_GI_SAMPLES'}
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
        st.write(gmhi_data.head(3))

        models = {'ORIGINAL MODEL':'RF_GMHI/model_data/taxonomy_original.csv', 'CAMDA MODEL':'RF_GMHI/model_data/taxonomy.csv', 'CAMDA MODEL 2025':'RF_GMHI/model_data/taxonomy_camda2025_all.csv'}
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

    with tab3:
        st.header("Ensemble Method Calculation")

        tax_file = st.file_uploader("Upload the taxonomy file")
        pathways_file = st.file_uploader("Upload the pathways file")

        if tax_file is not None and pathways_file is not None:
        # Delimiter detection and file reading
            tax_sample = tax_file.read(1024).decode('utf-8')
            tax_file.seek(0)
            pathways_sample = pathways_file.read(1024).decode('utf-8')
            pathways_file.seek(0)

            delimiter = '\t' if tax_sample.count('\t') > tax_sample.count(',') else ','
            tax_data = pd.read_csv(tax_file, delimiter=delimiter, index_col=0)
            delimiter = '\t' if pathways_sample.count('\t') > pathways_sample.count(',') else ','
            pathways_data = pd.read_csv(pathways_file, delimiter=delimiter, index_col=0)
            
            st.write("Preview of the taxonomy file:")
            st.write(tax_data.head(3))
            st.write("Preview of the pathways file:")
            st.write(pathways_data.head(3))

            # models = {'ORIGINAL MODEL':'RF_GMHI/model_data/taxonomy_original.csv', 'CAMDA MODEL':'RF_GMHI/model_data/taxonomy.csv', 'CAMDA MODEL 2025':'RF_GMHI/model_data/taxonomy_camda2025_all.csv'}
        
            taxonomy = pd.read_csv('RF_GMHI/model_data/taxonomy_camda2025_all.csv')
            MH_tax, MN_tax = set(taxonomy['Healthy']), set(taxonomy['Unhealthy'])



            if st.button("Run Calculation"):
                gmhi_results = get_all_GMHI(tax_data, MH_tax, MN_tax)
                hiPCA_results = calculate_hiPCA(f'hiPCA/model_data/CAMDA2025_ALL_SAMPLES',tax_data)

                new_data = pd.DataFrame(zip(gmhi_results, hiPCA_results['Combined Index']))
                new_data.columns = ['GMHI_taxonomy', 'hiPCA_taxonomy']
                pathways = pd.read_csv('ENSEMBLE/op_ensemble_model/pathways.tsv', sep = '\t')
                # st.write(pathways_data.T.columns)
                for path in pathways['pathways']:
                    try:
                        new_data[path.split(':')[0]] = list(pathways_data.T[path])
                    except:
                        new_data[path.split(':')[0]] = [0 for _ in range(len(new_data))]

                # st.write(new_data)

                model = joblib.load('ENSEMBLE/op_ensemble_model/model.pkl')
                preds = ['Healthy' if x == 1 else 'Unhealthy' for x in model.predict(new_data)]
                index = [x[1] if x[1] > x[0] else -x[0] for x in model.predict_proba(new_data)]
                new_data['Model Index'] = index
                new_data['Model Prediction'] = preds

                st.write("Results of GMHI calculation:")
                st.write(new_data)
                # st.write(hiPCA_results)

                results_csv = new_data.to_csv(index=False)

                st.download_button(
                    label="Download GMHI results as CSV",
                    data=results_csv,
                    file_name='results_ensemble_model.csv',
                    mime='text/csv',
                )
