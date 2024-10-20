import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from io import BytesIO

@st.cache_data
def load_data(file, config):
    try:
        if file.size > 10 * 1024 * 1024:
            st.error("File is too large. Please upload a file smaller than 10MB.")
            return None

        # Read file based on extension
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl', header=None)
        elif file.name.endswith('.xls'):
            df = pd.read_excel(file, engine='xlrd', header=None)
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file, header=None)
        else:
            st.error("Unsupported file format. Please upload an Excel (.xlsx or .xls) or CSV (.csv) file.")
            return None
        
        header_row = None
        for idx, row in df.iterrows():
            if row.astype(str).str.contains(config['identifier_column'], case=False).any():
                header_row = idx
                break
        
        if header_row is None:
            st.error(f"Could not find a header row containing {config['identifier_column']}. Please check the file format.")
            return None
        
        df.columns = df.iloc[header_row]
        df = df.iloc[header_row + 1:].reset_index(drop=True)
        
        df = df.rename(columns=lambda x: str(x).strip())
        
        columns_to_keep = [
            config['rank_column'],
            config['identifier_column']
        ] + config['subject_columns'] + [config['total_column']]
        
        try:
            df = df[columns_to_keep]
        except KeyError as e:
            st.error(f"Could not find column: {e}. Please check your column configurations.")
            return None
        
        column_mapping = {
            config['rank_column']: 'Rank',
            config['identifier_column']: 'Identifier',
            config['total_column']: 'Total'
        }
        
        column_mapping.update({
            old: new for old, new in zip(config['subject_columns'], config['subject_names'])
        })
        
        df = df.rename(columns=column_mapping)
        
        numeric_cols = ['Rank'] + config['subject_names'] + ['Total']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def main():
    st.title("Test Result Analysis App")
    
    with st.sidebar:
        st.header("Test Details")
        
        test_name = st.text_input("Test Name", "Examination")
        st.subheader("Column Settings")
        identifier_column = st.text_input("Identifier Column Name (e.g., Roll No, Enrollment)", "Enrol. No.")
        rank_column = st.text_input("Rank Column Name", "S. No.")
        total_column = st.text_input("Total Marks Column Name", "Total")
        
        st.subheader("Subject Settings")
        num_subjects = st.number_input("Number of Subjects", min_value=1, max_value=10, value=3)
        
        subject_columns = []
        subject_names = []
        for i in range(num_subjects):
            col1, col2 = st.columns(2)
            with col1:
                subject_col = st.text_input(f"Subject {i+1} Column Name", f"Subject{i+1}")
                subject_columns.append(subject_col)
            with col2:
                subject_name = st.text_input(f"Subject {i+1} Display Name", f"Subject {i+1}")
                subject_names.append(subject_name)
    
    config = {
        'test_name': test_name,
        'identifier_column': identifier_column,
        'rank_column': rank_column,
        'total_column': total_column,
        'subject_columns': subject_columns,
        'subject_names': subject_names
    }

    uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "xls", "csv"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file, config)
        
        if df is not None:
            st.subheader("Data Preview")
            st.dataframe(df.head())

            # Ask user if they want to remove specific columns/rows
            if st.checkbox("Would you like to remove specific columns/rows?"):
                columns_to_remove = st.multiselect("Select columns to remove from analysis", df.columns.tolist())
                if columns_to_remove:
                    df = df.drop(columns=columns_to_remove)
                    st.success(f"Removed columns: {', '.join(columns_to_remove)}")
                
                rows_to_remove = st.text_area("Enter rows to remove by Identifier (comma-separated)")
                if rows_to_remove:
                    identifiers_to_remove = [x.strip() for x in rows_to_remove.split(",")]
                    df = df[~df['Identifier'].isin(identifiers_to_remove)]
                    st.success(f"Removed rows with Identifiers: {', '.join(identifiers_to_remove)}")

                # Update the data preview after column/row removal
                st.subheader("Updated Data Preview")
                st.dataframe(df.head())

            identifier = st.text_input(f"Enter your {identifier_column}:")
            
            if identifier:
                student_data = df[df['Identifier'].astype(str) == identifier]
                
                if not student_data.empty:
                    st.subheader("Your Results")
                    st.dataframe(student_data)
                    
                    rank = student_data['Rank'].values[0]
                    total_students = len(df)
                    
                    st.write(f"Your Rank: {rank} out of {total_students}")

                    student_total = student_data['Total'].values[0]
                    class_average = df['Total'].mean()
                    max_total = df['Total'].max()

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=student_total,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Your Total Score"},
                        gauge={
                            'axis': {'range': [0, max_total]},
                            'steps': [
                                {'range': [0, class_average], 'color': "lightblue"},
                                {'range': [class_average, max_total], 'color': "lightgreen"}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': class_average}}))

                    st.plotly_chart(fig)

                    st.write(f"Your total score: {student_total}")
                    st.write(f"Class average score: {class_average:.2f}")

                    st.subheader("Subject-wise Comparison")
                    for subject in config['subject_names'] + ['Total']:
                        plt.figure(figsize=(10, 6))
                        
                        sns.violinplot(x=subject, data=df, inner='quartile', palette='muted')
                        plt.axvline(student_data[subject].values[0], color='red', linestyle='--', label='Your Score')
                        plt.axvline(df[subject].mean(), color='green', linestyle=':', label='Average Score')
                        
                        plt.title(f"Distribution of {subject} Marks", fontsize=16)
                        plt.xlabel(f"{subject} Marks", fontsize=14)
                        plt.ylabel("Density", fontsize=14)
                        plt.legend()
                        
                        st.pyplot(plt)
                        plt.clf()
                    
                    percentile = (1 - (rank - 1) / total_students) * 100
                    
                    st.subheader("Performance Summary")
                    st.write(f"Rank: {rank}")
                    st.write(f"Percentile: {percentile:.2f}%")

                else:
                    st.error(f"{identifier_column} not found in the data.")

if __name__ == "__main__":
    main()
