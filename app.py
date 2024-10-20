import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file, config):
    try:
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
        
        # Find the header row using the identifier column
        header_row = None
        for idx, row in df.iterrows():
            if row.astype(str).str.contains(config['identifier_column'], case=False).any():
                header_row = idx
                break
        
        if header_row is None:
            st.error(f"Could not find a header row containing {config['identifier_column']}. Please check the file format.")
            return None
        
        # Set the header and remove unnecessary rows
        df.columns = df.iloc[header_row]
        df = df.iloc[header_row + 1:].reset_index(drop=True)
        
        # Rename columns for consistency
        df = df.rename(columns=lambda x: str(x).strip())
        
        # Get columns based on configuration
        columns_to_keep = [
            config['rank_column'],
            config['identifier_column']
        ] + config['subject_columns'] + [config['total_column']]
        
        try:
            df = df[columns_to_keep]
        except KeyError as e:
            st.error(f"Could not find column: {e}. Please check your column configurations.")
            return None
        
        # Rename columns to standard names
        column_mapping = {
            config['rank_column']: 'Rank',
            config['identifier_column']: 'Identifier',
            config['total_column']: 'Total'
        }
        
        # Add subject mappings
        column_mapping.update({
            old: new for old, new in zip(config['subject_columns'], config['subject_names'])
        })
        
        df = df.rename(columns=column_mapping)
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['Rank'] + config['subject_names'] + ['Total']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def main():
    st.title("Test Result Analysis App")
    
    # Configuration sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Test name
        test_name = st.text_input("Test Name", "Examination")
        
        # Column configurations
        st.subheader("Column Settings")
        identifier_column = st.text_input("Identifier Column Name (e.g., Roll No, Enrollment)", "Enrollment")
        rank_column = st.text_input("Rank Column Name", "Rank")
        total_column = st.text_input("Total Marks Column Name", "Total")
        
        # Subject configuration
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
    
    # Create configuration dictionary
    config = {
        'test_name': test_name,
        'identifier_column': identifier_column,
        'rank_column': rank_column,
        'total_column': total_column,
        'subject_columns': subject_columns,
        'subject_names': subject_names
    }

    # File upload
    uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "xls", "csv"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file, config)
        
        if df is not None:
            st.subheader("Data Preview")
            st.dataframe(df.head())

            # Input identifier
            identifier = st.text_input(f"Enter your {identifier_column}:")
            
            if identifier:
                # Filter data for the specific student
                student_data = df[df['Identifier'].astype(str) == identifier]
                
                if not student_data.empty:
                    st.subheader("Your Results")
                    st.dataframe(student_data)
                    
                    rank = student_data['Rank'].values[0]
                    total_students = len(df)
                    
                    st.write(f"Your Rank: {rank} out of {total_students}")

                    # Overall performance gauge
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
                                {'range': [0, class_average], 'color': "lightgray"},
                                {'range': [class_average, max_total], 'color': "gray"}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': class_average}}))

                    st.plotly_chart(fig)

                    st.write(f"Your total score: {student_total}")
                    st.write(f"Class average score: {class_average:.2f}")

                    # Subject-wise comparison
                    st.subheader("Subject-wise Comparison")
                    for subject in config['subject_names'] + ['Total']:
                        plt.figure(figsize=(10, 6))
                        
                        # Create a violin plot
                        sns.violinplot(x=subject, data=df, inner='quartile', color='skyblue')
                        
                        # Add student's score
                        plt.axvline(student_data[subject].values[0], color='red', linestyle='--', label='Your Score')
                        
                        # Add average score
                        average_score = df[subject].mean()
                        plt.axvline(average_score, color='green', linestyle=':', label='Average Score')
                        
                        plt.title(f"Distribution of {subject} Marks", fontsize=16)
                        plt.xlabel(f"{subject} Marks", fontsize=14)
                        plt.ylabel("Density", fontsize=14)
                        plt.legend()
                        
                        st.pyplot(plt)
                        plt.clf()
                    
                    # Percentile calculation
                    percentile = (1 - (rank - 1) / total_students) * 100
                    
                    st.subheader("Performance Summary")
                    st.write(f"Rank: {rank}")
                    st.write(f"Percentile: {percentile:.2f}%")

                else:
                    st.error(f"{identifier_column} not found in the data.")

if __name__ == "__main__":
    main()