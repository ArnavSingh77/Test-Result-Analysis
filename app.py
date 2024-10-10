import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

def load_data(file):
    if file.name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file, header=None)
    elif file.name.endswith('.csv'):
        df = pd.read_csv(file, header=None)
    else:
        st.error("Unsupported file format. Please upload an Excel (.xlsx or .xls) or CSV (.csv) file.")
        return None
    
    # Find the row with column headers
    header_row = None
    for idx, row in df.iterrows():
        if row.astype(str).str.contains('Enrol|Roll', case=False).any():
            header_row = idx
            break
    
    if header_row is None:
        st.error("Could not find a proper header row in the file. Please check the file format.")
        return None
    
    # Set the header and remove unnecessary rows
    df.columns = df.iloc[header_row]
    df = df.iloc[header_row + 1:].reset_index(drop=True)
    
    # Rename columns for consistency
    df = df.rename(columns=lambda x: x.strip())
    
    # Identify important columns
    enrollment_col = next((col for col in df.columns if 'enrol' in col.lower() or 'roll' in col.lower()), None)
    rank_col = next((col for col in df.columns if 'no' in col.lower() or 'rank' in col.lower()), None)
    subject_cols = [col for col in df.columns if col.lower() in ['physics', 'chemistry', 'mathematics']]
    total_col = next((col for col in df.columns if 'total' in col.lower()), None)
    
    if not all([enrollment_col, rank_col, len(subject_cols) == 3, total_col]):
        st.error("Could not identify all required columns. Please check the file format.")
        return None
    
    # Select and rename columns
    columns_to_keep = [rank_col, enrollment_col] + subject_cols + [total_col]
    df = df[columns_to_keep]
    df.columns = ['Rank', 'Enrollment Number', 'Physics', 'Chemistry', 'Mathematics', 'Total']
    
    # Ensure numeric columns are properly typed
    numeric_cols = ['Rank', 'Physics', 'Chemistry', 'Mathematics', 'Total']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def main():
    st.title("FIITJEE Result Analysis App")

    # File upload
    uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "xls", "csv"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.subheader("Data Preview")
            st.dataframe(df.head())

            # Input enrollment number
            enrollment_number = st.text_input("Enter your enrollment number:")
            
            if enrollment_number:
                # Filter data for the specific student
                student_data = df[df['Enrollment Number'].astype(str) == enrollment_number]
                
                if not student_data.empty:
                    st.subheader("Your Results")
                    st.dataframe(student_data)
                    
                    rank = student_data['Rank'].values[0]
                    total_students = len(df)
                    
                    st.write(f"Your Rank: {rank} out of {total_students}")

                    # Overall performance gauge
                    student_total = student_data['Total'].values[0]
                    class_average = df['Total'].mean()
                    
                    # Calculate maximum score (highest marks of students)
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

                    # Improved comparison with other students using Seaborn
                    st.subheader("Comparison with Other Students")
                    subjects = ['Physics', 'Chemistry', 'Mathematics', 'Total']

                    for subject in subjects:
                        plt.figure(figsize=(10, 6))
                        
                        # Create a violin plot to show distribution
                        sns.violinplot(x=subject, data=df, inner='quartile', color='skyblue')
                        
                        # Add a line for the student's score
                        plt.axvline(student_data[subject].values[0], color='red', linestyle='--', label='Your Score')
                        
                        # Add a line for the average score
                        average_score = df[subject].mean()
                        plt.axvline(average_score, color='green', linestyle=':', label='Average Score')
                        
                        # Adding labels and title
                        plt.title(f"Distribution of {subject} Marks", fontsize=16)
                        plt.xlabel(f"{subject} Marks", fontsize=14)
                        plt.ylabel("Density", fontsize=14)
                        plt.legend()
                        
                        # Display the plot
                        st.pyplot(plt)
                        plt.clf()  # Clear the figure for the next plot
                    
                    # Percentile calculation
                    percentile = (1 - (rank - 1) / total_students) * 100
                    
                    st.subheader("Your Performance Summary")
                    st.write(f"Rank: {rank}")
                    st.write(f"Percentile: {percentile:.2f}%")

                else:
                    st.error("Enrollment number not found in the data.")

if __name__ == "__main__":
    main()