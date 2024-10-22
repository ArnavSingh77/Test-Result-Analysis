import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

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
        
        # Locate the header row
        header_row = None
        for idx, row in df.iterrows():
            if row.astype(str).str.contains(config['identifier_column'], case=False).any():
                header_row = idx
                break
        
        if header_row is None:
            st.error(f"Could not find a header row containing {config['identifier_column']}. Please check the file format.")
            return None
        
        # Set the DataFrame columns and remove the header row from the DataFrame
        df.columns = df.iloc[header_row]
        df = df.iloc[header_row + 1:].reset_index(drop=True)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Select relevant columns based on user configuration
        columns_to_keep = [
            config['rank_column'],
            config['identifier_column'],
            *config['subject_columns'],
            config['total_column']
        ]
        
        try:
            df = df[columns_to_keep]
        except KeyError as e:
            st.error(f"Could not find column: {e}. Please check your column configurations.")
            return None
        
        # Create a mapping for renaming columns
        column_mapping = {
            config['rank_column']: 'Rank',
            config['identifier_column']: 'Identifier',
            config['total_column']: 'Total'
        }
        
        # Map subject columns to display names
        for subject_col, subject_name in zip(config['subject_columns'], config['subject_names']):
            column_mapping[subject_col] = subject_name
            
        # Rename columns based on the mapping
        df = df.rename(columns=column_mapping)
        
        # Convert relevant columns to numeric
        numeric_cols = ['Rank', 'Total'] + config['subject_names']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def create_radar_chart(df, student_ids, subject_cols):
    """Create a radar chart comparing multiple students"""
    fig = go.Figure()
    
    for student_id in student_ids:
        student_data = df[df['Identifier'].astype(str) == student_id]
        if not student_data.empty:
            values = student_data[subject_cols].values[0].tolist()
            values.append(values[0])  # Complete the polygon
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=subject_cols + [subject_cols[0]],
                name=f"Student {student_id}",
                fill='toself'
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, df[subject_cols].max().max()]
            )),
        showlegend=True
    )
    
    return fig

def display_achievements(student_data, df):
    """Display achievement badges based on performance"""
    achievements = []
    
    # Top 10% Achievement
    percentile = (1 - (student_data['Rank'].values[0] - 1) / len(df)) * 100
    if percentile >= 90:
        achievements.append("🏆 Top 10% Performer!")
    
    # Subject Toppers
    for subject in df.columns[df.columns.str.startswith('Subject')]:
        if student_data[subject].values[0] >= df[subject].max() * 0.95:
            achievements.append(f"🌟 Excellence in {subject}!")
    
    # Overall Score Achievement
    if student_data['Total'].values[0] >= df['Total'].max() * 0.9:
        achievements.append("🎯 Outstanding Overall Performance!")
    
    return achievements

def main():
    st.title("Test Result Analysis App")
    
    # Instructions on how to use the app
    st.markdown("""
    ### Instructions
    1. **Upload Your Data:** Use the file uploader to upload an Excel or CSV file containing test results.
    2. **Configure Columns:** Enter the names of the columns that represent identifiers (e.g., Roll No), ranks, total scores, and subjects in the sidebar.
    3. **View Data:** After uploading, preview the data. You can remove any unwanted columns or rows if necessary.
    4. **Analyze Performance:** Enter your identifier to see your results and achievements.
    5. **Compare Peers:** Select other identifiers to compare your performance with peers visually.
    6. **Explore Insights:** Check out the distribution of scores and subject toppers to see where you stand.
    """)
    
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
                # Prompt user for the subject column name
                subject_col = st.text_input(f"Subject {i+1} Column Name", f"Subject{i+1}")
        
            with col2:
                # Ask if the display name is the same
                if st.checkbox(f"Is the display name the same as the column name for Subject {i+1}?", value=True):
                # If yes, use the same input
                  subject_name = subject_col  # Use the same name
                else:
                # If no, ask for the display name
                    subject_name = st.text_input(f"Subject {i+1} Display Name", f"Subject {i+1}")
        
            subject_columns.append(subject_col)
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
            # Initial data preview
            data_preview_placeholder = st.empty()
            data_preview_placeholder.subheader("Data Preview")
            data_preview_placeholder.dataframe(df.head())

            # Column and Row Removal Feature
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
                data_preview_placeholder.dataframe(df.head())

            # Top Performers List
            st.subheader("Top Performers")
            top_10 = df.nsmallest(10, 'Rank')[['Rank', 'Identifier', 'Total'] + config['subject_names']]
            st.dataframe(top_10, use_container_width=True)

            # Individual Analysis
            st.subheader("Individual Analysis")
            identifier = st.text_input(f"Enter your {identifier_column}:")
            
            if identifier:
                student_data = df[df['Identifier'].astype(str) == identifier]
                
                if not student_data.empty:
                    # Achievements
                    achievements = display_achievements(student_data, df)
                    if achievements:
                        for achievement in achievements:
                            st.markdown(f"### {achievement}")

                    # Basic Results
                    st.subheader("Your Results")
                    st.dataframe(student_data)
                    
                    rank = student_data['Rank'].values[0]
                    total_students = len(df)
                    
                    st.write(f"Your Rank: {rank} out of {total_students}")

                    # Score Gauge
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

                    # Detailed Performance Metrics - Moved Below "Your Results"
                    percentile = (1 - (rank - 1) / total_students) * 100
                    st.subheader("Detailed Performance Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Percentile", f"{percentile:.2f}%")
                    with col2:
                        st.metric("Total Score", f"{student_total}")
                    with col3:
                        st.metric("Class Average", f"{class_average:.2f}")

                    # Peer Comparison
                    st.subheader("Compare with Peers")
                    peer_ids = st.multiselect(
                        "Select peers to compare with",
                        df['Identifier'].unique(),
                    )
                    
                    if peer_ids:
                        comparison_ids = [identifier] + peer_ids
                        comparison_data = df[df['Identifier'].astype(str).isin(comparison_ids)]
                        
                        # Radar Chart Comparison
                        st.subheader("Subject-wise Peer-to-Peer Comparison")
                        radar_fig = create_radar_chart(df, comparison_ids, config['subject_names'])
                        st.plotly_chart(radar_fig)

                    # Performance Distribution (Violin Plots)
                    st.subheader("Subject-wise Performance Distribution")
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

                else:
                    st.error(f"{identifier_column} not found in the data.")

            # Leaderboard View
            st.subheader("🏆 Subject-wise Toppers")
            for subject in config['subject_names']:
                st.subheader(f"Top 5 in {subject}")
                st.dataframe(
                    df.nlargest(5, subject)[['Rank', 'Identifier', subject]],
                    use_container_width=True
                )

if __name__ == "__main__":
    main()
