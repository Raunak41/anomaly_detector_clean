from django.shortcuts import render
from .forms import UploadFileForm
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# For reading Word & PDF
import docx  # python-docx
import openpyxl # For Excel File
from PyPDF2 import PdfReader  # pip install PyPDF2

def index(request):
    return render(request, 'detector/index.html')


def read_uploaded_file(uploaded_file):
    """Detect file type and return a pandas DataFrame."""
    ext = os.path.splitext(uploaded_file.name)[1].lower()

    if ext == ".csv":
        return pd.read_csv(uploaded_file)

    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(uploaded_file)

    elif ext in [".doc", ".docx"]:
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return pd.DataFrame({"Text": text.split("\n")})

    elif ext == ".pdf":
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return pd.DataFrame({"Text": text.split("\n")})

    else:
        raise ValueError("Unsupported file format. Please upload CSV, Excel, Word, or PDF.")


def create_anomaly_chart(df, anomalies_df, numeric_df):
    """Create an interactive Plotly chart for anomaly visualization."""
    
    if numeric_df.shape[1] == 0:
        return None
    
    # Create figure based on number of numeric columns
    if numeric_df.shape[1] == 1:
        # Single numeric column - create a scatter plot with index
        col = numeric_df.columns[0]
        
        fig = go.Figure()
        
        # Normal points
        normal_df = df[df['Anomaly'] == 1]
        fig.add_trace(go.Scatter(
            x=normal_df.index,
            y=normal_df[col],
            mode='markers',
            name='Normal',
            marker=dict(color='#28a745', size=8),
            hovertemplate=f'<b>Index:</b> %{{x}}<br><b>{col}:</b> %{{y}}<extra></extra>'
        ))
        
        # Anomaly points
        fig.add_trace(go.Scatter(
            x=anomalies_df.index,
            y=anomalies_df[col],
            mode='markers',
            name='Anomaly',
            marker=dict(color='#dc3545', size=12, symbol='x'),
            hovertemplate=f'<b>Index:</b> %{{x}}<br><b>{col}:</b> %{{y}}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Anomaly Detection - {col}',
            xaxis_title='Record Index',
            yaxis_title=col,
            template='plotly_white',
            hovermode='closest',
            height=500
        )
        
    elif numeric_df.shape[1] >= 2:
        # Multiple columns - create 2D scatter plot with first two columns
        col1, col2 = numeric_df.columns[0], numeric_df.columns[1]
        
        fig = go.Figure()
        
        # Normal points
        normal_df = df[df['Anomaly'] == 1]
        fig.add_trace(go.Scatter(
            x=normal_df[col1],
            y=normal_df[col2],
            mode='markers',
            name='Normal',
            marker=dict(color='#28a745', size=8, opacity=0.6),
            hovertemplate=f'<b>{col1}:</b> %{{x}}<br><b>{col2}:</b> %{{y}}<extra></extra>'
        ))
        
        # Anomaly points
        fig.add_trace(go.Scatter(
            x=anomalies_df[col1],
            y=anomalies_df[col2],
            mode='markers',
            name='Anomaly',
            marker=dict(color='#dc3545', size=14, symbol='x', line=dict(width=2, color='darkred')),
            hovertemplate=f'<b>{col1}:</b> %{{x}}<br><b>{col2}:</b> %{{y}}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Anomaly Detection - {col1} vs {col2}',
            xaxis_title=col1,
            yaxis_title=col2,
            template='plotly_white',
            hovermode='closest',
            height=500,
            showlegend=True
        )
    
    # Add distribution plots if there are multiple numeric columns
    if numeric_df.shape[1] >= 3:
        # Create subplots for additional columns
        num_cols = min(4, numeric_df.shape[1])  # Show up to 4 columns
        
        fig2 = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'{col} Distribution' for col in numeric_df.columns[:num_cols]]
        )
        
        for idx, col in enumerate(numeric_df.columns[:num_cols]):
            row = (idx // 2) + 1
            col_pos = (idx % 2) + 1
            
            # Add histogram for normal data
            fig2.add_trace(
                go.Histogram(
                    x=df[df['Anomaly'] == 1][col],
                    name=f'Normal - {col}',
                    marker_color='#28a745',
                    opacity=0.7,
                    showlegend=(idx == 0)
                ),
                row=row, col=col_pos
            )
            
            # Add markers for anomalies
            fig2.add_trace(
                go.Scatter(
                    x=anomalies_df[col],
                    y=[0] * len(anomalies_df),
                    mode='markers',
                    name=f'Anomaly - {col}',
                    marker=dict(color='#dc3545', size=10, symbol='x'),
                    showlegend=(idx == 0)
                ),
                row=row, col=col_pos
            )
        
        fig2.update_layout(
            title_text='Feature Distributions with Anomalies',
            template='plotly_white',
            height=600,
            showlegend=True
        )
        
        # Combine both charts
        combined_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        combined_html += "<br><br>" + fig2.to_html(full_html=False, include_plotlyjs=False)
        return combined_html
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def detect_anomalies(request):
    anomalies_html = None
    chart_html = None
    message = ""
    stats = {}
    
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Read file
                df = read_uploaded_file(request.FILES['file'])
                total_records = len(df)
                
                # Only use numeric columns
                numeric_df = df.select_dtypes(include=[np.number])

                if numeric_df.empty:
                    message = "⚠️ No numeric columns found in the report."
                else:
                    # Train isolation forest
                    model = IsolationForest(contamination=0.1, random_state=42)
                    preds = model.fit_predict(numeric_df)
                    df['Anomaly'] = preds

                    anomalies_df = df[df['Anomaly'] == -1].copy()
                    anomaly_count = len(anomalies_df)
                    
                    stats = {
                        'total_records': total_records,
                        'anomaly_count': anomaly_count,
                        'normal_count': len(df[df['Anomaly'] == 1]),
                        'anomaly_percentage': round((anomaly_count / total_records) * 100, 2) if total_records > 0 else 0
                    }
                    
                    if anomalies_df.empty:
                        message = "✅ No anomalies detected!"
                        anomalies_html = "<p>No anomalies found in the data.</p>"
                    else:
                        message = f"🔍 Found {anomaly_count} anomalies ({stats['anomaly_percentage']}%)"

                        # Simple table without complex formatting first
                        anomalies_html = anomalies_df.to_html(
                            classes="table table-striped",
                            index=True,
                            escape=False
                        )
                        
                        print("=" * 50)
                        print("ANOMALIES HTML LENGTH:", len(anomalies_html))
                        print("FIRST 200 CHARS:", anomalies_html[:200])
                        print("=" * 50)
                        
                        # Try to generate chart
                        try:
                            chart_html = create_anomaly_chart(df, anomalies_df, numeric_df)
                            print("CHART GENERATED SUCCESSFULLY")
                        except Exception as chart_error:
                            print(f"Chart error: {chart_error}")
                            chart_html = None

            except Exception as e:
                message = f"❌ Error: {str(e)}"
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
    else:
        form = UploadFileForm()
    
    print(f"Rendering with anomalies: {anomalies_html is not None}")
    print(f"Rendering with chart: {chart_html is not None}")
    
    return render(request, 'detector/detect.html', {
        'form': form,
        'anomalies': anomalies_html,
        'chart': chart_html,
        'message': message,
        'stats': stats
    })