import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO,BytesIO
import base64

def main():
    
    file = st.file_uploader("upload your file",type=['CSV','EXCEL'])
    df = pd.read_csv(file)
    if df is not None:
        st.write(df.head(5))
        select_x_axis = st.selectbox('select the X-axis',df.columns)
        select_y_axis = st.selectbox('select the Y-axis',df.columns)
        fig = px.scatter(df,x=select_x_axis, y=select_y_axis)
        st.plotly_chart(fig)
        
        
     ##Download graph html file       
    mybuff = StringIO()
    fig.write_html(mybuff, include_plotlyjs='cdn')
    mybuff = BytesIO(mybuff.getvalue().encode())
    b64 = base64.b64encode(mybuff.read()).decode()
    href = f'<a href="data:text/html;charset=utf-8;base64, {b64}" download="plot.html">Download plot</a>'
    st.markdown(href, unsafe_allow_html=True)
    
        
    
    
if __name__ == '__main__':
    main()
