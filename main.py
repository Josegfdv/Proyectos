# Importación de todas las librerías 
#Ojo: se tienen que descargar todas las lbrerías para poder usarlas.
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from deep_translator import GoogleTranslator
from st_aggrid import AgGrid, GridOptionsBuilder
import re

# Configuración inicial para la web.
st.set_page_config(layout="wide", page_icon="👗", page_title="Fashion AI Recommender")

#Empezamos con la sidebar para redireccionar a las gráficas.
with st.sidebar:
    st.markdown("### 📈 Ver estadísticas")
    if st.button("Ir a estadísticas"):
        st.switch_page("pages/graficas.py")

# Constantes para el modelo
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
SENTIMENT_MAP = {
    'Positive': 2, 'Negative': -1, 'Neutral': 0.5,
    'Mixed': 1, 'Other': 0, 'Unknown': 0
}


@st.cache_data
def load_data():
    try:
        # Cargar datos con detección automática de formato
        df = pd.read_csv(
            "data_Set.csv",
            delimiter=None,  # Auto-detect
            engine='python',
            encoding_errors='replace'
        )
        
        # Mapeo de nombres de columnas alternativos
        column_mapping = {
            'precio': 'Price',
            'Precio': 'Price',
            'price': 'Price',
            'Purchase Price': 'Price',
            'PRECIO': 'Price',
            'prezzzo': 'Price'  # Ejemplo de posible typo
        }
        df = df.rename(columns=column_mapping)
        
        # Verificar columna Price
        if 'Price' not in df.columns:
            available_cols = ", ".join(df.columns)
            st.error(f"🚨 Columna 'Price' no encontrada. Columnas disponibles: {available_cols}")
            st.stop()

        # Limpieza avanzada de precios
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['Price'] = df['Price'].fillna(df.groupby('Category')['Price'].transform('median'))
        
        # Mapeo de Purchase History
        purchase_mapping = {
            'Negligible': 1, 'Low': 2, 'Medium': 3, 
            'High': 4, 'Very High': 5
        }
        df['Purchase_Score'] = df['Purchase History'].map(purchase_mapping).fillna(3)
        
        # Procesamiento de sentimientos
        df['Review_Sentiment'] = df['Customer Reviews'].map(SENTIMENT_MAP).fillna(0)
        df['Social_Sentiment'] = df['Social Media Comments'].map(SENTIMENT_MAP).fillna(0)
        
        # Generar embeddings
        model = SentenceTransformer(EMBEDDING_MODEL)
        df['Combined_Features'] = (
            df['Style Attributes'].fillna('') + " " +
            df['Category'].fillna('') + " " +
            df['Description'].fillna('') + " " +
            df['Brand'].fillna('') + " " +
            df['Season'].fillna('') + " " +
            df['Color'].fillna('')
        )
        df['Embeddings'] = df['Combined_Features'].apply(lambda x: model.encode(x))
        
        return df, model
    
    except Exception as e:
        st.error(f"💥 Error crítico al cargar datos: {str(e)}")
        st.stop()

def semantic_similarity(text, model, df):
    try:
        emb = model.encode(text)
        df['Similarity'] = df['Embeddings'].apply(lambda x: cosine_similarity([emb], [x])[0][0])
        return df
    except Exception as e:
        st.error(f"Error en similitud semántica: {str(e)}")
        st.stop()

def calculate_scores(df):
    try:
        scaler = MinMaxScaler()
        
        # Cálculo de métricas
        df['Popularity'] = np.log1p(df['Review Count']) * 0.8 + df['Purchase_Score'] * 0.2
        df['Sentiment_Score'] = (df['Review_Sentiment'] * 0.6 + df['Social_Sentiment'] * 0.4)
        
        # Normalización
        features = scaler.fit_transform(df[['Similarity', 'Rating', 'Popularity', 'Sentiment_Score']])
        df['Composite_Score'] = np.dot(features, [0.5, 0.2, 0.2, 0.1]) * 5
        
        # Ajuste final
        df['Composite_Score'] = df['Composite_Score'].clip(upper=5.0).round(2)
        return df.sort_values('Composite_Score', ascending=False)
    
    except Exception as e:
        st.error(f"Error en cálculo de scores: {str(e)}")
        st.stop()

def main_interface():
    st.title("🌟 Sistema Inteligente de Recomendación de Influencers 🚀")
    
    # Cargar datos
    df, model = load_data()
    
  
        
    

    # Filtros principales
    with st.sidebar:
        st.header("⚙️ Filtros Principales")
        price_range = st.slider(
            "💰 Rango de Precios",
            float(df['Price'].min()),
            float(df['Price'].max()),
            (float(df['Price'].quantile(0.25)), float(df['Price'].quantile(0.75)))
        )
        
        min_rating = st.slider("⭐ Rating Mínimo", 1.0, 5.0, 3.5, 0.1)
        selected_brands = st.multiselect("🏷️ Marcas", options=df['Brand'].unique())

      #  datos en sidebar
    with st.sidebar.expander("Autores de este trabajo", expanded=False):
        st.write("Becerra Lozano Demien")
        st.write("Webb Zulaica Maya Ángela")
        st.write("Izquierdo Preciado Mariana")
        st.write("García Vera Mariana")
        st.write("Fernández Del Valle José García")

    # Entrada de usuario
    with st.form("user_input"):
        user_style = st.text_area(
            "💬 Describe tu estilo personal:",
            placeholder="Ej: Me gusta el estilo urbano con colores neutros y materiales sostenibles...",
            height=100
        )
        submitted = st.form_submit_button("🚀 Obtener Recomendaciones")

    # Procesamiento
    if submitted and user_style:
        with st.spinner('🔍 Analizando tu estilo...'):
            try:
                # Traducción
                translated = GoogleTranslator(source='auto', target='en').translate(user_style)
                
                # Similitud semántica
                df_sim = semantic_similarity(translated, model, df)
                
                # Aplicar filtros
                filtered_df = df_sim[
                    (df_sim['Price'].between(*price_range)) &
                    (df_sim['Rating'] >= min_rating) &
                    (df_sim['Brand'].isin(selected_brands if selected_brands else df['Brand'].unique()))
                ].copy()
                
                # Calcular scores
                results = calculate_scores(filtered_df).head(6)

            except Exception as e:
                st.error(f"Error en procesamiento: {str(e)}")
                st.stop()

        # Mostrar resultados
        if not results.empty:
            st.success(f"🎉 Top {len(results)} Recomendaciones Personalizadas")
            
            # Tarjetas de influencers
            cols = st.columns(3)
            for idx, row in results.iterrows():
                with cols[idx % 3]:
                    with st.container():
                        st.markdown(f"""
                        <div style="padding:15px;border-radius:10px;margin:10px;box-shadow:0 2px 4px rgba(0,0,0,0.1)">
                            <h4 style="color:#2a9fd6;margin:0 0 10px 0">🌟 {row['Fashion Influencers']}</h4>
                            <p style="margin:5px 0">🏷️ <b>{row['Brand']}</b> | ⭐ {row['Rating']}/5</p>
                            <p style="margin:5px 0">💰 ${row['Price']:.2f} | 📈 Score: {row['Composite_Score']:.2f}/5</p>
                            <p style="margin:5px 0">🎨 {row['Style Attributes']} | 🌈 {row['Color']}</p>
                        </div>
                        """, unsafe_allow_html=True)

            # Análisis avanzado
            with st.expander("📊 Análisis Detallado", expanded=True):
                tab1, tab2 = st.tabs(["Distribución de Precios", "Relación Rating-Score"])
                
                with tab1:
                    fig = px.histogram(
                        results,
                        x='Price',
                        nbins=20,
                        title="Distribución de Precios en Recomendaciones"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    fig = px.scatter(
                        results,
                        x='Rating',
                        y='Composite_Score',
                        color='Brand',
                        size='Review Count',
                        hover_name='Fashion Influencers',
                        title="Relación entre Rating y Score"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("⚠️ No se encontraron coincidencias. Prueba con filtros más amplios.")

if __name__ == "__main__":
    main_interface()