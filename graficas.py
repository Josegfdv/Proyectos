import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


st.title("Análisis de Influencers de Moda")

st.title("📊 Estadísticas y Gráficas")



@st.cache_data
def load_data():
    df = pd.read_csv("data_Set.csv")
    influencer_stats = df.groupby('Fashion Influencers')['Price'].mean().reset_index(name='Precio_Promedio_Total')

    moda_descripcion = df.groupby('Fashion Influencers')['Description'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).reset_index(name='Moda_descripcion')
    influencer_stats = influencer_stats.merge(moda_descripcion, on='Fashion Influencers', how='left')

    rating = df.groupby('Fashion Influencers')['Rating'].mean().reset_index(name='Rating_avg')
    influencer_stats = influencer_stats.merge(rating, on='Fashion Influencers', how='left')

    style = df.groupby('Fashion Influencers')['Style Attributes'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).reset_index(name='Moda_estilo')
    influencer_stats = influencer_stats.merge(style, on='Fashion Influencers', how='left')

    review = df.groupby('Fashion Influencers')['Review Count'].mean().reset_index(name='Review_count_avg')
    influencer_stats = influencer_stats.merge(review, on='Fashion Influencers', how='left')

    color = df.groupby('Fashion Influencers')['Color'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).reset_index(name='Moda_color')
    influencer_stats = influencer_stats.merge(color, on='Fashion Influencers', how='left')

    season = df.groupby('Fashion Influencers')['Season'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).reset_index(name='Moda_Season')
    influencer_stats = influencer_stats.merge(season, on='Fashion Influencers', how='left')

    return df, influencer_stats

df, influencer_stats = load_data()

# Mostrar tabla general
st.subheader("Tabla de estadísticas por Fashion Influencer")
st.dataframe(influencer_stats)

# Seleccionar influencer
influencer_sel = st.selectbox("Selecciona un Fashion Influencer", influencer_stats['Fashion Influencers'])

# Mostrar datos del influencer seleccionado
sel_data = influencer_stats[influencer_stats['Fashion Influencers'] == influencer_sel].squeeze()
st.markdown(f"""
*Precio Promedio Total:* ${sel_data['Precio_Promedio_Total']:.2f}  
*Rating Promedio:* {sel_data['Rating_avg']:.2f}  
*Review Count Promedio:* {sel_data['Review_count_avg']:.0f}  
*Descripción Más Frecuente:* {sel_data['Moda_descripcion']}  
*Estilo Más Frecuente:* {sel_data['Moda_estilo']}  
*Color Más Frecuente:* {sel_data['Moda_color']}  
*Temporada Más Frecuente:* {sel_data['Moda_Season']}
""")

# Gráfico de proporciones de color
color_summary = df.groupby(['Fashion Influencers', 'Color']).size().reset_index(name='Cantidad')
totales = color_summary.groupby('Fashion Influencers')['Cantidad'].transform('sum')
color_summary['Proporcion'] = color_summary['Cantidad'] / totales
data_color = color_summary[color_summary['Fashion Influencers'] == influencer_sel]

st.subheader("Proporción de Colores")
fig1, ax1 = plt.subplots()
ax1.pie(data_color['Proporcion'], labels=data_color['Color'], autopct='%1.1f%%', startangle=90)
ax1.axis('equal')
st.pyplot(fig1)

# Gráfico de proporciones de estilo
style_summary = df.groupby(['Fashion Influencers', 'Style Attributes']).size().reset_index(name='Cantidad')
totales_style = style_summary.groupby('Fashion Influencers')['Cantidad'].transform('sum')
style_summary['Proporcion'] = style_summary['Cantidad'] / totales_style
data_style = style_summary[style_summary['Fashion Influencers'] == influencer_sel]

st.subheader("Proporción de Estilos")
fig2, ax2 = plt.subplots()
ax2.pie(data_style['Proporcion'], labels=data_style['Style Attributes'], autopct='%1.1f%%', startangle=90)
ax2.axis('equal')
st.pyplot(fig2)