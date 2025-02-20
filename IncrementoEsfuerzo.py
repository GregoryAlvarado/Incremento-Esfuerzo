import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configuración de la página
st.set_page_config(layout="wide", page_title="Cálculo de incremento de esfuerzo", page_icon=":chart_with_upwards_trend:")

def calcular_parametros_comunes(vertices, punto_analisis):
    x0, y0 = punto_analisis
    parametros = []

    for i in range(len(vertices)):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % len(vertices)]
        
        xi, yi = x1 - x0, y1 - y0
        xi1, yi1 = x2 - x0, y2 - y0
        
        F = xi * yi1 - xi1 * yi
        if F == 0:
            continue
        
        S = np.sign(F)
        L = np.sqrt((xi1 - xi) ** 2 + (yi1 - yi) ** 2)
        C1 = (xi * (xi1 - xi) + yi * (yi1 - yi)) / F
        C2 = (xi1 * (xi1 - xi) + yi1 * (yi1 - yi)) / F
        
        parametros.append((S, L, C1, C2, F))
    
    return parametros

def calcular_incremento_boussinesq(vertices, punto_analisis, q, z_max, dz):
    profundidades = np.arange(dz, z_max + dz, dz)
    esfuerzos = np.zeros_like(profundidades)
    parametros = calcular_parametros_comunes(vertices, punto_analisis)
    
    for S, L, C1, C2, F in parametros:
        A = profundidades * L / abs(F)
        G = A ** 2 + 1
        
        B1 = A * C1 / np.sqrt(G + C1 ** 2)
        B2 = A * C2 / np.sqrt(G + C2 ** 2)
        
        incremento = (q / (2 * np.pi)) * (np.arctan(C2) - np.arctan(C1) - np.arctan(B2) + np.arctan(B1) + (B2 - B1) / G)
        esfuerzos += S * incremento
    
    return profundidades, esfuerzos

def calcular_incremento_esfuerzo_westergaard(vertices, punto_analisis, q, z_max, dz, v):
    K = np.sqrt((1 - 2 * v) / (2 * (1 - v)))
    profundidades = np.arange(dz, z_max + dz, dz)
    esfuerzos = np.zeros_like(profundidades)
    parametros = calcular_parametros_comunes(vertices, punto_analisis)
    
    for S, L, C1, C2, F in parametros:
        A = profundidades * L / abs(F)
        G = A ** 2 + 1
        
        W1 = (K * A * C1) / np.sqrt(K ** 2 * A ** 2 + 1 + C1 ** 2)
        W2 = K * A * C2 / np.sqrt(K ** 2 * A ** 2 + 1 + C2 ** 2)
        
        incremento = (q / (2 * np.pi)) * (np.arctan(C2) - np.arctan(C1) - np.arctan(W2) + np.arctan(W1))
        esfuerzos += S * incremento
    
    return profundidades, esfuerzos

def calcular_incremento_frohlich(vertices, punto_analisis, q, z_max, dz, Xf):
    profundidades = np.arange(dz, z_max + dz, dz)
    esfuerzos = np.zeros_like(profundidades)
    parametros = calcular_parametros_comunes(vertices, punto_analisis)
    
    for S, L, C1, C2, F in parametros:
        A = profundidades * L / abs(F)
        G = A ** 2 + 1
        
        J1 = C1 / np.sqrt(G)
        J2 = C2 / np.sqrt(G)
        
        if Xf == 2:
            incremento = (q / (2 * np.pi*np.sqrt(G))) * (np.arctan(J2) - np.arctan(J1))
        elif Xf == 4:
            M = (2 * G + A ** 2) / np.sqrt(G)
            N1 = (A ** 2 * C1) / (G + C1 ** 2)
            N2 = (A ** 2 * C2) / (G + C2 ** 2)
            incremento = (q / (4 * np.pi * G)) * (M * (np.arctan(J2) - np.arctan(J1)) + N2 - N1)
        else:
            raise ValueError("Xf debe ser 2 o 4")
        
        esfuerzos += S * incremento
    
    return profundidades, esfuerzos

# Título de la aplicación
st.title("Incremento de Esfuerzo")
st.markdown("### Cálculos de incremento de esfuerzo para un polígono ")

# Dividimos la pantalla en tres columnas
col1, col2, col3 = st.columns([0.5, 1, 1])

with col1:
    st.header("INPUT")
    
    num_vertices = st.number_input("Número de vértices de la cimentación", min_value=3, value=4, step=1)
    
    vertices = []
    for i in range(num_vertices):
        col1_x, col2_x = st.columns(2)
        with col1_x:
            x = st.number_input(f"X{i+1}", value=0.0, key=f"x{i}")
        with col2_x:
            y = st.number_input(f"Y{i+1}", value=0.0, key=f"y{i}")
        vertices.append((x, y))
    
    col1_x, col2_x = st.columns(2)
    with col1_x:
        punto_x = st.number_input("X del punto de análisis", value=5.0)
    with col2_x:
        punto_y = st.number_input("Y del punto de análisis", value=5.0)
    punto_analisis = (punto_x, punto_y)
    
    q = st.number_input("Carga aplicada (t/m²)", value=100.0)
    z_max = st.number_input("Profundidad máxima (m)", value=200.0)
    dz = st.number_input("Incremento (m)", value=10.0, min_value=1.0)
    v = st.number_input("Coeficiente de Poisson (v)", value=0.3)
    Xf = st.selectbox("Frohlich (Xf)", options=[2, 4], index=0)
    
    calcular = st.button("Calcular", key="calcular")

with col2:
    st.markdown("#### Ecuación de Boussinesq")
    st.latex(r"\small \sigma_z = \sum_{o}^{N} \frac{Sq}{2\pi} \left\{ \alpha - S' \tan^{-1}(B_1) - \tan^{-1}(B_2) + \frac{S' B_1 + B_2}{A^2 + 1} \right\}")
    st.latex(r"\small B_i = \frac{\sqrt{q_i^2 - 1}}{\sqrt{r_i^2 + 1}}, \quad (i = 1,2)")

    st.markdown("#### Ecuación de Westergaard")
    st.latex(r"\sigma_z = \sum_{o}^{N} \frac{Sq}{2\pi} \left\{ \alpha - S' \tan^{-1}(W_1) - \tan^{-1}(W_2) \right\}")
    st.latex(r"W_i = k \frac{\sqrt{q_i^2 - 1}}{\sqrt{k^2 + r_i^2}}, \quad (i = 1,2)")
    st.latex(r"k = \text{constante}.")

    st.markdown("#### Ecuación de Fröhlich (χ = 2)")
    st.latex(r"\sigma_z = \sum_{o}^{N} \frac{Sq}{2\pi} \left( \frac{1}{A^2 + 1} \right) \left\{ S' \tan^{-1} \left( \frac{E_1}{\sqrt{A^2 + 1}} \right) + \tan^{-1} \left( \frac{E_2}{\sqrt{A^2 + 1}} \right) \right\}")
    st.latex(r"E_i = \sqrt{q_i^2 - 1}, \quad (i = 1,2)")

    st.markdown("#### Solución de Fröhlich (χ = 4)")
    st.latex(r"\sigma_z = \sum_{o}^{N} \frac{Sq}{2\pi} \left( \frac{1}{A^2 + 1} \right) \left\{ S' G_1 + G_2 \right\}")
    st.latex(r"G_i = \frac{3A^2 + 2}{\sqrt{A^2 + 1}} \tan^{-1} \left( \frac{\sqrt{q_i^2 - 1}}{\sqrt{A^2 + 1}} \right) + \frac{\sqrt{q_i^2 - 1}}{r_i^2 + 1}, \quad (i = 1,2)")
    
    if calcular:
        # Calcular los incrementos de esfuerzo para cada método
        profundidades_boussinesq, esfuerzos_boussinesq = calcular_incremento_boussinesq(vertices, punto_analisis, q, z_max, dz)
        profundidades_westergaard, esfuerzos_westergaard = calcular_incremento_esfuerzo_westergaard(vertices, punto_analisis, q, z_max, dz, v)
        profundidades_frohlich, esfuerzos_frohlich = calcular_incremento_frohlich(vertices, punto_analisis, q, z_max, dz, Xf)
        
        # Crear un DataFrame unificado
        df_resultados = pd.DataFrame({
            "Profundidad (m)": profundidades_boussinesq,
            "Boussinesq (t/m²)": esfuerzos_boussinesq,
            "Westergaard (t/m²)": esfuerzos_westergaard,
            "Frohlich (t/m²)": esfuerzos_frohlich
        })
        
        # Mostrar los resultados en una tabla unificada
        st.header("Resultados")
        st.dataframe(df_resultados, use_container_width=True)

        st.markdown("### Referencias")
        st.write("""
        [1] Damy, J. and Casales, G. (1985). "Soil stresses under a polygonal area uniformly loaded"  
        Proc. 11th Int. Conf. on Soil Mech. and Found. Engrg., ASCE, New York, N.Y., 2, 733-735.  

        [2] Joseph Boussinesq (1842-1929). "Application des Potentiels à l'Étude de l'Équilibre et du Mouvement des Solides Élastiques",  
        Gautier-Villars, Paris, 1885.  

        [3] H.M. Westergaard, "A Problem of Elasticity Suggested by a Problem in Soil Mechanics,  
        Soft Material Reinforced by Numerous Strong Horizontal Sheets", McMillan, 1939.  

        [4] Leonardo Zeevaert W., "Interacción Suelo-Estructura", Limusa, 1980.  

        [5] O.K. Frohlich. "Druckverteilung in Baugrunde" ("La repartición de presiones en suelos")  
        Springer Verlag, Berlín, 1934.  

        [6] N.M. Newmark, "Simplified Computation of Vertical Pressure in Elastic Foundations",  
        Circular 24, Eng. Exp. Station, Universidad de Illinois, 1935.  
        """)

with col3:
    if calcular:
        st.markdown("---")
        
        fig, ax = plt.subplots(figsize=(8,6))
        polygon = np.array(vertices + [vertices[0]])
        ax.plot(polygon[:, 0], polygon[:, 1], 'bo-', label="Polígono",linewidth=4, markersize=6, linestyle="--")
        ax.plot(punto_analisis[0], punto_analisis[1], 'ro', label="Punto de análisis", markersize=14)
        #ax.set_xlim(left=-0.1)
        #ax.set_ylim(bottom=-0.1)
        
        ax.set_xlabel("X (m)", fontsize=18)
        ax.set_ylabel("Y (m)", fontsize=18)
        ax.set_title("Geometría de la cimentación", fontsize=20, pad=20, fontweight='bold')
        ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0.1), fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(esfuerzos_boussinesq, profundidades_boussinesq, label="Boussinesq", linewidth=4, linestyle="--")
        ax.plot(esfuerzos_westergaard, profundidades_westergaard, label="Westergaard", linewidth=4,  linestyle="-")
        ax.plot(esfuerzos_frohlich, profundidades_frohlich, label=f"Frohlich Xf={Xf}", linewidth=4,  linestyle="-.")
        
        ax.set_xlabel("Incremento de esfuerzo (t/m²)", fontsize=18)
        ax.set_ylabel("Profundidad (m)", fontsize=18)
        ax.set_title("Distribución de esfuerzos", fontsize=20, pad=20, fontweight='bold')
        ax.legend(loc='lower right', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)

        max_y = max(profundidades_boussinesq)
        ax.set_ylim(0, max_y)
        ax.invert_yaxis()
        
        ax.set_xlim(left=-0.02)

        st.pyplot(fig)
