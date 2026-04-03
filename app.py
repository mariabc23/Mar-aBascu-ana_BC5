# ============================================================
# CABECERA
# ============================================================
# Alumno: María Bascuñana Castellanos
# URL Streamlit Cloud: 
# URL GitHub: 
 
# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json
import re
 
# ============================================================
# CONSTANTES
# ============================================================
MODEL = "gpt-4.1-mini"
 
# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
SYSTEM_PROMPT = """
Eres un asistente de análisis de datos de Spotify. Recibes preguntas en cualquier idioma y respondes analizando el historial de escucha del usuario.
 
## CONTEXTO DEL DATASET
Período: {fecha_min} → {fecha_max}
Plataformas: {plataformas}
reason_start posibles: {reason_start_values}
reason_end posibles: {reason_end_values}
 
## EL DATAFRAME `df` 
DataFrame de pandas ya cargado. Cada fila = una reproducción >= 30 segundos.
 
Campo                                    | Tipo    | Descripción
-----------------------------------------|---------|--------------------------------------------------------------------
ts                                       | string  | Timestamp de fin de reproducción (ISO 8601, UTC)
ms_played                                | int     | Milisegundos de reproducción efectiva
master_metadata_track_name               | string  | Nombre de la canción
master_metadata_album_artist_name        | string  | Artista principal
master_metadata_album_album_name         | string  | Álbum
spotify_track_uri                        | string  | Identificador único de la canción
reason_start                             | string  | Motivo de inicio (clickrow, trackdone, fwdbtn, etc.)
reason_end                               | string  | Motivo de fin (trackdone, fwdbtn, endplay, etc.)
shuffle                                  | bool    | Si el modo aleatorio estaba activado
skipped                                  | bool    | Si se saltó la canción (null = no se saltó)
platform                                 | string  | Plataforma (Android, iOS, Windows, web_player)
 
Columnas derivadas añadidas en el preprocesado:
fecha              | date    | Fecha calendario
año                | int     | Año
mes                | str     | Período "AAAA-MM" ej. "2024-03"
mes_nombre         | str     | Etiqueta legible ej. "Mar 2024"
semana             | str     | Período "AAAA-Www" ej. "2024-W12"
hora               | int     | Hora del día 0-23
dia_semana         | str     | Nombre del día en inglés ("Monday"..."Sunday")
es_finde           | bool    | True = sábado o domingo
estacion           | str     | "Invierno", "Primavera", "Verano", "Otoño"
minutos_escuchados | float   | ms_played / 60000
 
## NOTAS IMPORTANTES SOBRE COLUMNAS AMBIGUAS:
 
1. `shuffle` es booleana. True = modo aleatorio activado (el usuario no elige el orden). False = reproducción en orden. Usa df['shuffle'] == True / False para filtrar.
 
2. `skipped` es booleana. True = el usuario saltó la canción antes de que acabara. False = se escuchó completa. Nunca es nula (los nulos originales ya se convirtieron a False en el preprocesado).
 
3. `reason_start` indica cómo empezó la reproducción. Valores habituales: 'clickrow' (el usuario hizo clic), 'trackdone' (terminó la anterior), 'fwdbtn' (botón siguiente), 'backbtn' (botón anterior), 'playbtn' (botón play), 'appload' (al abrir la app), 'remote' (control remoto). Para saber si el usuario eligió activamente una canción, filtra por 'clickrow'.
 
4. `reason_end` indica por qué terminó la reproducción. Valores habituales: 'trackdone' (llegó al final), 'fwdbtn' (el usuario pulsó siguiente), 'backbtn' (pulsó anterior), 'endplay' (paró la reproducción), 'logout' (cerró sesión), 'remote'. Para detectar canciones saltadas activamente usa reason_end == 'fwdbtn'.
 
5. `ms_played` y `minutos_escuchados` miden lo mismo (ms_played / 60000 = minutos_escuchados). Usa siempre `minutos_escuchados` en los gráficos porque es más legible. Usa `ms_played` solo si necesitas precisión de milisegundos.
 
6. `spotify_track_uri` es el identificador único de cada canción. Úsalo para contar canciones distintas (nunca uses `master_metadata_track_name` para esto, ya que puede haber dos canciones diferentes con el mismo nombre). Ejemplo: df['spotify_track_uri'].nunique() para contar canciones únicas.
 
7. `dia_semana` contiene el nombre del día en inglés ('Monday', 'Tuesday', ..., 'Sunday'). Si necesitas ordenar los días correctamente usa: order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']. Para mostrarlos en español en el gráfico, aplica un map: {{'Monday':'Lunes','Tuesday':'Martes','Wednesday':'Miércoles','Thursday':'Jueves','Friday':'Viernes','Saturday':'Sábado','Sunday':'Domingo'}}.
 
8. `mes` ('2024-03') es útil para agrupar y ordenar cronológicamente. `mes_nombre` ('Mar 2024') es más legible para mostrar en ejes de gráficos. Para gráficos temporales, agrupa por `mes` (que ordena bien) y usa `mes_nombre` como etiqueta.
 
## FORMATO DE RESPUESTA — OBLIGATORIO
Tu respuesta debe ser EXCLUSIVAMENTE un objeto JSON. Ningún carácter fuera del JSON.
No escribas nada antes ni después. No uses backticks. No uses ```json.
Empieza directamente con {{ y termina con }}.
 
Para preguntas respondibles con los datos:
{{"tipo": "grafico", "codigo": "CODIGO_PYTHON_AQUI", "interpretacion": "TEXTO_EXPLICATIVO"}}
 
Para preguntas fuera de alcance (letras, recomendaciones, info externa):
{{"tipo": "fuera_de_alcance", "codigo": "", "interpretacion": "Explicación amable de por qué no puedo responder esto."}}
 
## INSTRUCCIONES OBLIGATORIAS
- Usa solo: df, pd, px, go (ya disponibles, no los importes)
- Crea una variable llamada exactamente `fig` con una figura Plotly
- Separa sentencias con punto y coma (;) para que quepan en una sola línea lógica
- Color principal: "#1DB954" (verde Spotify)
- Aplica siempre al final: fig.update_layout(template="plotly_dark", title_font_size=16)
- Títulos y etiquetas en español
 
## EJEMPLOS DE CADA CATEGORÍA
 
Pregunta: ¿Cuál es mi artista más escuchado?
{{"tipo": "grafico", "codigo": "top = df.groupby('master_metadata_album_artist_name')['minutos_escuchados'].sum().nlargest(10).reset_index().sort_values('minutos_escuchados'); fig = px.bar(top, x='minutos_escuchados', y='master_metadata_album_artist_name', orientation='h', title='Top 10 artistas por tiempo de escucha', color_discrete_sequence=['#1DB954']); fig.update_layout(template='plotly_dark', title_font_size=16, xaxis_title='Minutos', yaxis_title='')", "interpretacion": "Tu artista más escuchado es el que aparece en lo alto de la barra."}}
 
Pregunta: ¿Escucho más en shuffle o en orden?
{{"tipo": "grafico", "codigo": "counts = df['shuffle'].value_counts().reset_index(); counts.columns = ['modo', 'reproducciones']; counts['modo'] = counts['modo'].map({{True: 'Shuffle', False: 'En orden'}}); fig = px.pie(counts, names='modo', values='reproducciones', title='Shuffle vs. En orden', color_discrete_sequence=['#1DB954', '#535353']); fig.update_layout(template='plotly_dark', title_font_size=16)", "interpretacion": "El gráfico muestra la proporción de reproducciones con shuffle activado frente a en orden."}}
 
Pregunta: ¿A qué hora escucho más música?
{{"tipo": "grafico", "codigo": "por_hora = df.groupby('hora').size().reset_index(name='reproducciones'); fig = px.bar(por_hora, x='hora', y='reproducciones', title='Reproducciones por hora del día', color_discrete_sequence=['#1DB954']); fig.update_layout(template='plotly_dark', title_font_size=16, xaxis_title='Hora', yaxis_title='Reproducciones')", "interpretacion": "Cada barra muestra cuántas canciones empezaste en esa hora del día."}}
 
Recuerda: responde SOLO con el JSON. Ni una palabra fuera de las llaves.
"""
 
 
# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")
 
    # ----------------------------------------------------------
    # >>> TU PREPARACIÓN DE DATOS ESTÁ AQUÍ <<<
    # ----------------------------------------------------------
 
    # 1. Timestamp a datetime localizado a Madrid
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["ts"] = df["ts"].dt.tz_convert("Europe/Madrid")
 
    # 2. Columnas temporales derivadas
    df["fecha"]        = df["ts"].dt.date
    df["año"]          = df["ts"].dt.year
    df["mes"]          = df["ts"].dt.to_period("M").astype(str)
    df["mes_nombre"]   = df["ts"].dt.strftime("%b %Y")
    df["semana"]       = df["ts"].dt.to_period("W").astype(str)
    df["hora"]         = df["ts"].dt.hour
    df["dia_semana"]   = df["ts"].dt.day_name()
    df["es_finde"]     = df["ts"].dt.dayofweek >= 5
 
    # 3. Estación del año
    def estacion(mes):
        if mes in (12, 1, 2):  return "Invierno"
        if mes in (3, 4, 5):   return "Primavera"
        if mes in (6, 7, 8):   return "Verano"
        return "Otoño"
    df["estacion"] = df["ts"].dt.month.map(estacion)
 
    # 4. Minutos reproducidos
    df["minutos_escuchados"] = df["ms_played"] / 60_000
 
    # 5. Alias cortos
    df["artista"] = df["master_metadata_album_artist_name"]
    df["cancion"]  = df["master_metadata_track_name"]
    df["album"]   = df["master_metadata_album_album_name"]
 
    # 6. Filtrar reproducciones < 30 s
    df = df[df["ms_played"] >= 30_000].copy()
 
    # 7. Normalizar skipped
    df["saltada"] = df["skipped"].fillna(False).astype(bool)
 
    return df
 
 
def build_prompt(df):
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()
 
    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )
 
 
# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# No modifiques esta función.
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
 
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content
 
 
# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# Versión robusta: extrae el JSON aunque el LLM añada texto extra.
# No modifiques esta función.
def parse_response(raw):
    cleaned = raw.strip()
 
    # Caso 1: el LLM envolvió en backticks (```json ... ```)
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
 
    # Caso 2: el LLM añadió texto antes o después del JSON
    # Buscamos el primer { y el último } para extraer solo el objeto JSON
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
 
    return json.loads(cleaned)
 
 
# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# No modifiques esta función.
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")
 
 
# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
st.set_page_config(page_title="Spotify Analytics", layout="wide")
 
# --- Control de acceso ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
 
if not st.session_state.authenticated:
    st.title("Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()
 
# --- App principal ---
st.title("Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")
 
df = load_data()
system_prompt = build_prompt(df)
 
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):
 
    with st.chat_message("user"):
        st.write(prompt)
 
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                raw = get_response(prompt, system_prompt)
                parsed = parse_response(raw)
 
                if parsed["tipo"] == "fuera_de_alcance":
                    st.write(parsed["interpretacion"])
                else:
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")
 
            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")
 
 
# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    El LLM va a recibir únicamente el system prompt que se le ha desarrollado
#    (descripción de columnas,reglas de formato y ejemplos) añadido a eso, la pregunta que escriba el usuario.
#    Nunca va a ver el Dataframe real. Devuelve un JSON con código Plotly como string y una interpretación 
#    donde se muestra un texto explicativo (la respuesta en formato gráfico).
#    Ese código se ejecuta localmente en execute_chart() con exec(), que inyecta df,
#    pd, px y go en el namespace. No se envían datos al LLM por privacidad,
#    coste de tokens y porque el modelo genera código más fiable que si
#    intentase razonar sobre miles de filas crudas en el contexto.
#
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    El prompt nos va a servir para documentar las columnas del dataframe (nombres, tipos, semántica), 
#    el formato JSON de salida con llaves nos va a ayudar para escapar los literales y 3 ejemplos de
#    respuesta válida. 
#    Ejemplo que funciona: "¿A qué hora escucho más?" → el LLM usa
#    directamente `hora` porque está documentada; sin esa columna en el
#    prompt intentaría extraerla de `ts` y fallaría con un error tz-aware.
#    Ejemplo que fallaría sin los ejemplos JSON: el modelo devolvería
#    texto libre en lugar de JSON puro, rompiendo parse_response().
#
#
# 3. EL FLUJO COMPLETO
#    1. El usuario escribe la pregunta en st.chat_input().
#    2. get_response() la envía a la API de OpenAI junto al system prompt
#       previamente relleno con fechas y valores reales por build_prompt().
#    3. El modelo devuelve un string JSON con "tipo", "codigo" e
#       "interpretacion".
#    4. parse_response() extrae el JSON aunque haya texto extra alrededor.
#    5. Si tipo="fuera_de_alcance", se muestra solo el texto explicativo.
#    6. Si tipo="grafico", execute_chart() ejecuta el código con exec(),
#       inyectando df, pd, px y go en el namespace local.
#    7. La variable `fig` resultante se renderiza con st.plotly_chart()
#       y el texto de "interpretacion" se muestra debajo del gráfico.