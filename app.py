import streamlit as st
import os
import json
import logging
import time
from google.cloud import texttospeech
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips, VideoFileClip, ColorClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tempfile

logging.basicConfig(level=logging.INFO)

# Cargar credenciales de GCP desde secrets
credentials = dict(st.secrets.gcp_service_account)
with open("google_credentials.json", "w") as f:
    json.dump(credentials, f)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_credentials.json"

# Constantes
TEMP_DIR = "temp"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DEFAULT_FONT_SIZE = 50
VIDEO_FPS = 24
VIDEO_CODEC = 'libx264'
AUDIO_CODEC = 'aac'
VIDEO_PRESET = 'ultrafast'
VIDEO_THREADS = 4
IMAGE_SIZE_TEXT = (1080, 1920)
VIDEO_SIZE = (1080, 1920)

# Configuración de voces
VOCES_DISPONIBLES = {
    'es-ES-Standard-A': texttospeech.SsmlVoiceGender.FEMALE,
    'es-ES-Standard-B': texttospeech.SsmlVoiceGender.MALE,
    'es-ES-Standard-C': texttospeech.SsmlVoiceGender.FEMALE,
    'es-ES-Standard-D': texttospeech.SsmlVoiceGender.FEMALE,
    'es-ES-Standard-E': texttospeech.SsmlVoiceGender.FEMALE,
    'es-ES-Standard-F': texttospeech.SsmlVoiceGender.MALE,
    'es-ES-Neural2-A': texttospeech.SsmlVoiceGender.FEMALE,
    'es-ES-Neural2-B': texttospeech.SsmlVoiceGender.MALE,
    'es-ES-Neural2-C': texttospeech.SsmlVoiceGender.FEMALE,
    'es-ES-Neural2-D': texttospeech.SsmlVoiceGender.FEMALE,
    'es-ES-Neural2-E': texttospeech.SsmlVoiceGender.FEMALE,
    'es-ES-Neural2-F': texttospeech.SsmlVoiceGender.MALE,
    'es-ES-Polyglot-1': texttospeech.SsmlVoiceGender.MALE,
    'es-ES-Studio-C': texttospeech.SsmlVoiceGender.FEMALE,
    'es-ES-Studio-F': texttospeech.SsmlVoiceGender.MALE,
    'es-ES-Wavenet-B': texttospeech.SsmlVoiceGender.MALE,
    'es-ES-Wavenet-C': texttospeech.SsmlVoiceGender.FEMALE,
    'es-ES-Wavenet-D': texttospeech.SsmlVoiceGender.FEMALE,
    'es-ES-Wavenet-E': texttospeech.SsmlVoiceGender.MALE,
    'es-ES-Wavenet-F': texttospeech.SsmlVoiceGender.FEMALE,
}

def create_text_image(text, size=IMAGE_SIZE_TEXT, font_size=DEFAULT_FONT_SIZE,
                      text_color="white", background_video=None):
    """Creates a text image with the specified text and styles."""
    if background_video:
        img = Image.new('RGBA', size, (0, 0, 0, 0))  # Fondo completamente transparente
    else:
        img = Image.new('RGBA', size, (0, 0, 0, 0))  # Fondo completamente transparente
    
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception as e:
        logging.error(f"Error al cargar la fuente, usando la fuente predeterminada: {str(e)}")
        font = ImageFont.load_default()
    
    # Calculamos la altura de línea en función del tamaño de la fuente.
    line_height = font_size * 1.2

    words = text.split()
    lines = []
    current_line = []

    for word in words:
        current_line.append(word)
        test_line = ' '.join(current_line)
        left, top, right, bottom = draw.textbbox((0, 0), test_line, font=font)
        if right > size[0] - 120:
            current_line.pop()
            lines.append(' '.join(current_line))
            current_line = [word]
    lines.append(' '.join(current_line))

    total_height = len(lines) * line_height
    y = (size[1] - total_height) // 2

    for line in lines:
        left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
        x = (size[0] - (right - left)) // 2
        draw.text((x, y), line, font=font, fill=text_color)
        y += line_height
    return np.array(img)

def resize_and_center_video(video_clip, target_size):
    """Resizes and centers a video while maintaining its aspect ratio."""
    
    video_ratio = video_clip.size[0] / video_clip.size[1]
    target_ratio = target_size[0] / target_size[1]

    if video_ratio > target_ratio:
        # Video es más ancho, ajustar altura
        new_height = target_size[1]
        new_width = int(new_height * video_ratio)
        
    else:
        # Video es más alto, ajustar ancho
        new_width = target_size[0]
        new_height = int(new_width / video_ratio)

    resized_clip = video_clip.resize((new_width, new_height))

    # Centrar el video
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2
    
    # Creamos un clip negro de fondo del tamaño del video
    background_clip = ColorClip(size=target_size, color=(0,0,0)).set_duration(resized_clip.duration)

    # Pegamos el video redimensionado en el centro
    final_clip = CompositeVideoClip([background_clip,resized_clip.set_position((x_offset, y_offset))])
   
    return final_clip

    
def create_simple_video(texto, nombre_salida, voz, font_size, background_video):
    archivos_temp = []
    clips_audio = []
    clips_finales = []
    text_color = "white" # Color de texto por defecto
    bg_color = "black" # Color de fondo por defecto
    
    try:
        logging.info("Iniciando proceso de creación de video...")
        frases = [f.strip() + "." for f in texto.split('.') if f.strip()]
        client = texttospeech.TextToSpeechClient()
        
        tiempo_acumulado = 0
        
        # Agrupamos frases en segmentos
        segmentos_texto = []
        segmento_actual = ""
        for frase in frases:
          if len(segmento_actual) + len(frase) < 300:
            segmento_actual += " " + frase
          else:
            segmentos_texto.append(segmento_actual.strip())
            segmento_actual = frase
        segmentos_texto.append(segmento_actual.strip())
        
        # Cargar y procesar video de fondo (si existe) fuera del bucle
        if background_video:
            try:
              bg_clip_original = VideoFileClip(background_video)
              bg_clip_resized = resize_and_center_video(bg_clip_original, VIDEO_SIZE)
              bg_clip_resized = bg_clip_resized.set_opacity(0.5)
              
            except Exception as e:
              logging.error(f"Error al cargar o procesar el video de fondo: {e}")
              bg_clip_resized = None
        else:
             bg_clip_resized = None

        # Calcular duración total de los audios
        total_duration = 0
        for i, segmento in enumerate(segmentos_texto):
            logging.info(f"Procesando segmento {i+1} de {len(segmentos_texto)}")
            
            synthesis_input = texttospeech.SynthesisInput(text=segmento)
            voice = texttospeech.VoiceSelectionParams(
                language_code="es-ES",
                name=voz,
                ssml_gender=VOCES_DISPONIBLES[voz]
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            
            retry_count = 0
            max_retries = 3
            
            while retry_count <= max_retries:
              try:
                response = client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config
                )
                break
              except Exception as e:
                  logging.error(f"Error al solicitar audio (intento {retry_count + 1}): {str(e)}")
                  if "429" in str(e):
                    retry_count +=1
                    time.sleep(2**retry_count)
                  else:
                    raise
            
            if retry_count > max_retries:
                raise Exception("Maximos intentos de reintento alcanzado")
            
            temp_filename = f"temp_audio_{i}.mp3"
            archivos_temp.append(temp_filename)
            with open(temp_filename, "wb") as out:
                out.write(response.audio_content)
            
            audio_clip = AudioFileClip(temp_filename)
            clips_audio.append(audio_clip)
            total_duration += audio_clip.duration
        
        #  Creamos el clip de video de fondo en loop si existe y si no, un fondo negro
        if bg_clip_resized:
             bg_clip_looped = bg_clip_resized.loop(duration=total_duration)
        else:
             bg_clip_looped = ColorClip(size=VIDEO_SIZE, color=(0,0,0)).set_duration(total_duration)

        for i, segmento in enumerate(segmentos_texto):
            audio_clip = clips_audio[i]
            duracion = audio_clip.duration
          
            # Creamos una capa negra semitransparente
            black_clip = ColorClip(size=VIDEO_SIZE, color=(0, 0, 0)).set_opacity(0.5).set_duration(duracion)

            text_img = create_text_image(segmento, font_size=font_size,
                                    text_color=text_color,
                                    background_video=background_video
                                    )
            txt_clip = (ImageClip(text_img)
                        .set_duration(duracion)
                        .set_position('center'))
           
            video_segment = CompositeVideoClip([bg_clip_looped.subclip(tiempo_acumulado, tiempo_acumulado + duracion)
                                                , black_clip, txt_clip])
            video_segment = video_segment.set_audio(audio_clip)
            
            clips_finales.append(video_segment)
            
            tiempo_acumulado += duracion
            time.sleep(0.2)

        video_final = concatenate_videoclips(clips_finales, method="chain")
        
        video_final.write_videofile(
            nombre_salida,
            fps=VIDEO_FPS,
            codec=VIDEO_CODEC,
            audio_codec=AUDIO_CODEC,
            preset=VIDEO_PRESET,
            threads=VIDEO_THREADS
        )
        
        video_final.close()
        
        for clip in clips_audio:
            clip.close()
        
        for clip in clips_finales:
            clip.close()
            
        for temp_file in archivos_temp:
            try:
                if os.path.exists(temp_file):
                    os.close(os.open(temp_file, os.O_RDONLY))
                    os.remove(temp_file)
            except:
                pass
        if bg_clip_resized:
          bg_clip_original.close()
        
        return True, "Video generado exitosamente"
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        for clip in clips_audio:
            try:
                clip.close()
            except:
                pass
                
        for clip in clips_finales:
            try:
                clip.close()
            except:
                pass
                
        for temp_file in archivos_temp:
            try:
                if os.path.exists(temp_file):
                    os.close(os.open(temp_file, os.O_RDONLY))
                    os.remove(temp_file)
            except:
                pass
        
        return False, str(e)


def main():
    st.title("Creador de Videos Automático")
    
    uploaded_file = st.file_uploader("Carga un archivo de texto", type="txt")
    
    
    with st.sidebar:
        st.header("Configuración del Video")
        voz_seleccionada = st.selectbox("Selecciona la voz", options=list(VOCES_DISPONIBLES.keys()))
        font_size = st.slider("Tamaño de la fuente", min_value=10, max_value=200, value=DEFAULT_FONT_SIZE)
        background_type = st.radio("Tipo de fondo", ["Video"])

        background_video = None

        if background_type == "Video":
            background_video = st.file_uploader("Video de fondo (opcional)", type=["mp4", "mov", "avi"])

    
    if uploaded_file:
        texto = uploaded_file.read().decode("utf-8")
        nombre_salida = st.text_input("Nombre del Video (sin extensión)", "video_generado")
        
        if st.button("Generar Video"):
            with st.spinner('Generando video...'):
                nombre_salida_completo = f"{nombre_salida}.mp4"
                
                
                video_path = None

                if background_video and background_type == "Video":
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(background_video.name)[1]) as tmp_file:
                        tmp_file.write(background_video.read())
                        video_path = tmp_file.name


                success, message = create_simple_video(texto, nombre_salida_completo, voz_seleccionada,
                                                        font_size, video_path)
                if success:
                  st.success(message)
                  st.video(nombre_salida_completo)
                  with open(nombre_salida_completo, 'rb') as file:
                    st.download_button(label="Descargar video",data=file,file_name=nombre_salida_completo)
                    
                  st.session_state.video_path = nombre_salida_completo
                  if video_path:
                    os.remove(video_path)
                else:
                  st.error(f"Error al generar video: {message}")
                  if video_path:
                    os.remove(video_path)


if __name__ == "__main__":
    # Inicializar session state
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    main()
