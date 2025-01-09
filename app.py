import streamlit as st
import os
import json
import logging
import time
from google.cloud import texttospeech
from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
    CompositeVideoClip,
    ColorClip,
)
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tempfile
import requests
from io import BytesIO

# Configuración para nueva versión de Pillow
import PIL.Image

def ANTIALIAS():
    return PIL.Image.Resampling.LANCZOS

PIL.Image.ANTIALIAS = ANTIALIAS()

logging.basicConfig(level=logging.INFO)

# Cargar credenciales de GCP desde secrets
credentials = dict(st.secrets.gcp_service_account)
with open("google_credentials.json", "w") as f:
    json.dump(credentials, f)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_credentials.json"

# Constantes
TEMP_DIR = "temp"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DEFAULT_FONT_SIZE = 40
VIDEO_FPS = 30
VIDEO_CODEC = "libx264"
AUDIO_CODEC = "aac"
VIDEO_PRESET = "fast"  # Use "fast" or "faster" for faster encoding
VIDEO_THREADS = 4
IMAGE_SIZE_TEXT = (1080, 1350)  # Tamaño para los textos (vertical)
IMAGE_SIZE_SUBSCRIPTION = (1080, 1920)  # Tamaño para la suscripción (vertical)
SUBSCRIPTION_DURATION = 5
LOGO_SIZE = (150, 150)
VIDEO_SIZE = (1080, 1920)  # Tamaño vertical para shorts
TEXT_MARGIN = 50
# Configuración de voces
VOCES_DISPONIBLES = {
    "es-ES-Standard-A": texttospeech.SsmlVoiceGender.FEMALE,
    "es-ES-Standard-B": texttospeech.SsmlVoiceGender.MALE,
    "es-ES-Standard-C": texttospeech.SsmlVoiceGender.FEMALE,
    "es-ES-Standard-D": texttospeech.SsmlVoiceGender.FEMALE,
    "es-ES-Standard-E": texttospeech.SsmlVoiceGender.FEMALE,
    "es-ES-Standard-F": texttospeech.SsmlVoiceGender.MALE,
    "es-ES-Neural2-A": texttospeech.SsmlVoiceGender.FEMALE,
    "es-ES-Neural2-B": texttospeech.SsmlVoiceGender.MALE,
    "es-ES-Neural2-C": texttospeech.SsmlVoiceGender.FEMALE,
    "es-ES-Neural2-D": texttospeech.SsmlVoiceGender.FEMALE,
    "es-ES-Neural2-E": texttospeech.SsmlVoiceGender.FEMALE,
    "es-ES-Neural2-F": texttospeech.SsmlVoiceGender.MALE,
    "es-ES-Polyglot-1": texttospeech.SsmlVoiceGender.MALE,
    "es-ES-Studio-C": texttospeech.SsmlVoiceGender.FEMALE,
    "es-ES-Studio-F": texttospeech.SsmlVoiceGender.MALE,
    "es-ES-Wavenet-B": texttospeech.SsmlVoiceGender.MALE,
    "es-ES-Wavenet-C": texttospeech.SsmlVoiceGender.FEMALE,
    "es-ES-Wavenet-D": texttospeech.SsmlVoiceGender.FEMALE,
    "es-ES-Wavenet-E": texttospeech.SsmlVoiceGender.MALE,
    "es-ES-Wavenet-F": texttospeech.SsmlVoiceGender.FEMALE,
}

def create_text_image(
    text,
    size=IMAGE_SIZE_TEXT,
    font_size=DEFAULT_FONT_SIZE,
    bg_color="black",
    text_color="white",
    background_video=None,
    background_image=None,
    full_size_background=False,
    stretch_background=False,
):
    if full_size_background:
        size = VIDEO_SIZE
        
    if background_video:
        return None
    
    if background_image:
        try:
            img = Image.open(background_image).convert("RGB")
            if stretch_background:
                img = img.resize(size)
            else:
                img.thumbnail(size)
                new_img = Image.new('RGB', size, bg_color)
                new_img.paste(img, ((size[0]-img.width)//2, (size[1]-img.height)//2))
                img = new_img
        except Exception as e:
            logging.error(f"Error al cargar imagen de fondo: {str(e)}, usando fondo {bg_color}.")
            img = Image.new('RGB', size, bg_color)
    else:
        img = Image.new('RGB', size, bg_color)

    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception as e:
        logging.error(f"Error al cargar la fuente, usando la fuente predeterminada: {str(e)}")
        font = ImageFont.load_default()

    line_height = font_size * 1.5

    words = text.split()
    lines = []
    current_line = []

    for word in words:
        current_line.append(word)
        test_line = " ".join(current_line)
        left, top, right, bottom = draw.textbbox((0, 0), test_line, font=font)
        if right > size[0] - (TEXT_MARGIN * 2):  # Ancho de la imagen menos dos veces el margen
            current_line.pop()
            lines.append(" ".join(current_line))
            current_line = [word]
    lines.append(" ".join(current_line))

    total_height = len(lines) * line_height
    y = (size[1] - total_height) // 2

    for line in lines:
        left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
        x = (size[0] - (right - left)) // 2
        draw.text((x, y), line, font=font, fill=text_color)
        y += line_height
    return np.array(img)

def create_subscription_image(logo_url, size=IMAGE_SIZE_SUBSCRIPTION, font_size=60):
    """Creates an image for the subscription message."""
    img = Image.new("RGB", size, (255, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
        font2 = ImageFont.truetype(FONT_PATH, font_size // 2)
    except:
        font = ImageFont.load_default()
        font2 = ImageFont.load_default()

    try:
        response = requests.get(logo_url)
        response.raise_for_status()
        logo_img = Image.open(BytesIO(response.content)).convert("RGBA")
        logo_img = logo_img.resize(LOGO_SIZE)
        logo_position = (20, 20)
        img.paste(logo_img, logo_position, logo_img)
    except Exception as e:
        logging.error(f"Error al cargar el logo: {str(e)}")

    text1 = "¡SUSCRÍBETE A LECTOR DE SOMBRAS!"
    left1, top1, right1, bottom1 = draw.textbbox((0, 0), text1, font=font)
    x1 = (size[0] - (right1 - left1)) // 2
    y1 = (size[1] - (bottom1 - top1)) // 2 - (bottom1 - top1) // 2 - 20
    draw.text((x1, y1), text1, font=font, fill="white")

    text2 = "Dale like y activa la campana 🔔"
    left2, top2, right2, bottom2 = draw.textbbox((0, 0), text2, font=font2)
    x2 = (size[0] - (right2 - left2)) // 2
    y2 = (size[1] - (bottom2 - top2)) // 2 + (bottom1 - top1) // 2 + 20
    draw.text((x2, y2), text2, font=font2, fill="white")
    return np.array(img)


def create_simple_video(
    texto,
    nombre_salida,
    voz,
    logo_url,
    font_size,
    bg_color,
    text_color,
    background_video,
    background_image,
    stretch_background
):
    archivos_temp = []
    clips_audio = []
    clips_finales = []
    try:
        logging.info("Iniciando proceso de creación de video...")
        frases = [f.strip() + "." for f in texto.split(".") if f.strip()]
        client = texttospeech.TextToSpeechClient()

        segmentos_texto = []
        segmento_actual = ""
        for frase in frases:
            if len(segmento_actual) + len(frase) < 400:
                segmento_actual += " " + frase
            else:
                segmentos_texto.append(segmento_actual.strip())
                segmento_actual = frase
        segmentos_texto.append(segmento_actual.strip())

        if background_video:
             try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(background_video)[1]) as tmp_file:
                     tmp_file.write(requests.get(background_video).content)
                     video_path = tmp_file.name

                bg_clip_original = VideoFileClip(video_path)
                bg_clip_resized = bg_clip_original.resize(VIDEO_SIZE)
                bg_clip_resized = bg_clip_resized.set_opacity(0.5)
                archivos_temp.append(video_path)

             except Exception as e:
                logging.error(f"Error al cargar o procesar el video de fondo: {e}")
                bg_clip_resized = None
        else:
            bg_clip_resized = None

        # Generar audio para todo el texto
        full_text = " ".join(segmentos_texto)
        synthesis_input = texttospeech.SynthesisInput(text=full_text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="es-ES",
            name=voz,
            ssml_gender=VOCES_DISPONIBLES[voz],
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        retry_count = 0
        max_retries = 3

        while retry_count <= max_retries:
            try:
                response = client.synthesize_speech(
                    input=synthesis_input, voice=voice, audio_config=audio_config
                )
                break
            except Exception as e:
                logging.error(
                    f"Error al solicitar audio (intento {retry_count + 1}): {str(e)}"
                )
                if "429" in str(e):
                    retry_count += 1
                    time.sleep(2**retry_count)
                else:
                    raise
        if retry_count > max_retries:
            raise Exception("Maximos intentos de reintento alcanzado")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
            temp_audio_file.write(response.audio_content)
            archivos_temp.append(temp_audio_file.name)
            audio_clip = AudioFileClip(temp_audio_file.name)
            duracion_total_audio = audio_clip.duration
            
        
        tiempo_acumulado = 0
        for i, segmento in enumerate(segmentos_texto):
            logging.info(f"Procesando segmento {i+1} de {len(segmentos_texto)}")
            
            if bg_clip_resized:
                # Usar el video de fondo redimensionado
                bg_clip_segment = bg_clip_resized.loop(duration=duracion_total_audio)

                text_img = create_text_image(
                     segmento, font_size=font_size, text_color=text_color
                 )
                txt_clip = (
                     ImageClip(text_img)
                     .set_duration(duracion_total_audio)
                     .set_position("center")
                 )
                video_segment = CompositeVideoClip([bg_clip_segment, txt_clip])

            else:
                text_img = create_text_image(
                    segmento,
                    font_size=font_size,
                    bg_color=bg_color,
                    text_color=text_color,
                    background_image=background_image,
                    full_size_background=True,
                    stretch_background = stretch_background
                )
                txt_clip = (
                    ImageClip(text_img)
                    .set_duration(duracion_total_audio)
                    .set_position("center")
                )
                video_segment = txt_clip
            
            start_time = tiempo_acumulado
            end_time = tiempo_acumulado + (duracion_total_audio / len(segmentos_texto))

            if video_segment:
                video_segment = video_segment.set_start(start_time).set_end(end_time).set_audio(audio_clip.set_start(start_time))
                
            clips_finales.append(video_segment)
            tiempo_acumulado = end_time
            time.sleep(0.1)

        subscribe_img = create_subscription_image(logo_url)
        duracion_subscribe = 5

        subscribe_clip = (
            ImageClip(subscribe_img)
            .set_start(tiempo_acumulado)
            .set_duration(duracion_subscribe)
            .set_position("center")
        )
        clips_finales.append(subscribe_clip)

        video_final = concatenate_videoclips(clips_finales, method="compose")

        video_final.write_videofile(
            nombre_salida,
            fps=VIDEO_FPS,
            codec=VIDEO_CODEC,
            audio_codec=AUDIO_CODEC,
            preset=VIDEO_PRESET,
            threads=VIDEO_THREADS,
        )

        video_final.close()
        audio_clip.close()

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

        if bg_clip_resized:
             bg_clip_original.close()
        return True, "Video generado exitosamente"
    except Exception as e:
        logging.error(f"Error: {str(e)}")
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
    st.title("Creador de Shorts para YouTube")

    uploaded_file = st.file_uploader("Carga un archivo de texto", type="txt")

    with st.sidebar:
        st.header("Configuración del Video")
        voz_seleccionada = st.selectbox(
            "Selecciona la voz", options=list(VOCES_DISPONIBLES.keys())
        )
        font_size = st.slider(
            "Tamaño de la fuente", min_value=10, max_value=100, value=DEFAULT_FONT_SIZE
        )
        bg_color = st.color_picker("Color de fondo", value="#000000")
        text_color = st.color_picker("Color de texto", value="#ffffff")
        background_type = st.radio("Tipo de fondo", ["Color sólido", "Imagen", "Video"])
        stretch_background = st.checkbox("Estirar imagen de fondo", value=False)

        background_image = None
        background_video = None

        if background_type == "Imagen":
            background_image = st.file_uploader(
                "Imagen de fondo", type=["png", "jpg", "jpeg", "webp"]
            )
        elif background_type == "Video":
            background_video = st.text_input(
                "URL del video de fondo (mp4, mov, avi)"
            )
    logo_url = "https://yt3.ggpht.com/pBI3iT87_fX91PGHS5gZtbQi53nuRBIvOsuc-Z-hXaE3GxyRQF8-vEIDYOzFz93dsKUEjoHEwQ=s176-c-k-c0x00ffffff-no-rj"  # Logo del canal

    if uploaded_file:
        texto = uploaded_file.read().decode("utf-8")
        nombre_salida = st.text_input("Nombre del Video (sin extensión)", "video_generado")

        if st.button("Generar Video"):
            with st.spinner("Generando video..."):
                nombre_salida_completo = f"{nombre_salida}.mp4"

                img_path = None
                
                if background_image:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=os.path.splitext(background_image.name)[1]
                    ) as tmp_file:
                        tmp_file.write(background_image.read())
                        img_path = tmp_file.name
                

                success, message = create_simple_video(
                    texto,
                    nombre_salida_completo,
                    voz_seleccionada,
                    logo_url,
                    font_size,
                    bg_color,
                    text_color,
                    background_video,
                    img_path,
                    stretch_background
                )

                if success:
                    st.success(message)
                    st.video(nombre_salida_completo)
                    with open(nombre_salida_completo, "rb") as file:
                        st.download_button(
                            label="Descargar video",
                            data=file,
                            file_name=nombre_salida_completo,
                        )

                    st.session_state.video_path = nombre_salida_completo

                    if img_path:
                         os.remove(img_path)

                else:
                    st.error(f"Error al generar video: {message}")
                    if img_path:
                       os.remove(img_path)

        if st.session_state.get("video_path"):
            st.markdown(
                '<a href="https://www.youtube.com/upload" target="_blank">Subir video a YouTube</a>',
                unsafe_allow_html=True,
            )

if __name__ == "__main__":
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    main()
