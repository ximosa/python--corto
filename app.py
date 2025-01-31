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

# Configuración versión de Pillow2
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
DEFAULT_FONT_SIZE = 45
VIDEO_FPS = 24
VIDEO_CODEC = "libx264"
AUDIO_CODEC = "aac"
VIDEO_PRESET = "ultrafast"
VIDEO_THREADS = 4
IMAGE_SIZE_TEXT = (1080, 540)
IMAGE_SIZE_SUBSCRIPTION = (1080, 1920)
SUBSCRIPTION_DURATION = 5
LOGO_SIZE = (200, 200)
VIDEO_SIZE = (1080, 1920)

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
    bg_color=(0, 0, 0, 0),
    text_color="white",
    background_video=None,
    background_image=None,
    full_size_background=False,
):
    if full_size_background:
        size = VIDEO_SIZE

    if background_video:
        return None

    img = Image.new("RGBA", size, (0, 0, 0, 0))
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
        if right > size[0] - 60:
            current_line.pop()
            lines.append(" ".join(current_line))
            current_line = [word]
    lines.append(" ".join(current_line))

    total_height = len(lines) * line_height
    if total_height > size[1]:
        # Reducing font size if text height exceeds image height
        font_size = int(font_size * (size[1] / total_height))
        font = ImageFont.truetype(FONT_PATH, font_size)
        line_height = font_size * 1.5
        lines = []
        current_line = []

        for word in words:
            current_line.append(word)
            test_line = " ".join(current_line)
            left, top, right, bottom = draw.textbbox((0, 0), test_line, font=font)
            if right > size[0] - 60:
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

    # Ajustar tamaño del texto principal
    text1 = "¡SUSCRÍBETE A LECTOR DE SOMBRAS!"
    max_width = size[0] - 40
    while draw.textbbox((0, 0), text1, font=font)[2] > max_width:
        font_size -= 1
        font = ImageFont.truetype(FONT_PATH, font_size)

    left1, top1, right1, bottom1 = draw.textbbox((0, 0), text1, font=font)
    x1 = (size[0] - (right1 - left1)) // 2
    y1 = (size[1] - (bottom1 - top1)) // 2 - (bottom1 - top1) // 2 - 20
    draw.text((x1, y1), text1, font=font, fill="white")

    # Ajustar tamaño del texto secundario
    text2 = "Dale like y activa la campana 🔔"
    while draw.textbbox((0, 0), text2, font=font2)[2] > max_width:
        font2_size -= 1
        font2 = ImageFont.truetype(FONT_PATH, font2_size)

    left2, top2, right2, bottom2 = draw.textbbox((0, 0), text2, font=font2)
    x2 = (size[0] - (right2 - left2)) // 2
    y2 = (size[1] - (bottom2 - top2)) // 2 + (bottom1 - top1) // 2 + 20
    draw.text((x2, y2), text2, font=font2, fill="white")

    return np.array(img)
import threading

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
):
    archivos_temp = []
    clips_audio = []
    clips_finales = []
    lock = threading.Lock()

    def process_segment(i, segmento, client, voz, archivos_temp, clips_audio, lock):
        try:
            synthesis_input = texttospeech.SynthesisInput(text=segmento)
            voice = texttospeech.VoiceSelectionParams(
                language_code="es-ES",
                name=voz,
                ssml_gender=VOCES_DISPONIBLES[voz],
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )

            response = client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            temp_filename = f"temp_audio_{i}.mp3"
            with open(temp_filename, "wb") as out:
                out.write(response.audio_content)

            audio_clip = AudioFileClip(temp_filename)
            with lock:
                archivos_temp.append(temp_filename)
                clips_audio.append(audio_clip)
        except Exception as e:
            logging.error(f"Error in segment {i}: {str(e)}")

    try:
        logging.info("Iniciando proceso de creación de video...")
        frases = [f.strip() + "." for f in texto.split(".") if f.strip()]
        client = texttospeech.TextToSpeechClient()
        tiempo_acumulado = 0

        segmentos_texto = []
        segmento_actual = ""
        for frase in frases:
            if len(segmento_actual) + len(frase) < 400:
                segmento_actual += " " + frase
            else:
                segmentos_texto.append(segmento_actual.strip())
                segmento_actual = frase
        segmentos_texto.append(segmento_actual.strip())

        bg_clip_base = None
        if background_video:
            try:
                bg_clip_original = VideoFileClip(background_video)
                aspect_ratio = VIDEO_SIZE[0] / VIDEO_SIZE[1]
                original_aspect = bg_clip_original.size[0] / bg_clip_original.size[1]

                if original_aspect > aspect_ratio:
                    new_height = VIDEO_SIZE[1]
                    new_width = int(new_height * original_aspect)
                    x_center = (new_width - VIDEO_SIZE[0]) // 2
                    bg_clip_base = (bg_clip_original
                                    .resize(height=new_height)
                                    .crop(x1=x_center, y1=0,
                                         x2=x_center + VIDEO_SIZE[0],
                                         y2=VIDEO_SIZE[1]))
                else:
                    new_width = VIDEO_SIZE[0]
                    new_height = int(new_width / original_aspect)
                    y_center = (new_height - VIDEO_SIZE[1]) // 2
                    bg_clip_base = (bg_clip_original
                                    .resize(width=new_width)
                                    .crop(x1=0, y1=y_center,
                                         x2=VIDEO_SIZE[0],
                                         y2=y_center + VIDEO_SIZE[1]))

                duracion_total = sum(len(segmento.strip().split()) * 0.3 for segmento in segmentos_texto) + SUBSCRIPTION_DURATION
                bg_clip_base = bg_clip_base.loop(duration=duracion_total)

            except Exception as e:
                logging.error(f"Error al cargar o procesar el video de fondo: {e}")
                bg_clip_base = None

        threads = []
        for i, segmento in enumerate(segmentos_texto):
            t = threading.Thread(target=process_segment, args=(i, segmento, client, voz, archivos_temp, clips_audio, lock))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        for i, segmento in enumerate(segmentos_texto):
            audio_clip = clips_audio[i]
            duracion = audio_clip.duration

            if bg_clip_base:
                start_time = tiempo_acumulado % bg_clip_base.duration
                bg_clip_segment = bg_clip_base.subclip(start_time, start_time + duracion)
                black_clip = ColorClip(size=VIDEO_SIZE, color=(0, 0, 0)).set_opacity(0.5).set_duration(duracion)
                text_img = create_text_image(segmento, font_size=font_size, text_color=text_color)
                txt_clip = ImageClip(text_img).set_duration(duracion).set_position("center")
                video_segment = CompositeVideoClip([bg_clip_segment, black_clip, txt_clip])
                video_segment = video_segment.set_audio(audio_clip)
            else:
                text_img = create_text_image(
                    segmento,
                    font_size=font_size,
                    bg_color=bg_color,
                    text_color=text_color,
                    background_image=background_image,
                    full_size_background=True,
                )
                txt_clip = ImageClip(text_img).set_duration(duracion).set_position("center")
                video_segment = txt_clip.set_audio(audio_clip)

            clips_finales.append(video_segment)
            tiempo_acumulado += duracion

        subscribe_img = create_subscription_image(logo_url)
        subscribe_clip = (
            ImageClip(subscribe_img)
            .set_duration(SUBSCRIPTION_DURATION)
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

        for clip in clips_audio + clips_finales:
            try:
                clip.close()
            except:
                pass

        for temp_file in archivos_temp:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

        if 'bg_clip_original' in locals():
            bg_clip_original.close()

        return True, "Video generado exitosamente"

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        for clip in clips_audio + clips_finales:
            try:
                clip.close()
            except:
                pass

        for temp_file in archivos_temp:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

        return False, str(e)

def main():
    st.title("Creador de Videos Automático")

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

        background_image = None
        background_video = None

        if background_type == "Imagen":
            background_image = st.file_uploader(
                "Imagen de fondo", type=["png", "jpg", "jpeg", "webp"]
            )
        elif background_type == "Video":
            background_video = st.file_uploader(
                "Video de fondo", type=["mp4", "mov", "avi"]
            )

    logo_url = "https://yt3.ggpht.com/pBI3iT87_fX91PGHS5gZtbQi53nuRBIvOsuc-Z-hXaE3GxyRQF8-vEIDYOzFz93dsKUEjoHEwQ=s176-c-k-c0x00ffffff-no-rj"

    if uploaded_file:
        texto = uploaded_file.read().decode("utf-8")
        nombre_salida = st.text_input("Nombre del Video (sin extensión)", "video_generado")

        if st.button("Generar Video"):
            with st.spinner("Generando video..."):
                nombre_salida_completo = f"{nombre_salida}.mp4"

                img_path = None
                video_path = None

                if background_image:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=os.path.splitext(background_image.name)[1]
                    ) as tmp_file:
                        tmp_file.write(background_image.read())
                        img_path = tmp_file.name

                if background_video:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=os.path.splitext(background_video.name)[1]
                    ) as tmp_file:
                        tmp_file.write(background_video.read())
                        video_path = tmp_file.name

                success, message = create_simple_video(
                    texto,
                    nombre_salida_completo,
                    voz_seleccionada,
                    logo_url,
                    font_size,
                    bg_color,
                    text_color,
                    video_path,
                    img_path,
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
                    if video_path:
                        os.remove(video_path)
                else:
                    st.error(f"Error al generar video: {message}")
                    if img_path:
                        os.remove(img_path)
                    if video_path:
                        os.remove(video_path)

        if st.session_state.get("video_path"):
            st.markdown(
                '<a href="https://www.youtube.com/upload" target="_blank">Subir video a YouTube</a>',
                unsafe_allow_html=True,
            )

if __name__ == "__main__":
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    main()
