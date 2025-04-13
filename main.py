"""
Бот для транскрипции с использованием AssemblyAI API
с управлением кэшем в отдельной директории,
разделением аудио на части по 2 минуты с помощью ffmpeg,
возможностью выбора языка для транскрипции
и обработкой текста через ChatGPT
"""
import os
import asyncio
import time
import uuid
import requests
import shutil
import signal
import sys
import subprocess
import json
import logging
from typing import Tuple, List, Optional, Dict, Any
import math
from dotenv import load_dotenv
from openai import OpenAI

from pyrogram import Client, filters
from pyrogram.types import Message
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton

# Загрузка переменных окружения из файла .env
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log")
    ]
)
logger = logging.getLogger("assemblyai_bot")

# Учетные данные API из переменных окружения
API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")
BOT_TOKEN = os.getenv("BOT_TOKEN")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Настройка директорий
CACHE_DIR = os.path.join(os.getcwd(), "cache")
AUDIO_FILES_DIR = os.path.join(CACHE_DIR, "audio_files")
SEGMENTS_DIR = os.path.join(CACHE_DIR, "audio_segments")
SESSION_DIR = os.path.join(CACHE_DIR, "sessions")
PROMPTS_FILE = os.path.join(CACHE_DIR, "user_prompts.json")

# Создаём все необходимые директории
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(AUDIO_FILES_DIR, exist_ok=True)
os.makedirs(SEGMENTS_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)

# Создаем клиент OpenAI
openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Максимальная длина сегмента в секундах (2 минуты)
MAX_SEGMENT_LENGTH_SEC = 2 * 60

# Промпт по умолчанию для ChatGPT
DEFAULT_PROMPT = "Суммируй следующий текст в краткой форме, выделяя основные мысли и ключевые моменты."

# Поддерживаемые языки для выбора
SUPPORTED_LANGUAGES = {
    "ru": "Русский🇷🇺",
    "en": "Английский🇬🇧",
}

# Загрузка пользовательских промптов из файла
def load_user_prompts():
    if os.path.exists(PROMPTS_FILE):
        try:
            with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка при загрузке промптов: {e}")
    return {}

# Сохранение пользовательских промптов в файл
def save_user_prompts(prompts):
    try:
        with open(PROMPTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка при сохранении промптов: {e}")

# Инициализация промптов при запуске
user_prompts = load_user_prompts()

# Хранение состояний пользователей
user_states = {}

def clean_cache():
    """Очистка всех кэш-директорий"""
    print("Очистка кэша...")
    try:
        # Удаляем содержимое директорий, но сохраняем сами директории
        for dir_path in [AUDIO_FILES_DIR, SEGMENTS_DIR]:
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        
        # Очищаем сессии Pyrogram
        for filename in os.listdir(SESSION_DIR):
            file_path = os.path.join(SESSION_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Очищаем все состояния пользователей
        for username in list(user_states.keys()):
            user_states[username] = {}
            
        print("Кэш очищен")
    except Exception as e:
        print(f"Ошибка при очистке кэша: {e}")

# Функция для обработки сигналов (Ctrl+C и т.д.)
def signal_handler(sig, frame):
    print("\nЗавершение работы и очистка кэша...")
    clean_cache()
    sys.exit(0)

# Регистрируем обработчик сигналов
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Очищаем кэш при запуске
clean_cache()

# Инициализация клиента с указанием директории для сессий
app = Client(
    "assemblyai_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN,
    workdir=SESSION_DIR,  # Директория для хранения файлов сессий
)

def get_state(username: str) -> Dict[str, Any]:
    """Получение состояния пользователя"""
    if username not in user_states:
        user_states[username] = {}
    return user_states[username]

def set_state(username: str, key: str, value: Any) -> None:
    """Установка значения в состоянии пользователя"""
    if username not in user_states:
        user_states[username] = {}
    user_states[username][key] = value

def clear_state(username: str, key: str = None) -> None:
    """Очистка состояния пользователя"""
    if username in user_states:
        if key:
            if key in user_states[username]:
                del user_states[username][key]
        else:
            user_states[username] = {}

def get_audio_duration(file_path):
    """Получает длительность аудиофайла в секундах с помощью ffprobe"""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception as e:
        print(f"Ошибка при получении длительности аудиофайла: {e}")
        # Если ошибка, возвращаем большое значение, чтобы разделить файл
        return MAX_SEGMENT_LENGTH_SEC + 1

def split_audio_file(file_path: str) -> List[str]:
    """
    Разделяет аудиофайл на сегменты по 2 минуты с помощью ffmpeg
    с перекодированием для обеспечения совместимости
    
    Args:
        file_path: Путь к исходному аудиофайлу
        
    Returns:
        List[str]: Список путей к созданным сегментам
    """
    try:
        print(f"Разделение аудиофайла: {file_path}")
        
        # Получаем длительность аудио
        duration_sec = get_audio_duration(file_path)
        
        # Проверка, что файл не поврежден и имеет длительность
        if duration_sec <= 0:
            print("Файл поврежден или имеет нулевую длительность")
            # Конвертируем в MP3 для исправления возможных ошибок
            converted_path = os.path.join(AUDIO_FILES_DIR, f"{uuid.uuid4()}.mp3")
            convert_cmd = [
                "ffmpeg",
                "-i", file_path,
                "-c:a", "libmp3lame",
                "-q:a", "4",
                converted_path
            ]
            subprocess.run(convert_cmd, capture_output=True)
            
            # Получаем длительность конвертированного файла
            duration_sec = get_audio_duration(converted_path)
            file_path = converted_path
        
        # Если файл меньше 2 минут, конвертируем в MP3 и возвращаем
        if duration_sec <= MAX_SEGMENT_LENGTH_SEC:
            # Конвертируем в MP3 для обеспечения совместимости
            converted_path = os.path.join(AUDIO_FILES_DIR, f"{uuid.uuid4()}.mp3")
            convert_cmd = [
                "ffmpeg",
                "-i", file_path,
                "-c:a", "libmp3lame",
                "-q:a", "4",  # Качество MP3 (0-9, где 0 лучшее)
                converted_path
            ]
            subprocess.run(convert_cmd, capture_output=True)
            return [converted_path]
        
        # Определение количества сегментов
        num_segments = math.ceil(duration_sec / MAX_SEGMENT_LENGTH_SEC)
        
        print(f"Длительность аудио: {duration_sec} сек, создание {num_segments} сегментов")
        
        # Создание сегментов
        segment_paths = []
        for i in range(num_segments):
            start_sec = i * MAX_SEGMENT_LENGTH_SEC
            segment_id = str(uuid.uuid4())
            segment_path = os.path.join(SEGMENTS_DIR, f"{segment_id}.mp3")
            
            # Команда ffmpeg для вырезания сегмента с перекодированием
            cmd = [
                "ffmpeg",
                "-i", file_path,
                "-ss", str(start_sec),
                "-t", str(MAX_SEGMENT_LENGTH_SEC),
                "-c:a", "libmp3lame",  # Перекодируем в MP3
                "-q:a", "4",  # Качество MP3 (0-9, где 0 лучшее)
                "-ar", "44100",  # Частота дискретизации
                segment_path
            ]
            
            # Выполнение команды
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Проверка размера созданного файла
            if os.path.exists(segment_path) and os.path.getsize(segment_path) > 0:
                segment_paths.append(segment_path)
                print(f"Создан сегмент {i+1}/{num_segments}: {segment_path}")
            else:
                print(f"Ошибка при создании сегмента {i+1}/{num_segments}: {result.stderr}")
                # Если возникла ошибка, попробуем снова с другими параметрами
                retry_cmd = [
                    "ffmpeg",
                    "-i", file_path,
                    "-ss", str(start_sec),
                    "-t", str(MAX_SEGMENT_LENGTH_SEC),
                    "-c:a", "aac",  # Используем AAC кодек как альтернативу
                    "-b:a", "128k",  # Битрейт
                    "-ar", "44100",
                    segment_path
                ]
                subprocess.run(retry_cmd, capture_output=True)
                
                if os.path.exists(segment_path) and os.path.getsize(segment_path) > 0:
                    segment_paths.append(segment_path)
                    print(f"Создан сегмент {i+1}/{num_segments} (повторная попытка): {segment_path}")
        
        # Если не удалось создать сегменты, возвращаем исходный файл
        if not segment_paths:
            print("Не удалось создать сегменты, используем исходный файл")
            # Конвертируем исходный файл в MP3
            converted_path = os.path.join(AUDIO_FILES_DIR, f"{uuid.uuid4()}.mp3")
            convert_cmd = [
                "ffmpeg",
                "-i", file_path,
                "-c:a", "libmp3lame",
                "-q:a", "4",
                converted_path
            ]
            subprocess.run(convert_cmd, capture_output=True)
            return [converted_path]
        
        return segment_paths
    
    except Exception as e:
        print(f"Ошибка при разделении аудио: {e}")
        # В случае ошибки конвертируем исходный файл в MP3
        try:
            converted_path = os.path.join(AUDIO_FILES_DIR, f"{uuid.uuid4()}.mp3")
            convert_cmd = [
                "ffmpeg",
                "-i", file_path,
                "-c:a", "libmp3lame",
                "-q:a", "4",
                converted_path
            ]
            subprocess.run(convert_cmd, capture_output=True)
            if os.path.exists(converted_path) and os.path.getsize(converted_path) > 0:
                return [converted_path]
        except:
            pass
        
        # Если все методы не сработали, возвращаем исходный файл
        return [file_path]

async def download_large_file(message: Message, file_id: str, save_path: str) -> Tuple[bool, str]:
    """
    Скачивание большого файла с обработкой ошибок
    
    Args:
        message: Сообщение пользователя
        file_id: Идентификатор файла
        save_path: Путь для сохранения файла
        
    Returns:
        Tuple[bool, str]: Успех и сообщение/путь к файлу
    """
    try:
        # Отправка сообщения о загрузке
        status_msg = await message.reply("⏳ Скачиваю файл...")
        
        # Создаем директорию, если её нет
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Скачивание файла
        start_time = time.time()
        await app.download_media(message=file_id, file_name=save_path)
        download_time = time.time() - start_time
        
        # Проверка, что файл существует и не пустой
        if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
            await status_msg.edit_text("❌ Ошибка: файл не скачался или пустой")
            return False, "Файл не скачался или пустой"
        
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        logger.info(f"Файл скачан: {save_path}, размер: {file_size_mb:.2f} МБ, время: {download_time:.2f}с")
        
        # Обновление статуса
        await status_msg.edit_text("✅ Файл успешно загружен. Подготовка к обработке...")
        
        return True, save_path
    
    except Exception as e:
        logger.error(f"Ошибка при скачивании файла: {e}", exc_info=True)
        try:
            await status_msg.edit_text(f"❌ Ошибка при скачивании файла: {str(e)}")
        except:
            await message.reply(f"❌ Ошибка при скачивании файла: {str(e)}")
        return False, str(e)

async def transcribe_with_assemblyai(audio_path: str, message: Message, status_msg: Message = None, target_language: str = "auto") -> Tuple[bool, str]:
    """
    Транскрибация аудио с помощью AssemblyAI API с возможностью перевода результата
    
    Args:
        audio_path: Путь к аудиофайлу
        message: Исходное сообщение пользователя
        status_msg: Сообщение со статусом (опционально)
        target_language: Код целевого языка для транскрипции (по умолчанию "auto" - оставить оригинальный)
        
    Returns:
        Tuple[bool, str]: Успех и текст транскрипции/сообщение об ошибке
    """
    try:
        if status_msg:
            await status_msg.edit_text("🔄 Отправка аудио на транскрипцию...")
        else:
            status_msg = await message.reply("🔄 Отправка аудио на транскрипцию...")
        
        # Проверяем размер файла
        file_size = os.path.getsize(audio_path)
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"Размер файла для транскрипции: {file_size_mb:.2f} МБ")
        
        # Если файл больше 20МБ, сжимаем его
        if file_size > 20 * 1024 * 1024:
            await status_msg.edit_text(f"⚙️ Файл слишком большой ({file_size_mb:.2f} МБ). Оптимизирую для загрузки...")
            compressed_path = os.path.join(AUDIO_FILES_DIR, f"{uuid.uuid4()}.mp3")
            
            compress_cmd = [
                "ffmpeg",
                "-i", audio_path,
                "-c:a", "libmp3lame",
                "-b:a", "64k",  # Низкий битрейт для уменьшения размера
                compressed_path
            ]
            
            result = subprocess.run(compress_cmd, capture_output=True, text=True)
            
            if os.path.exists(compressed_path) and os.path.getsize(compressed_path) > 0:
                compressed_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
                logger.info(f"Файл сжат до {compressed_size_mb:.2f} МБ")
                await status_msg.edit_text(f"✅ Файл оптимизирован ({compressed_size_mb:.2f} МБ). Отправляю на транскрипцию...")
                audio_path = compressed_path
            else:
                logger.error(f"Ошибка при сжатии файла: {result.stderr}")
                await status_msg.edit_text("⚠️ Не удалось оптимизировать файл. Пробую отправить как есть...")
        
        # Стартовое время для отслеживания
        start_time = time.time()
        
        # Загрузка файла на сервер AssemblyAI
        headers = {
            "authorization": ASSEMBLYAI_API_KEY,
            "content-type": "application/json"
        }
        
        await status_msg.edit_text("📤 Загрузка аудио на сервер распознавания...")
        
        # Загрузка файла (выполняется в отдельном потоке)
        def upload_file():
            with open(audio_path, "rb") as f:
                response = requests.post(
                    "https://api.assemblyai.com/v2/upload",
                    headers=headers,
                    data=f,
                    timeout=300  # 5 минут таймаут
                )
            
            if response.status_code != 200:
                logger.error(f"Ошибка загрузки файла: {response.text}")
                raise Exception(f"Ошибка загрузки файла: {response.text}")
            
            return response.json()["upload_url"]
        
        try:
            upload_url = await asyncio.get_event_loop().run_in_executor(None, upload_file)
            logger.info(f"Файл успешно загружен, URL: {upload_url}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла: {e}")
            await status_msg.edit_text(f"❌ Ошибка при загрузке файла: {str(e)}")
            return False, f"Ошибка при загрузке файла: {str(e)}"
        
        # Отправка задания на транскрипцию
        # Формируем сообщение в зависимости от режима (определение или перевод)
        if target_language == "auto":
            await status_msg.edit_text(f"⏳ Файл загружен. Начинаю транскрипцию с автоопределением языка...")
        else:
            target_lang_name = SUPPORTED_LANGUAGES.get(target_language, target_language)
            await status_msg.edit_text(f"⏳ Файл загружен. Начинаю транскрипцию на {target_lang_name}...")
        
        # Формируем данные для API
        data = {
            "audio_url": upload_url,
        }
        
        # Если выбран конкретный язык, добавляем его в запрос
        if target_language != "auto":
            data["language_code"] = target_language
        
        response = requests.post(
            "https://api.assemblyai.com/v2/transcript",
            json=data,
            headers=headers
        )
        
        if response.status_code != 200:
            logger.error(f"Ошибка создания задания: {response.text}")
            await status_msg.edit_text(f"❌ Ошибка создания задания транскрипции: {response.text}")
            return False, f"Ошибка создания задания: {response.text}"
        
        transcript_id = response.json()["id"]
        logger.info(f"ID транскрипции: {transcript_id}")
        
        # Ожидание завершения транскрипции
        progress_updates = 0
        last_update_time = time.time()
        
        while True:
            current_time = time.time()
            # Обновляем статус каждые 10 секунд
            if current_time - last_update_time >= 10:
                progress_updates += 1
                elapsed_time = current_time - start_time
                await status_msg.edit_text(f"⏳ Транскрибирую аудио... ({elapsed_time:.0f}с)")
                last_update_time = current_time
            
            response = requests.get(
                f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                headers=headers
            )
            
            if response.status_code != 200:
                logger.error(f"Ошибка при проверке статуса: {response.text}")
                await status_msg.edit_text(f"❌ Ошибка при проверке статуса транскрипции: {response.text}")
                return False, f"Ошибка при проверке статуса: {response.text}"
            
            status = response.json()["status"]
            logger.info(f"Статус транскрипции: {status}")
            
            if status == "completed":
                total_time = time.time() - start_time
                # Получаем определенный язык из результата
                detected_language = response.json().get("language_code", "неизвестен")
                detected_lang_name = SUPPORTED_LANGUAGES.get(detected_language, detected_language)
                
                # Получаем текст транскрипции
                transcription_text = response.json()["text"]
                
                # Формируем сообщение о результате
                if target_language == "auto":
                    # Если был запрошен оригинальный язык
                    await status_msg.edit_text(f"✅ Транскрипция завершена!\n"
                                            f"👂 Определен язык аудио: {detected_lang_name}")
                else:
                    # Если был запрошен конкретный язык
                    await status_msg.edit_text(f"✅ Транскрипция завершена!\n"
                                            f"👂 Исходный язык аудио: {detected_lang_name}")
                
                return True, transcription_text
                
            elif status == "error":
                error_msg = response.json().get("error", "Неизвестная ошибка")
                logger.error(f"Ошибка транскрипции: {error_msg}")
                await status_msg.edit_text(f"❌ Ошибка при транскрипции: {error_msg}")
                return False, f"Ошибка транскрипции: {error_msg}"
            
            # Пауза перед следующей проверкой
            await asyncio.sleep(3)
    
    except Exception as e:
        logger.error(f"Ошибка при транскрипции: {e}", exc_info=True)
        if status_msg:
            await status_msg.edit_text(f"❌ Произошла ошибка: {str(e)}")
        else:
            await message.reply(f"❌ Произошла ошибка: {str(e)}")
        return False, f"Ошибка при транскрипции: {str(e)}"
    
    finally:
        # Удаляем временные файлы
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"Удален временный файл: {audio_path}")
        except Exception as e:
            logger.error(f"Ошибка при удалении файла: {e}")

# Функция для вызова ChatGPT API
async def process_with_chatgpt(text, prompt, message):
    """
    Обрабатывает текст через ChatGPT API с указанным промптом
    
    Args:
        text: Текст транскрипции для обработки
        prompt: Системный промпт для ChatGPT
        message: Сообщение для обновления статуса
    
    Returns:
        str: Ответ от ChatGPT или сообщение об ошибке
    """
    status_msg = await message.reply("🧠 Обрабатываю текст с помощью ChatGPT...")
    
    try:
        if not openai_client:
            return "❌ Ошибка: API ключ OpenAI не настроен в конфигурации бота."
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        # Извлекаем текст ответа
        result = response.choices[0].message.content
        await status_msg.edit_text("✅ Обработка через ChatGPT завершена!")
        return result
    
    except Exception as e:
        logger.error(f"Ошибка при вызове ChatGPT API: {e}", exc_info=True)
        await status_msg.edit_text(f"❌ Ошибка при обработке текста: {str(e)}")
        return f"Ошибка при обработке текста: {str(e)}"

# Обновление текста для выбора языка
async def show_language_selection(message: Message, file_path: str) -> None:
    """
    Показывает клавиатуру выбора языка для транскрипции
    
    Args:
        message: Исходное сообщение пользователя
        file_path: Путь к аудиофайлу
    """
    # Получаем username для хранения пути к файлу в состоянии
    username = message.from_user.username or str(message.from_user.id)
    
    # Сохраняем путь к файлу в состоянии пользователя
    set_state(username, "file_path", file_path)
    
    # Формируем клавиатуру с языками
    keyboard = []
    
    # Добавляем языки для выбора, по одному на строку
    for lang_code, lang_name in SUPPORTED_LANGUAGES.items():
        # Пропускаем auto, если не хотим его показывать
        if lang_code == "auto":
            continue
            
        keyboard.append([
            InlineKeyboardButton(
                lang_name, 
                callback_data=f"lang_{lang_code}"
            )
        ])
    
    # Добавляем кнопку отмены процесса
    keyboard.append([
        InlineKeyboardButton("🔙 Отмена", callback_data="cancel_transcription")
    ])
    
    # Отправляем сообщение с клавиатурой
    await message.reply(
        "Выберите язык для транскрипции:\n",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

# Обработчик для кнопок выбора языка
@app.on_callback_query(filters.regex(r"^lang_(.+)$"))
async def handle_language_selection(client, callback_query):
    try:
        # Получаем данные из callback_data
        data_parts = callback_query.data.split("_")
        if len(data_parts) != 2:
            await callback_query.answer("Некорректные данные")
            return
            
        language_code = data_parts[1]  # Код языка
        
        # Получаем username и путь к файлу из состояния
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        file_path = get_state(username).get("file_path")
        
        # Проверяем существование файла и состояния
        if not file_path or not os.path.exists(file_path):
            await callback_query.answer("Файл не найден, повторите загрузку")
            await callback_query.message.edit_text("❌ Файл не найден или удален. Пожалуйста, загрузите файл заново.")
            return
        
        # Обновляем состояние пользователя
        set_state(username, "language", language_code)
        
        # Сообщаем, что язык выбран
        lang_name = SUPPORTED_LANGUAGES.get(language_code, "оригинальный")
        await callback_query.answer(f"Выбран язык: {lang_name}")
        
        # Редактируем сообщение, убираем клавиатуру
        status_msg = await callback_query.message.edit_text(
            f"🔤 Выбран язык: {lang_name}\n⏳ Начинаю транскрипцию...",
            reply_markup=None
        )
        
        # Запуск транскрипции с выбранным языком
        success, transcription = await transcribe_with_assemblyai(
            file_path, 
            callback_query.message, 
            status_msg, 
            language_code
        )
        
        if success:
            # Сохраняем транскрипцию в состоянии пользователя
            set_state(username, "transcription", transcription)
            
            # Предлагаем пользователю выбор действий с транскрипцией
            # В функции handle_language_selection меняем код создания клавиатуры:
            # Предлагаем пользователю выбор действий с транскрипцией
            keyboard = [
                [InlineKeyboardButton("📝 Показать транскрипцию", callback_data="action_show_transcription")],
                [InlineKeyboardButton("🧠 Обработать через ChatGPT", callback_data="action_process_gpt")],
                [InlineKeyboardButton("🔙 Вернуться", callback_data="back_to_language")]
            ]
            
            await callback_query.message.reply(
                "✅ Транскрипция завершена!\n\n"
                "Выберите действие с полученным текстом:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
            logger.info(f"Транскрипция успешно получена для {username}, длина: {len(transcription)} символов")
        else:
            # В случае ошибки
            await callback_query.message.reply(f"❌ Не удалось выполнить транскрипцию: {transcription}")
            logger.error(f"Ошибка транскрипции: {transcription}")
            
            # Очищаем состояние пользователя
            clear_state(username, "transcribing")
            clear_state(username, "file_path")
            clear_state(username, "language")
            
    except Exception as e:
        logger.error(f"Ошибка при обработке выбора языка: {e}", exc_info=True)
        await callback_query.message.reply(f"❌ Произошла ошибка: {str(e)}")
        
        # Очищаем состояние пользователя в случае ошибки
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        clear_state(username, "transcribing")
        clear_state(username, "file_path")
        clear_state(username, "language")

# Обработчик кнопок действий с транскрипцией
@app.on_callback_query(filters.regex(r"^action_(.+)$"))
async def handle_action_selection(client, callback_query):
    try:
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        action = callback_query.data.split("_", 1)[1]
        
        # Получаем сохраненную транскрипцию
        transcription = get_state(username).get("transcription")
        
        if not transcription:
            await callback_query.answer("Транскрипция не найдена")
            await callback_query.message.edit_text(
                "❌ Транскрипция не найдена или устарела. Пожалуйста, загрузите файл заново.",
                reply_markup=None
            )
            return
        
        # Обрабатываем выбранное действие
        if action == "show_transcription":
            # Показываем транскрипцию
            await callback_query.answer("Показываю транскрипцию")
            
            # Разбиваем текст на страницы по 3000 символов (чтобы уместиться в лимиты сообщений)
            page_size = 3000
            pages = [transcription[i:i+page_size] for i in range(0, len(transcription), page_size)]
            total_pages = len(pages)
            
            # Сохраняем информацию о страницах в состоянии пользователя
            set_state(username, "transcription_pages", pages)
            set_state(username, "current_page", 0)
            
            # Создаем клавиатуру с навигацией по страницам
            keyboard = []
            
            # Если страниц больше одной, добавляем кнопки навигации
            if total_pages > 1:
                nav_buttons = []
                nav_buttons.append(InlineKeyboardButton("◀️", callback_data="page_prev"))
                nav_buttons.append(InlineKeyboardButton(f"1/{total_pages}", callback_data="page_info"))
                nav_buttons.append(InlineKeyboardButton("▶️", callback_data="page_next"))
                keyboard.append(nav_buttons)
            
            # Добавляем кнопку "Назад"
            keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_actions")])
            
            # Показываем первую страницу
            await callback_query.message.edit_text(
                f"📝 **Результат транскрипции** (страница 1/{total_pages}):\n\n{pages[0]}",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        elif action == "process_gpt":
            # Проверяем настройку API ключа
            if not openai_client:
                await callback_query.answer("ChatGPT недоступен")
                await callback_query.message.edit_text(
                    "❌ Обработка через ChatGPT недоступна: API ключ не настроен.\n"
                    "Обратитесь к администратору бота.",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔙 Назад", callback_data="back_to_actions")]
                    ])
                )
                return
            
            await callback_query.answer("Отправляю в ChatGPT")
            
            # Получаем промпт пользователя или используем промпт по умолчанию
            prompt = user_prompts.get(username, DEFAULT_PROMPT)
            
            # Обновляем статус
            await callback_query.message.edit_text(
                "🧠 Обрабатываю текст с помощью ChatGPT...",
                reply_markup=None
            )
            
            # Отправляем текст на обработку в ChatGPT
            try:
                if not openai_client:
                    result = "❌ Ошибка: API ключ OpenAI не настроен в конфигурации бота."
                else:
                    response = openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": transcription}
                        ],
                        temperature=0.7,
                        max_tokens=2000
                    )
                    result = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Ошибка при вызове ChatGPT API: {e}", exc_info=True)
                result = f"Ошибка при обработке текста: {str(e)}"
            
            # Разбиваем результат на страницы, как и для транскрипции
            page_size = 3000
            pages = [result[i:i+page_size] for i in range(0, len(result), page_size)]
            total_pages = len(pages)
            
            # Сохраняем информацию о страницах в состоянии пользователя
            set_state(username, "gpt_result_pages", pages)
            set_state(username, "current_gpt_page", 0)
            
            # Создаем клавиатуру с навигацией по страницам
            keyboard = []
            
            # Если страниц больше одной, добавляем кнопки навигации
            if total_pages > 1:
                nav_buttons = []
                nav_buttons.append(InlineKeyboardButton("◀️", callback_data="gpt_page_prev"))
                nav_buttons.append(InlineKeyboardButton(f"1/{total_pages}", callback_data="gpt_page_info"))
                nav_buttons.append(InlineKeyboardButton("▶️", callback_data="gpt_page_next"))
                keyboard.append(nav_buttons)
            
            # Добавляем кнопку "Назад"
            keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_actions")])
            
            # Показываем первую страницу результата
            await callback_query.message.edit_text(
                f"🧠 **Результат обработки ChatGPT** (страница 1/{total_pages}):\n\n{pages[0]}",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        
    except Exception as e:
        logger.error(f"Ошибка при обработке действия: {e}", exc_info=True)
        await callback_query.answer(f"Произошла ошибка: {str(e)}")
        
        # В случае ошибки добавляем кнопку для возврата
        await callback_query.message.edit_text(
            f"❌ Произошла ошибка: {str(e)}",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="back_to_actions")]
            ])
        )
# Обработчик для добавления/изменения промпта
@app.on_message(filters.command("setprompt"))
async def set_prompt_command(client, message):
    try:
        username = message.from_user.username or str(message.from_user.id)
        
        # Проверяем, есть ли текст промпта после команды
        if len(message.text.split(' ', 1)) < 2:
            # Если текста нет, отправляем текущий промпт и инструкцию
            current_prompt = user_prompts.get(username, DEFAULT_PROMPT)
            await message.reply(
                f"📝 **Ваш текущий промпт:**\n\n`{current_prompt}`\n\n"
                "Чтобы изменить промпт, отправьте команду в формате:\n"
                "`/setprompt Текст вашего нового промпта`"
            )
            return
        
        # Получаем текст промпта из сообщения
        prompt_text = message.text.split(' ', 1)[1].strip()
        
        # Сохраняем промпт пользователя
        user_prompts[username] = prompt_text
        save_user_prompts(user_prompts)
        
        await message.reply(
            "✅ Ваш промпт успешно сохранен!\n\n"
            f"📝 **Новый промпт:**\n\n`{prompt_text}`\n\n"
            "Этот промпт будет использоваться при отправке транскрипций в ChatGPT."
        )
    
    except Exception as e:
        logger.error(f"Ошибка при установке промпта: {e}", exc_info=True)
        await message.reply(f"❌ Произошла ошибка: {str(e)}")

# Обработчик для сброса промпта к значению по умолчанию
@app.on_message(filters.command("resetprompt"))
async def reset_prompt_command(client, message):
    try:
        username = message.from_user.username or str(message.from_user.id)
        
        # Удаляем промпт пользователя, если он был
        if username in user_prompts:
            del user_prompts[username]
            save_user_prompts(user_prompts)
        
        await message.reply(
            "✅ Ваш промпт сброшен к значению по умолчанию!\n\n"
            f"📝 **Промпт по умолчанию:**\n\n`{DEFAULT_PROMPT}`"
        )
    
    except Exception as e:
        logger.error(f"Ошибка при сбросе промпта: {e}", exc_info=True)
        await message.reply(f"❌ Произошла ошибка: {str(e)}")

@app.on_message(filters.command("start"))
async def start_command(client, message):
    username = message.from_user.username or str(message.from_user.id)
    
    # Получаем текущий промпт пользователя или дефолтный
    current_prompt = user_prompts.get(username, DEFAULT_PROMPT)
    
    # Создаем клавиатуру для главного меню
    keyboard = [
        [InlineKeyboardButton("📝 Изменить промпт", callback_data="change_prompt")],
        [InlineKeyboardButton("❓ Помощь", callback_data="show_help")]
    ]
    
    await message.reply(
        "👋 Привет! Я бот для транскрипции аудио с возможностью обработки результата через ChatGPT.\n\n"
        "📱 **Что я умею:**\n"
        "• Автоматически определять язык аудио\n"
        "• Переводить текст на разные языки\n"
        "• Обрабатывать полученный текст через ChatGPT с вашим промптом\n\n"
        "🎧 Отправьте мне голосовое сообщение или аудиофайл, чтобы начать.\n\n"
        f"📝 **Ваш текущий промпт для ChatGPT:**\n`{current_prompt}`",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

@app.on_message(filters.command("help"))
async def help_command(client, message):
    # Собираем список языков для помощи
    languages_list = "\n".join([f"• {name}" for code, name in SUPPORTED_LANGUAGES.items()])
    
    await message.reply(
        "📋 **Как пользоваться этим ботом:**\n\n"
        "1. Отправьте голосовое сообщение или аудиофайл (MP3, OGG, WAV и др.)\n"
        "2. Бот автоматически определит язык аудио\n"
        "3. Выберите язык для транскрипции:\n"
        "   - 'Оригинальный язык' - текст останется на языке аудио\n"
        "   - Любой другой язык - текст будет переведен на выбранный язык\n"
        "4. После обработки вы сможете:\n"
        "   - Просто получить текст транскрипции\n"
        "   - Отправить текст в ChatGPT с вашим собственным промптом\n\n"
        "**Команды бота:**\n"
        "/setprompt [текст] - установить промпт для ChatGPT\n"
        "/resetprompt - сбросить промпт к значению по умолчанию\n"
        "/languages - показать поддерживаемые языки\n"
        "/status - проверить статус текущих задач\n"
        "/cancel - отменить текущую задачу\n\n"
        "**Поддерживаемые языки для перевода:**\n"
        f"{languages_list}\n\n"
        "Бот использует технологию AssemblyAI для точного распознавания речи и технологию ChatGPT для обработки текста."
    )

@app.on_message(filters.command("languages"))
async def languages_command(client, message):
    """Показать список поддерживаемых языков для перевода"""
    # Собираем список языков
    languages_list = "\n".join([f"• {name} ({code})" for code, name in SUPPORTED_LANGUAGES.items() if code != "auto"])
    
    await message.reply(
        "🌐 **Поддерживаемые языки для перевода:**\n\n"
        f"{languages_list}\n\n"
        "При выборе 'Оригинальный язык' система определит язык аудио автоматически и "
        "предоставит транскрипцию на исходном языке без перевода."
    )

@app.on_message(filters.command("status"))
async def status_command(client, message):
    """Показать статус активных транскрипций"""
    username = message.from_user.username or str(message.from_user.id)
    
    if username in user_states and "transcribing" in user_states[username]:
        # Проверяем, выбран ли язык
        language_info = ""
        if "language" in user_states[username]:
            lang_code = user_states[username]["language"]
            lang_name = SUPPORTED_LANGUAGES.get(lang_code, "оригинальном")
            language_info = f" на {lang_name} языке"
        
        await message.reply(f"⏳ У вас есть активная задача транскрипции{language_info} в обработке.")
    else:
        await message.reply("✅ У вас нет активных задач транскрипции.")

@app.on_message(filters.command("cancel"))
async def cancel_command(client, message):
    """Отменить текущую транскрипцию"""
    username = message.from_user.username or str(message.from_user.id)
    
    if username in user_states and "transcribing" in user_states[username]:
        clear_state(username, "transcribing")
        clear_state(username, "file_path")
        clear_state(username, "language")
        clear_state(username, "transcription")
        await message.reply("✅ Задача транскрипции отменена.")
    else:
        await message.reply("У вас нет активных задач для отмены.")

async def process_audio_file(message: Message) -> None:
    """
    Обработка аудиофайла из сообщения
    
    Args:
        message: Сообщение с аудиофайлом
    """
    try:
        username = message.from_user.username or str(message.from_user.id)
        logger.info(f"Начинаю обработку аудиофайла от пользователя {username}")
        
        # Получение данных файла
        file_id = None
        file_name = None
        
        if message.audio:
            file_id = message.audio.file_id
            file_name = message.audio.file_name or f"audio_{message.id}.mp3"
            logger.info(f"Получен аудиофайл: {file_id}")
        elif message.voice:
            file_id = message.voice.file_id
            file_name = f"voice_{message.id}.ogg"
            logger.info(f"Получено голосовое сообщение: {file_id}")
        elif message.document:
            file_id = message.document.file_id
            file_name = message.document.file_name or f"document_{message.id}"
            logger.info(f"Получен документ: {file_id}, mime_type: {message.document.mime_type}")
        else:
            logger.warning("Сообщение не содержит медиафайлов")
            await message.reply("Пожалуйста, отправьте аудиофайл.")
            return
        
        # Формирование уникального имени файла
        file_name = f"{int(time.time())}_{file_name}"
        save_path = os.path.join(AUDIO_FILES_DIR, file_name)
        
        # Скачивание файла
        logger.info(f"Сохранение в {save_path}")
        success, result = await download_large_file(message, file_id, save_path)
        
        if success:
            # Если скачивание успешно, предлагаем выбор языка
            logger.info("Файл успешно скачан, предлагаем выбор языка")
            await show_language_selection(message, save_path)
            
            # Сохраняем в состоянии пользователя информацию о файле
            set_state(username, "file_path", save_path)
            set_state(username, "transcribing", True)
        else:
            # Если скачивание не удалось, сообщаем об ошибке
            logger.error(f"Ошибка скачивания: {result}")
            await message.reply(f"❌ Ошибка при скачивании файла: {result}")
            clear_state(username, "transcribing")
    
    except Exception as e:
        logger.error(f"Ошибка обработки аудиофайла: {e}", exc_info=True)
        await message.reply(f"❌ Произошла ошибка при обработке аудиофайла: {str(e)}")
        clear_state(username, "transcribing")

@app.on_message(filters.audio | filters.voice | filters.document)
async def handle_audio(client, message):
    """
    Обработчик аудиосообщений, голосовых сообщений и документов
    """
    try:
        username = message.from_user.username or str(message.from_user.id)
        
        # Проверка если это не аудио-документ
        if message.document and not message.document.mime_type.startswith("audio/"):
            # Проверяем расширение на наличие аудио-форматов
            file_ext = os.path.splitext(message.document.file_name)[1].lower() if message.document.file_name else ""
            if file_ext not in [".mp3", ".ogg", ".wav", ".flac", ".m4a", ".aac"]:
                await message.reply("Этот документ не похож на аудиофайл. Отправьте аудиофайл для транскрипции.")
                return
        
        # Проверка активной транскрипции
        if username in user_states and "transcribing" in user_states[username]:
            await message.reply(
                "У вас уже есть активная задача транскрипции. "
                "Дождитесь ее завершения или отмените командой /cancel."
            )
            return
        
        # Установка статуса транскрипции
        set_state(username, "transcribing", True)
        
        # Обработка аудиофайла
        await process_audio_file(message)
        
    except Exception as e:
        logger.error(f"Ошибка при обработке аудио: {e}", exc_info=True)
        await message.reply(f"❌ Произошла ошибка при обработке аудио: {str(e)}")
        
        # Очистка статуса пользователя в случае ошибки
        clear_state(username, "transcribing")

@app.on_message(filters.text)
async def handle_text(client, message):
    # Пропускаем сообщения с командами
    if message.text.startswith('/'):
        return
    
    await message.reply(
        "Пожалуйста, отправьте мне голосовое сообщение или аудиофайл для транскрипции.\n"
        "Используйте /help для получения справки или /languages для списка поддерживаемых языков."
    )

@app.on_callback_query(filters.regex(r"^change_prompt$"))
async def change_prompt_callback(client, callback_query):
    try:
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        
        # Создаем клавиатуру с кнопкой "Назад"
        keyboard = [
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]
        ]
        
        # Редактируем текущее сообщение
        await callback_query.message.edit_text(
            "📝 **Изменение промпта для ChatGPT**\n\n"
            "Чтобы установить новый промпт, отправьте команду:\n"
            "`/setprompt Ваш новый промпт`\n\n"
            "Например:\n"
            "`/setprompt Выдели ключевые моменты и сделай краткий конспект по этому тексту.`\n\n"
            "Чтобы вернуться к стандартному промпту, используйте команду:\n"
            "`/resetprompt`",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
        # Подтверждаем нажатие кнопки
        await callback_query.answer("Открываю настройки промпта")
        
    except Exception as e:
        logger.error(f"Ошибка при обработке изменения промпта: {e}", exc_info=True)
        await callback_query.answer(f"Произошла ошибка: {str(e)}")

@app.on_callback_query(filters.regex(r"^show_help$"))
async def show_help_callback(client, callback_query):
    try:
        # Собираем список языков для помощи
        languages_list = "\n".join([f"• {name}" for code, name in SUPPORTED_LANGUAGES.items()])
        
        # Создаем клавиатуру с кнопкой "Назад"
        keyboard = [
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]
        ]
        
        # Редактируем текущее сообщение
        await callback_query.message.edit_text(
            "📋 **Как пользоваться этим ботом:**\n\n"
            "1. Отправьте голосовое сообщение или аудиофайл (MP3, OGG, WAV и др.)\n"
            "2. Бот автоматически определит язык аудио\n"
            "3. Выберите язык для транскрипции\n"
            "4. После обработки вы сможете:\n"
            "   - Просто получить текст транскрипции\n"
            "   - Отправить текст в ChatGPT с вашим собственным промптом\n\n"
            "**Команды бота:**\n"
            "/setprompt [текст] - установить промпт для ChatGPT\n"
            "/resetprompt - сбросить промпт к значению по умолчанию\n"
            "/languages - показать поддерживаемые языки\n"
            "/status - проверить статус текущих задач\n"
            "/cancel - отменить текущую задачу\n\n"
            "**Поддерживаемые языки для перевода:**\n"
            f"{languages_list}",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
        # Подтверждаем нажатие кнопки
        await callback_query.answer("Открываю справку")
        
    except Exception as e:
        logger.error(f"Ошибка при показе справки: {e}", exc_info=True)
        await callback_query.answer(f"Произошла ошибка: {str(e)}")

# Обработчик для кнопки "Назад"
@app.on_callback_query(filters.regex(r"^back_to_main$"))
async def back_to_main_callback(client, callback_query):
    try:
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        
        # Получаем текущий промпт пользователя или дефолтный
        current_prompt = user_prompts.get(username, DEFAULT_PROMPT)
        
        # Создаем клавиатуру для главного меню
        keyboard = [
            [InlineKeyboardButton("📝 Изменить промпт", callback_data="change_prompt")],
            [InlineKeyboardButton("❓ Помощь", callback_data="show_help")]
        ]
        
        # Редактируем сообщение, возвращая главное меню
        await callback_query.message.edit_text(
            "👋 Привет! Я бот для транскрипции аудио с возможностью обработки результата через ChatGPT.\n\n"
            "📱 **Что я умею:**\n"
            "• Автоматически определять язык аудио\n"
            "• Переводить текст на разные языки\n"
            "• Обрабатывать полученный текст через ChatGPT с вашим промптом\n\n"
            "🎧 Отправьте мне голосовое сообщение или аудиофайл, чтобы начать.\n\n"
            f"📝 **Ваш текущий промпт для ChatGPT:**\n`{current_prompt}`",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
        # Подтверждаем нажатие кнопки
        await callback_query.answer("Возвращаюсь в главное меню")
        
    except Exception as e:
        logger.error(f"Ошибка при возврате в главное меню: {e}", exc_info=True)
        await callback_query.answer(f"Произошла ошибка: {str(e)}")

@app.on_callback_query(filters.regex(r"^back_to_language$"))
async def back_to_language_callback(client, callback_query):
    try:
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        file_path = get_state(username).get("file_path")
        
        if not file_path or not os.path.exists(file_path):
            await callback_query.answer("Файл не найден, начните заново")
            await callback_query.message.edit_text(
                "❌ Файл не найден или удален. Пожалуйста, загрузите файл заново."
            )
            return
            
        # Формируем клавиатуру с языками заново
        keyboard = []
        
        # Добавляем языки для выбора, по одному на строку
        for lang_code, lang_name in SUPPORTED_LANGUAGES.items():
            if lang_code == "auto":
                continue
                
            keyboard.append([
                InlineKeyboardButton(
                    lang_name, 
                    callback_data=f"lang_{lang_code}"
                )
            ])
        
        # Добавляем кнопку отмены процесса
        keyboard.append([
            InlineKeyboardButton("🔙 Отмена", callback_data="cancel_transcription")
        ])
        
        # Редактируем сообщение с выбором языка
        await callback_query.message.edit_text(
            "Выберите язык для транскрипции:\n",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
        # Подтверждаем нажатие кнопки
        await callback_query.answer("Возвращаюсь к выбору языка")
        
    except Exception as e:
        logger.error(f"Ошибка при возврате к выбору языка: {e}", exc_info=True)
        await callback_query.answer(f"Произошла ошибка: {str(e)}")
    
@app.on_callback_query(filters.regex(r"^cancel_transcription$"))
async def cancel_transcription_callback(client, callback_query):
    try:
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        
        # Очищаем состояние пользователя
        clear_state(username, "transcribing")
        clear_state(username, "file_path")
        clear_state(username, "language")
        
        # Редактируем сообщение
        await callback_query.message.edit_text(
            "✅ Транскрипция отменена.\n\n"
            "Вы можете отправить другой аудиофайл для обработки или использовать команду /start для возврата в главное меню."
        )
        
        # Подтверждаем нажатие кнопки
        await callback_query.answer("Транскрипция отменена")
        
    except Exception as e:
        logger.error(f"Ошибка при отмене транскрипции: {e}", exc_info=True)
        await callback_query.answer(f"Произошла ошибка: {str(e)}")

@app.on_callback_query(filters.regex(r"^back_to_actions$"))
async def back_to_actions_callback(client, callback_query):
    try:
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        transcription = get_state(username).get("transcription")
        
        if not transcription:
            await callback_query.answer("Транскрипция не найдена")
            await callback_query.message.edit_text(
                "❌ Транскрипция не найдена или устарела. Пожалуйста, загрузите файл заново.",
                reply_markup=None
            )
            return
        
        # Создаем клавиатуру с действиями заново
        keyboard = [
            [InlineKeyboardButton("📝 Показать транскрипцию", callback_data="action_show_transcription")],
            [InlineKeyboardButton("🧠 Обработать через ChatGPT", callback_data="action_process_gpt")],
            [InlineKeyboardButton("🔙 Вернуться", callback_data="back_to_language")]
        ]
        
        # Редактируем сообщение, возвращая меню действий
        await callback_query.message.edit_text(
            "✅ Транскрипция завершена!\n\n"
            "Выберите действие с полученным текстом:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
        # Подтверждаем нажатие кнопки
        await callback_query.answer("Возвращаюсь к выбору действий")
        
    except Exception as e:
        logger.error(f"Ошибка при возврате к выбору действий: {e}", exc_info=True)
        await callback_query.answer(f"Произошла ошибка: {str(e)}")

# Обработчик для навигации по страницам транскрипции
@app.on_callback_query(filters.regex(r"^page_(prev|next|info)$"))
async def handle_transcription_pagination(client, callback_query):
    try:
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        action = callback_query.data.split("_")[1]
        
        # Получаем страницы и текущую страницу из состояния
        pages = get_state(username).get("transcription_pages", [])
        current_page = get_state(username).get("current_page", 0)
        total_pages = len(pages)
        
        if not pages:
            await callback_query.answer("Информация о страницах не найдена")
            return
            
        # Обрабатываем действие
        if action == "prev":
            # Переходим на предыдущую страницу, если возможно
            if current_page > 0:
                current_page -= 1
                set_state(username, "current_page", current_page)
            else:
                await callback_query.answer("Вы уже на первой странице")
                return
        elif action == "next":
            # Переходим на следующую страницу, если возможно
            if current_page < total_pages - 1:
                current_page += 1
                set_state(username, "current_page", current_page)
            else:
                await callback_query.answer("Вы уже на последней странице")
                return
        elif action == "info":
            # Просто информационная кнопка, ничего не делаем
            await callback_query.answer(f"Страница {current_page + 1} из {total_pages}")
            return
            
        # Создаем клавиатуру с навигацией
        keyboard = []
        
        # Если страниц больше одной, добавляем кнопки навигации
        if total_pages > 1:
            nav_buttons = []
            nav_buttons.append(InlineKeyboardButton("◀️", callback_data="page_prev"))
            nav_buttons.append(InlineKeyboardButton(f"{current_page + 1}/{total_pages}", callback_data="page_info"))
            nav_buttons.append(InlineKeyboardButton("▶️", callback_data="page_next"))
            keyboard.append(nav_buttons)
        
        # Добавляем кнопку "Назад"
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_actions")])
        
        # Обновляем сообщение с текущей страницей
        await callback_query.message.edit_text(
            f"📝 **Результат транскрипции** (страница {current_page + 1}/{total_pages}):\n\n{pages[current_page]}",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
        # Подтверждаем нажатие кнопки
        if action != "info":
            await callback_query.answer(f"Страница {current_page + 1} из {total_pages}")
            
    except Exception as e:
        logger.error(f"Ошибка при навигации по страницам: {e}", exc_info=True)
        await callback_query.answer(f"Произошла ошибка: {str(e)}")

# Обработчик для навигации по страницам результата ChatGPT
@app.on_callback_query(filters.regex(r"^gpt_page_(prev|next|info)$"))
async def handle_gpt_pagination(client, callback_query):
    try:
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        action = callback_query.data.split("_")[2]
        
        # Получаем страницы и текущую страницу из состояния
        pages = get_state(username).get("gpt_result_pages", [])
        current_page = get_state(username).get("current_gpt_page", 0)
        total_pages = len(pages)
        
        if not pages:
            await callback_query.answer("Информация о страницах не найдена")
            return
            
        # Обрабатываем действие
        if action == "prev":
            # Переходим на предыдущую страницу, если возможно
            if current_page > 0:
                current_page -= 1
                set_state(username, "current_gpt_page", current_page)
            else:
                await callback_query.answer("Вы уже на первой странице")
                return
        elif action == "next":
            # Переходим на следующую страницу, если возможно
            if current_page < total_pages - 1:
                current_page += 1
                set_state(username, "current_gpt_page", current_page)
            else:
                await callback_query.answer("Вы уже на последней странице")
                return
        elif action == "info":
            # Просто информационная кнопка, ничего не делаем
            await callback_query.answer(f"Страница {current_page + 1} из {total_pages}")
            return
            
        # Создаем клавиатуру с навигацией
        keyboard = []
        
        # Если страниц больше одной, добавляем кнопки навигации
        if total_pages > 1:
            nav_buttons = []
            nav_buttons.append(InlineKeyboardButton("◀️", callback_data="gpt_page_prev"))
            nav_buttons.append(InlineKeyboardButton(f"{current_page + 1}/{total_pages}", callback_data="gpt_page_info"))
            nav_buttons.append(InlineKeyboardButton("▶️", callback_data="gpt_page_next"))
            keyboard.append(nav_buttons)
        
        # Добавляем кнопку "Назад"
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_actions")])
        
        # Обновляем сообщение с текущей страницей
        await callback_query.message.edit_text(
            f"🧠 **Результат обработки ChatGPT** (страница {current_page + 1}/{total_pages}):\n\n{pages[current_page]}",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
        # Подтверждаем нажатие кнопки
        if action != "info":
            await callback_query.answer(f"Страница {current_page + 1} из {total_pages}")
            
    except Exception as e:
        logger.error(f"Ошибка при навигации по страницам ChatGPT: {e}", exc_info=True)
        await callback_query.answer(f"Произошла ошибка: {str(e)}")

# Запуск бота
if __name__ == "__main__":
    try:
        print("Запуск бота...")
        app.run()
    except KeyboardInterrupt:
        print("\nОстановка бота...")
    except Exception as e:
        print(f"Критическая ошибка: {e}")
    finally:
        # Очищаем кэш при завершении
        clean_cache()
        print("Бот остановлен")