"""
Бот для транскрипции с использованием AssemblyAI API
с управлением кэшем, выбором языка для транскрипции
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

from typing import Tuple, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from pyrogram import Client, filters
from pyrogram.types import Message
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton

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

API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")
BOT_TOKEN = os.getenv("BOT_TOKEN")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CACHE_DIR = os.path.join(os.getcwd(), "cache")
AUDIO_FILES_DIR = os.path.join(CACHE_DIR, "audio_files")
SESSION_DIR = os.path.join(CACHE_DIR, "sessions")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(AUDIO_FILES_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)

# Создаем клиент OpenAI
openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

DEFAULT_PROMPT = "Суммируй следующий текст в краткой форме, выделяя основные мысли и ключевые моменты."

SUPPORTED_LANGUAGES = {
    "ru": "Русский🇷🇺",
    "en": "Английский🇬🇧",
}


user_prompts = {} 
user_current_prompts = {}  

MAX_STORED_PROMPTS = 5
user_states = {}

def clean_cache():
    """Очистка всех кэш-директорий"""
    print("Очистка кэша...")
    try:
        for dir_path in [AUDIO_FILES_DIR]:
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        
        user_states.clear()
        print("Кэш очищен")
    except Exception as e:
        print(f"Ошибка при очистке кэша: {e}")

def signal_handler(sig, frame):
    print("\nЗавершение работы и очистка кэша...")
    clean_cache()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

clean_cache()

app = Client(
    "assemblyai_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN,
    workdir=SESSION_DIR,
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
        if key and key in user_states[username]:
            del user_states[username][key]
        else:
            user_states[username] = {}

def get_audio_duration(file_path):
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
        logger.error(f"Ошибка при получении длительности аудиофайла: {e}")
        return 0

async def download_file(message: Message, file_id: str, save_path: str) -> Tuple[bool, str]:
    try:
        status_msg = await message.reply("⏳ Скачиваю файл...")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        await app.download_media(message=file_id, file_name=save_path)
        
        # Проверка, что файл существует и не пустой
        if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
            await status_msg.edit_text("❌ Ошибка: файл не скачался или пустой")
            return False, "Файл не скачался или пустой"
        
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        logger.info(f"Файл скачан: {save_path}, размер: {file_size_mb:.2f} МБ")
        
        await status_msg.edit_text("✅ Файл успешно загружен. Выберите язык для транскрипции:")
        
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
    Транскрибация аудио с помощью AssemblyAI API
    """
    try:
        if status_msg:
            await status_msg.edit_text("🔄 Отправка аудио на транскрипцию...")
        else:
            status_msg = await message.reply("🔄 Отправка аудио на транскрипцию...")
        
        start_time = time.time()
        
        file_size = os.path.getsize(audio_path)
        file_size_mb = file_size / (1024 * 1024)
        
        if file_size > 20 * 1024 * 1024:
            await status_msg.edit_text(f"⚙️ Файл слишком большой ({file_size_mb:.2f} МБ). Оптимизирую...")
            compressed_path = os.path.join(AUDIO_FILES_DIR, f"{uuid.uuid4()}.mp3")
            
            compress_cmd = [
                "ffmpeg",
                "-i", audio_path,
                "-c:a", "libmp3lame",
                "-b:a", "64k",
                compressed_path
            ]
            
            result = subprocess.run(compress_cmd, capture_output=True, text=True)
            
            if os.path.exists(compressed_path) and os.path.getsize(compressed_path) > 0:
                compressed_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
                logger.info(f"Файл сжат до {compressed_size_mb:.2f} МБ")
                audio_path = compressed_path
                await status_msg.edit_text(f"✅ Файл оптимизирован ({compressed_size_mb:.2f} МБ). Отправляю на транскрипцию...")
        
        headers = {
            "authorization": ASSEMBLYAI_API_KEY,
            "content-type": "application/json"
        }
        
        with open(audio_path, "rb") as f:
            response = requests.post(
                "https://api.assemblyai.com/v2/upload",
                headers=headers,
                data=f,
                timeout=300
            )
        
        if response.status_code != 200:
            logger.error(f"Ошибка загрузки файла: {response.text}")
            await status_msg.edit_text(f"❌ Ошибка при загрузке файла: {response.text}")
            return False, f"Ошибка при загрузке файла: {response.text}"
            
        upload_url = response.json()["upload_url"]
        logger.info(f"Файл успешно загружен, URL: {upload_url}")
   
        target_lang_name = SUPPORTED_LANGUAGES.get(target_language, target_language)
        await status_msg.edit_text(f"⏳ Файл загружен. Начинаю транскрипцию на {target_lang_name}...")
        
        data = {"audio_url": upload_url}
        
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
        
        last_update_time = time.time()
        
        while True:
            current_time = time.time()
            # Обновляем статус каждые 10 секунд
            if current_time - last_update_time >= 10:
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
            
            if status == "completed":
                detected_language = response.json().get("language_code", "неизвестен")
                detected_lang_name = SUPPORTED_LANGUAGES.get(detected_language, detected_language)
                
                transcription_text = response.json()["text"]
                
                await status_msg.edit_text(f"✅ Транскрипция завершена!\n"
                                        f"👂 Исходный язык аудио: {detected_lang_name}")
                
                return True, transcription_text
                
            elif status == "error":
                error_msg = response.json().get("error", "Неизвестная ошибка")
                logger.error(f"Ошибка транскрипции: {error_msg}")
                await status_msg.edit_text(f"❌ Ошибка при транскрипции: {error_msg}")
                return False, f"Ошибка транскрипции: {error_msg}"
            
            await asyncio.sleep(3)
    
    except Exception as e:
        logger.error(f"Ошибка при транскрипции: {e}", exc_info=True)
        if status_msg:
            await status_msg.edit_text(f"❌ Произошла ошибка: {str(e)}")
        else:
            await message.reply(f"❌ Произошла ошибка: {str(e)}")
        return False, f"Ошибка при транскрипции: {str(e)}"
    
    finally:
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"Удален временный файл: {audio_path}")
        except Exception as e:
            logger.error(f"Ошибка при удалении файла: {e}")

async def process_with_chatgpt(text, prompt, message):
    try:
        if not openai_client:
            return False, "❌ Ошибка: API ключ OpenAI не настроен в конфигурации бота."
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompt}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": text}]
            }
        ]
        
        logger.info(f"Отправляю запрос в OpenAI с новым форматом")
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        
        result = response.choices[0].message.content

        return True, result
    
    except Exception as e:
        logger.error(f"Ошибка при вызове ChatGPT API: {e}", exc_info=True)
        return False, f"Ошибка при обработке текста: {str(e)}"

async def show_language_selection(message: Message, file_path: str) -> None:

    username = message.from_user.username or str(message.from_user.id)
    
    set_state(username, "file_path", file_path)
    
    keyboard = []
    
    for lang_code, lang_name in SUPPORTED_LANGUAGES.items():
        keyboard.append([
            InlineKeyboardButton(lang_name, callback_data=f"lang_{lang_code}")
        ])
    
    keyboard.append([
        InlineKeyboardButton("🔙 Отмена", callback_data="cancel_transcription")
    ])
    
    await message.reply(
        "Выберите язык для транскрипции:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

def get_user_prompts(username):
    """Получить список предыдущих промптов пользователя"""
    return user_prompts.get(username, [])

def add_user_prompt(username, prompt):
    """Добавить новый промпт в историю пользователя"""
    if username not in user_prompts:
        user_prompts[username] = []
    
    # Проверяем, есть ли уже такой промпт в истории
    if prompt in user_prompts[username]:
        # Если есть - удаляем его, чтобы потом добавить в конец (как самый свежий)
        user_prompts[username].remove(prompt)
    
    # Добавляем промпт в конец списка
    user_prompts[username].append(prompt)
    
    # Ограничиваем количество хранимых промптов
    if len(user_prompts[username]) > MAX_STORED_PROMPTS:
        user_prompts[username] = user_prompts[username][-MAX_STORED_PROMPTS:]
    
    # Устанавливаем текущий промпт
    user_current_prompts[username] = prompt

def set_user_prompt(username, prompt):
    """Установить текущий промпт пользователя"""
    user_current_prompts[username] = prompt
    # Также добавляем его в историю
    add_user_prompt(username, prompt)

def get_current_prompt(username):
    """Получить текущий промпт пользователя"""
    return user_current_prompts.get(username, "")

@app.on_callback_query(filters.regex(r"^lang_(.+)$"))
async def handle_language_selection(client, callback_query):
    try:
        language_code = callback_query.data.split("_")[1]
        
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        file_path = get_state(username).get("file_path")
        
        if not file_path or not os.path.exists(file_path):
            await callback_query.answer("Файл не найден, загрузите файл заново")
            return
        
        set_state(username, "language", language_code)
        
        lang_name = SUPPORTED_LANGUAGES.get(language_code, "оригинальный")
        await callback_query.answer(f"Выбран язык: {lang_name}")
        
        status_msg = await callback_query.message.edit_text(
            f"🔤 Выбран язык: {lang_name}\n⏳ Начинаю транскрипцию...",
            reply_markup=None
        )
        
        success, transcription = await transcribe_with_assemblyai(
            file_path, 
            callback_query.message, 
            status_msg, 
            language_code
        )
        
        if success:
            set_state(username, "transcription", transcription)
            
            keyboard = [
                [InlineKeyboardButton("📝 Показать транскрипцию", callback_data="show_transcription")],
                [InlineKeyboardButton("🧠 Обработать через ChatGPT", callback_data="process_gpt")],
                [InlineKeyboardButton("🔙 В главное меню", callback_data="back_to_menu")]
            ]
            
            await callback_query.message.reply(
                "✅ Транскрипция завершена!\nВыберите действие с полученным текстом:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
            logger.info(f"Транскрипция успешно получена для {username}, длина: {len(transcription)} символов")
        else:
            await callback_query.message.reply(f"❌ Не удалось выполнить транскрипцию: {transcription}")
            
            clear_state(username)
                
    except Exception as e:
        logger.error(f"Ошибка при обработке выбора языка: {e}", exc_info=True)
        await callback_query.answer(f"Произошла ошибка: {str(e)}")
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        clear_state(username)

@app.on_callback_query(filters.regex(r"^show_transcription$"))
async def show_transcription(client, callback_query):
    try:
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        
        transcription = get_state(username).get("transcription")
        
        if not transcription:
            await callback_query.answer("Транскрипция не найдена")
            return
        
        page_size = 3000
        pages = [transcription[i:i+page_size] for i in range(0, len(transcription), page_size)]
        total_pages = len(pages)
        
        set_state(username, "pages", pages)
        set_state(username, "current_page", 0)
        
        keyboard = []
        
        if total_pages > 1:
            nav_buttons = []
            nav_buttons.append(InlineKeyboardButton("◀️", callback_data="prev_page"))
            nav_buttons.append(InlineKeyboardButton(f"1/{total_pages}", callback_data="page_info"))
            nav_buttons.append(InlineKeyboardButton("▶️", callback_data="next_page"))
            keyboard.append(nav_buttons)
        
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")])
        
        await callback_query.message.edit_text(
            f"📝 **Результат транскрипции** (страница 1/{total_pages}):\n\n{pages[0]}",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
        await callback_query.answer("Показываю транскрипцию")
        
    except Exception as e:
        logger.error(f"Ошибка при показе транскрипции: {e}", exc_info=True)
        await callback_query.answer(f"Произошла ошибка: {str(e)}")

@app.on_callback_query(filters.regex(r"^(prev|next)_page$"))
async def navigate_pages(client, callback_query):
    try:
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        action = callback_query.data.split("_")[0]  # prev или next
        
        # Получаем страницы и текущую страницу
        pages = get_state(username).get("pages", [])
        current_page = get_state(username).get("current_page", 0)
        total_pages = len(pages)
        
        if not pages:
            await callback_query.answer("Информация о страницах не найдена")
            return
            
        if action == "prev":
            if current_page > 0:
                current_page -= 1
                set_state(username, "current_page", current_page)
            else:
                await callback_query.answer("Вы уже на первой странице")
                return
        elif action == "next":
            if current_page < total_pages - 1:
                current_page += 1
                set_state(username, "current_page", current_page)
            else:
                await callback_query.answer("Вы уже на последней странице")
                return
            
        keyboard = []
        
        if total_pages > 1:
            nav_buttons = []
            nav_buttons.append(InlineKeyboardButton("◀️", callback_data="prev_page"))
            nav_buttons.append(InlineKeyboardButton(f"{current_page + 1}/{total_pages}", callback_data="page_info"))
            nav_buttons.append(InlineKeyboardButton("▶️", callback_data="next_page"))
            keyboard.append(nav_buttons)
        
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")])
        
        await callback_query.message.edit_text(
            f"📝 **Результат транскрипции** (страница {current_page + 1}/{total_pages}):\n\n{pages[current_page]}",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
        await callback_query.answer(f"Страница {current_page + 1} из {total_pages}")
    
    except Exception as e:
        logger.error(f"Ошибка при навигации по страницам: {e}", exc_info=True)
        await callback_query.answer(f"Произошла ошибка: {str(e)}")

@app.on_callback_query(filters.regex(r"^page_info$"))
async def page_info(client, callback_query):
    try:
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        
        pages = get_state(username).get("pages", [])
        current_page = get_state(username).get("current_page", 0)
        total_pages = len(pages)
        
        await callback_query.answer(f"Страница {current_page + 1} из {total_pages}")
    except Exception as e:
        logger.error(f"Ошибка при показе информации о странице: {e}", exc_info=True)
        await callback_query.answer("Произошла ошибка")

@app.on_callback_query(filters.regex(r"^process_gpt$"))
async def process_gpt(client, callback_query):
    try:
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        
        transcription = get_state(username).get("transcription")
        
        if not transcription:
            await callback_query.answer("Транскрипция не найдена")
            return
            
        if not openai_client:
            await callback_query.answer("ChatGPT недоступен")
            await callback_query.message.edit_text(
                "❌ Обработка через ChatGPT недоступна: API ключ не настроен.\n"
                "Обратитесь к администратору бота.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]
                ])
            )
            return
        
        # Получаем промпт пользователя или используем стандартный
        prompt = user_prompts.get(username, DEFAULT_PROMPT)
        
        # Показываем текущий промпт и предлагаем действия
        keyboard = [
            [InlineKeyboardButton("✅ Отправить на обработку", callback_data="start_gpt_process")],
            [InlineKeyboardButton("✏️ Изменить промпт", callback_data="change_prompt")],
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]
        ]
        
        await callback_query.message.edit_text(
            "🧠 **Подготовка к обработке через ChatGPT**\n\n"
            f"📝 **Текущий промпт:**\n`{prompt}`\n\n"
            "Выберите действие:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
        await callback_query.answer("Подготовка к обработке")
        
    except Exception as e:
        logger.error(f"Ошибка при подготовке к обработке через ChatGPT: {e}", exc_info=True)
        await callback_query.answer(f"Произошла ошибка: {str(e)}")

def get_main_keyboard(username):
    """Возвращает основную клавиатуру для обработки промптов"""
    # Получаем текущий промпт пользователя
    prompt = get_current_prompt(username)
    
    # Короткое отображение промпта (если он есть)
    prompt_display = None
    if prompt:
        prompt_display = prompt[:25] + "..." if len(prompt) > 25 else prompt
    
    # Создаем клавиатуру
    keyboard = [
        [InlineKeyboardButton("✅ Отправить на обработку", callback_data="start_gpt_process")],
        [InlineKeyboardButton("✏️ Изменить промпт", callback_data="change_prompt")],
        [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]
    ]
    
    # Если есть текущий промпт, показываем его в первой строке
    if prompt_display:
        keyboard.insert(0, [InlineKeyboardButton(f"📝 Текущий промпт: {prompt_display}", callback_data="show_current_prompt")])
    
    return InlineKeyboardMarkup(keyboard)

@app.on_callback_query(filters.regex(r"^change_prompt$"))
async def change_prompt(client, callback_query):
    try:
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        set_state(username, "awaiting_prompt", True)
        previous_prompts = get_user_prompts(username)
        keyboard = [[InlineKeyboardButton("🔙 Отмена", callback_data="back_to_gpt")]]
        
        for i, prompt in enumerate(previous_prompts):
            display_prompt = (prompt[:40] + "...") if len(prompt) > 40 else prompt
            keyboard.append([InlineKeyboardButton(f"📜 {display_prompt}", callback_data=f"use_prompt_{i}")])
        
        await callback_query.message.edit_text(
            "📝 **Введите новый промпт**\n\n"
            "Отправьте текст вашего нового промпта в следующем сообщении.\n\n"
            "Примеры промптов:\n"
            "• `Суммируй этот текст кратко, выделяя ключевые моменты.`\n"
            "• `Преобразуй этот транскрипт в четкий конспект с заголовками.`\n"
            "• `Извлеки из текста список действий и задач.`\n\n"
            + (("Или выберите один из ваших предыдущих промптов:") if previous_prompts else ""),
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        await callback_query.answer("Введите новый промпт" + (" или выберите из истории" if previous_prompts else ""))
    except Exception as e:
        logger.error(f"Ошибка при изменении промпта: {e}", exc_info=True)
        await callback_query.answer(f"Произошла ошибка: {str(e)}")

@app.on_callback_query(filters.regex(r"^use_prompt_(\d+)$"))
async def use_previous_prompt(client, callback_query):
    try:
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        prompt_index = int(callback_query.data.split("_")[-1])
        
        # Получаем промпты пользователя
        previous_prompts = get_user_prompts(username)
        
        # Проверяем, что индекс в пределах списка
        if 0 <= prompt_index < len(previous_prompts):
            selected_prompt = previous_prompts[prompt_index]
            
            # Устанавливаем выбранный промпт как текущий
            set_user_prompt(username, selected_prompt)
            
            # Возвращаемся к основному интерфейсу
            await callback_query.message.edit_text(
                f"✅ Промпт выбран:\n\n`{selected_prompt}`\n\nТеперь отправьте текст для обработки.",
                reply_markup=get_main_keyboard(username)  # Предполагается, что у вас есть эта функция
            )
            await callback_query.answer("Промпт успешно выбран")
        else:
            await callback_query.answer("Промпт не найден")
    except Exception as e:
        logger.error(f"Ошибка при выборе предыдущего промпта: {e}", exc_info=True)
        await callback_query.answer(f"Произошла ошибка: {str(e)}")

@app.on_callback_query(filters.regex(r"^start_gpt_process$"))
async def start_gpt_process(client, callback_query):
    try:
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        
        transcription = str(get_state(username).get("transcription"))
        
        if not transcription:
            await callback_query.answer("Транскрипция не найдена")
            return
            
        prompt = str(user_prompts.get(username, DEFAULT_PROMPT))
        
        status_msg = await callback_query.message.edit_text(
            "🧠 Обрабатываю текст с помощью ChatGPT...",
            reply_markup=None
        )
        
        success, response = await process_with_chatgpt(transcription, prompt, callback_query.message)
        
        await status_msg.edit_text("✅ Обработка через ChatGPT завершена!")
        
        if not success:
            await callback_query.message.reply(
                f"❌ {response}",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]
                ])
            )
            return
        
        page_size = 3000
        pages = [response[i:i+page_size] for i in range(0, len(response), page_size)]
        total_pages = len(pages)
        
        set_state(username, "gpt_pages", pages)
        set_state(username, "current_gpt_page", 0)
        
        keyboard = []
        
        if total_pages > 1:
            nav_buttons = []
            nav_buttons.append(InlineKeyboardButton("◀️", callback_data="prev_gpt_page"))
            nav_buttons.append(InlineKeyboardButton(f"1/{total_pages}", callback_data="gpt_page_info"))
            nav_buttons.append(InlineKeyboardButton("▶️", callback_data="next_gpt_page"))
            keyboard.append(nav_buttons)
        
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")])
        
        await callback_query.message.reply(
            f"🧠 Страница 1/{total_pages}:\n\n{pages[0]}",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
    except Exception as e:
        logger.error(f"Ошибка при обработке текста через ChatGPT: {e}", exc_info=True)
        await callback_query.answer(f"Произошла ошибка: {str(e)}")
        
        await callback_query.message.edit_text(
            f"❌ Произошла ошибка при обработке текста: {str(e)}",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]
            ])
        )

@app.on_callback_query(filters.regex(r"^(prev|next)_gpt_page$"))
async def navigate_gpt_pages(client, callback_query):
    try:
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        action = callback_query.data.split("_")[0]  # prev или next
        
        # Получаем страницы и текущую страницу
        pages = get_state(username).get("gpt_pages", [])
        current_page = get_state(username).get("current_gpt_page", 0)
        total_pages = len(pages)
        
        if not pages:
            await callback_query.answer("Информация о страницах не найдена")
            return
            
        # Обрабатываем действие
        if action == "prev":
            if current_page > 0:
                current_page -= 1
                set_state(username, "current_gpt_page", current_page)
            else:
                await callback_query.answer("Вы уже на первой странице")
                return
        elif action == "next":
            if current_page < total_pages - 1:
                current_page += 1
                set_state(username, "current_gpt_page", current_page)
            else:
                await callback_query.answer("Вы уже на последней странице")
                return
            
        # Создаем клавиатуру с навигацией
        keyboard = []
        
        # Если страниц больше одной, добавляем кнопки навигации
        if total_pages > 1:
            nav_buttons = []
            nav_buttons.append(InlineKeyboardButton("◀️", callback_data="prev_gpt_page"))
            nav_buttons.append(InlineKeyboardButton(f"{current_page + 1}/{total_pages}", callback_data="gpt_page_info"))
            nav_buttons.append(InlineKeyboardButton("▶️", callback_data="next_gpt_page"))
            keyboard.append(nav_buttons)
        
        # Добавляем кнопку "Назад"
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")])
        
        # Обновляем сообщение с текущей страницей
        await callback_query.message.edit_text(
            f"🧠 **Результат обработки ChatGPT** (страница {current_page + 1}/{total_pages}):\n\n{pages[current_page]}",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
        # Подтверждаем нажатие кнопки
        await callback_query.answer(f"Страница {current_page + 1} из {total_pages}")
    
    except Exception as e:
        logger.error(f"Ошибка при навигации по страницам ChatGPT: {e}", exc_info=True)
        await callback_query.answer(f"Произошла ошибка: {str(e)}")

@app.on_callback_query(filters.regex(r"^gpt_page_info$"))
async def gpt_page_info(client, callback_query):
    try:
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        
        # Получаем информацию о страницах
        pages = get_state(username).get("gpt_pages", [])
        current_page = get_state(username).get("current_gpt_page", 0)
        total_pages = len(pages)
        
        await callback_query.answer(f"Страница {current_page + 1} из {total_pages}")
    except Exception as e:
        logger.error(f"Ошибка при показе информации о странице: {e}", exc_info=True)
        await callback_query.answer("Произошла ошибка")

@app.on_callback_query(filters.regex(r"^back_to_gpt$"))
async def back_to_gpt(client, callback_query):
    try:
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        
        # Сбрасываем флаг ожидания промпта
        set_state(username, "awaiting_prompt", False)
        
        # Получаем промпт пользователя
        prompt = user_prompts.get(username, DEFAULT_PROMPT)
        
        # Показываем текущий промпт и предлагаем действия
        keyboard = [
            [InlineKeyboardButton("✅ Отправить на обработку", callback_data="start_gpt_process")],
            [InlineKeyboardButton("✏️ Изменить промпт", callback_data="change_prompt")],
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]
        ]
        
        await callback_query.message.edit_text(
            "🧠 **Подготовка к обработке через ChatGPT**\n\n"
            f"📝 **Текущий промпт:**\n`{prompt}`\n\n"
            "Выберите действие:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
        await callback_query.answer("Вернулся к настройкам обработки")
        
    except Exception as e:
        logger.error(f"Ошибка при возврате к GPT: {e}", exc_info=True)
        await callback_query.answer(f"Произошла ошибка: {str(e)}")

@app.on_callback_query(filters.regex(r"^back_to_menu$"))
async def back_to_menu(client, callback_query):
    try:
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        clear_state(username)
        
        await show_main_menu(client, callback_query.message, username)
        await callback_query.answer("Вернулся в главное меню")
        
    except Exception as e:
        logger.error(f"Ошибка при возврате в меню: {e}", exc_info=True)
        await callback_query.answer(f"Произошла ошибка: {str(e)}")
        
        # В случае ошибки возвращаемся в главное меню
        await show_main_menu(client, callback_query.message, username)

@app.on_callback_query(filters.regex(r"^cancel_transcription$"))
async def cancel_transcription(client, callback_query):
    try:
        username = callback_query.from_user.username or str(callback_query.from_user.id)
        
        # Очищаем состояние пользователя
        clear_state(username)
        
        # Возвращаемся в главное меню
        await show_main_menu(client, callback_query.message, username)
        
        await callback_query.answer("Транскрипция отменена")
        
    except Exception as e:
        logger.error(f"Ошибка при отмене транскрипции: {e}", exc_info=True)
        await callback_query.answer(f"Произошла ошибка: {str(e)}")

async def show_main_menu(client, message, username):
    """Показывает главное меню"""
    prompt = user_prompts.get(username, DEFAULT_PROMPT)
    
    keyboard = [
        [InlineKeyboardButton("❓ Помощь", callback_data="show_help")]
    ]
    
    try:
        await message.edit_text(
            "👋 Привет! Я бот для транскрипции аудио с возможностью обработки результата через ChatGPT.\n\n"
            "📱 **Что я умею:**\n"
            "• Автоматически определять язык аудио\n"
            "• Переводить текст на разные языки\n"
            "• Обрабатывать полученный текст через ChatGPT\n\n"
            "🎧 Отправьте мне голосовое сообщение или аудиофайл, чтобы начать.\n\n",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    except Exception:
        await message.reply(
            "👋 Привет! Я бот для транскрипции аудио с возможностью обработки результата через ChatGPT.\n\n"
            "📱 **Что я умею:**\n"
            "• Автоматически определять язык аудио\n"
            "• Переводить текст на разные языки\n"
            "• Обрабатывать полученный текст через ChatGPT\n\n"
            "🎧 Отправьте мне голосовое сообщение или аудиофайл, чтобы начать.\n\n",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

@app.on_callback_query(filters.regex(r"^show_help$"))
async def show_help(client, callback_query):
    try:
        languages_list = "\n".join([f"• {name}" for code, name in SUPPORTED_LANGUAGES.items()])
        keyboard = [
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]
        ]
        
        await callback_query.message.edit_text(
            "📋 **Как пользоваться этим ботом:**\n\n"
            "1. Отправьте голосовое сообщение или аудиофайл (MP3, OGG, WAV и др.)\n"
            "2. Выберите язык для транскрипции\n"
            "3. После обработки вы сможете:\n"
            "   - Просто получить текст транскрипции\n"
            "   - Отправить текст в ChatGPT с вашим собственным промптом\n\n"
            "**Команды бота:**\n"
            "/start - начать работу с ботом\n"
            "/help - показать эту справку\n\n"
            "**Поддерживаемые языки для транскрипции:**\n"
            f"{languages_list}",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
        await callback_query.answer("Открываю справку")
        
    except Exception as e:
        logger.error(f"Ошибка при показе справки: {e}", exc_info=True)
        await callback_query.answer(f"Произошла ошибка: {str(e)}")

@app.on_message(filters.command("start"))
async def start_command(client, message):
    username = message.from_user.username or str(message.from_user.id)
    await show_main_menu(client, message, username)

@app.on_message(filters.command("help"))
async def help_command(client, message):
    languages_list = "\n".join([f"• {name}" for code, name in SUPPORTED_LANGUAGES.items()])
    
    await message.reply(
        "📋 **Как пользоваться этим ботом:**\n\n"
        "1. Отправьте голосовое сообщение или аудиофайл (MP3, OGG, WAV и др.)\n"
        "2. Выберите язык для транскрипции\n"
        "3. После обработки вы сможете:\n"
        "   - Просто получить текст транскрипции\n"
        "   - Отправить текст в ChatGPT с вашим собственным промптом\n\n"
        "**Поддерживаемые языки для транскрипции:**\n"
        f"{languages_list}\n\n"
        "Бот использует технологию AssemblyAI для распознавания речи и ChatGPT для обработки текста."
    )

@app.on_message(filters.text)
async def handle_text(client, message):
    if message.text.startswith('/'):
        return
    
    username = message.from_user.username or str(message.from_user.id)
    
    if get_state(username).get("awaiting_prompt", False):
        new_prompt = message.text
        add_user_prompt(username, new_prompt)
        set_state(username, "awaiting_prompt", False)

        await message.reply(
            f"✅ Ваш промпт успешно сохранен!\n\n"
            f"📝 **Новый промпт:**\n`{new_prompt}`",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Вернуться к ChatGPT", callback_data="back_to_gpt")]
            ])
        )
        return
    
    await message.reply(
        "Пожалуйста, отправьте мне голосовое сообщение или аудиофайл для транскрипции.\n"
        "Используйте /help для получения справки."
    )

@app.on_message(filters.audio | filters.voice | filters.document)
async def handle_audio(client, message):
    """Обработчик аудиосообщений, голосовых сообщений и документов"""
    try:
        username = message.from_user.username or str(message.from_user.id)
        
        if message.document and not message.document.mime_type.startswith("audio/"):
            file_ext = os.path.splitext(message.document.file_name)[1].lower() if message.document.file_name else ""
            if file_ext not in [".mp3", ".ogg", ".wav", ".flac", ".m4a", ".aac"]:
                await message.reply("Этот документ не похож на аудиофайл. Отправьте аудиофайл для транскрипции.")
                return
        
        if get_state(username).get("transcribing", False):
            await message.reply(
                "У вас уже есть активная задача транскрипции. "
                "Дождитесь ее завершения или отправьте новый файл для обработки."
            )
            return
        
        set_state(username, "transcribing", True)
        
        file_id = None
        file_name = None
        
        if message.audio:
            file_id = message.audio.file_id
            file_name = message.audio.file_name or f"audio_{message.id}.mp3"
        elif message.voice:
            file_id = message.voice.file_id
            file_name = f"voice_{message.id}.ogg"
        elif message.document:
            file_id = message.document.file_id
            file_name = message.document.file_name or f"document_{message.id}"
        else:
            await message.reply("Пожалуйста, отправьте аудиофайл.")
            return
        
        file_name = f"{int(time.time())}_{file_name}"
        save_path = os.path.join(AUDIO_FILES_DIR, file_name)
        
        # Скачивание файла
        success, result = await download_file(message, file_id, save_path)
        
        if success:
            await show_language_selection(message, save_path)
        else:
            await message.reply(f"❌ Ошибка при скачивании файла: {result}")
            clear_state(username)
    
    except Exception as e:
        logger.error(f"Ошибка обработки аудиофайла: {e}", exc_info=True)
        await message.reply(f"❌ Произошла ошибка при обработке аудиофайла: {str(e)}")
        clear_state(username)

if __name__ == "__main__":
    try:
        print("Запуск бота...")
        app.run()
    except KeyboardInterrupt:
        print("\nОстановка бота...")
    except Exception as e:
        print(f"Критическая ошибка: {e}")
    finally:
        clean_cache()
        print("Бот остановлен")