import base64
import logging
import asyncio
import time
import aiohttp
import requests
from dataclasses import dataclass
from typing import Optional, Dict
from contextlib import suppress
from functools import wraps
import uuid

import g4f
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import FSInputFile
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage
from g4f.client import Client as G4FClient
import random
from config import BOT_TOKEN, OLLAMA_TOKEN, OLLAMA_HOST

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BotConfig:
    """Конфигурация бота"""
    MAX_RESPONSE_WORDS = 150
    MAX_ATTEMPTS = 3
    REQUEST_TIMEOUT = 30
    IMAGE_TIMEOUT = 10
    MAX_CONCURRENT_REQUESTS = 20

    SYSTEM_PROMPT = (
        "Ты - Владимир Вольфович Жириновский! Отвечай эмоционально, с сарказмом и характерной экспрессией. "
        "Используй острые политические комментарии, исторические аналогии и эмоциональные восклицания. "
        "Так же, твои сообщения должны быть не длинее 150 слов. Быстро и чётко "
        "Вам будут даны 2 сообщения, 1-е сообщение на которое ответили, и 2-е сообщение, ответ на 1-е сообщение, "
        "с обращением к вам, обычно какая-то прозыба или вопрос, вы должны ответить КАК ЖИРИНОВСКИЙ "
        "второму пользователю (что отправил 2-е сообщение) "
        "Пиши сразу как жириновский, будто ты уже и есть сам жириновский "
        "Не используй смайлики "
        "Отвечай максимально агрессивно по отношению ко всем"
        "По желанию цитируй фразы пользователей, используются"
        "Parse_mode: HTML - это значит что стиль "
)


@dataclass
class ConversationContext:
    """Контекст разговора"""
    original_sender: str
    reply_sender: str
    original_text: str
    reply_text: str
    chat_id: int
    message_id: int
    request_id: str
    original_message: types.Message = None
    has_image: bool = False
    image_description: str = ""


class AIClient:
    """Клиент для работы с ИИ"""

    def __init__(self, ollama_host: str, ollama_token: str):
        self.ollama_host = ollama_host
        self.ollama_token = ollama_token

    async def analyze_image(self, image_data: bytes, request_id: str) -> Optional[str]:
        """Анализирует изображение через Ollama"""
        try:
            image_base64 = base64.b64encode(image_data).decode('utf-8')

            payload = {
                "model": "qwen3-vl:235b",
                "prompt": "Опиши это изображение максимально подробно на русском языке: что изображено, цвета, надписи, детали, эмоции если есть люди",
                "stream": False,
                "images": [image_base64],
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 1500,
                    "num_ctx": 4096
                }
            }

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.ollama_token}'
            }

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.post(
                    f"{self.ollama_host}/api/generate",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
            )

            print(f"Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print("-" * 10)
                print(f"Done reason: {data.get('done_reason')}")
                print(f"Eval count: {data.get('eval_count')}")

                # Сначала пробуем взять thinking, если есть и не пустой
                thinking = data.get('thinking', '').strip()
                if thinking:
                    print(f"Thinking length: {len(thinking)}")
                    return thinking

                # Если thinking пустой, берем response
                response_text = data.get('response', '').strip()
                print(f"Response length: {len(response_text)}")
                return response_text

            print(f"Error: {response.text}")
            return None

        except Exception as e:
            logger.warning(f"[{request_id}] Ошибка анализа изображения: {e}")
            return None
    async def generate_text_response(self, prompt: str, request_id: str) -> Optional[str]:
        """Генерирует текстовый ответ через g4f"""
        try:
            logger.info(f"[{request_id}] Генерируем текст через g4f")

            client = G4FClient()

            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model="",
                        messages=[{"role": "user", "content": prompt}],
                        timeout=30
                    )

                    result = response.choices[0].message.content
                    if result and len(result.split()) < 200:
                        logger.info(f"[{request_id}] ✅ Ответ от g4f получен")
                        return result

                except Exception as e:
                    logger.warning(f"[{request_id}] g4f попытка {attempt + 1} ошибка: {e}")
                    if attempt < 2:
                        await asyncio.sleep(1)

            return None

        except Exception as e:
            logger.error(f"[{request_id}] g4f критическая ошибка: {e}")
            return None

class RequestManager:
    """Менеджер параллельных запросов"""

    def __init__(self, max_concurrent: int = 5):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_requests: Dict[str, asyncio.Task] = {}
        self.lock = asyncio.Lock()

    async def add_request(self, request_id: str, task: asyncio.Task):
        """Добавить активный запрос"""
        async with self.lock:
            self.active_requests[request_id] = task

    async def remove_request(self, request_id: str):
        """Удалить завершенный запрос"""
        async with self.lock:
            if request_id in self.active_requests:
                del self.active_requests[request_id]

    async def get_active_count(self) -> int:
        """Получить количество активных запросов"""
        async with self.lock:
            return len(self.active_requests)

    async def process_request(self, request_id: str, coro):
        """Обработать запрос с ограничением параллелизма"""
        async with self.semaphore:
            task = asyncio.create_task(coro)
            await self.add_request(request_id, task)
            try:
                return await task
            finally:
                await self.remove_request(request_id)


class MessageProcessor:
    """Процессор сообщений"""

    def __init__(self):
        self.trigger_phrases = [
            "жириновский", "ввж", "володя",
            "владимир вольфович", "жирик", "зириновский"
        ]

    def should_respond(self, message: types.Message, bot_username: str) -> bool:
        """Определить, должен ли бот отвечать на сообщение"""
        if not message.text:
            return False

        message_text = message.text.lower()
        triggers = self.trigger_phrases + [f"@{bot_username}"]

        return any(phrase in message_text for phrase in triggers)


def log_execution_time(func):
    """Декоратор для логирования времени выполнения"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"[{request_id}] Начало выполнения {func.__name__}")

        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"[{request_id}] {func.__name__} выполнена за {execution_time:.2f}с")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"[{request_id}] {func.__name__} завершена с ошибкой за {execution_time:.2f}с: {e}")
            raise

    return wrapper


# Инициализация компонентов
config = BotConfig()
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode='HTML'))
dp = Dispatcher(storage=MemoryStorage())
request_manager = RequestManager(max_concurrent=config.MAX_CONCURRENT_REQUESTS)
processor = MessageProcessor()
ai_client = AIClient(OLLAMA_HOST, OLLAMA_TOKEN)


async def analyze_image(image_url: str, request_id: str) -> Optional[str]:
    """Анализирует изображение и возвращает описание"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url, timeout=config.IMAGE_TIMEOUT) as response:
                if response.status == 200:
                    image_data = await response.read()

                    # ТОЛЬКО Ollama для изображений
                    ollama_description = await ai_client.analyze_image(image_data, request_id)
                    return ollama_description

        return None

    except Exception as e:
        logger.error(f"[{request_id}] Ошибка анализа изображения: {e}")
        return None


async def get_image_url(message: types.Message, request_id: str) -> Optional[str]:
    """Получить URL изображения из сообщения"""
    try:
        if message.photo:
            file = await bot.get_file(message.photo[-1].file_id)
            image_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file.file_path}"
            logger.info(f"[{request_id}] Получен URL изображения: {image_url}")
            return image_url
        return None
    except Exception as e:
        logger.error(f"[{request_id}] Ошибка получения URL изображения: {e}")
        return None


def build_conversation_context(original: types.Message, reply: types.Message, request_id: str) -> ConversationContext:
    """Построить контекст разговора"""
    original_sender = original.from_user.full_name
    reply_sender = reply.from_user.full_name
    original_text = original.text or original.caption or ""
    reply_text = reply.text or reply.caption or ""

    return ConversationContext(
        original_sender=original_sender,
        reply_sender=reply_sender,
        original_text=original_text,
        reply_text=reply_text,
        chat_id=reply.chat.id,
        message_id=reply.message_id,
        request_id=request_id,
        original_message=original
    )


@log_execution_time
async def generate_jirinovsky_response(context: ConversationContext) -> str:
    """Генерирует ответ в стиле Жириновского"""
    try:
        image_analysis = ""

        if context.original_message and (context.original_message.photo or context.original_message.document):
            logger.info(f"[{context.request_id}] Обнаружено изображение, начинаем анализ")

            image_url = await get_image_url(context.original_message, context.request_id)
            if image_url:
                image_desc = await analyze_image(image_url, context.request_id)
                if image_desc:
                    image_analysis = f"\n[На изображении: {image_desc}]"
                    logger.info(f"[{context.request_id}] Изображение успешно проанализировано")
                else:
                    logger.warning(f"[{context.request_id}] Не удалось получить описание изображения")
                    image_analysis = "\n[Изображение не удалось проанализировать]"

        # Формируем промпт для ИИ
        prompt = (
            f"{config.SYSTEM_PROMPT}\n\n"
            f"КОНТЕКСТ БЕСЕДЫ:\n"
            f"1. Оригинальное сообщение от {context.original_sender}: \"{context.original_text}\""
        )

        if image_analysis:
            prompt += f"{image_analysis}"

        prompt += (
            f"\n2. Ответ от {context.reply_sender} (тебе): \"{context.reply_text}\"\n\n"
            f"ТВОЯ ЗАДАЧА: Ответь {context.reply_sender} в стиле Жириновского! Будь эмоционален, саркастичен и агрессивен!"
        )

        logger.info(f"[{context.request_id}] Генерируем ответ для {context.reply_sender}")

        # ТОЛЬКО g4f для текста
        response = await ai_client.generate_text_response(prompt, context.request_id)

        if response:
            logger.info(f"[{context.request_id}] ✅ Успешно сгенерирован ответ")
            return response

        logger.warning(f"[{context.request_id}] Все попытки провалились")
        return f"{context.reply_sender}, дорогой! Сейчас не до твоих вопросов! Система перегружена провокаторами!"

    except Exception as e:
        logger.error(f"[{context.request_id}] Критическая ошибка генерации: {e}")
        return "Эх, система дала сбой! Провокаторы нажимают не на те кнопки! Попробуй позже!"


async def process_message_with_context(context: ConversationContext):
    """Асинхронная обработка сообщения с контекстом"""
    try:
        # Отправляем статус "печатает"
        await bot.send_chat_action(context.chat_id, "typing")

        # Генерируем ответ
        response = await generate_jirinovsky_response(context)

        # Отправляем ответ
        await bot.send_message(
            chat_id=context.chat_id,
            text=response,
            reply_to_message_id=context.message_id
        )

        logger.info(f"[{context.request_id}] Успешно отправлен ответ пользователю {context.reply_sender}")

    except Exception as e:
        logger.error(f"[{context.request_id}] Ошибка отправки ответа: {e}")
        with suppress(Exception):
            await bot.send_message(
                chat_id=context.chat_id,
                text="Пфф! Не могу ответить! Система загружена провокациями!",
                reply_to_message_id=context.message_id
            )


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    """Обработчик команды /start"""

    await message.answer(
        f"Дорогой! Я бот-Жириновский!"
        "Добавь меня в группу и упомяни в ответе на сообщение - "
        "и я дам свой острый комментарий!"
    )


@dp.message(Command("status"))
async def cmd_status(message: types.Message):
    """Показать статус бота"""
    active_requests = await request_manager.get_active_count()

    await message.answer(
        f"Дорогой! Я работаю на полную мощность!\n"
        f"Активных запросов: {active_requests}\n"
        f"Готов отвечать всем желающим получить порцию правды!"
    )


@dp.message(Command("ptichko"))
async def cmd_ptichko(message: types.Message):
    """Обработчик команды /ptichko"""
    try:
        rand_num = random.randint(1, 10)
        if rand_num >= 9:
            photo = FSInputFile("explode.jpg")
            await message.answer_photo(photo, caption="Птичка взорвалась! Видимо, ты её взбесил, поганец")
        else:
            photo = FSInputFile("images.jpg")
            await message.answer_photo(photo, caption="Вот тебе птичка, дорогой! Не отвлекай от важных дел!")
    except FileNotFoundError:
        await message.answer("Птичка улетела! Видимо, испугалась твоих вопросов!")
    except Exception as e:
        logger.error(f"Ошибка отправки изображения: {e}")
        await message.answer("Что-то пошло не так с птичкой! Проверь логи, дорогой!")


@dp.message(F.chat.type.in_({"group", "supergroup"}))
async def handle_group_messages(message: types.Message):
    """Обработчик сообщений в группах"""
    request_id = str(uuid.uuid4())[:8]

    try:
        bot_username = (await bot.me()).username.lower()
        logger.info(f"[{request_id}] Сообщение от {message.from_user.full_name} в чате {message.chat.id}")

        if not processor.should_respond(message, bot_username):
            return

        if not message.reply_to_message:
            return

        context = build_conversation_context(
            message.reply_to_message,
            message,
            request_id
        )

        logger.info(f"[{request_id}] Запускаем обработку для {context.reply_sender}")

        # Запускаем асинхронную обработку
        await asyncio.create_task(
            request_manager.process_request(
                request_id,
                process_message_with_context(context)
            )
        )

        await bot.send_chat_action(message.chat.id, "typing")

    except Exception as e:
        logger.error(f"[{request_id}] Ошибка обработки сообщения: {e}")
        await message.reply("Пфф! Провокация! Не могу обработать этот запрос!")


@dp.message(F.chat.type == "private")
async def handle_private_messages(message: types.Message):
    """Обработчик личных сообщений"""
    await message.answer(
        "Дорогой! Я работаю только в группах! "
        "Добавь меня в группу, собери аудиторию, и тогда я покажу всю свою мощь!"
    )


async def on_startup():
    """Действия при запуске бота"""
    bot_info = await bot.me()
    logger.info("=" * 50)
    logger.info("🚀 Бот запущен!")
    logger.info(f"🤖 Username: @{bot_info.username}")
    logger.info("=" * 50)


async def on_shutdown():
    """Действия при остановке бота"""
    logger.info("🛑 Бот останавливается...")
    active_count = await request_manager.get_active_count()
    if active_count > 0:
        logger.info(f"⏳ Ожидаем завершения {active_count} активных запросов...")
        await asyncio.sleep(2)
    logger.info("✅ Бот успешно остановлен!")


async def main():
    """Основная функция"""
    try:
        dp.startup.register(on_startup)
        dp.shutdown.register(on_shutdown)

        await bot.delete_webhook(drop_pending_updates=True)
        logger.info("🌐 Начинаем поллинг...")
        await dp.start_polling(bot)

    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"💥 Неожиданная ошибка: {e}")
