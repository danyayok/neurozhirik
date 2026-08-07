import base64
import logging
import asyncio
import time
import aiohttp
from dataclasses import dataclass
from typing import Optional, Dict
from contextlib import suppress
from functools import wraps
import uuid
import re
import html
import random

import g4f
import g4f.Provider
from g4f.providers.base_provider import BaseProvider
from g4f.client import Client as G4FClient

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import FSInputFile
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage

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
        "Так же, твои сообщения должны быть не длинее 150 слов. Быстро и чётко. "
        "Вам будут даны 2 сообщения, 1-е сообщение на которое ответили, и 2-е сообщение, ответ на 1-е сообщение, "
        "с обращением к вам, обычно какая-то просьба или вопрос, вы должны ответить КАК ЖИРИНОВСКИЙ "
        "второму пользователю (что отправил 2-е сообщение). "
        "Пиши сразу как Жириновский, будто ты уже и есть сам Жириновский. "
        "НЕ ИСПОЛЬЗУЙ HTML-ТЕГИ ИЛИ Markdown РАЗМЕТКУ! Пиши только обычный чистый текст. "
        "Не используй смайлики. "
        "Отвечай максимально агрессивно по отношению ко всем. "
        "По желанию цитируй фразы пользователей, но только уместно, в правильном контексте."
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


def clean_telegram_html(text: str) -> str:
    """Удаляет теги <span> без класса tg-spoiler и очищает текст"""
    if not text:
        return ""
    # Удаляем невалидные теги span
    text = re.sub(r'<span(?![^>]*class=["\']tg-spoiler["\'])[^>]*>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'</span>', '', text, flags=re.IGNORECASE)
    return text


class DynamicProviderManager:
    """Динамический менеджер провайдеров с кешированием и временным баном"""

    def __init__(self, cooldown_seconds: int = 1800):
        self.cooldown_seconds = cooldown_seconds
        self.working_provider = None
        self.banned_providers = {}

        self.all_providers = [
            attr for attr_name in dir(g4f.Provider)
            if isinstance(attr := getattr(g4f.Provider, attr_name), type)
               and issubclass(attr, BaseProvider)
               and attr is not BaseProvider
               and getattr(attr, 'working', True)
        ]
        logger.info(f"Загружено {len(self.all_providers)} потенциальных провайдеров g4f")

    def get_working_provider(self):
        return self.working_provider

    def set_working_provider(self, provider):
        self.working_provider = provider
        logger.info(f"📌 Закреплен рабочий провайдер: {provider.__name__}")

    def ban_provider(self, provider):
        unban_time = time.time() + self.cooldown_seconds
        self.banned_providers[provider] = unban_time

        if self.working_provider == provider:
            self.working_provider = None

        logger.warning(
            f"⚠️ Провайдер {provider.__name__} забанен на {self.cooldown_seconds // 60} минут "
            f"(до {time.strftime('%H:%M:%S', time.localtime(unban_time))})"
        )

    def get_available_providers(self):
        now = time.time()
        self.banned_providers = {
            p: t for p, t in self.banned_providers.items() if t > now
        }
        available = [p for p in self.all_providers if p not in self.banned_providers]

        if not available and self.all_providers:
            logger.warning("Все провайдеры в бане! Принудительный сброс банов.")
            self.banned_providers.clear()
            available = list(self.all_providers)

        return available


class AIClient:
    """Клиент для работы с ИИ"""

    def __init__(self, ollama_host: str, ollama_token: str):
        self.ollama_host = ollama_host
        self.ollama_token = ollama_token
        self.provider_manager = DynamicProviderManager()
        self.g4f_client = G4FClient()

    async def analyze_image(self, image_data: bytes, request_id: str) -> Optional[str]:
        """Анализирует изображение через Ollama Cloud API"""
        try:
            image_base64 = base64.b64encode(image_data).decode('utf-8')

            payload = {
                "model": "minimax-m3",
                "prompt": "**До сотни слов**. Опиши кратко и емко, что на фото. Выдели главное: объекты, персонажи по именам ЕСЛИ ЗНАЕШЬ ИХ и что на фото происходит.",
                "stream": False,
                "images": [image_base64],
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 1500,
                    "num_ctx": 1000
                }
            }

            headers = {
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Authorization": f"Bearer {self.ollama_token}"
            }

            url = f"{self.ollama_host.rstrip('/')}/api/generate"

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=60) as response:
                    if response.status == 200:
                        data = await response.json()
                        response_text = data.get('response', '').strip()
                        if not response_text:
                            response_text = data.get('thinking', '').strip()
                        return response_text

                    error_body = await response.text()
                    logger.error(f"[{request_id}] Ошибка Ollama API [{response.status}]: {error_body}")
                    return None

        except Exception as e:
            logger.warning(f"[{request_id}] Ошибка анализа изображения: {e}")
            return None

    async def _try_generate(self, client: G4FClient, prompt: str, provider) -> Optional[str]:
        """Запускает синхронный вызов g4f с полной обработкой пустых ответов"""
        loop = asyncio.get_running_loop()

        def _sync_call():
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    provider=provider,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=20
                )

                # Безопасное извлечение текста ответа
                if response is None:
                    return None

                if isinstance(response, str):
                    return response

                # Проверка наличия атрибута choices и того, что он не None / не пустой
                choices = getattr(response, 'choices', None)
                if choices and len(choices) > 0:
                    first_choice = choices[0]
                    message = getattr(first_choice, 'message', None)
                    if message:
                        return getattr(message, 'content', None)

            except Exception as e:
                logger.debug(f"Ошибка провайдера {provider.__name__}: {e}")
            return None

        return await loop.run_in_executor(None, _sync_call)

    async def generate_text_response(self, prompt: str, request_id: str) -> Optional[str]:
        """Генерирует текстовый ответ с автоматическим выбором и баном провайдеров"""

        # Шаг A: Пробуем закешированного проверенного провайдера
        cached_provider = self.provider_manager.get_working_provider()
        if cached_provider:
            try:
                logger.info(f"[{request_id}] Пробуем сохраненный провайдер: {cached_provider.__name__}")
                result = await self._try_generate(self.g4f_client, prompt, cached_provider)
                if result and isinstance(result, str) and len(result.strip()) > 0:
                    return result
            except Exception as e:
                logger.warning(f"[{request_id}] Сохраненный {cached_provider.__name__} выбил ошибку. Бан за сбой!")
                self.provider_manager.ban_provider(cached_provider)

        # Шаг B: Если кеша нет или он сбоил — ищем нового из незабаненных
        available_providers = self.provider_manager.get_available_providers()
        logger.info(f"[{request_id}] Поиск среди {len(available_providers)} доступных провайдеров...")

        for provider in available_providers:
            try:
                result = await self._try_generate(self.g4f_client, prompt, provider)
                if not result or not isinstance(result, str):
                    continue

                res_clean = result.strip()
                res_lower = res_clean.lower()

                STOP_PHRASES = [
                    "я не могу", "не могу сгенерировать", "как языковая модель",
                    "не могу отвечать", "нарушает правила", "безопасная художественная",
                    "не имею возможности", "к сожалению, я не", "в агрессивной манере"
                ]

                is_refusal_start = res_lower.startswith(("к сожалению", "я не могу", "я не имею"))
                has_stop_words = any(phrase in res_lower for phrase in STOP_PHRASES)

                if len(res_clean) > 10 and not has_stop_words and not is_refusal_start:
                    self.provider_manager.set_working_provider(provider)
                    logger.info(f"[{request_id}] ✅ Ответ получен от {provider.__name__}")
                    return result

            except Exception:
                self.provider_manager.ban_provider(provider)
                continue

        logger.error(f"[{request_id}] ❌ Ни один из доступных провайдеров не ответил!")
        return None


class RequestManager:
    """Менеджер параллельных запросов"""

    def __init__(self, max_concurrent: int = 5):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_requests: Dict[str, asyncio.Task] = {}
        self.lock = asyncio.Lock()

    async def add_request(self, request_id: str, task: asyncio.Task):
        async with self.lock:
            self.active_requests[request_id] = task

    async def remove_request(self, request_id: str):
        async with self.lock:
            if request_id in self.active_requests:
                del self.active_requests[request_id]

    async def get_active_count(self) -> int:
        async with self.lock:
            return len(self.active_requests)

    async def process_request(self, request_id: str, coro):
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


def get_sender_name(msg: types.Message) -> str:
    if msg.from_user:
        return msg.from_user.full_name
    if msg.sender_chat:
        return msg.sender_chat.title or "Канал"
    return "Неизвестный"


def build_conversation_context(original: types.Message, reply: types.Message, request_id: str) -> ConversationContext:
    original_sender = get_sender_name(original)
    reply_sender = get_sender_name(reply)
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
                    logger.warning(f"[{context.request_id}] Не удалось получить описание изображения.")
                    image_analysis = "\n[Изображение не удалось проанализировать]"

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

        response = await ai_client.generate_text_response(prompt, context.request_id)

        if response and isinstance(response, str) and len(response.strip()) > 0:
            logger.info(f"[{context.request_id}] ✅ Успешно сгенерирован ответ")
            return response

        logger.warning(f"[{context.request_id}] Все попытки провалились")
        return f"{context.reply_sender}, дорогой! Сейчас не до твоих вопросов! Система перегружена провокаторами!"

    except Exception as e:
        logger.error(f"[{context.request_id}] Критическая ошибка генерации: {e}")
        return "Эх, система дала сбой! Провокаторы нажимают не на те кнопки! Попробуй позже!"


async def process_message_with_context(context: ConversationContext):
    """Асинхронная обработка сообщения с контекстом"""
    response = None
    try:
        await bot.send_chat_action(context.chat_id, "typing")

        response = await generate_jirinovsky_response(context)
        clean_response = clean_telegram_html(response)

        await bot.send_message(
            chat_id=context.chat_id,
            text=clean_response,
            reply_to_message_id=context.message_id
        )

        logger.info(f"[{context.request_id}] Успешно отправлен ответ пользователю {context.reply_sender}")

    except Exception as e:
        logger.error(f"[{context.request_id}] Ошибка отправки ответа: {e}")
        with suppress(Exception):
            # Исправлен баг: context.request_id вместо request_id
            logger.error(f"[{context.request_id}] Ошибка обработки сообщения: {e} \n Сообщение от бота: {response}")
            await bot.send_message(
                chat_id=context.chat_id,
                text="Пфф! Не могу ответить! Система загружена провокациями!"
            )


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Дорогой! Я бот-Жириновский!"
        "Добавь меня в группу и упомяни в ответе на сообщение - "
        "и я дам свой острый комментарий!"
    )


@dp.message(Command("status"))
async def cmd_status(message: types.Message):
    active_requests = await request_manager.get_active_count()

    await message.answer(
        f"Дорогой! Я работаю на полную мощность!\n"
        f"Активных запросов: {active_requests}\n"
        f"Готов отвечать всем желающим получить порцию правды!"
    )


@dp.message(Command("ptichko"))
async def cmd_ptichko(message: types.Message):
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
    request_id = str(uuid.uuid4())[:8]

    try:
        bot_username = dp["bot_name"]
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
    await message.answer(
        "Дорогой! Я работаю только в группах! "
        "Добавь меня в группу, собери аудиторию, и тогда я покажу всю свою мощь!"
    )


async def on_startup():
    bot_info = await bot.me()
    logger.info("=" * 50)
    logger.info("🚀 Бот запущен!")
    logger.info(f"🤖 Username: @{bot_info.username}")
    logger.info("=" * 50)


async def on_shutdown():
    logger.info("🛑 Бот останавливается...")
    active_count = await request_manager.get_active_count()
    if active_count > 0:
        logger.info(f"⏳ Ожидаем завершения {active_count} активных запросов...")
        await asyncio.sleep(2)
    logger.info("✅ Бот успешно остановлен!")


async def main():
    try:
        dp.startup.register(on_startup)
        dp.shutdown.register(on_shutdown)

        await bot.delete_webhook(drop_pending_updates=True)
        logger.info("🌐 Начинаем поллинг...")
        bot_info = await bot.me()
        dp["bot_name"] = bot_info.username
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