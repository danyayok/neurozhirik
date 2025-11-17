import logging
import asyncio
import time
import aiohttp
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
from config import BOT_TOKEN

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


async def analyze_image(image_url: str, request_id: str) -> Optional[str]:
    """Анализирует изображение и возвращает описание"""
    try:
        logger.info(f"[{request_id}] Начинаем анализ изображения: {image_url}")

        async with aiohttp.ClientSession() as session:
            async with session.get(image_url, timeout=config.IMAGE_TIMEOUT) as response:
                if response.status == 200:
                    image_data = await response.read()
                    logger.info(f"[{request_id}] Изображение загружено, размер: {len(image_data)} байт")

                    client = G4FClient(provider=g4f.Provider.PollinationsAI)

                    # Пробуем разные модели для анализа изображения
                    models_to_try = ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "flux"]

                    for model in models_to_try:
                        try:
                            logger.info(f"[{request_id}] Пробуем модель {model}")

                            response = client.chat.completions.create(
                                model=model,
                                messages=[{
                                    "role": "user",
                                    "content": "Опиши это изображение максимально подробно: что изображено, цвета, надписи, детали, эмоции если есть люди"
                                }],
                                image=image_data,
                                timeout=config.IMAGE_TIMEOUT
                            )

                            description = response.choices[0].message.content
                            if description and len(description) > 10:
                                logger.info(f"[{request_id}] Успешно проанализировано с моделью {model}")
                                return description

                        except Exception as model_error:
                            logger.warning(f"[{request_id}] Модель {model} не сработала: {model_error}")
                            continue

                    logger.warning(f"[{request_id}] Ни одна модель не смогла проанализировать изображение")
                    return None

                else:
                    logger.error(f"[{request_id}] Ошибка загрузки изображения: статус {response.status}")
                    return None

    except asyncio.TimeoutError:
        logger.error(f"[{request_id}] Таймаут при анализе изображения")
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
    """Генерирует ответ в стиле Жириновского с обработкой изображений"""
    try:
        # Обработка изображений
        image_analysis = ""

        # Проверяем и анализируем изображение в оригинальном сообщении
        if context.original_message and (context.original_message.photo or context.original_message.document):
            logger.info(f"[{context.request_id}] Обнаружено изображение, начинаем анализ")

            image_url = await get_image_url(context.original_message, context.request_id)
            if image_url:
                image_desc = await analyze_image(image_url, context.request_id)
                if image_desc:
                    image_analysis = f"\n\n[На изображении: {image_desc}]"
                    logger.info(f"[{context.request_id}] Изображение успешно проанализировано")
                else:
                    logger.warning(f"[{context.request_id}] Не удалось получить описание изображения")
                    image_analysis = "\n\n[Изображение не удалось проанализировать]"
            else:
                logger.info(f"[{context.request_id}] Изображение не найдено в сообщении")

        # Формируем финальный промпт
        prompt = (
            f"{config.SYSTEM_PROMPT}\n\n"
            f"1. Оригинальное сообщение от {context.original_sender}: \"{context.original_text}{image_analysis}\"\n"
            f"2. Ответ от {context.reply_sender} (тебе): \"{context.reply_text}\"\n\n"
            f"Ответь {context.reply_sender} в стиле Жириновского!"
        )

        logger.info(f"[{context.request_id}] Генерируем ответ для {context.reply_sender}")

        # Генерируем ответ
        client = G4FClient()
        for attempt in range(config.MAX_ATTEMPTS):
            try:
                logger.info(f"[{context.request_id}] Попытка генерации {attempt + 1}")

                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    timeout=config.REQUEST_TIMEOUT
                )
                result = response.choices[0].message.content

                if result and len(result.split()) < 200:
                    logger.info(f"[{context.request_id}] Успешно сгенерирован ответ")
                    return result
                else:
                    logger.warning(f"[{context.request_id}] Ответ слишком длинный или пустой")

            except Exception as e:
                logger.error(f"[{context.request_id}] Попытка {attempt + 1} ошибка: {e}")
                if attempt < config.MAX_ATTEMPTS - 1:
                    await asyncio.sleep(1)

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
        # Пытаемся отправить сообщение об ошибке
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
        "Дорогой! Я бот-Жириновский! Добавь меня в группу и упомяни в ответе на сообщение - "
        "и я дам свой острый комментарий! Могу отвечать сразу нескольким провокаторам одновременно!"
    )


@dp.message(Command("status"))
async def cmd_status(message: types.Message):
    """Показать статус бота"""
    active_requests = await request_manager.get_active_count()
    await message.answer(
        f"Дорогой! Я работаю на полную мощность!\n"
        f"Активных запросов: {active_requests}\n"
        f"Максимум параллельно: {config.MAX_CONCURRENT_REQUESTS}\n"
        f"Готов отвечать всем желающим получить порцию правды!"
    )


@dp.message(Command("ptichko"))
async def cmd_ptichko(message: types.Message):
    """Обработчик команды /ptichko"""
    try:
        photo = FSInputFile("images.jpg")
        await message.answer_photo(photo, caption="Вот тебе птичка, дорогой! Не отвлекай от важных дел!")
    except FileNotFoundError:
        await message.answer("Птичка улетела! Видимо, испугалась твоих вопросов!")
    except Exception as e:
        logger.error(f"Ошибка отправки изображения: {e}")
        await message.answer("Что-то пошло не так с птичкой! Проверь логи, дорогой!")


@dp.message(F.chat.type.in_({"group", "supergroup"}))
async def handle_group_messages(message: types.Message):
    """Обработчик сообщений в группах - полностью асинхронный"""
    request_id = str(uuid.uuid4())[:8]

    try:
        bot_username = (await bot.me()).username.lower()
        logger.info(f"[{request_id}] Сообщение от {message.from_user.full_name} в чате {message.chat.id}")

        if not processor.should_respond(message, bot_username):
            return

        if not message.reply_to_message:
            # Создаем задачу для отправки ответа без блокировки
            await asyncio.create_task(
                bot.send_message(
                    chat_id=message.chat.id,
                    text="Дорогой, ты что, не знаешь как работает система? Ответь на сообщение и тогда зови меня!",
                    reply_to_message_id=message.message_id
                )
            )
            return

        # Создаем контекст для асинхронной обработки
        context = build_conversation_context(
            message.reply_to_message,
            message,
            request_id
        )

        logger.info(f"[{request_id}] Запускаем асинхронную обработку для {context.reply_sender}")

        # Запускаем асинхронную обработку через менеджер запросов
        await asyncio.create_task(
            request_manager.process_request(
                request_id,
                process_message_with_context(context)
            )
        )

        # Сразу отправляем подтверждение получения запроса
        await asyncio.create_task(
            bot.send_chat_action(message.chat.id, "typing")
        )

    except Exception as e:
        logger.error(f"[{request_id}] Ошибка обработки сообщения: {e}")
        # Асинхронно отправляем сообщение об ошибке
        await asyncio.create_task(
            bot.send_message(
                chat_id=message.chat.id,
                text="Пфф! Провокация! Не могу обработать этот запрос!",
                reply_to_message_id=message.message_id
            )
        )


@dp.message(F.chat.type == "private")
async def handle_private_messages(message: types.Message):
    """Обработчик личных сообщений"""
    await message.answer(
        "Дорогой! Я работаю только в группах! Ты что, одинокий волк? "
        "Добавь меня в группу, собери аудиторию, и тогда я покажу всю свою мощь! "
        "Могу отвечать сразу нескольким участникам - проверь мою скорость!"
    )


@dp.message(Command("test_image"))
async def cmd_test_image(message: types.Message):
    """Тестовая команда для проверки анализа изображений"""
    if message.reply_to_message and (message.reply_to_message.photo or message.reply_to_message.document):
        request_id = "test_image_" + str(uuid.uuid4())[:8]
        context = build_conversation_context(
            message.reply_to_message,
            message,
            request_id
        )

        await asyncio.create_task(
            request_manager.process_request(
                request_id,
                process_message_with_context(context)
            )
        )
        await message.reply("Тестирую анализ изображения...")
    else:
        await message.reply("Ответь на сообщение с изображением для теста!")


async def on_startup():
    """Действия при запуске бота"""
    bot_info = await bot.me()
    logger.info("=" * 50)
    logger.info("🚀 Бот запущен!")
    logger.info(f"🤖 Username: @{bot_info.username}")
    logger.info(f"🆔 ID: {bot_info.id}")
    logger.info(f"📝 Имя: {bot_info.full_name}")
    logger.info(f"⚡ Максимум параллельных запросов: {config.MAX_CONCURRENT_REQUESTS}")
    logger.info("=" * 50)


async def on_shutdown():
    """Действия при остановке бота"""
    logger.info("🛑 Бот останавливается... Отменяем активные задачи...")

    # Ждем завершения активных задач (с таймаутом)
    active_count = await request_manager.get_active_count()
    if active_count > 0:
        logger.info(f"⏳ Ожидаем завершения {active_count} активных запросов...")
        await asyncio.sleep(2)  # Даем время на завершение

    logger.info("✅ Бот успешно остановлен!")


async def main():
    """Основная функция"""
    try:
        dp.startup.register(on_startup)
        dp.shutdown.register(on_shutdown)

        await bot.delete_webhook(drop_pending_updates=True)
        logger.info("🌐 Вебхук удален, начинаем асинхронный поллинг...")
        await dp.start_polling(bot)

    except Exception as e:
        logger.error(f"💥 Критическая ошибка при запуске бота: {e}")
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"💥 Неожиданная ошибка: {e}")