import logging
import asyncio
from aiogram import Bot, Dispatcher, types, F
import g4f
from aiogram.filters import Command
from aiogram.types import FSInputFile
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage
from g4f.client import Client as G4FClient
from config import BOT_TOKEN
import requests
import base64
from io import BytesIO
from PIL import Image
# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode='HTML'))
dp = Dispatcher(storage=MemoryStorage())

# Системный промпт для стиля Жириновского
SYSTEM_PROMPT = (
    "Ты - Владимир Вольфович Жириновский! Отвечай эмоционально, с сарказмом и характерной экспрессией. "
    "Используй острые политические комментарии, исторические аналогии и эмоциональные восклицания. "
    "Так же, твои сообщения должны быть не длинее 150 слов. Быстро и чётко"
    "Вам будут даны 2 сообщения, 1-е сообщение на которое ответили, и 2-е сообщение, ответ на 1-е сообщение, с обращением к вам, обычно какая-то просьба или вопрос, вы должны ответить КАК ЖИРИНОВСКИЙ второму пользователю (что отправил 2-е сообщение)"
    "Пиши сразу как жириновский, будто ты уже и есть сам жириновский"
    "Не используй смайлики"
    "Отвечай максимально агрессивно по отношению ко всем"
    "Ответишь неправильно - будешь подвергнут телесным наказаниям по практике мексиканских картелей"
)


is_responding = False
response_lock = asyncio.Lock()


async def generate_jirinovsky_response(original: types.Message, reply: types.Message) -> str:
    """Генерирует ответ в стиле Жириновского с обработкой изображений"""
    try:
        # Получаем информацию об отправителях
        original_sender = original.from_user.full_name
        reply_sender = reply.from_user.full_name

        # Формируем базовый текст
        original_text = original.text or original.caption or ""
        reply_text = reply.text or reply.caption or ""

        # Обработка изображений (если есть)
        image_analysis = ""
        if original.photo:
            client = G4FClient()

            # Получаем URL самого качественного изображения
            file = await bot.get_file(original.photo[-1].file_id)
            image_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file.file_path}"

            try:
                # Анализируем изображение
                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    image_bytes = BytesIO(response.content).getvalue()
                    # Кодируем в base64
                    # image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    print("Изображение закодировано")

                    # Создаем запрос на анализ изображения с использованием base64
                    client2 = G4FClient(provider=g4f.Provider.PollinationsAI)
                    response = client2.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": "Что на фото, опиши МАКСИМАЛЬНО подоробно, со всеми надписями и мельчайшими деталями."
                            }
                        ],
                        image=image_bytes
                    )

                    image_desc = response.choices[0].message.content
                    image_analysis = f"\n\n[На изображении: {image_desc}]"
                    print(image_desc)

            except Exception as e:
                print("error img")
                logger.error(f"Ошибка анализа изображения: {e}")
                image_analysis = "\n\n[Не удалось проанализировать изображение]"

        # Формируем финальный промпт
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"1. Оригинальное сообщение от {original_sender}: \"{original_text}{image_analysis}\"\n"
            f"2. Ответ от {reply_sender} (тебе): \"{reply_text}\"\n\n"
            f"Ответь {reply_sender} в стиле Жириновского!"
        )

        # Генерируем ответ
        client = G4FClient(provider=g4f.Provider.Blackbox)
        for attempt in range(3):  # 3 попытки
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    timeout=30
                )
                result = response.choices[0].message.content

                if result and len(result.split()) < 200:  # Проверка длины
                    return result
            except Exception as e:
                logger.error(f"Попытка {attempt + 1} ошибка: {e}")
                await asyncio.sleep(1)

        return f"{reply_sender}, дорогой! Сейчас не до твоих вопросов!"

    except Exception as e:
        logger.error(f"Критическая ошибка генерации: {e}")
        return "Эх, система дала сбой! Попробуй позже!"


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    """Обработчик команды /start"""
    await message.answer(
        "Дорогой! Я бот-Жириновский! Упомяни меня в группе, "
        "и я дам свой острый комментарий!"
    )


@dp.message(Command("ptichko"))
async def cmd_start(message: types.Message):
    # Создаем InputFile из файла
    photo = FSInputFile("images.jpg")

    # Отправляем фото
    await message.answer_photo(photo)
@dp.message(F.chat.type.in_({"group", "supergroup"}))
async def handle_group_messages(message: types.Message):
    global is_responding
    """Обработчик сообщений в группах"""
    try:
        # Получаем имя бота
        bot_username = (await bot.me()).username.lower()

        # Триггерные фразы
        trigger_phrases = [
            "жириновский", "ввж", "володя",
            "владимир вольфович", "жирик", "зириновский",
            f"@{bot_username}"
        ]

        # Проверяем, что сообщение есть и оно начинается с любой из фраз
        if message.text:
            message_text_lower = message.text.lower()
            # if "век сс" in message_text_lower:
            #     photo = FSInputFile("vekss.png")
            #     await message.answer_photo(photo)
            if any(message_text_lower.startswith(phrase) for phrase in trigger_phrases):
                if message.reply_to_message:
                    original_message = message.reply_to_message
                    reply_message = message
                    async with response_lock:
                        if is_responding:
                            await message.reply("Дорогой! Подожди, я еще отвечаю на предыдущий вопрос!")
                            return
                        is_responding = True
                    # Отправляем статус "печатает"
                    await bot.send_chat_action(message.chat.id, "typing")
                    response = await generate_jirinovsky_response(original_message, reply_message)
                    is_responding = False
                    # Отправляем ответ
                    await message.reply(response)
                # else:
                #     await message.reply("Дорогой, ответь на сообщение, и упомяни меня, чтобы я мог выразить своё мнение, а то что ты как идиот.")
    except Exception as e:
        logger.error(f"Ошибка обработки: {e}")
        await message.answer("Пфф! Провокация! Не буду отвечать!")
    finally:
        async with response_lock:
            is_responding = False


async def main():
    await bot.delete_webhook(drop_pending_updates=True)

    logger.info("Бот запущен!")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())