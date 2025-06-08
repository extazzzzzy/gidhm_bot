import telebot
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
from datetime import datetime

bot = telebot.TeleBot('7784410418:AAH5Ov-g4FbRNiDnlAN1gf_0NkWIpikhDS4')

def predict(src_to_model, src_to_labels, file_name):
    model = load_model(src_to_model, compile=False)
    with open(src_to_labels, "r", encoding="utf-8") as file:
        class_names = [line.strip() for line in file.readlines()]
    
    image = Image.open(file_name).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]

    # confidence_score = prediction[0][index]
    # print(confidence_score)

    if ("isSnowModel" in src_to_model):
        return class_name

    with open("src/models/descriptionObjects.txt", "r", encoding="utf-8") as file:
        description_objects = [line.strip().replace('\\n', '\n') for line in file]

    return class_name, description_objects[index]

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, "Приветствую тебя, мой друг! Я был создан с целью сохранения исторического наследия города Ханты-Мансийска, а также развития туризма в нём. По фото я могу определить достопримечательность г. Ханты-Мансийска и рассказать тебе о ней!")
    bot.send_message(message.chat.id, "Отправь мне фото 📸\nУбедись, что фото не размыто и на нём хорошее освещение📌")

@bot.message_handler(content_types=['photo'])
def get_photo(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"photo_{timestamp}.jpg"

        with open(file_name, 'wb') as new_file:
            new_file.write(downloaded_file)
        bot.send_message(message.chat.id, "Фото получено! Обработка фотографии...")

        predict_snow = predict('src/models/isSnowModel.h5', 'src/models/labelsIsSnow.txt', file_name)
        #bot.send_message(message.chat.id, f"Степень заснеженности: {predict_snow} ❄️")

        if (predict_snow == "Значительная"):
            predict_object, description_object = predict('src/models/winterObjects.h5', 'src/models/labelsObjects.txt', file_name)
        else:
            predict_object, description_object = predict('src/models/summerObjects.h5', 'src/models/labelsObjects.txt', file_name)
        
        bot.send_message(
            message.chat.id,
            f"*{predict_object}* 🧭\n\n{description_object}",
            parse_mode='Markdown'
        )


    except Exception as e:
        bot.send_message(message.chat.id, f"Ошибка при обработке фото: {e}")
    
    finally:
        if os.path.exists(file_name):
            os.remove(file_name)



@bot.message_handler(func=lambda message: True, content_types=['text', 'sticker', 'document', 'audio', 'video', 'voice', 'location', 'contact'])
def handle_other_messages(message):
    bot.send_message(message.chat.id, "Отправь мне фото 📸\nУбедись, что фото не размыто и на нём хорошее освещение📌")

bot.polling(none_stop=True)
