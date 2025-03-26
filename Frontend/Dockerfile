#большинство здесь из образца - Э. Гоулд
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Сначала копируем только requirements.txt и устанавливаем зависимости
COPY ./streamlit-service/requirements.txt /app
RUN pip3 install -r requirements.txt

# Затем копируем остальной код
# Нужно изменить если называние папки изменяется - Э. Гоулд
COPY /Frontend /app

# Нужно выбрать и открыть порт - Э. Гоулд
# Порт здесь тоже открыт для хакера. (Есть причина, почему нужно открывать порты, чтобы получить данные.) - Э. Гоулд
EXPOSE 8501

# Нужно выбрать порт - Э. Гоулд
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Нужно изменить если называние файла изменяется - Э. Гоулд
# Нужно выбрать порт - Э. Гоулд
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]