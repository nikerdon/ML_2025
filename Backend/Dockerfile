#большинство здесь из образца - Э. Гоулд
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Сначала копируем только requirements.txt и устанавливаем зависимости
COPY ./titanic-api/requirements.txt /app
RUN pip3 install -r requirements.txt

# Затем копируем остальной код
# Нужно изменить если называние папки изменяется - Э. Гоулд
COPY ./Backend /app

# Нужно изменить если называние файла изменяется - Э. Гоулд
CMD ["python3", "app_api.py"]