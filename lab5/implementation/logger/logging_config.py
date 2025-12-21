import logging
from rich.logging import RichHandler

def setup_logger():
    """
    Настройка логгера:
    - В файл app.log: уровень DEBUG, подробные логи с временем, файлом и строкой
    - В консоль: уровень INFO, краткие логи с RichHandler
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # минимальный уровень

    # ---------- File Handler ----------
    file_handler = logging.FileHandler("app.log", mode="a", encoding="utf-8")
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    # ---------- Console Handler ----------
    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    # ---------- Добавляем хэндлеры ----------
    logger.handlers = []  # убираем старые, если были
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
