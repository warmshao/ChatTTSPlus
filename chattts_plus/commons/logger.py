import platform
import logging
from datetime import datetime, timezone
from pathlib import Path

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("wetext-zh_normalizer").setLevel(logging.WARNING)
logging.getLogger("NeMo-text-processing").setLevel(logging.WARNING)

# from https://github.com/FloatTech/ZeroBot-Plugin/blob/c70766a989698452e60e5e48fb2f802a2444330d/console/console_windows.go#L89-L96
colorCodePanic = "\x1b[1;31m"
colorCodeFatal = "\x1b[1;31m"
colorCodeError = "\x1b[31m"
colorCodeWarn = "\x1b[33m"
colorCodeInfo = "\x1b[37m"
colorCodeDebug = "\x1b[32m"
colorCodeTrace = "\x1b[36m"
colorReset = "\x1b[0m"

log_level_color_code = {
    logging.DEBUG: colorCodeDebug,
    logging.INFO: colorCodeInfo,
    logging.WARN: colorCodeWarn,
    logging.ERROR: colorCodeError,
    logging.FATAL: colorCodeFatal,
}

log_level_msg_str = {
    logging.DEBUG: "DEBU",
    logging.INFO: "INFO",
    logging.WARN: "WARN",
    logging.ERROR: "ERRO",
    logging.FATAL: "FATL",
}


class Formatter(logging.Formatter):
    def __init__(self, color=platform.system().lower() != "windows"):
        # https://stackoverflow.com/questions/2720319/python-figure-out-local-timezone
        self.tz = datetime.now(timezone.utc).astimezone().tzinfo
        self.color = color

    def format(self, record: logging.LogRecord):
        logstr = "[" + datetime.now(self.tz).strftime("%z %Y%m%d %H:%M:%S") + "] ["
        if self.color:
            logstr += log_level_color_code.get(record.levelno, colorCodeInfo)
        logstr += log_level_msg_str.get(record.levelno, record.levelname)
        if self.color:
            logstr += colorReset
        fn = record.filename.removesuffix(".py")
        logstr += f"] {str(record.name)} | {fn} | {str(record.msg) % record.args}"
        return logstr


def get_logger(name: str, lv=logging.INFO, remove_exist=False, format_root=False, log_file=None):
    """
    Configure and return a logger with specified settings.

    Args:
        name (str): The name of the logger.
        lv (int): The logging level (default: logging.INFO).
        remove_exist (bool): Whether to remove existing handlers (default: False).
        format_root (bool): Whether to format the root logger as well (default: False).
        log_file (str | Path | None): Path to the log file. If provided, logs will also be written to this file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(lv)

    if remove_exist and logger.hasHandlers():
        logger.handlers.clear()

    formatter = Formatter()

    # Add console handler if no handlers exist
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    else:
        # Update formatter for existing handlers
        for h in logger.handlers:
            h.setFormatter(formatter)

    # Add file handler if log_file is specified
    if log_file is not None:
        log_file = Path(log_file)
        # Create directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(str(log_file), encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if format_root:
        for h in logger.root.handlers:
            h.setFormatter(formatter)

    return logger
