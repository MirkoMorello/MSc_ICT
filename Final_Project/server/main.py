from .server import Server  # Use relative import: .server
from .config import SERVER_IP, SERVER_PORT  # Use relative import: .config
from .utils import logging_utils  # Use relative import

logger = logging_utils.get_logger(__name__)

if __name__ == "__main__":
    server = Server(SERVER_IP, SERVER_PORT)
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Stopping server (KeyboardInterrupt).")
    except Exception as e:
        logger.exception("An unexpected error occurred: %s", e)
    finally:
        server.shutdown()