import logging

FORMAT = '%(asctime)s:%(process)d:%(levelname)s::%(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("data")
logger.setLevel(logging.INFO)