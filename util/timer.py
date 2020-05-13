import logging
from datetime import datetime
from contextlib import ContextDecorator

logging.basicConfig()


class Timer(ContextDecorator):
    def __init__(self, name: str = 'anonymous'):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

    def __enter__(self):
        self.begin = datetime.now()

    def __exit__(self, *args):
        self.end = datetime.now()
        self.elapse = (self.end - self.begin).total_seconds()
        self.logger.info('cost %.3f seconds' % self.elapse)
