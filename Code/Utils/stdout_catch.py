import sys
import os
import time


class Logger(object):
    def __init__(self, file_name: str):
        self.terminal = sys.stdout
        self.log = open(file_name, 'w')

    def write(self, s):
        self.terminal.write(s)
        self.log.write(s)

    def flush(self):
        pass
