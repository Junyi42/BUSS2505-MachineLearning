import sys


class Logger(object):
    def __init__(self, file_path: str = "./Default.log"):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == '__main__':
    sys.stdout = Logger('./log.txt')
    print('hello world!,hello boy')