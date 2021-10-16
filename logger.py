from datetime import datetime
import os


class Logger:
    def __init__(self):
        path = './log'
        filename = f'fed-match - {datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}.txt'

        if not os.path.isdir(path):
            os.makedirs(path)

        self.file_path = os.path.join(path, filename)

    def log(self, content: str):
        print(content)

        with open(self.file_path, 'a+') as outfile:
            outfile.write(content)
            outfile.write('\n')
