import pathlib
from sys import argv

def main():
    i = 1
    for path in pathlib.Path(argv[1]).iterdir():
        if path.is_file():
            old_name = path.stem
            old_extension = path.suffix
            directory = path.parent
            new_name = arv[2] + str(i) + old_extension
            path.rename(pathlib.Path(directory, new_name))
            i += 1

if __name__ == '__main__':
    main()
