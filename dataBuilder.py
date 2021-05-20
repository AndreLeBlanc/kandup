import pathlib
from sys import argv

def rename():
    i = 1
    for path in pathlib.Path(argv[1]).iterdir():
        if path.is_file():
            old_name = path.stem
            old_extension = path.suffix
            directory = path.parent
            new_name = argv[2] + str(i) + old_extension
            path.rename(pathlib.Path(directory, new_name))
            i += 1

def main():
    if argv[1].upper() == "HELP":
        print("This script is used to rename images in order to create datasets where\n"
              "images must be named in a certain way. The program takes two strings as\n"
              "arguments, the first string is the directory of the images that will be\n"
              "renamed and the second argument is the name that the files are given.\n"
              "Example:\n"
              "python3 dataBuilder.py /a_dir/ bob\n"
              "Will rename all files in the \"a_dir\" directory bob1, bob2, bob3 etc\n")
    else:
        rename()

if __name__ == '__main__':
    main()
