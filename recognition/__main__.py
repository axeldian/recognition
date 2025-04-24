import sys
from recognition.cli import parse_and_execute

if __name__ == "__main__":
    print(sys.argv)
    parse_and_execute(sys.argv[1:])

