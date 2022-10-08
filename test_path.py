import  os
import sys


def test_curr_path():
    print('getcwd:   ', os.getcwd())
    os.chdir(sys.path[0])

    print('abspath:   ', os.path.abspath('.'))

    current_work_dir = os.path.dirname(__file__)
    print('curr_path', current_work_dir)