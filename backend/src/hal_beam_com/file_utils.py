import os


def get_root_dir():
    """
    Return the root project directory.
    """
    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
    print(root_dir)
    return root_dir

def get_data_dir():
    target_dir = os.path.join(get_root_dir(), "data")
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

def get_temp_dir():
    target_dir = os.path.join(get_root_dir(), "temp_data")
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

def get_command_dir():
    command_dir = os.path.join(get_data_dir(), "command_examples")
    os.makedirs(command_dir, exist_ok=True)
    return command_dir

def get_db_dir():
    command_dir = os.path.join(get_data_dir(), "db")
    os.makedirs(command_dir, exist_ok=True)
    return command_dir

