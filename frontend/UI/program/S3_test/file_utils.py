import os


class FileUtil:
    def __init__(self):
        """
        Initialize with the project root directory
        """
        self.root_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        print(self.root_dir)

    def get_root_dir(self):
        """
        Return the root project directory.
        """
        return self.root_dir

    def get_temp_dir(self):
        """
        Append a sub-path to the current directory and return full path.
        Creates the folder if it doesn't exist.
        """
        target_dir = os.path.join(self.root_dir, "temp_data")
        os.makedirs(target_dir, exist_ok=True)
        return target_dir


# if __name__ == "__main__":
#     util = FileUtil()
#     print("Current Directory:", util.get_root_dir())
