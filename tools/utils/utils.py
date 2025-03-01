import os

def create_empty_dir(directory):
    """
    Creates an empty directory. If the directory already exists, it removes all files within it.
    Args:
        directory (str): The path of the directory to create or empty.
    Raises:
        OSError: If the directory cannot be created or files cannot be removed.
    """

    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        # Remove all files in the directory
        for file in os.listdir(directory):
            os.remove(os.path.join(directory, file))