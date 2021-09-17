import os


def handle_saving_dir(save_to_dir, error_msg=None):
    if not os.path.exists(save_to_dir):
        os.mkdir(save_to_dir)
    elif not os.path.isdir(save_to_dir):
        if error_msg is None:
            error_msg = f"{save_to_dir} should be a directory"
        raise ValueError(error_msg)
