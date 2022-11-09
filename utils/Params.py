import json


class ParamsLoader(object):
    def __init__(self, file_path):
        # file path
        self.file_path = file_path
        self.params_data = None

        # load the offline_dataset
        self._load_json_file()

    def _load_json_file(self):
        try:
            with open(self.file_path, "r") as f_in:
                self.params_data = json.load(f_in)
        except FileNotFoundError as error:
            print(error)