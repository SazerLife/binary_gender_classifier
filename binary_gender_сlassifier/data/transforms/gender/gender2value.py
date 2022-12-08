class Gender2Value:
    def __init__(self):
        self.__dictionary_path = None

    def __call__(self, gender: str) -> int:
        if gender == "F":
            label: int = 0
        elif gender == "M":
            label: int = 1
        else:
            raise RuntimeError(f"Unexpected gender: {gender}")

        return label
