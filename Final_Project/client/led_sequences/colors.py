import enum

class Colors(enum.Enum):
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    ORANGE = (255, 165, 0)
    YELLOW = (255, 255, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    INDIGO = (75, 0, 130)
    VIOLET = (238, 130, 238)
    BLACK = (0, 0, 0)
    LIGHT_BLUE = (173, 216, 230)
    LIGHT_GREEN = (144, 238, 144)
    LIGHT_YELLOW = (255, 255, 224)
    LIGHT_PURPLE = (221, 160, 221)
    LIGHT_ORANGE = (255, 160, 122)
    LIGHT_RED = (255, 99, 71)
    LIGHT_PINK = (255, 182, 193)
    LIGHT_BROWN = (210, 105, 30)
    LIGHT_GRAY = (211, 211, 211)
    DARK_GRAY = (169, 169, 169)
    DARK_BROWN = (139, 69, 19)
    DARK_PINK = (255, 20, 147)
    DARK_RED = (139, 0, 0)
    DARK_ORANGE = (255, 140, 0)
    DARK_YELLOW = (255, 215, 0)
    DARK_GREEN = (0, 100, 0)
    DARK_BLUE = (0, 0, 139)
    DARK_VIOLET = (148, 0, 211)

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)
    
    def __iter__(self):
        return iter(self.value)