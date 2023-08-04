from enum import Enum


class Status(Enum):
    NULL = 0
    REPOSITION = 1
    CATCH_CENTER_1 = 2
    CATCH_CENTER_2 = 3
    CATCH_SHAPE_1 = 4
    CATCH_SHAPE_2 = 5
    QUIT = 6


class Events(Enum):
    NULL = 0
    # event1,2,3,4
