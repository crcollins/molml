import os

from molml.io import read_file_data


DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

METHANE_PATH = os.path.join(DATA_PATH, "methane.out")
METHANE_VALS = read_file_data(METHANE_PATH)
METHANE_ELEMENTS = METHANE_VALS.elements.tolist()
METHANE_NUMBERS = METHANE_VALS.numbers.tolist()
METHANE_COORDS = METHANE_VALS.coords
METHANE = (METHANE_ELEMENTS, METHANE_COORDS)
METHANE_CONNECTIONS = {
    0: {1: "1", 2: "1", 3: "1", 4: "1"},
    1: {0: "1"},
    2: {0: "1"},
    3: {0: "1"},
    4: {0: "1"},
}

BIG_PATH = os.path.join(DATA_PATH, "big.out")
BIG_VALS = read_file_data(BIG_PATH)
BIG_ELEMENTS = BIG_VALS.elements.tolist()
BIG_NUMBERS = BIG_VALS.numbers.tolist()
BIG_COORDS = BIG_VALS.coords
BIG = (BIG_ELEMENTS, BIG_COORDS)

MID_PATH = os.path.join(DATA_PATH, "mid.out")
MID_VALS = read_file_data(MID_PATH)
MID_ELEMENTS = MID_VALS.elements.tolist()
MID_NUMBERS = MID_VALS.numbers.tolist()
MID_COORDS = MID_VALS.coords
MID = (MID_ELEMENTS, MID_COORDS)

ALL_DATA = [METHANE, MID, BIG]
