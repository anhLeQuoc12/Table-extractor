from dataclasses import dataclass


@dataclass
class InitialMainTexts:
    """
    Class represents for 2 lines of initial main texts in the image
    """
    line1: str = ''
    line2: str = ''
