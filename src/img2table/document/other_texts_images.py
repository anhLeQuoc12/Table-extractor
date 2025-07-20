from dataclasses import dataclass

@dataclass
class OtherTextsAndImages:
    """
    Class represents for 2 lines of initial main texts and other texts and images in the image
    """
    initial_text_line1: str = ''
    initial_text_line2: str = ''
    other_texts_and_images: tuple = None
