# coding: utf-8
import typing
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List

import cv2
import numpy as np

from img2table.document.base import Document, MockDocument
from img2table.document.base.rotation import fix_rotation_image
from img2table.document.other_texts_images import InitialMainTexts
from img2table.tables.objects.extraction import ExtractedTable
from img2table.tables.objects.table import Table

if typing.TYPE_CHECKING:
    from img2table.ocr.base import OCRInstance


@dataclass
class Image(Document):
    detect_rotation: bool = False

    def __post_init__(self):
        self.pages = None

        super(Image, self).__post_init__()

    @cached_property
    def images(self) -> List[np.ndarray]:
        img = cv2.imdecode(np.frombuffer(self.bytes, np.uint8), cv2.IMREAD_COLOR)
        if self.detect_rotation:
            rotated_img, _ = fix_rotation_image(img=img)
            return [rotated_img]
        else:
            return [img]
        

    def extract_content(self, ocr: "OCRInstance" = None, implicit_rows: bool = False, implicit_columns: bool = False,
                       borderless_tables: bool = False, min_confidence: int = 50) -> tuple[InitialMainTexts, List[ExtractedTable]]:
        """
        Extract initial main texts and tables from document
        :param ocr: OCRInstance object used to extract table content
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :param implicit_columns: boolean indicating if implicit columns are splitted
        :param borderless_tables: boolean indicating if borderless tables should be detected
        :param min_confidence: minimum confidence level from OCR in order to process text, from 0 (worst) to 99 (best)
        :return: The initial main texts and list of extracted tables in the image
        """
        extracted_tables = super(Image, self).extract_content(ocr=ocr,
                                                             implicit_rows=implicit_rows,
                                                             implicit_columns=implicit_columns,
                                                             borderless_tables=borderless_tables,
                                                             min_confidence=min_confidence)
        return extracted_tables.get(0)
