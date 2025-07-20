# coding: utf-8
from typing import Any

import polars as pl

from img2table.document.base import Document
from img2table.document.other_texts_images import OtherTextsAndImages
from img2table.ocr.data import OCRDataframe


class OCRInstance:
    @property
    def pl_schema(self):
        schema = {
            "page": pl.Int64,
            "class": str,
            "id": str,
            "parent": str,
            "value": str,
            "confidence": pl.Int64,
            "x1": pl.Int64,
            "y1": pl.Int64,
            "x2": pl.Int64,
            "y2": pl.Int64
        }
        return schema

    def content(self, document: Document) -> Any:
        raise NotImplementedError

    def get_ocr_dataframes(self, content: Any) -> list[OCRDataframe]:
        raise NotImplementedError

    def of(self, document: Document) -> list[OCRDataframe]:
        """
        Extract text from Document to list of OCRDataframe objects for each page
        :param document: Document object
        :return: list of OCRDataframe object for each page
        """
        # Extract content from document
        content = self.content(document=document)

        # Create list OCRDataframe from content
        return self.get_ocr_dataframes(content=content)
