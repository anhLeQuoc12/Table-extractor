# coding: utf-8
from typing import Any

import polars as pl

from img2table.document.base import Document
from img2table.document.initial_main_texts import InitialMainTexts
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

    def get_initial_main_texts_and_ocr_dataframes(self, content: Any) -> list[tuple[InitialMainTexts, OCRDataframe]]:
        raise NotImplementedError

    def of(self, document: Document) -> list[tuple[InitialMainTexts, OCRDataframe]]:
        """
        Extract text from Document to a tuple of initial main texts and OCRDataframe object for each page
        :param document: Document object
        :return: list of tuple containing initial main texts and OCRDataframe object for each page
        """
        # Extract content from document
        content = self.content(document=document)

        # Create OCRDataframe from content
        return self.get_initial_main_texts_and_ocr_dataframes(content=content)
