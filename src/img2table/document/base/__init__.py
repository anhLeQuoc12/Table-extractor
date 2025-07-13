# coding: utf-8
import io
import typing
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Union, Dict, List, Optional

import numpy as np
import xlsxwriter

from img2table import Validations
from img2table.document.initial_main_texts import InitialMainTexts
from img2table.tables.objects.extraction import CellPosition, ExtractedTable, create_all_rectangles

if typing.TYPE_CHECKING:
    from img2table.ocr.base import OCRInstance
    from img2table.tables.objects.table import Table


@dataclass
class MockDocument:
    images: List[np.ndarray]


@dataclass
class Document(Validations):
    src: Union[str, Path, io.BytesIO, bytes]

    def validate_src(self, value, **_) -> Union[str, Path, io.BytesIO, bytes]:
        if not isinstance(value, (str, Path, io.BytesIO, bytes)):
            raise TypeError(f"Invalid type {type(value)} for src argument")
        return value

    def validate_detect_rotation(self, value, **_) -> int:
        if not isinstance(value, bool):
            raise TypeError(f"Invalid type {type(value)} for detect_rotation argument")
        return value

    def __post_init__(self):
        super(Document, self).__post_init__()
        # Initialize list_imts_dfs
        self.list_imts_dfs = None

        if not hasattr(self, "pages"):
            self.pages = None

        if isinstance(self.pages, list):
            self.pages = sorted(self.pages)

    @cached_property
    def bytes(self) -> bytes:
        if isinstance(self.src, bytes):
            return self.src
        elif isinstance(self.src, io.BytesIO):
            self.src.seek(0)
            return self.src.read()
        elif isinstance(self.src, str):
            with io.open(self.src, 'rb') as f:
                return f.read()

    @property
    def images(self) -> List[np.ndarray]:
        raise NotImplementedError

    def get_initial_main_texts_and_table_content(self, tables: Dict[int, List["Table"]], ocr: "OCRInstance",
                          min_confidence: int) -> Dict[int, tuple[InitialMainTexts, List[ExtractedTable]]]:
        """
        Retrieve 2 initial lines of main texts and table content with OCR
        :param tables: dictionary containing extracted tables by page
        :param ocr: OCRInstance object used to extract table content
        :param min_confidence: minimum confidence level from OCR in order to process text, from 0 (worst) to 99 (best)
        :return: dictionary with page number as key and list of initial main texts and extracted tables as values
        """
        # Get pages where tables have been detected
        # table_pages = [k for k, v in tables.items() if len(v) > 0]

        if (self.list_imts_dfs is None and ocr is None):
            return {k: (None, [tb.extracted_table for tb in v]) for k, v in tables.items()}

        # Create document containing only pages
        ocr_doc = MockDocument(images=[self.images[page] for page in tables.keys()])

        # Get OCRDataFrame object
        if self.list_imts_dfs is None and ocr is not None:
            self.list_imts_dfs = ocr.of(document=ocr_doc)

        if self.list_imts_dfs is None:
            return {k: (None, []) for k in tables.keys()}

        # Retrieve table contents with ocr
        for page in tables.keys():
            ocr_df_page = self.list_imts_dfs[page][1]

            # Get table content
            tables[page] = [table.get_content(ocr_df=ocr_df_page, min_confidence=min_confidence)
                            for table in tables[page]]

            # Filter relevant tables
            tables[page] = [table for table in tables[page] if max(table.nb_rows, table.nb_columns) >= 2]

            # Retrieve titles
            from img2table.tables.processing.text.titles import get_title_tables
            tables[page] = get_title_tables(img=self.images[page],
                                            tables=tables[page],
                                            ocr_df=ocr_df_page)

        # Reset OCR
        # self.ocr_df = None

        return {k: (self.list_imts_dfs[k][0], [tb.extracted_table for tb in v
                    # if (max(tb.nb_rows, tb.nb_columns) >= 2 and not tb._borderless)
                    # or (tb.nb_rows >= 2 and tb.nb_columns >= 3)
                    ])
                for k, v in tables.items()}

    def extract_content(self, ocr: "OCRInstance" = None, implicit_rows: bool = False, implicit_columns: bool = False,
                       borderless_tables: bool = False, min_confidence: int = 50) -> Dict[int, tuple[InitialMainTexts, List[ExtractedTable]]]:
        """
        Extract dictionary of initial main texts and tables from document with pages as keys
        :param ocr: OCRInstance object used to extract table content
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :param implicit_columns: boolean indicating if implicit columns are splitted
        :param borderless_tables: boolean indicating if borderless tables should be detected
        :param min_confidence: minimum confidence level from OCR in order to process text, from 0 (worst) to 99 (best)
        :return: dictionary with page number as key and the initial main texts and extracted tables as values
        """
        # Extract tables from document
        from img2table.tables.image import TableImage
        tables = {idx: TableImage(img=img,
                                  min_confidence=min_confidence).extract_tables(implicit_rows=implicit_rows,
                                                                                implicit_columns=implicit_columns,
                                                                                borderless_tables=borderless_tables)
                  for idx, img in enumerate(self.images)}


        # Update table content with OCR if possible
        dict_page_content = self.get_initial_main_texts_and_table_content(tables=tables,
                                        ocr=ocr,
                                        min_confidence=min_confidence)

        # If pages have been defined, modify tables keys
        if self.pages:
            dict_page_content = {self.pages[k]: v for k, v in dict_page_content.items()}

        for table in dict_page_content[0][1]:
            print(f"{table}, borderless: {table.is_borderless}")

        return dict_page_content

    def to_xlsx(self, dest: Union[str, Path, io.BytesIO], ocr: "OCRInstance" = None, implicit_rows: bool = False,
                implicit_columns: bool = False, borderless_tables: bool = False,
                min_confidence: int = 50) -> Optional[io.BytesIO]:
        """
        Create xlsx file containing all extracted tables from document
        :param dest: destination for xlsx file
        :param ocr: OCRInstance object used to extract table content
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :param implicit_columns: boolean indicating if implicit columns are splitted
        :param borderless_tables: boolean indicating if borderless tables should be detected
        :param min_confidence: minimum confidence level from OCR in order to process text, from 0 (worst) to 99 (best)
        :return: if a buffer is passed as dest arg, it is returned containing xlsx data
        """
        # Extract initial main texts and tables
        dict_page_content = self.extract_content(ocr=ocr,
                                               implicit_rows=implicit_rows,
                                               implicit_columns=implicit_columns,
                                               borderless_tables=borderless_tables,
                                               min_confidence=min_confidence)
        dict_page_content = {0: dict_page_content} if isinstance(dict_page_content, tuple) else dict_page_content

        # Create workbook
        workbook = xlsxwriter.Workbook(dest, {'in_memory': True})

        # Create generic cell format
        title_format = workbook.add_format({'align': 'left', 'valign': 'vcenter', 'text_wrap': True, 'bold': True, 'font_size': '13'})
        header_format = workbook.add_format({'align': 'left', 'valign': 'top', 'text_wrap': True, 'bold': True, 'bg_color': '#f3f2f4'})
        cell_format = workbook.add_format({'align': 'left', 'valign': 'top', 'text_wrap': True})
        final_row_format = workbook.add_format({'align': 'left', 'valign': 'top', 'text_wrap': True, 'bg_color': '#f6f6f6'})

        header_format.set_border()
        cell_format.set_border()
        final_row_format.set_border()

        # For each extracted table, create a corresponding worksheet and populate it
        is_one_page = len(dict_page_content) == 1
        for page, content in dict_page_content.items():
            imt, tables = content
            if (isinstance(dest, str)):
                dest_file_name = dest[dest.rfind("/")+1 : dest.rfind(".")]
            elif (isinstance(dest, Path)):
                dest_file_name = dest.stem
            else:
                dest_file_name = ''

            for idx, table in enumerate(tables):
                # Create worksheet
                if (is_one_page):
                    sheet_name = f"{dest_file_name} - Table {idx+1}" if dest_file_name else f"Page {page + 1} - Table {idx + 1}"
                else:
                    sheet_name = f"Page {page + 1} - Table {idx + 1}"

                sheet = workbook.add_worksheet(name=sheet_name)

                # Populate worksheet
                if (idx == 0):
                    # Write 2 initial lines of main texts to the sheet
                    line1_format = workbook.add_format({'align': 'left', 'valign': 'bottom', 'text_wrap': True, 'font_size': '13'})
                    line2_format = workbook.add_format({'align': 'left', 'valign': 'bottom', 'text_wrap': True, 'bold': True, 'font_size': '13'})

                    sheet.merge_range('A1:Z1', imt.line1, line1_format)
                    sheet.merge_range('A2:Z2', imt.line2, line2_format)

                    table._to_worksheet(sheet=sheet, nb_upper_rows_existed=3, title_format=title_format, header_format=header_format, cell_fmt=cell_format, final_row_format=final_row_format)
                else:
                    table._to_worksheet(sheet=sheet, title_format=title_format, header_format=header_format, cell_fmt=cell_format, final_row_format=final_row_format)


            # If no tables have been detected, write 2 initial lines of main texts
            if (len(tables) == 0):
                sheet = workbook.add_worksheet(name=dest_file_name)
                line1_format = workbook.add_format({'align': 'left', 'valign': 'bottom', 'text_wrap': True, 'font_size': '13'})
                line2_format = workbook.add_format({'align': 'left', 'valign': 'bottom', 'text_wrap': True, 'bold': True, 'font_size': '13'})

                sheet.merge_range('A1:Z1', imt.line1, line1_format)
                sheet.merge_range('A2:Z2', imt.line2, line2_format)
        

        # Close workbook
        workbook.close()

        # If destination is a BytesIO object, return it
        if isinstance(dest, io.BytesIO):
            dest.seek(0)
            return dest
