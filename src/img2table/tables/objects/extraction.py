# coding: utf-8

from dataclasses import dataclass
from typing import Optional, List, OrderedDict, NamedTuple

import pandas as pd
from bs4 import BeautifulSoup
from xlsxwriter.format import Format
from xlsxwriter.worksheet import Worksheet


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class TableCell:
    bbox: BBox
    value: Optional[str]

    def __hash__(self):
        return hash(repr(self))


class CellPosition(NamedTuple):
    cell: TableCell
    row: int
    col: int


@dataclass
class CellSpan:
    top_row: int
    bottom_row: int
    col_left: int
    col_right: int
    value: Optional[str]

    @property
    def colspan(self) -> int:
        return self.col_right - self.col_left + 1

    @property
    def rowspan(self) -> int:
        return self.bottom_row - self.top_row + 1

    @property
    def html_value(self) -> str:
        if self.value is not None:
            return self.value.replace("\n", "<br>")
        else:
            return ""

    @property
    def html(self) -> str:
        return f'<td colspan="{self.colspan}" rowspan="{self.rowspan}">{self.html_value}</td>'

    def html_cell_span(self) -> List["CellSpan"]:
        if self.colspan > 1 and self.rowspan > 1:
            # Check largest coordinate and split
            if self.colspan > self.rowspan:
                return [CellSpan(top_row=row_idx,
                                 bottom_row=row_idx,
                                 col_left=self.col_left,
                                 col_right=self.col_right,
                                 value=self.value)
                        for row_idx in range(self.top_row, self.bottom_row + 1)]
            else:
                return [CellSpan(top_row=self.top_row,
                                 bottom_row=self.bottom_row,
                                 col_left=col_idx,
                                 col_right=col_idx,
                                 value=self.value)
                        for col_idx in range(self.col_left, self.col_right + 1)]

        return [self]


def create_all_rectangles(cell_positions: List[CellPosition]) -> List[CellSpan]:
    """
    Create all possible rectangles from list of cell positions
    :param cell_positions: list of cell positions
    :return: list of CellSpan objects representing rectangle coordinates
    """
    # Get cell value
    cell_value = cell_positions[0].cell.value

    # Get bounding coordinates
    min_col = min(map(lambda x: x.col, cell_positions))
    max_col = max(map(lambda x: x.col, cell_positions))
    min_row = min(map(lambda x: x.row, cell_positions))
    max_row = max(map(lambda x: x.row, cell_positions))

    # Get largest rectangle fully covered by cell positions
    largest_area, area_cell_pos = 0, None
    for col_left in range(min_col, max_col + 1):
        for col_right in range(col_left, max_col + 1):
            for top_row in range(min_row, max_row + 1):
                for bottom_row in range(top_row, max_row + 1):
                    # Get matching cell positions
                    matching_cell_pos = [cp for cp in cell_positions if col_left <= cp.col <= col_right
                                         and top_row <= cp.row <= bottom_row]

                    # Check if the rectangle is fully covered
                    fully_covered = len(matching_cell_pos) == (col_right - col_left + 1) * (bottom_row - top_row + 1)

                    # If rectangle is the largest, update values
                    if fully_covered and (len(matching_cell_pos) > largest_area):
                        largest_area = len(matching_cell_pos)
                        area_cell_pos = matching_cell_pos
                        cell_span = CellSpan(col_left=col_left,
                                             top_row=top_row,
                                             col_right=col_right,
                                             bottom_row=bottom_row,
                                             value=cell_value)

    # Get remaining cell positions
    remaining_cell_positions = [cp for cp in cell_positions if cp not in area_cell_pos]

    if remaining_cell_positions:
        # Get remaining rectangles
        return [cell_span] + create_all_rectangles(remaining_cell_positions)
    else:
        # Return coordinates
        return [cell_span]


@dataclass
class ExtractedTable:
    bbox: BBox
    title: Optional[str]
    content: OrderedDict[int, List[TableCell]]
    is_borderless: bool

    @property
    def df(self) -> pd.DataFrame:
        """
        Create pandas DataFrame representation of the table
        :return: pandas DataFrame containing table data
        """
        values = [[cell.value for cell in row] for k, row in self.content.items()]
        return pd.DataFrame(values)

    @property
    def html(self) -> str:
        """
        Create HTML representation of the table
        :return: HTML table
        """
        # Group cells based on hash (merged cells are duplicated over multiple rows/columns in content)
        dict_cells = dict()
        for id_row, row in self.content.items():
            for id_col, cell in enumerate(row):
                cell_pos = CellPosition(cell=cell, row=id_row, col=id_col)
                dict_cells[hash(cell)] = dict_cells.get(hash(cell), []) + [cell_pos]

        # Get list of cell spans
        cell_span_list = [cell_span for _, cells in dict_cells.items()
                          for cell_span in create_all_rectangles(cell_positions=cells)]
        cell_span_list = [span for cell_span in cell_span_list for span in cell_span.html_cell_span()]

        # Create HTML rows
        rows_html = list()
        for row_idx in range(len(self.content)):
            # Get cells in row
            row_cells = sorted([cell_span for cell_span in cell_span_list if cell_span.top_row == row_idx],
                               key=lambda cs: cs.col_left)
            html_row = "<tr>" + "".join([cs.html for cs in row_cells]) + "</tr>"
            rows_html.append(html_row)

        # Create HTML table
        table_html = "<table>" + "".join(rows_html) + "</table>"

        return BeautifulSoup(table_html, "html.parser").prettify().strip()

    def _to_worksheet(self, sheet: Worksheet, nb_upper_rows_existed: int = 0, title_format: Optional[Format] = None, 
                      header_format: Optional[Format] = None, cell_fmt: Optional[Format] = None, 
                      final_row_format: Optional[Format] = None):
        """
        Populate xlsx worksheet with table data
        :param sheet: xlsxwriter Worksheet
        :param cell_fmt: xlsxwriter cell format
        """        
        # Write table title if is bordered table
        if (not self.is_borderless):
            sheet.merge_range(f"A{nb_upper_rows_existed+1}:Z{nb_upper_rows_existed+1}", self.title, title_format)
            nb_upper_rows_existed += 1

        # Group cells based on hash (merged cells are duplicated over multiple rows/columns in content)
        dict_cells = dict()
        for id_row, row in self.content.items():
            for id_col, cell in enumerate(row):
                cell_pos = CellPosition(cell=cell, row=nb_upper_rows_existed + id_row, col=id_col)
                dict_cells[hash(cell)] = dict_cells.get(hash(cell), []) + [cell_pos]

        # Write all cells to sheet
        nb_rows = len(self.content.keys())
        nb_cols = len(self.content[0])
        for c in dict_cells.values():
            format = header_format if (not self.is_borderless) and (c[0].row == nb_upper_rows_existed) else cell_fmt
            if len(c) == 1:
                cell_pos = c.pop()
                sheet.write(cell_pos.row, cell_pos.col, cell_pos.cell.value, format)
            # Detect rows that have all columns merge in 1 column
            elif len(c) == nb_cols:
                cell_pos = c.pop()
                row_1_col_format = header_format if (cell_pos.row != nb_rows - 1 + nb_upper_rows_existed) else final_row_format
                sheet.merge_range(first_row=cell_pos.row,
                                first_col=0,
                                last_row=cell_pos.row,
                                last_col=nb_cols-1,
                                data=cell_pos.cell.value,
                                cell_format=row_1_col_format)
            else:
                # Get all rectangles
                for cell_span in create_all_rectangles(cell_positions=c):
                    # Case of merged cells
                    sheet.merge_range(first_row=cell_span.top_row,
                                      first_col=cell_span.col_left,
                                      last_row=cell_span.bottom_row,
                                      last_col=cell_span.col_right,
                                      data=cell_span.value,
                                      cell_format=format)

        # Autofit worksheet
        # sheet.autofit()
            
        # Detect merged columns and set each column size corresponding to each column in the header row
        row_0 = self.content.get(0)
        list_row0_merged_cols = list()
        col = 0
        while (col < len(row_0)):
            list_merged_cols = [col]
            for next_col in range(col+1, len(row_0)):
                if (hash(row_0[col]) == hash(row_0[next_col])):
                    list_merged_cols.append(next_col)
                else:
                    break
            if (len(list_merged_cols) > 1):
                list_row0_merged_cols.append(list_merged_cols)
            else:
                list_row0_merged_cols.append(col)
            col += len(list_merged_cols)
        
        # Here 0.35 is the ratio of the input images to fit the computer screen (15'6 inch), can be changed
        for id_col in list_row0_merged_cols:
            if (isinstance(id_col, int)):
                # sheet.setro
                cell = row_0[id_col]
                sheet.set_column_pixels(first_col=id_col,
                                        last_col=id_col,
                                        width=0.35*(cell.bbox.x2 - cell.bbox.x1))
            else:
                cell = row_0[id_col[0]]
                nb_merged_cols = len(id_col)
                sheet.set_column_pixels(first_col=id_col[0],
                                        last_col=id_col[-1],
                                        width=0.35*(cell.bbox.x2 - cell.bbox.x1)/nb_merged_cols)
                

    def html_repr(self, title: Optional[str] = None) -> str:
        """
        Create HTML representation of the table
        :param title: title of HTML paragraph
        :return: HTML string
        """
        html = f"""{rf'<h3 style="text-align: center">{title}</h3>' if title else ''}
                   <p style=\"text-align: center\">
                       <b>Title:</b> {self.title or 'No title detected'}<br>
                       <b>Bounding box:</b> x1={self.bbox.x1}, y1={self.bbox.y1}, x2={self.bbox.x2}, y2={self.bbox.y2}
                   </p>
                   <div align=\"center\">{self.df.to_html().replace("None", "")}</div>
                   <hr>
                """
        return html

    def __repr__(self):
        return f"ExtractedTable(title={self.title}, bbox=({self.bbox.x1}, {self.bbox.y1}, {self.bbox.x2}, " \
               f"{self.bbox.y2}),shape=({len(self.content)}, {len(self.content[0])}))".strip()
