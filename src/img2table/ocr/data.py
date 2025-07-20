# coding: utf-8
from dataclasses import dataclass
from pathlib import Path

import cv2
import polars as pl
import numpy as np
from PIL import Image

from img2table.document.other_texts_images import InitialMainTexts, OtherTextsAndImages
from img2table.tables import threshold_dark_areas
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.table import Table


@dataclass
class OCRDataframe:
    df: pl.DataFrame

    def page(self, page_number: int = 0) -> "OCRDataframe":
        # Filter dataframe on specific page
        df_page = self.df.filter(pl.col('page') == page_number)
        return OCRDataframe(df=df_page)

    def get_text_cell(self, cell: Cell, margin: int = 0, page_number: int = None, min_confidence: int = 50) -> str:
        """
        Get text corresponding to cell
        :param cell: Cell object in document
        :param margin: margin to take around cell
        :param page_number: page number of the cell
        :param min_confidence: minimum confidence in order to include a word, from 0 (worst) to 99 (best)
        :return: text contained in cell
        """
        # Define relevant bounding box
        bbox = cell.bbox(margin=margin)

        # Filter dataframe on relevant page
        df_words = self.df.filter(pl.col('class') == "ocrx_word")
        if page_number:
            df_words = df_words.filter(pl.col('page') == page_number)
        # Filter dataframe on relevant words
        df_words = df_words.filter(pl.col('value').is_not_null() & (pl.col('confidence') >= min_confidence))

        # Compute coordinates of intersection
        df_words = (df_words.with_columns([pl.lit(bbox[0]).alias('x1_bbox'),
                                           pl.lit(bbox[1]).alias('y1_bbox'),
                                           pl.lit(bbox[2]).alias('x2_bbox'),
                                           pl.lit(bbox[3]).alias('y2_bbox')]
                                          )
                    .with_columns([pl.max_horizontal(['x1', 'x1_bbox']).alias('x_left'),
                                   pl.max_horizontal(['y1', 'y1_bbox']).alias('y_top'),
                                   pl.min_horizontal(['x2', 'x2_bbox']).alias('x_right'),
                                   pl.min_horizontal(['y2', 'y2_bbox']).alias('y_bottom'),
                                   ])
                    )

        # Filter where intersection is not empty
        df_intersection = (df_words.filter(pl.col("x_right") > pl.col("x_left"))
                           .filter(pl.col("y_bottom") > pl.col("y_top"))
                           )

        # Compute area of word bbox and intersection
        df_areas = (df_intersection.with_columns([
            ((pl.col('x2') - pl.col('x1')) * (pl.col('y2') - pl.col('y1'))).alias('w_area'),
            ((pl.col('x_right') - pl.col('x_left')) * (pl.col('y_bottom') - pl.col('y_top'))).alias('int_area')
        ])
        )

        # Filter on words where its bbox is contained in area
        df_words_contained = df_areas.filter(pl.col('int_area') / pl.col('w_area') > 0.5)

        # Group text by parents
        df_text_parent = (df_words_contained
                          .group_by('parent')
                          .agg([pl.col('x1').min(),
                                pl.col('x2').max(),
                                pl.col('y1').min(),
                                pl.col('y2').max(),
                                pl.col('value').alias('value')])
                          .sort([pl.col("y1"), pl.col("x1")])
                          )

        # Concatenate all rows
        text_lines = (df_text_parent.select(pl.col('value'))
                      .get_column('value')
                      .to_list()
                      )

        return "\n".join([" ".join(line).strip() for line in text_lines]).strip() or None

    def get_text_table(self, table: Table, src_img: np.ndarray, dest_path: Path, page_number: int = None, min_confidence: int = 50) -> Table:
        """
        Identify texts and images located in Table object
        :param table: Table object
        :param src_img: The source image in which table was detected
        :param dest_path: The Path object containing destination path
        :param page_number: page number of the cell
        :param min_confidence: minimum confidence in order to include a word, from 0 (worst) to 99 (best)
        :return: table with content set on all cells
        """
        # Filter dataframe on relevant page
        df_words = self.df.filter(pl.col('class') == "ocrx_word")

        if page_number:
            df_words = df_words.filter(pl.col('page') == page_number)
        # Filter dataframe on relevant words
        df_words = df_words.filter(pl.col('value').is_not_null() & (pl.col('confidence') >= min_confidence))

        # Create dataframe containing all coordinates of Cell objects
        list_cells = [{"row": id_row, "col": id_col, "x1_cell": cell.x1, "x2_cell": cell.x2, "y1_cell": cell.y1, "y2_cell": cell.y2}
                      for id_row, row in enumerate(table.items)
                      for id_col, cell in enumerate(row.items)]
        df_cells = pl.DataFrame(data=list_cells)

        # Cartesian product between two dataframes
        df_word_cells = df_words.join(other=df_cells, how="cross")

        # Compute coordinates of intersection
        df_word_cells = df_word_cells.with_columns([pl.max_horizontal(['x1', 'x1_cell']).alias('x_left'),
                                                    pl.max_horizontal(['y1', 'y1_cell']).alias('y_top'),
                                                    pl.min_horizontal(['x2', 'x2_cell']).alias('x_right'),
                                                    pl.min_horizontal(['y2', 'y2_cell']).alias('y_bottom'),
                                                    ])

        # Filter where intersection is not empty
        df_intersection = (df_word_cells.filter(pl.col("x_right") > pl.col("x_left"))
                           .filter(pl.col("y_bottom") > pl.col("y_top"))
                           )

        # Compute area of word bbox and intersection
        df_areas = (df_intersection.with_columns([
            ((pl.col('x2') - pl.col('x1')) * (pl.col('y2') - pl.col('y1'))).alias('w_area'),
            ((pl.col('x_right') - pl.col('x_left')) * (pl.col('y_bottom') - pl.col('y_top'))).alias('int_area')
        ])
        )

        # Filter on words where its bbox is contained in area
        df_words_contained = df_areas.filter(pl.col('int_area') / pl.col('w_area') > 0.5)

        # Group text by parent
        df_text_parent = (df_words_contained
                          .group_by(['row', 'col', 'parent'])
                          .agg([pl.col('x1').min(),
                                pl.col('x2').max(),
                                pl.col('y1').min(),
                                pl.col('y2').max(),
                                pl.col('value').map_elements(lambda x: ' '.join(x), return_dtype=str).alias('value'),
                                pl.col('x1_cell').first(),
                                pl.col('y1_cell').first(),
                                pl.col('x2_cell').first(),
                                pl.col('y2_cell').first()])
                          .sort([pl.col("row"), pl.col("col"), pl.col('y1'), pl.col('x1')])
                          .group_by(['row', 'col'])
                          .agg([pl.col('x1'),
                                pl.col('y1'),
                                pl.col('x2'),
                                pl.col('y2'),
                                pl.col('value'),
                                pl.col('x1').min().alias('cell_words_x1_min'),
                                pl.col('y1').min().alias('cell_words_y1_min'),
                                pl.col('x2').max().alias('cell_words_x2_max'),
                                pl.col('y2').max().alias('cell_words_y2_max'),
                                pl.col('x1_cell').first(),
                                pl.col('y1_cell').first(),
                                pl.col('x2_cell').first(),
                                pl.col('y2_cell').first()])
                          )
        
        # Identify list of cells that don't have text in to create table dataframe
        list_none_text_cells = list()
        for cell in list_cells:
            row = cell['row']
            col = cell['col']
            has_in_df = False
            for df_row in df_text_parent.iter_rows(named=True):
                if (row == df_row['row'] and col == df_row['col']):
                    has_in_df = True
                    break
            if (not has_in_df):
                list_none_text_cells.append({"row": row, "col": col, "x1": [], "y1": [], "x2": [], "y2": [], "value": None,
                                             "cell_words_x1_min": -1, "cell_words_y1_min": -1, "cell_words_x2_max": -1, "cell_words_y2_max": -1,
                                             "x1_cell": cell['x1_cell'], "y1_cell": cell['y1_cell'], "x2_cell": cell['x2_cell'], "y2_cell": cell['y2_cell']})
        
        df_table = (df_text_parent.vstack(pl.DataFrame(list_none_text_cells))
        .sort([pl.col("row"), pl.col("col"), pl.col('y1_cell'), pl.col('x1_cell')]))


        # Implement found texts to table cells content and process to find images in table
        thresh_img = threshold_dark_areas(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB), 11)
        _, _, stats, _ = cv2.connectedComponentsWithStats(thresh_img, 8, cv2.CV_32S)

        for rec in df_table.to_dicts():
            list_x1 = rec['x1']
            list_x2 = rec['x2']
            list_y1 = rec['y1']
            list_y2 = rec['y2']
            list_texts = rec['value']
            cw_min_x1 = rec['cell_words_x1_min']
            cw_min_y1 = rec['cell_words_y1_min']
            cw_max_x2 = rec['cell_words_x2_max']
            cw_max_y2 = rec['cell_words_y2_max']
            cell_x1 = rec['x1_cell']
            cell_y1 = rec['y1_cell']
            cell_x2 = rec['x2_cell']
            cell_y2 = rec['y2_cell']
            cell_text = list_texts[0] if list_texts != None else None
            nb_texts = len(list_texts) if list_texts != None else -1
            
            # Get all texts in the cell
            if (cell_text != None and nb_texts > 1):
                for index in range(0, nb_texts):
                    if (index == nb_texts-1):
                        break
                    else:
                        x_distance = list_x1[index+1] - list_x2[index]
                        y_distance = list_y1[index+1] - list_y2[index]
                        if (x_distance > 0) and (y_distance < 0):
                            nb_spaces = x_distance // 20
                            for i in range(0, nb_spaces):
                                cell_text += '  '
                        elif (x_distance < 0) and (y_distance > 0):
                            cell_text += "\n"
                        cell_text += list_texts[index+1]

            # Process to get the potential image in the cell, this processing assumes that the images will stay under the text
            # and will get all the images cropped in one image (if there are more than one image)
            img_x1, img_y1, img_x2, img_y2 = (10000, 10000, -1, -1)
            for stat in stats:
                x1, y1, w, h, area = stat
                if (cell_text == None):
                    if (x1 > cell_x1 and y1 > cell_y1 and x1+w < cell_x2 and y1+h < cell_y2):
                        if (x1 < img_x1):
                            img_x1 = x1
                        if (y1 < img_y1):
                            img_y1 = y1
                        if (x1+w > img_x2):
                            img_x2 = x1+w
                        if (y1+h > img_y2):
                            img_y2 = y1+h
                    else:
                        continue
                else:
                    if (x1 > cell_x1) and (y1 > cw_max_y2) and (x1+w < cell_x2) and (y1+h < cell_y2):
                        if (x1 < img_x1):
                            img_x1 = x1
                        if (y1 < img_y1):
                            img_y1 = y1
                        if (x1+w > img_x2):
                            img_x2 = x1+w
                        if (y1+h > img_y2):
                            img_y2 = y1+h
                    else:
                        continue

            # If found the potential image, check whether it is real image or not, if yes save it in destination folder
            crop_img_path = None
            if (img_x1 != 10000):
                width = img_x2 - img_x1
                height = img_y2 - img_y1
                if (max(width, height) > 50 and max(width, height) / min(width, height) <= 8):
                    crop_img = src_img[img_y1:img_y2, img_x1:img_x2]
                    output_files_paths = dest_path.parent.iterdir()
                    dest_file_name = dest_path.stem
                    index = 0
                    for file_path in output_files_paths:
                        if file_path.stem.find(f'{dest_file_name}_image_') != -1:
                            index += 1

                    crop_img_path = f"{dest_path.absolute()}_image_{index+1}"
                    print(crop_img_path)
                    cv2.imwrite(crop_img_path, crop_img)
                

            table.items[rec.get('row')].items[rec.get('col')].content = (cell_text, crop_img_path)

        # Get images in table
        
        return table
    
    def get_other_texts_and_images(self, src_img: np.ndarray, existed_tables: list[Table], dest_path: Path, min_confidence: int = 50) -> OtherTextsAndImages:
        df_words = self.df.filter(pl.col("class") == "ocrx_word")
        list_words_not_in_tables = list()
        for row in df_words.iter_rows(named=True):
            x1 = row['x1']
            y1 = row['y1']
            x2 = row['x2']
            y2 = row['y2']
            not_in_tables = True
            for table in existed_tables:
                if (table.x1 <= x1 <= table.x2) or (table.x1 <= x2 <= table.x2) or (table.y1 <= y1 <= table.y2) or (table.y1 <= y2 <= table.y2):
                    not_in_tables = False
                    break
            if (not_in_tables):
                list_words_not_in_tables.append(row)

        df_texts_not_in_tables = (pl.DataFrame(data=list_words_not_in_tables)
                                  .group_by(pl.col("parent"))
                                  .agg([pl.col("x1").min(),
                                        pl.col("y1").min(),
                                        pl.col("x2").max(),
                                        pl.col('y2').max(),
                                        pl.col('value').map_elements(lambda x: ' '.join(x), return_dtype=str).alias('value')])
                                    .sort([pl.col('y1'), pl.col('x1')]))
        otherTextsImages = OtherTextsAndImages()
        prev_x = 0
        prev_y = 0
        for row in df_texts_not_in_tables.iter_rows(named=True):
            x1 = row['x1']
            y1 = row['y1']
            x2 = row['x2']
            y2 = row['y2']
            text = row['value']
            if (not otherTextsImages.initial_text_line1):
                otherTextsImages.initial_text_line1 = text
            elif not otherTextsImages.initial_text_line2:
                otherTextsImages.initial_text_line2 = text
                prev_x = x2
                prev_y = y2
            elif (y2 >= prev_y - 10) and (y2 <= prev_y + 10):
                text = ''
                nb_tabs = int((x1 - prev_x) / 50)
                tabs = ''
                for i in range (0, nb_tabs):
                    tabs += '    '
                otherTextsImages.initial_text_line2 += tabs + text
                prev_x = x2




    def __eq__(self, other):
        if isinstance(other, self.__class__):
            try:
                assert self.df.sort(by=['id']).equals(other.df.sort(by=['id']))
                return True
            except AssertionError:
                return False
        return False
