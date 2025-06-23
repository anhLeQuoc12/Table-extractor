import os
from pathlib import Path

from img2table.document import Image
from img2table.ocr import TesseractOCR

dir_name = "../input_images/"
images_paths = Path(dir_name).iterdir()
tesseract_ocr = TesseractOCR(n_threads=1, lang="eng")

for index, image_path in enumerate(images_paths):
    image_from_path = Image(src=str(image_path))
    output_file_name = f"../output/{image_path.stem}.xlsx"
    image_from_path.to_xlsx(output_file_name,
                        ocr=tesseract_ocr,
                        implicit_rows=False,
                        implicit_columns=False,
                        borderless_tables=True,
                        min_confidence=50
                        )

# extracted_tables = img_from_path.extract_tables(ocr=tesseract_ocr)
# for table in extracted_tables:
#     for id_row, row in enumerate(table.content.values()):
#         for id_col, col in enumerate(row):
#             x1 = col.bbox.x1
#             y1 = col.bbox.y1
#             x2 = col.bbox.x2
#             y2 = col.bbox.y2
#             print(f"Cell: ({x1}, {y1}, {x2}, {y2}), with text: {col.value}") 
