# Table-extractor
A repository for table extractor from image project
Not yet finished.

To run the project users first have to install "Python" and "Tesseract OCR" and put them in the "PATH" environment variable.
Steps to run the project:
- Clone the repo into your prefered directory.
- Open a terminal and type "python -m venv .venv" to create a virtual environment named ".venv".
- Next type ".venv/Scripts/activate" to activate virtual environment.
- Type "pip install -r requirements.txt" to install dependency packages.
- If the terminal return Error with "Module not found: No module named 'PIL'", then type "pip install Pillow" to install the Pillow package.
- Finally, run the scripts by "cd src" and "python main.py", the input images is stored in "input_images" folder, and the output excel files will be in the "output" folder.
