import tkinter as tk
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from detector import *


# initialize Detector() class object
detector = Detector()
detector.load_model()
prediction = ''


def analyze_file():
    """
    function to run analyzing image process
    """
    global prediction
    file_path = askopenfilename(filetypes=[("image", ".jpeg"), ("image", ".png"), ("image", ".jpg")])
    logger.info(f'Analyze file from {file_path}...')
    if not file_path:
        return
    prediction = detector.predict_image(image_path=file_path, threshold=THRESHOLD)
    prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)
    image = ImageTk.PhotoImage(image=Image.fromarray(prediction))
    label.configure(image=image)
    label.image = image
    label.grid(row=0, column=1)


def save_file():
    """
    function to save analyzed file to the chosen path
    """
    global prediction
    filepath = asksaveasfilename(
        defaultextension=".jpg",
        filetypes=[("image", ".jpeg"), ("image", ".png"), ("image", ".jpg")],
    )
    if not filepath:
        return
    logger.info(f'Write analyzed file to {filepath}')
    image = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(filepath), image)


def close_program():
    """
    function to exit the program
    """
    logger.info('Bye!')
    window.destroy()


def crete_buttons():
    """
    function to create all buttons for UI
    """
    fr_buttons = tk.Frame(window, relief=tk.RAISED, bd=2)
    btn_analyze = tk.Button(fr_buttons, text='Аналіз зображення', command=analyze_file)
    btn_save = tk.Button(fr_buttons, text="Зберегти як...", command=save_file)
    btn_exit = tk.Button(fr_buttons, text="Вихід", command=close_program)
    btn_analyze.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
    btn_save.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
    btn_exit.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
    fr_buttons.grid(row=0, column=0, sticky="ns")


if __name__ == '__main__':
    window = tk.Tk()
    window.title("Face Detection System")
    window.rowconfigure(0, minsize=600, weight=1)
    window.columnconfigure(1, minsize=800, weight=1)
    window.configure(background='white')
    crete_buttons()
    logo = Image.open('logo.png')
    logo = ImageTk.PhotoImage(logo)
    label = tk.Label(window, image=logo)
    label.image = logo
    label.grid(row=0, column=1)
    window.mainloop()
