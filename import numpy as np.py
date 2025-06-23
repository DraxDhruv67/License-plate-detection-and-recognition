import numpy as np
import cv2
import imutils
import pytesseract
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class NumberPlateDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Number Plate Detector")
        
        self.label = tk.Label(root, text="Select an image to detect number plate:")
        self.label.pack(pady=10)

        self.btn_select = tk.Button(root, text="Select Image", command=self.load_image)
        self.btn_select.pack(pady=10)

        self.result_label = tk.Label(root, text="")
        self.result_label.pack(pady=10)

        self.canvas = tk.Canvas(root, width=500, height=500)
        self.canvas.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        image = cv2.imread(file_path)
        image = imutils.resize(image, width=500)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 170, 200)

        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        NumberPlateCnt = None
        idx = 7 
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:30]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4: 
                NumberPlateCnt = approx
                x, y, w, h = cv2.boundingRect(c)
                new_img = gray[y:y + h, x:x + w]
                cv2.imwrite(f'Cropped Images-Text/{idx}.png', new_img)
                idx += 1
                break

        if NumberPlateCnt is not None:
            cv2.drawContours(image, [NumberPlateCnt], -1, (0, 255, 0), 3)
            cv2.imshow("Final Image With Number Plate Detected", image)
            cv2.waitKey(0)

            cropped_img_loc = f'Cropped Images-Text/{idx - 1}.png'
            text = pytesseract.image_to_string(cropped_img_loc, lang='eng')
            self.result_label.config(text=f"Detected Number: {text.strip()}")
            self.show_image(cropped_img_loc)
        else:
            messagebox.showerror("Error", "No number plate detected.")

    def show_image(self, img_path):
        img = Image.open(img_path)
        img = img.resize((250, 150), Image.ANTIALIAS)
        img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk 

if __name__ == "__main__":
    root = tk.Tk()
    app = NumberPlateDetectorApp(root)
    root.mainloop()