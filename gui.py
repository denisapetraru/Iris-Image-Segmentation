"""
@author: Ana-Maria
"""
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from PIL import ImageDraw
import tkinter as tk
from IPython.display import Image as IPImage, display
import cv2
import numpy as np
import matplotlib.pyplot as plt

selected_image = None
bw_img = None
flood_img = None
sobel_img = None

def open_image():
    global selected_image
    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff")])
    if file_path:
        try:
            image = Image.open(file_path)

            # Convert BMP to PNG
            if image.format == "BMP":
                image = image.convert("RGBA")
                image.save("temp_image.png", "PNG")
                file_path = "temp_image.png"

            img = IPImage(filename=file_path)
            display(img)

            # Store the opened image
            selected_image = Image.open(file_path)
        except Exception as e:
            messagebox.showerror("Error", str(e))

def threshold_image():
    global bw_img
    if selected_image:
        try:
            # Convert the image to grayscale
            grayscale_image = selected_image.convert("L")

            # Convert the PIL image to a NumPy array
            np_image = np.array(grayscale_image)

            # Apply thresholding
            ret, bw_img = cv2.threshold(np_image, 128, 255, cv2.THRESH_BINARY)

            # Convert the NumPy array back to a PIL image
            binary_image = Image.fromarray(bw_img)

            plt.figure()

            # Original Image
            plt.subplot(121)
            plt.title("Original Image,I")
            plt.imshow(np_image, cmap="gray")

            # Thresholded Image
            plt.subplot(122)
            plt.title("Thresholded Image,It")
            plt.imshow(bw_img, cmap="gray")
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", str(e))
    else:
        messagebox.showerror("Error", "No image selected.")


def flood_fill_algorithm():
    global flood_img, bw_img
    if selected_image:
        try:
            # Convert the image to grayscale
            grayscale_image = selected_image.convert("L")

            # Convert the PIL image to a NumPy array
            np_image = np.array(grayscale_image)

            # Invert the image (make black pixels white and vice versa)
            inverted_image = cv2.bitwise_not(bw_img)

            _, labels, stats, _ = cv2.connectedComponentsWithStats(inverted_image)

            # Find the label corresponding to the pupil=>assuming it's the second largest component
            pupil_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) +1

            # Create a mask for the pupil
            pupil_mask = (labels == pupil_label).astype(np.uint8) * 255

            flood_img = cv2.bitwise_and(inverted_image, inverted_image, mask=pupil_mask)

            # Invert the result back to the original format
            flood_img = cv2.bitwise_not(flood_img)

            plt.figure(figsize=(15, 5))

            # Original Image
            plt.subplot(131)
            plt.title("Original Image,I")
            plt.imshow(np_image, cmap="gray")

            # Thresholded Image
            plt.subplot(132)
            plt.title("Thresholded Image,It")
            plt.imshow(bw_img, cmap="gray")

            # Processed Image-flood fill alg
            plt.subplot(133)
            plt.title("Processed Image-flood fill alg,Itf")
            plt.imshow(flood_img, cmap="gray")

        except Exception as e:
            messagebox.showerror("Error", str(e))
    else:
        messagebox.showerror("Error", "No image selected.")

def edge_detection():
    global sobel_img, flood_img, bw_img
    if selected_image:
        try:
            # Convert the image to grayscale
            grayscale_image = selected_image.convert("L")
            
            # Convert the PIL image to a NumPy array
            np_image = np.array(grayscale_image)
            
            # Apply Sobel edge detection
            Igh = cv2.Sobel(np_image, cv2.CV_64F, 1, 0, ksize=3)
            Igv = cv2.Sobel(np_image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate Igb and normalize it
            sobel_combined = np.sqrt(Igh**2 + Igv**2)
            sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Calculate gradient orientations for (Igh, Igv)
            gradient_orientation = np.arctan2(Igv, Igh)
          
            # Generate the binary edge image (Igb) with only strong edges using a threshold
            threshold_value = 50
            _, Igb = cv2.threshold(sobel_combined, threshold_value, 255, cv2.THRESH_BINARY)
            
            plt.figure(figsize=(20, 5))
            # Original Image
            plt.subplot(141)
            plt.title("Original Image,I")
            plt.imshow(np_image, cmap="gray")
           
            # Thresholded Image
            plt.subplot(142)
            plt.title("Thresholded Image,It")
            plt.imshow(bw_img, cmap="gray")
           
            # Processed Image-flood fill alg
            plt.subplot(143)
            plt.title("Processed Image-flood fill alg,Itf")
            plt.imshow(flood_img, cmap="gray")
            
            # Sobel  Edge Detection
            plt.subplot(144)
            plt.title('Sobel Edge Detection,Igb')
            plt.imshow(sobel_combined, cmap='gray')
              
        except Exception as e:
            messagebox.showerror("Error", str(e))
    else:
        messagebox.showerror("Error", "No image selected.")
        
def detect_iris():
    global selected_image
    if selected_image:
        try:
            # Convert the image to grayscale
            grayscale_image = selected_image.convert("L")

            # Convert the PIL image to a NumPy array
            np_image = np.array(grayscale_image)

            # Apply Sobel edge detection or other techniques to enhance edges if needed
            edges = cv2.Canny(np_image, 100, 200)

            # Apply Hough Circle Transform to detect circles (iris boundary)
            min_radius = 10
            max_radius = 80
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=200, param1=100, param2=20, minRadius=min_radius, maxRadius=max_radius)

            # Overlay the detected circles on the original image
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    center = (circle[0], circle[1])
                    radius = circle[2]

                    # Draw the circle detected as the iris boundary
                    cv2.circle(np_image, center, radius,(240,255,255), 3)

                # Convert NumPy array back to PIL image for display
                result_image = Image.fromarray(np_image)
              
                plt.figure()
                plt.title('Detected Iris')
                plt.imshow(result_image, cmap='gray')
                plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))
    else:
        messagebox.showerror("Error", "No image selected.")

def exit_app():
    result = messagebox.askquestion("Exit", "Are you sure you want to exit the program?")
    if result == "yes":
        root.destroy()

# Create the main window
root = tk.Tk()
root.title("Prelucrarea numerica a imaginilor-team VSP")
root.geometry("1920x1080")

# Load and set the background image
background_image = Image.open("images/Iris-Makro für Paare.png")
background_image = ImageTk.PhotoImage(background_image)
background_label = tk.Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create the Select button
select_button = tk.Button(root, text="Select Image", command=open_image, width=17, height=2, fg="white", bg="#000000", borderwidth=0, relief="flat", font=("Segoe Script", 12))
select_button.place(x=840, y=355)

# Create the Threshold button
threshold_button = tk.Button(root, text="Threshold Image", command=threshold_image, width=17, height=2, fg="white", bg="#000000", borderwidth=0, relief="flat", font=("Segoe Script", 12))
threshold_button.place(x=840, y=455)

# Create the Flood Fill button
flood_fill_button = tk.Button(root, text="Flood Fill Algorithm", command=flood_fill_algorithm, width=17, height=2, fg="white", bg="#000000", borderwidth=0, relief="flat", font=("Segoe Script", 12))
flood_fill_button.place(x=840, y=555)


# Create the Sobel Edge Detection button
edge_button = tk.Button(root, text="Sobel Edge Detection", command=edge_detection, width=17, height=2, fg="white", bg="#000000", borderwidth=0, relief="flat", font=("Segoe Script", 12))
edge_button.place(x=840, y=655)


# Create the Exit button
exit_button = tk.Button(root, text="Exit", command=exit_app, width=17, height=2, fg="white", bg="#000000", borderwidth=0, relief="flat", font=("Segoe Script", 12))
exit_button.place(x=840, y=855)

hough_button = tk.Button(root, text="Final Image,Ipr", command=detect_iris,width=17, height=2, fg="white", bg="#000000", borderwidth=0, relief="flat",font=("Segoe Script", 12))
hough_button.place(x=840, y=755)

# Text label
text_label = tk.Label(root, text="IRIS IMAGE SEGMENTATION", fg="white", bg="#000000", font=("Segoe Script", 20))
text_label.place(x=700, y=90)

# Main loop start
root.mainloop()
