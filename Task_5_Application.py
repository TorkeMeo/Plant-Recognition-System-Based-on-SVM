
# this fulfill Task 5 requirement
import tkinter as tk
from tkinter import filedialog, Label, Frame, Button
import cv2
import numpy as np
import joblib
import json
import os
from PIL import Image, ImageTk
from skimage.feature import hog, local_binary_pattern
#resize
IMG_SIZE = (128, 128)
GT_FOLDER = "GT"                     #Folder containing ground truth images
GT_FILE_PATTERN = "{}_ground_truth.jpg"  # e.g. Cordyline_fruticosa_ground_truth.jpg

# same as training
def gabor_features(gray):
    gabor_feats = []
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    lambdas = [4.0, 8.0, 16.0]
    for theta in thetas:
        for lambd in lambdas:
            kernel = cv2.getGaborKernel((21,21), sigma=5.0, theta=theta, lambd=lambd, gamma=0.5, psi=0)
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            mean = np.mean(filtered)
            std = np.std(filtered)
            gabor_feats.extend([mean, std])
    return np.array(gabor_feats)

def color_moments(hsv):
    moments = []
    for i in range(3):
        channel = hsv[:,:,i].flatten().astype(np.float32)
        mean = np.mean(channel)
        moments.append(mean)
        std = np.std(channel)
        moments.append(std)
        skew = np.mean(((channel - mean) ** 3))
        moments.append(skew)
    return np.array(moments)

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # HOG
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(36,36), cells_per_block=(2,2),
                   block_norm='L2-Hys', transform_sqrt=True, feature_vector=True)

    # LBP
    lbp = local_binary_pattern(gray, 8, 1, 'uniform')
    n_bins = 8 + 2
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins+1), range=(0, n_bins))
    lbp_hist = lbp_hist.astype('float') / (lbp_hist.sum() + 1e-6)

    # Color histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_feat = []
    for i in range(3):
        hist = cv2.calcHist([hsv], [i], None, [64], [0,256])
        hist = hist / (hist.sum() + 1e-6)
        color_feat.extend(hist.flatten())
    color_feat = np.array(color_feat)

    # Color moments
    color_mom_feat = color_moments(hsv)

    # Gabor
    gabor_feat = gabor_features(gray)

    return np.hstack([hog_feat, lbp_hist, color_feat, color_mom_feat, gabor_feat])
model = None
scaler = None
class_names = None
gt_images = {}

def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if not file_path:
        return

    # Display selected image
    img = Image.open(file_path)
    img.thumbnail((400, 400))
    photo = ImageTk.PhotoImage(img)
    img_display.config(image=photo)
    img_display.image = photo

    # Classification
    try:
        feat = extract_features(file_path)
        feat_scaled = scaler.transform([feat])
        pred_id = model.predict(feat_scaled)[0]
        pred_name = class_names[pred_id]
        prob = model.predict_proba(feat_scaled)[0]
        confidence = prob[pred_id]
        result_label.config(text=f"Prediction: {pred_name}\nConfidence: {confidence:.4f}")

        # Show ground truth image for the predicted class
        gt_path = gt_images.get(pred_name)
        if gt_path and os.path.exists(gt_path):
            gt_img = Image.open(gt_path)
            gt_img.thumbnail((400, 400))
            gt_photo = ImageTk.PhotoImage(gt_img)
            gt_display.config(image=gt_photo)
            gt_display.image = gt_photo
        else:
            gt_display.config(image='')
            gt_display.image = None
    except Exception as e:
        result_label.config(text=f"Prediction failed: {e}")

def exit_app():
    root.quit()

if __name__ == "__main__":
    # Load model, scaler, class names
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("class_names.json", "r", encoding='utf-8') as f:
        class_names = json.load(f)

    # Build ground truth image mapping
    for cls in class_names:
        gt_path = os.path.join(GT_FOLDER, GT_FILE_PATTERN.format(cls))
        if os.path.exists(gt_path):
            gt_images[cls] = gt_path
        else:
            print(f"Warning: Ground truth image not found for {cls}: {gt_path}")

    # Create main window: 900x750
    root = tk.Tk()
    root.title("Campus Vegetation Classification")
    root.geometry("900x750")

    #user guideline
    guide_text = (
        "User instruction\n"
        "1. Click 'Load Image' to select a plant photo (jpg, jpeg, png, bmp).\n"
        "2. The system will automatically extract features and predict the species.\n"
        "3. The predicted name and confidence score will appear below.\n"
        "4. A reference image (ground truth) of the predicted species will be shown on the right.\n"
        "5. To exit, click 'Exit'."
    )
    guide_label = tk.Label(root, text=guide_text, font=("Arial", 10), justify=tk.LEFT, bg="#f0f0f0",fg="blue", relief=tk.GROOVE)
    guide_label.pack(pady=5, padx=10, fill=tk.X)

    #botton setting
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    btn_load = Button(button_frame, text="Load Image", command=load_image, bg="pink", fg="black", font=("Arial", 12))
    btn_load.pack(side=tk.LEFT, padx=10)

    btn_exit = Button(button_frame, text="Exit", command=exit_app, bg="lightgray", fg="red", font=("Arial", 12))
    btn_exit.pack(side=tk.LEFT, padx=10)

    img_frame = Frame(root)
    img_frame.pack(pady=10)

    # Show selected pictutre
    img_left_label = Label(img_frame, text="Selected Image", font=("Arial", 10))
    img_left_label.grid(row=0, column=0, padx=10)
    img_display = Label(img_frame)
    img_display.grid(row=1, column=0, padx=10)

    #Show ground truth
    img_right_label = Label(img_frame, text="Reference (Ground Truth)", font=("Arial", 10))
    img_right_label.grid(row=0, column=1, padx=10)
    gt_display = Label(img_frame)
    gt_display.grid(row=1, column=1, padx=10)

    #result
    result_label = Label(root, text="", font=("Arial", 14), fg="green")
    result_label.pack(pady=10)

    root.mainloop()
