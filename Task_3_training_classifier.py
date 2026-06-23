# from 24110119d YU Fenggang
# this fulfill Task 3 requirement
import os
import cv2
import numpy as np
import joblib
import json
import time
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#get data
DATA_ROOT = "Task_1-2_Total_Dataset/Task_1-2_train"
TRAIN_PATH = os.path.join(DATA_ROOT)
IMG_SIZE = (128, 128)

# For HOG
HOG_ORIENTATIONS = 9                # gradient direction
HOG_PIXELS_PER_CELL = (36, 36)        # cell size
HOG_CELLS_PER_BLOCK = (2, 2)        # Block size
HOG_BLOCK_NORM = 'L2-Hys'            # Normalization method
HOG_TRANSFORM_SQRT = True

# For LBP
LBP_RADIUS = 1                       # radius
LBP_N_POINTS = 8                    # get point
LBP_METHOD = 'uniform'             # uniform mode

# HSV
COLOR_BINS = 64                      #bin number
COLOR_RANGE = [0, 256]               # range of channel

# I use random seed to make sure my result can be show again
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# SVM with RBF kernel
PARAM_GRID = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.001, 0.01, 0.1],
    'kernel': ['rbf']
}

#get 20% of my dataset as val set
VAL_SPLIT = 0.2

MODEL_SAVE_PATH = "model.pkl"
SCALER_SAVE_PATH = "scaler.pkl"
CLASS_NAMES_SAVE_PATH = "class_names.json"
LOG_SAVE_PATH = "train_log.json"

#feature extraction
def extract_features(image_path):
    #read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"There is problem with {image_path}")
    img = cv2.resize(img, IMG_SIZE)

    # Turn to Black&Gray&white picture
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # HOG
    hog_feat = hog(gray, orientations=HOG_ORIENTATIONS,
                   pixels_per_cell=HOG_PIXELS_PER_CELL,
                   cells_per_block=HOG_CELLS_PER_BLOCK,
                   block_norm=HOG_BLOCK_NORM,
                   transform_sqrt=HOG_TRANSFORM_SQRT,
                   feature_vector=True)

    # LBP
    lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, LBP_METHOD)
    n_bins = LBP_N_POINTS + 2
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins+1), range=(0, n_bins))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-6)

    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # color hist
    color_hist = []
    for i in range(3):
        hist = cv2.calcHist([hsv], [i], None, [COLOR_BINS], COLOR_RANGE)
        hist = hist / (hist.sum() + 1e-6)
        color_hist.extend(hist.flatten())
    color_hist = np.array(color_hist)

    # color moments
    color_mom = color_moments(hsv)

    # gobar
    gabor_feat = gabor_features(gray)

    #add them together as feature vector
    features = np.hstack([hog_feat, lbp_hist, color_hist, color_mom, gabor_feat])
    return features


def load_dataset(data_path):
    X = []
    y = []
    image_paths = []  # 可选，用于调试

    # get my label
    class_names = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    print(f"Get: {class_names}")

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_path, class_name)
        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        for fname in tqdm(image_files):
            path = os.path.join(class_dir, fname)
            try:
                feat = extract_features(path)
                X.append(feat)
                y.append(class_idx)
                image_paths.append(path)
            except Exception as e:
                print(f"mistake {path}: {e}")

    return np.array(X), np.array(y), class_names

# Get Gabor feature vector
def gabor_features(gray):
    gabor_feats = []
    # Direction and size
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    lambdas = [4.0, 8.0, 16.0]
    for theta in thetas:
        for lambd in lambdas:
            # choose kernel
            kernel = cv2.getGaborKernel((21,21), sigma=5.0, theta=theta, lambd=lambd, gamma=0.5, psi=0)
            # filter
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            # Get vector
            mean = np.mean(filtered)
            std = np.std(filtered)
            gabor_feats.extend([mean, std])
    return np.array(gabor_feats)

# color moments
def color_moments(hsv):
    moments = []
    for i in range(3):  # H/S/V in hsv
        channel = hsv[:,:,i].flatten().astype(np.float32)
        #calculation
        mean = np.mean(channel)
        moments.append(mean)
        std = np.std(channel)
        moments.append(std)
        skew = np.mean(((channel - mean) ** 3))
        moments.append(skew)
    return np.array(moments)

if __name__ == "__main__":
    X_train_full, y_train_full, class_names = load_dataset(TRAIN_PATH)
    print(f"feature dimension: {X_train_full.shape[1]}")
    unique, counts = np.unique(y_train_full, return_counts=True)
    min_samples = counts.min() #get minimum sample

    #make sure I can do the cross-validation
    if min_samples < 2:
        min_class_idx = np.argmin(counts)
        min_class_name = class_names[min_class_idx]
        raise ValueError(
            f"Too little samples in '{min_class_name}', {min_samples} is less than 2\n"
        )
    elif min_samples < 3:
        n_splits = 2
    else:
        n_splits = 3

    # get val set
    X_train, X_val, y_train, y_val = train_test_split(
          X_train_full, y_train_full, test_size=VAL_SPLIT, random_state=RANDOM_SEED, stratify=y_train_full
    )
    print(f"train set: {len(X_train)}")
    print(f"val set: {len(X_val)}")

    # normalized
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # SVM training
    svm_base = SVC(class_weight='balanced', random_state=RANDOM_SEED, probability=True)
    grid_search = GridSearchCV(
        svm_base, PARAM_GRID, cv=n_splits, scoring='accuracy', n_jobs=-1, verbose=1
    )
    start_time = time.time()
    grid_search.fit(X_train_scaled, y_train)
    end_time = time.time()
    print(f"Time: {end_time - start_time:.2f} 秒")
    print(f"Parameter choose: {grid_search.best_params_}")
    print(f"CV accuracy: {grid_search.best_score_:.4f}")

    # Evaluate on val set to get best model
    best_model = grid_search.best_estimator_
    y_val_pred = best_model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"validation set accuracy: {val_acc:.4f}")
    # save output
    joblib.dump(best_model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    with open(CLASS_NAMES_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)

    #save log for debugging
    log = {
        "dataset": {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "total_classes": len(class_names),
            "class_names": class_names,
        },
        "feature_extraction": {
            "image_size": IMG_SIZE,
            "hog": {
                "orientations": HOG_ORIENTATIONS,
                "pixels_per_cell": HOG_PIXELS_PER_CELL,
                "cells_per_block": HOG_CELLS_PER_BLOCK,
                "block_norm": HOG_BLOCK_NORM,
                "transform_sqrt": HOG_TRANSFORM_SQRT,
            },
            "lbp": {
                "radius": LBP_RADIUS,
                "n_points": LBP_N_POINTS,
                "method": LBP_METHOD,
            },
            "color_histogram": {
                "bins_per_channel": COLOR_BINS,
                "color_space": "HSV",
            },
            "feature_dimension": X_train.shape[1],
        },
        "model_training": {
            "model_type": "SVM with RBF kernel",
            "grid_search_params": PARAM_GRID,
            "cv_folds": n_splits,
            "best_params": grid_search.best_params_,
            "best_cv_accuracy": grid_search.best_score_,
            "validation_accuracy": val_acc,
            "training_time_sec": end_time - start_time,
        },
        "random_seed": RANDOM_SEED,
    }
    with open(LOG_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"model : {MODEL_SAVE_PATH}")
    print(f"scaler: {SCALER_SAVE_PATH}")
    print(f"class/label name: {CLASS_NAMES_SAVE_PATH}")
    print(f"log {LOG_SAVE_PATH}")