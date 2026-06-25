
import os
import cv2
import numpy as np
import joblib
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

# Setting path
TEST_PATH = "Task_1-2_Total_Dataset/Task_1-2_test"                     # test inside has test dataset
#model location
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
CLASS_NAMES_PATH = "class_names.json"
OUTPUT_DIR = "evaluation_results"
#resize img
IMG_SIZE = (128, 128)

# gabor (same as code in task 3)
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
# color moments (same as code in task 3)
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
    if img is None:
        raise ValueError("img error")
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

    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_feat = []
    for i in range(3):
        hist = cv2.calcHist([hsv], [i], None, [64], [0,256])
        hist = hist / (hist.sum() + 1e-6)
        color_feat.extend(hist.flatten())
    color_feat = np.array(color_feat)

    # color moments
    color_mom_feat = color_moments(hsv)

    # Gabor
    gabor_feat = gabor_features(gray)

    # add together
    return np.hstack([hog_feat, lbp_hist, color_feat, color_mom_feat, gabor_feat])

def load_test_set(data_path, class_names):
    X, y, paths = [], [], []
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_path, class_name)
        if not os.path.isdir(class_dir):
            print(f"No file {class_dir}")
            continue
        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        for fname in tqdm(image_files):
            path = os.path.join(class_dir, fname)
            try:
                feat = extract_features(path)
                X.append(feat)
                y.append(class_idx)
                paths.append(path)
            except Exception as e:
                print(f"Error on {path}: {e}")
    return np.array(X), np.array(y), paths

if __name__ == "__main__":
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    examples_correct = os.path.join(OUTPUT_DIR, "examples/correct")
    examples_wrong = os.path.join(OUTPUT_DIR, "examples/wrong")
    os.makedirs(examples_correct, exist_ok=True)
    os.makedirs(examples_wrong, exist_ok=True)

    # Load model we get from task 4
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Loading success")
    with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
        class_names = json.load(f)
    print(f"Label: {class_names}")

    X_test, y_true, img_paths = load_test_set(TEST_PATH, class_names)
    print(f"Test set pic amount: {len(X_test)}")

    #normalized
    X_test_scaled = scaler.transform(X_test)

    # classify
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    # acuuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"accuracy: {acc:.4f}")

    # save report
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
    print("save report to classification_report.txt")

    # Get the report of each label
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # get P,R,F1, paint
    per_class_metrics = []
    for cls in class_names:
        metrics = report_dict[cls]
        per_class_metrics.append({
            'Class': cls,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-score': metrics['f1-score'],
            'Support': metrics['support']
        })

    # save as .csv
    df_metrics = pd.DataFrame(per_class_metrics)
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, "per_class_metrics.csv"), index=False)
    print("The detailed report of each category saved to per_class_metrics.csv")

    # Draw Precision, Recall, F1-score
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(class_names))
    width = 0.25

    bars1 = ax.bar(x - width, df_metrics['Precision'], width, label='Precision', color='skyblue')
    bars2 = ax.bar(x, df_metrics['Recall'], width, label='Recall', color='lightgreen')
    bars3 = ax.bar(x + width, df_metrics['F1-score'], width, label='F1-score', color='salmon')

    ax.set_xlabel('Plant Species')
    ax.set_ylabel('Score')
    ax.set_title('Per-class Precision, Recall, and F1-score')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=100, ha='right')
    ax.legend()

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "per_class_metrics_bar.png"))
    print("Picture save to per_class_metrics_bar.png")
    plt.close(fig)

    #confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    print("Confusion matrix saved to evaluation_results/confusion_matrix.png")

    # save classification result
    df = pd.DataFrame({
        'image_path': img_paths,
        'true_label': [class_names[i] for i in y_true],
        'pred_label': [class_names[i] for i in y_pred],
        'confidence': np.max(y_prob, axis=1)
    })
    df.to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)
    print("Result saved: evaluation_results/results.csv")

    # Error classified picture
    wrong_count = 0
    for i, (true, pred, path) in enumerate(zip(y_true, y_pred, img_paths)):
        if true != pred and wrong_count < 40:
            dst = os.path.join(examples_wrong, f"wrong_{i}_true_{class_names[true]}_pred_{class_names[pred]}.jpg")
            shutil.copy2(path, dst)
            wrong_count += 1
    print(f"Total {wrong_count} wrong classification example are saved to {examples_wrong}")

    # Correct part
    correct_count = 0
    for i, (true, pred, path) in enumerate(zip(y_true, y_pred, img_paths)):
        if true == pred and correct_count < 40:
            dst = os.path.join(examples_correct, f"correct_{i}_{class_names[true]}.jpg")
            shutil.copy2(path, dst)
            correct_count += 1
    print(f"Total {correct_count} correct classification examples are saved to {examples_correct}")
