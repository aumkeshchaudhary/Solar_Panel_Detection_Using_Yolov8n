from ultralytics import YOLO
import matplotlib.pyplot as plt
import supervision as sv
from supervision.draw.color import ColorPalette
import cv2
import numpy as np
from supervision.metrics.detection import (
    MeanAveragePrecision,
    ConfusionMatrix
)

# Initialize and train the model
def train_model():
    # Load a pretrained YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    results = model.train(
        data='solar_panel_dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,  # Early stopping patience
        save=True,    # Save best model
        device='0'    # Use GPU if available
    )
    
    # Plot training results
    plt.figure(figsize=(12, 8))
    results.plot()
    plt.savefig('training_results.png')
    plt.close()
    
    return model

def evaluate_model(model, test_images, ground_truth):
    predictions = []
    for img_path in test_images:
        results = model.predict(img_path)
        predictions.append(results[0])
    
    # Initialize metrics
    metric_map = MeanAveragePrecision()
    
    # Update metrics with predictions and ground truth
    metric_map.update(ground_truth, predictions)
    
    # Get mAP50 score
    map_dict = metric_map.compute()
    print(f"mAP50: {map_dict['map50']:.4f}")
    
    return map_dict

def compute_metrics_matrix(model, test_images, ground_truth, iou_thresholds, conf_thresholds):
    results = []
    
    for iou_thresh in iou_thresholds:
        row = []
        for conf_thresh in conf_thresholds:
            confusion_matrix = ConfusionMatrix(iou_threshold=iou_thresh)
            
            # Get predictions with current confidence threshold
            predictions = []
            for img_path in test_images:
                pred = model.predict(img_path, conf=conf_thresh)
                predictions.append(pred[0])
            
            # Update confusion matrix
            confusion_matrix.update(ground_truth, predictions)
            
            # Calculate metrics
            tp, fp, fn = confusion_matrix.matrix.ravel()[:3]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            row.append({
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            })
        results.append(row)
    
    return results

def visualize_predictions(model, test_images, num_samples=4):
    # Select random test images
    sample_images = np.random.choice(test_images, num_samples, replace=False)
    
    for img_path in sample_images:
        # Load and predict
        img = cv2.imread(img_path)
        results = model.predict(img_path)
        
        # Get ground truth
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        with open(label_path, 'r') as f:
            gt_labels = f.readlines()
        
        # Create annotators
        pred_annotator = sv.BoxAnnotator(color=ColorPalette.default[0])
        gt_annotator = sv.BoxAnnotator(color=ColorPalette.default[1])
        
        # Draw predictions and ground truth
        img_with_preds = pred_annotator.annotate(img, detections=results[0].boxes)
        img_with_all = gt_annotator.annotate(img_with_preds, detections=gt_labels)
        
        # Display
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img_with_all, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # Train model
    model = train_model()
    
    # Define thresholds for evaluation
    iou_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    conf_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Get test images
    import glob
    test_images = glob.glob('/content/dataset/test/images/*.jpg')
    
    # Evaluate and visualize
    metrics = compute_metrics_matrix(model, test_images, None, iou_thresholds, conf_thresholds)
    visualize_predictions(model, test_images)
