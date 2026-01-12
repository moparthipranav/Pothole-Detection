import torch
import cv2
from pathlib import Path
from src.model.YoloModel import YOLOModel
from utils.logger import logging

class PredictPipeline:
    """Inference pipeline for object detection"""
    
    def __init__(self, model_path: str, device: str = None, n_classes: int = 80):
        """
        Initialize prediction pipeline
        
        :param model_path: Path to saved model checkpoint
        :param device: Device to use ('cuda' or 'cpu')
        :param n_classes: Number of classes
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_classes = n_classes
        
        self.model = YOLOModel(
            base_channels=64,
            base_depth=2,
            n_classes=n_classes
        ).to(self.device)
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            logging.info(f"Loaded model from {model_path}")
        else:
            logging.warning(f"Model checkpoint not found at {model_path}")
        
        self.model.eval()
    
    def preprocess_image(self, image_path: str, img_size: int = 416):
        """
        Load and preprocess image
        
        :param image_path: Path to image
        :param img_size: Target image size
        :return: Preprocessed tensor
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        image = cv2.resize(image, (img_size, img_size))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image.unsqueeze(0).to(self.device), (h, w)
    
    def predict(self, image_path: str, conf_threshold: float = 0.5, img_size: int = 640):
        """
        Run prediction on image
        
        :param image_path: Path to image
        :param conf_threshold: Confidence threshold for detections
        :param img_size: Target image size
        :return: Boxes and scores
        """
        image, orig_size = self.preprocess_image(image_path, img_size)
        
        with torch.no_grad():
            boxes, scores = self.model(image)
        
        # Filter by confidence
        max_scores = scores.max(dim=-1)[0]
        valid_mask = max_scores > conf_threshold
        
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        
        return boxes, scores

def predict_pipeline(image_path: str, model_path: str, conf_threshold: float = 0.5):
    """Convenience function for single prediction"""
    pipeline = PredictPipeline(model_path)
    boxes, scores = pipeline.predict(image_path, conf_threshold)
    return boxes, scores
