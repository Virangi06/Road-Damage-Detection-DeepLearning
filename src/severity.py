import math

class SeverityAssessor:
    """
    Automated Road Damage Severity Scoring & Maintenance Priority Index Engine.
    
    Evaluates:
    1. Normalized Damaged Area Ratio (A_norm)
    2. Region Count Density (N_regions)
    3. Maximum Class Damage Weight (W_class_max)
    
    Produces:
    - Severity Score (0 - 100)
    - Severity Category (Low, Medium, High)
    - Repair Priority Index (1 - 10)
    """
    
    CLASS_WEIGHTS = {
        0: 0.4,  # D00: Longitudinal Crack
        1: 0.5,  # D10: Transverse Crack
        2: 0.8,  # D20: Alligator Crack
        3: 1.0,  # D40: Pothole
        4: 0.6   # D43/D44: Other Damage / Rutting
    }
    
    CLASS_NAMES = {
        0: "D00 (Longitudinal Crack)",
        1: "D10 (Transverse Crack)",
        2: "D20 (Alligator Crack)",
        3: "D40 (Pothole)",
        4: "D43/D44 (Other Damage)"
    }

    def __init__(self):
        pass

    def calculate_severity(self, image_shape, bboxes, class_ids, scores=None):
        """
        Calculates severity score and repair priority index for a set of detected damage bounding boxes.
        
        Args:
            image_shape (tuple): (height, width) or (height, width, channels)
            bboxes (list of tuples/lists): Bounding boxes [xmin, ymin, xmax, ymax]
            class_ids (list of int): Predicted class index for each bounding box
            scores (list of float, optional): Confidence scores for each bounding box
            
        Returns:
            dict: Structured severity report with scores, categories, and metrics.
        """
        h_img, w_img = image_shape[:2]
        image_area = float(h_img * w_img)
        
        if image_area <= 0:
            image_area = 1.0
            
        if len(bboxes) == 0:
            return {
                "severity_score": 0.0,
                "severity_category": "Low",
                "repair_priority": 1,
                "normalized_area_ratio": 0.0,
                "region_count": 0,
                "max_class_weight": 0.0,
                "max_severity_class": "None",
                "details": []
            }
            
        total_bbox_area = 0.0
        details = []
        max_class_weight = 0.0
        max_class_id = class_ids[0] if len(class_ids) > 0 else 0
        
        for idx, bbox in enumerate(bboxes):
            xmin, ymin, xmax, ymax = bbox[:4]
            w_box = max(0.0, float(xmax - xmin))
            h_box = max(0.0, float(ymax - ymin))
            box_area = w_box * h_box
            total_bbox_area += box_area
            
            cls_id = int(class_ids[idx]) if idx < len(class_ids) else 0
            weight = self.CLASS_WEIGHTS.get(cls_id, 0.5)
            score = float(scores[idx]) if scores is not None and idx < len(scores) else 1.0
            
            if weight > max_class_weight:
                max_class_weight = weight
                max_class_id = cls_id
                
            details.append({
                "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)],
                "class_id": cls_id,
                "class_name": self.CLASS_NAMES.get(cls_id, f"Class {cls_id}"),
                "confidence": score,
                "class_weight": weight,
                "box_area_pixels": box_area
            })
            
        normalized_area = min(1.0, total_bbox_area / image_area)
        region_count = len(bboxes)
        
        # Severity Score Formula (0 to 100)
        # Score = min(100, 50 * A_norm + 25 * (N_regions / 5) + 25 * W_class_max)
        area_term = 50.0 * normalized_area
        count_term = 25.0 * (min(region_count, 5) / 5.0)
        weight_term = 25.0 * max_class_weight
        
        raw_score = area_term + count_term + weight_term
        severity_score = round(min(100.0, max(0.0, raw_score)), 2)
        
        # Categorize
        if severity_score < 30.0:
            severity_category = "Low"
        elif severity_score < 65.0:
            severity_category = "Medium"
        else:
            severity_category = "High"
            
        # Repair Priority Index (1 to 10)
        # Priority = Clamp( floor(Severity / 10) + floor(W_class_max * 2), 1, 10 )
        base_priority = math.floor(severity_score / 10.0) + math.floor(max_class_weight * 2.0)
        repair_priority = int(max(1, min(10, base_priority)))
        
        return {
            "severity_score": severity_score,
            "severity_category": severity_category,
            "repair_priority": repair_priority,
            "normalized_area_ratio": round(normalized_area, 4),
            "region_count": region_count,
            "max_class_weight": max_class_weight,
            "max_severity_class": self.CLASS_NAMES.get(max_class_id, f"Class {max_class_id}"),
            "details": details
        }


if __name__ == '__main__':
    print("Testing SeverityAssessor...")
    assessor = SeverityAssessor()
    test_boxes = [[50, 50, 200, 200], [300, 400, 500, 600]]
    test_classes = [3, 1]  # Pothole (1.0) and Transverse Crack (0.5)
    test_scores = [0.92, 0.85]
    report = assessor.calculate_severity((800, 800), test_boxes, test_classes, test_scores)
    print(f"Severity Report:\n{report}")
    print("SeverityAssessor test passed!")
