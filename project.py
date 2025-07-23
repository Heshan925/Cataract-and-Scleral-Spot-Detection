import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
from PIL import Image, ImageTk
import threading

class CataractDetection:
    """
    Your existing cataract detection class
    Expected to return: original_image, processed_image, details
    """
    def __init__(self):
        pass
    
    def detect(self, image_path):

        original_image, processed_image, details = self.cataract_detection(image_path)
        return original_image, processed_image, details
    
    def cataract_detection(self,image_address):
        """
        Enhanced cataract detection with improved accuracy
        """
        # Load and validate image
        image = CataractDetection.load_image(image_address)
        if image is None:
            return
        
        # Detect iris with multiple methods for better accuracy
        iris_detected, iris_image, iris_info = CataractDetection.detect_iris_enhanced(image)
        
        
        if iris_info is None:
            print("Error: Could not detect iris in the image")
            return
        
        # Calculate affected area with improved methods
        affected_iris_image, affected_info = CataractDetection.analyze_affected_area(iris_image, iris_info)
        
        # Calculate percentage
        percentage = CataractDetection.calculate_affected_percentage(iris_info, affected_info)
        
        CataractDetection.displayResults(iris_info,affected_info,percentage)
       
        return image, affected_iris_image, percentage
        
    @staticmethod
    def load_image(image_address):
        try:
            image = cv2.imread(image_address, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Error: Could not load image from {image_address}")
                return None
            
            # Resize if image is too large (for processing efficiency)
            height, width = image.shape[:2]
            if max(height, width) > 1000:
                scale = 1000 / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    @staticmethod
    def detect_iris_enhanced(image):
        """
        Enhanced iris detection
        """
        processed_image, original_gray = CataractDetection.preprocess_image(image)
        
        # Method 1: Enhanced Hough Circle Transform with multiple parameter sets
        circles_hough = CataractDetection.detect_circles_hough_enhanced(processed_image, original_gray, image)
        
        # Method 2: Contour-based detection focusing on circular dark regions
        circles_contour = CataractDetection.detect_circles_contour_enhanced(original_gray, image)
        
        # Method 3: Edge-based circular detection
        circles_edge = CataractDetection.detect_circles_edge_based(original_gray, image)
        
        # Combine results and select best circle
        best_circle = CataractDetection.select_best_circle([circles_hough, circles_contour, circles_edge], image)
        
        if best_circle is None:
            return image, None, None
        
        x, y, r = best_circle
        
        # Draw circle on image
        iris_detected = image.copy()
        cv2.circle(iris_detected, (x, y), r, (0, 255, 0), 3)
        cv2.circle(iris_detected, (x, y), 2, (0, 0, 255), -1) 
        
        # Extract iris region with proper masking
        iris_image, iris_info = CataractDetection.extract_iris_region(image, x, y, r)
        
        return iris_detected, iris_image, iris_info

    @staticmethod
    def preprocess_image(image):
        """
        Enhanced image preprocessing specifically for iris detection in eye images
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE to enhance contrast between iris and sclera
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Enhance iris-sclera boundary using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Gaussian blur optimized for iris detection
        blurred = cv2.GaussianBlur(filtered, (7, 7), 0)
        
        return blurred, gray

    @staticmethod
    def detect_circles_hough_enhanced(processed_image, original_gray, original_image):
        """
        Enhanced Hough Circle Transform
        """
        height, width = original_image.shape[:2]
        
        
        # For close-up eye images, iris typically occupies 30-70% of image width
        min_radius = max(15, int(width * 0.15))
        max_radius = min(int(width * 0.4), int(height * 0.4))
        
        # Multiple parameter sets for different image conditions
        parameter_sets = [
            # Standard parameters for clear images
            {
                'dp': 1,
                'minDist': int(min(width, height) * 0.3),
                'param1': 50,
                'param2': 25,
                'minRadius': min_radius,
                'maxRadius': max_radius
            },
            # Sensitive parameters for challenging images
            {
                'dp': 1,
                'minDist': int(min(width, height) * 0.25),
                'param1': 40,
                'param2': 20,
                'minRadius': min_radius,
                'maxRadius': max_radius
            },
            # High precision parameters
            {
                'dp': 1,
                'minDist': int(min(width, height) * 0.35),
                'param1': 60,
                'param2': 30,
                'minRadius': min_radius,
                'maxRadius': max_radius
            }
        ]
        
        all_circles = []
        
        # Try multiple preprocessing approaches
        images_to_try = [
            processed_image,
            original_gray,
            CataractDetection.enhance_iris_boundary(original_gray),
            CataractDetection.create_iris_mask(original_gray)
        ]
        
        for img in images_to_try:
            if img is None:
                continue
                
            for params in parameter_sets:
                circles = cv2.HoughCircles(
                    img,
                    cv2.HOUGH_GRADIENT,
                    dp=params['dp'],
                    minDist=params['minDist'],
                    param1=params['param1'],
                    param2=params['param2'],
                    minRadius=params['minRadius'],
                    maxRadius=params['maxRadius']
                )
                
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    for circle in circles:
                        # Validate circle quality
                        if CataractDetection.validate_iris_circle(circle, original_gray):
                            all_circles.append(circle)
        
        return np.array(all_circles) if all_circles else None

    @staticmethod
    def enhance_iris_boundary(gray_image):
        """
        Enhance the boundary between iris and sclera for better circle detection
        """
        # Apply median filter to reduce noise
        median = cv2.medianBlur(gray_image, 5)
        
        # Use morphological gradient to highlight boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        gradient = cv2.morphologyEx(median, cv2.MORPH_GRADIENT, kernel)
        
        # Apply threshold to create binary edge image
        _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh

    @staticmethod
    def create_iris_mask(gray_image):
        """
        Create a mask highlighting potential iris regions (darker areas)
        """
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Use adaptive threshold to find darker regions
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
        )
        
        # Clean up using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned

    @staticmethod
    def validate_iris_circle(circle, gray_image):
        """
        validation for iris
        """
        x, y, r = circle
        height, width = gray_image.shape
        
        # Check if circle is within image bounds
        if x - r < 0 or x + r >= width or y - r < 0 or y + r >= height:
            return False
        
        # Check if circle center is not at image edges
        border_threshold = min(width, height) * 0.1
        if (x < border_threshold or x > width - border_threshold or 
            y < border_threshold or y > height - border_threshold):
            return False
        
        # Analyze intensity distribution inside and outside the circle
        mask_inside = np.zeros(gray_image.shape, dtype=np.uint8)
        cv2.circle(mask_inside, (x, y), r, 255, -1)
        
        mask_outside = np.zeros(gray_image.shape, dtype=np.uint8)
        cv2.circle(mask_outside, (x, y), int(r * 1.5), 255, -1)
        mask_outside = cv2.bitwise_and(mask_outside, cv2.bitwise_not(mask_inside))
        
        # Calculate mean intensities
        mean_inside = cv2.mean(gray_image, mask=mask_inside)[0]
        mean_outside = cv2.mean(gray_image, mask=mask_outside)[0]
        
        # Iris should be darker than surrounding sclera
        intensity_ratio = mean_inside / (mean_outside + 1e-6)
        
        # Valid iris should have intensity ratio < 0.85 
        return intensity_ratio < 0.85

    @staticmethod
    def detect_circles_contour_enhanced(gray_image, original_image):
        """
        Enhanced contour-based detection focusing on circular dark regions
        """
        height, width = original_image.shape[:2]
        
        # Create binary image highlighting darker regions 
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        min_area = (width * height) * 0.05  
        max_area = (width * height) * 0.6   
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > 0.5:  # Reasonably circular
                # Get enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                # Additional validation
                if (15 < radius < min(width, height) * 0.4 and
                    CataractDetection.validate_iris_circle([int(x), int(y), int(radius)], gray_image)):
                    circles.append([int(x), int(y), int(radius)])
        
        return np.array(circles) if circles else None

    @staticmethod
    def detect_circles_edge_based(gray_image, original_image):
        """
        Edge-based circular detection using Canny edge detection
        """
        height, width = original_image.shape[:2]
        
        # Apply Canny edge detection with multiple thresholds
        edges1 = cv2.Canny(gray_image, 50, 150)
        edges2 = cv2.Canny(gray_image, 30, 100)
        edges3 = cv2.Canny(gray_image, 70, 200)
        
        circles = []
        
        for edges in [edges1, edges2, edges3]:
            # Hough Circle Transform on edge image
            detected = cv2.HoughCircles(
                edges,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=int(min(width, height) * 0.3),
                param1=50,
                param2=15,  
                minRadius=max(15, int(width * 0.15)),
                maxRadius=min(int(width * 0.4), int(height * 0.4))
            )
            
            if detected is not None:
                detected = np.round(detected[0, :]).astype("int")
                for circle in detected:
                    if CataractDetection.validate_iris_circle(circle, gray_image):
                        circles.append(circle)
        
        return np.array(circles) if circles else None

    @staticmethod
    def detect_circles_contour(processed_image):
        """
        Contour-based circle detection
        """
        # Edge detection
        edges = cv2.Canny(processed_image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        for contour in contours:
            # Filter contours by area
            area = cv2.contourArea(contour)
            if area < 1000 or area > 50000:
                continue
            
            # Check if contour is roughly circular
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > 0.6:  # Threshold for circularity
                # Get enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if 20 < radius < 150:
                    circles.append([int(x), int(y), int(radius)])
        
        return np.array(circles) if circles else None

    @staticmethod
    def detect_circles_template(processed_image):
        """
        Template matching for circular patterns
        """
        # Create circular templates of different sizes
        templates = []
        for r in range(30, 120, 10):
            template = np.zeros((2*r+10, 2*r+10), dtype=np.uint8)
            cv2.circle(template, (r+5, r+5), r, 255, 2)
            templates.append((template, r))
        
        best_matches = []
        for template, radius in templates:
            result = cv2.matchTemplate(processed_image, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > 0.3: 
                x = max_loc[0] + radius + 5
                y = max_loc[1] + radius + 5
                best_matches.append([x, y, radius, max_val])
        
        if best_matches:
            # Sort by match quality and return best
            best_matches.sort(key=lambda x: x[3], reverse=True)
            return np.array([[best_matches[0][:3]]])
        
        return None

    @staticmethod
    def select_best_circle(circle_results, image):
        """
        Select the best circle from multiple detection methods with enhanced scoring
        """
        all_circles = []
        
        for circles in circle_results:
            if circles is not None:
                for circle in circles:
                    if len(circle) >= 3:
                        all_circles.append(circle[:3])
        
        if not all_circles:
            return None
        
        # Remove duplicate circles (similar position and size)
        unique_circles = CataractDetection.remove_duplicate_circles(all_circles)
        
        # Score circles based on multiple criteria
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        scored_circles = []
        
        for x, y, r in unique_circles:
            # 1. Position score 
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            position_score = 1 - (dist_from_center / max_dist)
            
            # 2. Size score 
            expected_radius = min(width, height) * 0.25  
            size_diff = abs(r - expected_radius) / expected_radius
            size_score = 1 / (1 + size_diff)
            
            # 3. Contrast score 
            contrast_score = CataractDetection.calculate_contrast_score(gray, x, y, r)
            
            # 4. Circularity score 
            circularity_score = CataractDetection.calculate_circularity_score(gray, x, y, r)
            
            # 5. Boundary score 
            boundary_score = CataractDetection.calculate_boundary_score(gray, x, y, r)
            
            # Weighted total score
            total_score = (position_score * 0.15 + 
                        size_score * 0.20 + 
                        contrast_score * 0.30 + 
                        circularity_score * 0.20 + 
                        boundary_score * 0.15)
            
            scored_circles.append((x, y, r, total_score))
        
        # Return circle with highest score
        if scored_circles:
            best_circle = max(scored_circles, key=lambda x: x[3])
            return best_circle[:3]
        
        return None

    @staticmethod
    def remove_duplicate_circles(circles):
        """
        Remove duplicate circles that are too similar
        """
        if not circles:
            return []
        
        unique_circles = []
        
        for circle in circles:
            x, y, r = circle
            is_duplicate = False
            
            for ux, uy, ur in unique_circles:
                # Check if circles are similar
                center_dist = np.sqrt((x - ux)**2 + (y - uy)**2)
                radius_diff = abs(r - ur)
                
                if center_dist < 20 and radius_diff < 10:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_circles.append(circle)
        
        return unique_circles

    @staticmethod
    def calculate_contrast_score(gray_image, x, y, r):
        """
        Calculate contrast between iris and surrounding sclera
        """
        # Create masks for iris and sclera regions
        mask_iris = np.zeros(gray_image.shape, dtype=np.uint8)
        cv2.circle(mask_iris, (x, y), r, 255, -1)
        
        mask_sclera = np.zeros(gray_image.shape, dtype=np.uint8)
        cv2.circle(mask_sclera, (x, y), int(r * 1.3), 255, -1)
        mask_sclera = cv2.bitwise_and(mask_sclera, cv2.bitwise_not(mask_iris))
        
        # Calculate mean intensities
        mean_iris = cv2.mean(gray_image, mask=mask_iris)[0]
        mean_sclera = cv2.mean(gray_image, mask=mask_sclera)[0]
        
        # Iris should be significantly darker than sclera
        if mean_sclera > mean_iris:
            contrast_ratio = (mean_sclera - mean_iris) / (mean_sclera + 1e-6)
            return min(contrast_ratio * 2, 1.0)  
        else:
            return 0.0

    @staticmethod
    def calculate_circularity_score(gray_image, x, y, r):
        """
        Calculate how circular the detected region is using edge analysis
        """
        # Get edge image
        edges = cv2.Canny(gray_image, 50, 150)
        
        # Sample points on the circle circumference
        angles = np.linspace(0, 2*np.pi, 72)  
        edge_votes = 0
        
        for angle in angles:
            # Calculate point on circumference
            px = int(x + r * np.cos(angle))
            py = int(y + r * np.sin(angle))
            
            # Check if point is within image bounds
            if 0 <= px < edges.shape[1] and 0 <= py < edges.shape[0]:
                # Check for edge in a small neighborhood
                neighborhood = edges[max(0, py-1):py+2, max(0, px-1):px+2]
                if np.any(neighborhood > 0):
                    edge_votes += 1
        
        return edge_votes / len(angles)

    @staticmethod
    def calculate_boundary_score(gray_image, x, y, r):
        """
        Calculate the strength of the iris-sclera boundary
        """
        # Create annular region around the circle boundary
        mask_outer = np.zeros(gray_image.shape, dtype=np.uint8)
        cv2.circle(mask_outer, (x, y), r + 5, 255, -1)
        
        mask_inner = np.zeros(gray_image.shape, dtype=np.uint8)
        cv2.circle(mask_inner, (x, y), r - 5, 255, -1)
        
        boundary_mask = cv2.bitwise_and(mask_outer, cv2.bitwise_not(mask_inner))
        
        # Calculate gradient magnitude in boundary region
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Average gradient in boundary region
        boundary_gradient = cv2.mean(gradient_magnitude, mask=boundary_mask)[0]
        
        # Normalize to 0-1 scale
        return min(boundary_gradient / 50.0, 1.0)

    @staticmethod
    def calculate_edge_score(image, x, y, r):
        """
        Calculate how well a circle aligns with edges in the image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Sample points on the circle circumference
        angles = np.linspace(0, 2*np.pi, 36)
        edge_count = 0
        
        for angle in angles:
            px = int(x + r * np.cos(angle))
            py = int(y + r * np.sin(angle))
            
            if 0 <= px < edges.shape[1] and 0 <= py < edges.shape[0]:
                if edges[py, px] > 0:
                    edge_count += 1
        
        return edge_count / len(angles)

    @staticmethod
    def extract_iris_region(image, x, y, r):
        """
        Extract iris region with proper masking and calculate pixel count
        """
        # Create circular mask
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # Apply mask
        result = cv2.bitwise_and(image, image, mask=mask)
        
        # Crop to bounding box
        x1, y1 = max(0, x - r), max(0, y - r)
        x2, y2 = min(image.shape[1], x + r), min(image.shape[0], y + r)
        cropped = result[y1:y2, x1:x2]
        mask_cropped = mask[y1:y2, x1:x2]
        
        # Convert to RGBA for transparency
        cropped_rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
        cropped_rgba[:, :, 3] = mask_cropped
        
        # Calculate actual iris pixel count
        iris_pixel_count = np.count_nonzero(mask_cropped)
        
        iris_info = {
            'center': (x, y),
            'radius': r,
            'pixel_count': iris_pixel_count,
            'mask': mask_cropped
        }
        
        return cropped_rgba, iris_info

    @staticmethod
    def analyze_affected_area(iris_image, iris_info):
        hsv = cv2.cvtColor(iris_image, cv2.COLOR_BGR2HSV)

        # Define gray range in HSV
        lower_gray = np.array([0, 0, 40])
        upper_gray = np.array([200, 50, 255])

        # Mask for gray pixels
        mask = cv2.inRange(hsv, lower_gray, upper_gray)

        # Morphological cleaning to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the cropped iris image
        result = iris_image.copy()
        result = cv2.cvtColor(result, cv2.COLOR_BGRA2RGB)
        cv2.drawContours(result, contours, -1, (173, 16, 37), -1)
        
        affected_p_count=CataractDetection.affected_pixel_count(mask_clean,contours)
        
        affected_info = {
            'pixel_count': affected_p_count,
            'mask': mask_clean,
            'contours': contours
        }
        return result, affected_info

    @staticmethod
    def affected_pixel_count(mask_clean,contours):
        # 1) Create an empty mask (same size as your image)
        mask_filled = np.zeros(mask_clean.shape, dtype=np.uint8)

        # 2) Draw the contours filled onto this mask
        cv2.drawContours(mask_filled, contours, -1, 255, -1)  

        # 3) Count non-zero pixels in the filled mask
        num_pixels = cv2.countNonZero(mask_filled)

        
        return num_pixels

    @staticmethod
    def detect_affected_hsv(image):
        """
        HSV-based detection of affected areas
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Multiple ranges for different types of cataracts
        ranges = [
            ([0, 0, 180], [180, 30, 255]),  
            ([0, 0, 100], [180, 50, 200]),    
            ([10, 50, 50], [25, 255, 255])    
        ]
        
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        return combined_mask

    @staticmethod
    def detect_affected_kmeans(image, iris_mask):
        """
        K-means clustering to identify affected areas
        """
        # Only process pixels within the iris
        iris_pixels = image[iris_mask > 0]
        
        if len(iris_pixels) < 10:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Reshape for K-means
        pixel_values = iris_pixels.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 3
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Find the cluster representing affected areas 
        brightness = np.sum(centers, axis=1)
        affected_cluster = np.argmax(brightness)
        
        # Create mask for affected areas
        affected_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        iris_coords = np.where(iris_mask > 0)
        
        for i, (y, x) in enumerate(zip(iris_coords[0], iris_coords[1])):
            if labels[i] == affected_cluster:
                affected_mask[y, x] = 255
        
        return affected_mask

    @staticmethod
    def detect_affected_texture(image):
        """
        Texture-based detection of affected areas
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate local standard deviation 
        kernel = np.ones((5, 5), np.float32) / 25
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        std_dev = np.sqrt(sqr_mean - mean**2)
        
        # Threshold for low texture areas 
        _, affected_mask = cv2.threshold(std_dev, np.mean(std_dev) * 0.5, 255, cv2.THRESH_BINARY_INV)
        
        return affected_mask.astype(np.uint8)

    @staticmethod
    def combine_masks(masks):
        """
        Combine multiple detection masks using weighted voting
        """
        if not masks:
            return np.zeros((100, 100), dtype=np.uint8)
        
        # Ensure all masks have the same size
        target_shape = masks[0].shape
        normalized_masks = []
        
        for mask in masks:
            if mask.shape != target_shape:
                mask = cv2.resize(mask, (target_shape[1], target_shape[0]))
            normalized_masks.append(mask.astype(np.float32) / 255.0)
        
        # Weighted combination
        weights = [0.4, 0.4, 0.2]  
        combined = np.zeros(target_shape, dtype=np.float32)
        
        for i, mask in enumerate(normalized_masks):
            if i < len(weights):
                combined += weights[i] * mask
        
        # Threshold the combined result
        _, result = cv2.threshold(combined, 0.3, 255, cv2.THRESH_BINARY)
        
        return result.astype(np.uint8)
    
    @staticmethod
    def displayResults(iris_info,affected_info,percentage):
        print("CATARACT DETECTION RESULTS")
        print("="*50)
        print(f"Iris Center: {iris_info['center']}")
        print(f"Iris Radius: {iris_info['radius']} pixels")
        print(f"Total Iris Area: {iris_info['pixel_count']} pixels")
        print(f"Affected Area: {affected_info['pixel_count']} pixels")
        print(f"Affected Percentage: {percentage}%")

    


    @staticmethod
    def calculate_affected_percentage(iris_info, affected_info):
        """
        Calculate the percentage of affected area
        """
        if iris_info['pixel_count'] == 0:
            return 0
        
        percentage = (affected_info['pixel_count'] / iris_info['pixel_count']) * 100
        return round(percentage, 2)

    

class ScleraSpotDetection:
    """
    Your existing sclera spot detection class
    Expected to return: processed_image_1, processed_image_2, details
    """
    def __init__(self):
        pass
    
    def detect(self, image_path_1, image_path_2):
        """
        Process two images for sclera spot detection and comparison
        Args:
            image_path_1 (str): Path to the first input image
            image_path_2 (str): Path to the second input image
        Returns:
            tuple: (processed_image_1, processed_image_2, details)
        """
        processed_image_1, processed_image_2, details = self.spot_detection(image_path_1, image_path_2)
        return processed_image_1, processed_image_2, details
    
    def spot_detection(self, image_path_1, image_path_2):
    
        processed_image_1,area_image_1=ScleraSpotDetection.detect_eye_spots(image_path_1)
        processed_image_2,area_image_2=ScleraSpotDetection.detect_eye_spots(image_path_2)
        details=ScleraSpotDetection.spot_growth(area_image_1,area_image_2)
    
        return processed_image_1, processed_image_2,details

    @staticmethod
    def detect_eye_spots(image_path,dark_threshold=120,white_threshold=100):
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image from {image_path}")
            return
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Create masks
        white_mask = (blurred > white_threshold).astype(np.uint8) * 255
        dark_mask = (blurred < dark_threshold).astype(np.uint8) * 255
        
        # Clean up dark mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_mask_cleaned = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        dark_mask_cleaned = cv2.morphologyEx(dark_mask_cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(dark_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Print statistics
        print(f"Image size: {gray.shape}")
        print(f"Pixel intensity range: {gray.min()} - {gray.max()}")
        print(f"Dark threshold: {dark_threshold}")
        print(f"White threshold: {white_threshold}")
        print(f"Number of dark regions found: {len(contours)}")
        
        # Analyze each contour
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            print(f"Contour {i+1}: Area = {area}")
        


        contour_image = image.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        return contour_image, area

    @staticmethod
    def spot_growth(area_1,area_2):
        growth=((area_2-area_1)/area_1)*100
        if growth > 0 :
            print(f"The growth of the spot :  {growth}%")
            return round(growth, 2)
        return 0

class MedicalDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cataract And Scleral Spot Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize detection classes
        self.cataract_detector = CataractDetection()
        self.sclera_detector = ScleraSpotDetection()
        
        # Current page state
        self.current_page = "home"
        self.selected_image_path = None
        self.selected_image_path_2 = None 
        self.detection_results = None
        
        # Create main container
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Show home page initially
        self.show_home_page()
    
    def clear_frame(self):
        """Clear all widgets from main frame"""
        for widget in self.main_frame.winfo_children():
            widget.destroy()
    
    def show_home_page(self):
        """Display the home page with two detection options"""
        self.clear_frame()
        self.current_page = "home"
        
        # Title
        title_label = ttk.Label(self.main_frame, text="Cataract And Scleral Spot Detection System", 
                               font=('Arial', 24, 'bold'))
        title_label.pack(pady=30)
        
        subtitle_label = ttk.Label(self.main_frame, text="AI-FREE Image Processing based solution for efficient eye screening",
                                  font=('Arial', 12))
        subtitle_label.pack(pady=10)
        
        # Create buttons frame
        buttons_frame = ttk.Frame(self.main_frame)
        buttons_frame.pack(expand=True, fill=tk.BOTH, padx=50, pady=50)
        
        # Configure grid weights
        buttons_frame.grid_columnconfigure(0, weight=1)
        buttons_frame.grid_columnconfigure(1, weight=1)
        buttons_frame.grid_rowconfigure(0, weight=1)
        
        # Cataract Detection Button
        cataract_frame = ttk.LabelFrame(buttons_frame, text="Cataract Detection", 
                                       padding=20)
        cataract_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        cataract_desc = ttk.Label(cataract_frame, 
                                 text="Detect and analyze cataracts in eye images\nusing image processing techniques",
                                 font=('Arial', 10), justify=tk.CENTER)
        cataract_desc.pack(pady=10)
        
        cataract_btn = ttk.Button(cataract_frame, text="Start Cataract Detection",
                                 command=lambda: self.show_detection_page("cataract"),
                                 style='Accent.TButton')
        cataract_btn.pack(pady=10)
        
        # Sclera Spot Detection Button
        sclera_frame = ttk.LabelFrame(buttons_frame, text="Sclera Spot Detection", 
                                     padding=20)
        sclera_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        sclera_desc = ttk.Label(sclera_frame,
                               text="Identify and Compare spots on the sclera\nusing image processing techniques",
                               font=('Arial', 10), justify=tk.CENTER)
        sclera_desc.pack(pady=10)
        
        sclera_btn = ttk.Button(sclera_frame, text="Start Sclera Spot Detection",
                               command=lambda: self.show_detection_page("sclera"),
                               style='Accent.TButton')
        sclera_btn.pack(pady=10)
    
    def show_detection_page(self, detection_type):
        """Display the detection page for the selected type"""
        self.clear_frame()
        self.current_page = detection_type
        self.selected_image_path = None
        self.selected_image_path_2 = None
        self.detection_results = None
        
        # Header frame
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, pady=10)
        
        # Back button
        back_btn = ttk.Button(header_frame, text="← Back to Home",
                             command=self.show_home_page)
        back_btn.pack(side=tk.LEFT)
        
        # Title
        title_text = "Cataract Detection" if detection_type == "cataract" else "Sclera Spot Detection"
        title_label = ttk.Label(header_frame, text=title_text, font=('Arial', 18, 'bold'))
        title_label.pack(side=tk.LEFT, padx=20)
        
        # Main content frame
        content_frame = ttk.Frame(self.main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Configure grid
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=2)
        content_frame.grid_rowconfigure(0, weight=1)
        
        # Left panel - Controls
        left_panel = ttk.LabelFrame(content_frame, text="Controls", padding=10)
        left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # File selection
        file_frame = ttk.Frame(left_panel)
        file_frame.pack(fill=tk.X, pady=10)
        
        if detection_type == "cataract":
            self.setup_cataract_controls(file_frame)
        else:
            self.setup_sclera_controls(file_frame)
        
        # Process button
        self.process_btn = ttk.Button(left_panel, text="Analyze Image",
                                     command=self.process_image,
                                     state=tk.DISABLED)
        self.process_btn.pack(fill=tk.X, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(left_panel, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(left_panel, text="Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Add scrollbar to results
        results_scroll_frame = ttk.Frame(results_frame)
        results_scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(results_scroll_frame, height=8, width=30, 
                                   state=tk.DISABLED, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(results_scroll_frame, orient=tk.VERTICAL, 
                                 command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right panel - Visualization
        right_panel = ttk.LabelFrame(content_frame, text="Visualization", padding=10)
        right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize empty plot
        self.update_visualization()
    
    def setup_cataract_controls(self, parent_frame):
        """Setup controls for cataract detection"""
        ttk.Label(parent_frame, text="Select Image:").pack(anchor=tk.W)
        select_btn = ttk.Button(parent_frame, text="Browse Image", 
                               command=self.select_image)
        select_btn.pack(fill=tk.X, pady=5)
        
        # Selected file label
        self.file_label = ttk.Label(parent_frame, text="No file selected", 
                                   foreground="gray")
        self.file_label.pack(anchor=tk.W, pady=5)
    
    def setup_sclera_controls(self, parent_frame):
        """Setup controls for sclera spot detection (two images)"""
        # First image selection
        ttk.Label(parent_frame, text="Select First Image:").pack(anchor=tk.W)
        select_btn_1 = ttk.Button(parent_frame, text="Browse First Image", 
                                 command=lambda: self.select_image(1))
        select_btn_1.pack(fill=tk.X, pady=5)
        
        self.file_label = ttk.Label(parent_frame, text="No file selected", 
                                   foreground="gray")
        self.file_label.pack(anchor=tk.W, pady=5)
        
        # Second image selection
        ttk.Label(parent_frame, text="Select Second Image:").pack(anchor=tk.W, pady=(10, 0))
        select_btn_2 = ttk.Button(parent_frame, text="Browse Second Image", 
                                 command=lambda: self.select_image(2))
        select_btn_2.pack(fill=tk.X, pady=5)
        
        self.file_label_2 = ttk.Label(parent_frame, text="No file selected", 
                                     foreground="gray")
        self.file_label_2.pack(anchor=tk.W, pady=5)
    
    def select_image(self, image_number=1):
        """Open file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            title=f"Select image {image_number}",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            filename = os.path.basename(file_path)
            
            if image_number == 1:
                self.selected_image_path = file_path
                self.file_label.config(text=filename, foreground="black")
            else:
                self.selected_image_path_2 = file_path
                self.file_label_2.config(text=filename, foreground="black")
            
            # Enable process button based on detection type
            if self.current_page == "cataract":
                if self.selected_image_path:
                    self.process_btn.config(state=tk.NORMAL)
                    self.show_original_image()
            else:  # sclera detection
                if self.selected_image_path and self.selected_image_path_2:
                    self.process_btn.config(state=tk.NORMAL)
                    self.show_original_images()
    
    def show_original_image(self):
        """Display the selected image in the visualization panel (for cataract detection)"""
        if self.selected_image_path:
            image = cv2.imread(self.selected_image_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                self.fig.clear()
                ax = self.fig.add_subplot(111)
                ax.imshow(image_rgb)
                ax.set_title("Selected Image")
                ax.axis('off')
                self.canvas.draw()
    
    def show_original_images(self):
        """Display both selected images in the visualization panel (for sclera detection)"""
        if self.selected_image_path and self.selected_image_path_2:
            image1 = cv2.imread(self.selected_image_path)
            image2 = cv2.imread(self.selected_image_path_2)
            
            if image1 is not None and image2 is not None:
                image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                
                self.fig.clear()
                ax1 = self.fig.add_subplot(121)
                ax1.imshow(image1_rgb)
                ax1.set_title("First Image")
                ax1.axis('off')
                
                ax2 = self.fig.add_subplot(122)
                ax2.imshow(image2_rgb)
                ax2.set_title("Second Image")
                ax2.axis('off')
                
                self.fig.tight_layout()
                self.canvas.draw()
    
    def process_image(self):
        """Process the selected image(s) using the appropriate detection method"""
        if self.current_page == "cataract":
            if not self.selected_image_path:
                messagebox.showerror("Error", "Please select an image first")
                return
        else:  # sclera detection
            if not self.selected_image_path or not self.selected_image_path_2:
                messagebox.showerror("Error", "Please select both images first")
                return
        
        # Start processing in a separate thread to avoid blocking UI
        self.process_btn.config(state=tk.DISABLED)
        self.progress.start()
        
        thread = threading.Thread(target=self._process_image_thread)
        thread.daemon = True
        thread.start()
    
    def _process_image_thread(self):
        """Process image in a separate thread"""
        try:
            if self.current_page == "cataract":
                # For cataract detection: returns (original_image, processed_image, details)
                original_image, processed_image, details = self.cataract_detector.detect(self.selected_image_path)
                processed_image=cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                self.detection_results = {
                    'original_image': original_image,
                    'processed_image': processed_image,
                    'details': f"{details}% of iris is affected"
                }
            else:
                # For sclera detection: returns (processed_image_1, processed_image_2, details)
                processed_image_1, processed_image_2, details = self.sclera_detector.detect(
                    self.selected_image_path, self.selected_image_path_2)
                self.detection_results = {
                    'processed_image_1': processed_image_1,
                    'processed_image_2': processed_image_2,
                    'details': f"The spot has spread by {details}%  "
                }
            
            # Update UI in main thread
            self.root.after(0, self._update_results)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {str(e)}"))
            self.root.after(0, self._processing_finished)
    
    def _update_results(self):
        """Update the results display"""
        if self.detection_results is None:
            messagebox.showerror("Error", "No results received from detection")
            self._processing_finished()
            return
        
        # Update results text
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # Display the details from your detection classes
        if 'details' in self.detection_results:
            self.results_text.insert(tk.END, self.detection_results['details'])
        else:
            self.results_text.insert(tk.END, "No details available")
        
        self.results_text.config(state=tk.DISABLED)
        
        # Update visualization
        self.update_visualization()
        
        self._processing_finished()
    
    def update_visualization(self):
        """Update the matplotlib visualization"""
        self.fig.clear()
        
        if self.detection_results is None:
            # Show empty plot
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Select and analyze an image\\nto see results', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=14, color='gray')
            ax.axis('off')
        else:
            if self.current_page == "cataract":
                # For cataract detection: show original and processed images
                original_img = self.detection_results.get('original_image')
                processed_img = self.detection_results.get('processed_image')
                
                if original_img is not None and processed_img is not None:
                    # Convert images to RGB if they're in BGR format
                    if len(original_img.shape) == 3 and original_img.shape[2] == 3:
                        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                    else:
                        original_rgb = original_img
                    
                    if len(processed_img.shape) == 3 and processed_img.shape[2] == 3:
                        processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                    else:
                        processed_rgb = processed_img
                    
                    # Show original and processed images side by side
                    ax1 = self.fig.add_subplot(121)
                    ax1.imshow(original_rgb)
                    ax1.set_title('Original Image')
                    ax1.axis('off')
                    
                    ax2 = self.fig.add_subplot(122)
                    ax2.imshow(processed_rgb)
                    ax2.set_title('Processed Image')
                    ax2.axis('off')
                else:
                    # Show error message if images are not available
                    ax = self.fig.add_subplot(111)
                    ax.text(0.5, 0.5, 'Images not available in results', 
                           ha='center', va='center', transform=ax.transAxes, 
                           fontsize=14, color='red')
                    ax.axis('off')
            
            else:  # sclera detection
                # For sclera detection: show both processed images
                processed_img_1 = self.detection_results.get('processed_image_1')
                processed_img_2 = self.detection_results.get('processed_image_2')
                
                if processed_img_1 is not None and processed_img_2 is not None:
                    # Convert images to RGB if they're in BGR format
                    if len(processed_img_1.shape) == 3 and processed_img_1.shape[2] == 3:
                        processed_rgb_1 = cv2.cvtColor(processed_img_1, cv2.COLOR_BGR2RGB)
                    else:
                        processed_rgb_1 = processed_img_1
                    
                    if len(processed_img_2.shape) == 3 and processed_img_2.shape[2] == 3:
                        processed_rgb_2 = cv2.cvtColor(processed_img_2, cv2.COLOR_BGR2RGB)
                    else:
                        processed_rgb_2 = processed_img_2
                    
                    # Show both processed images side by side
                    ax1 = self.fig.add_subplot(121)
                    ax1.imshow(processed_rgb_1)
                    ax1.set_title('Processed Image 1')
                    ax1.axis('off')
                    
                    ax2 = self.fig.add_subplot(122)
                    ax2.imshow(processed_rgb_2)
                    ax2.set_title('Processed Image 2')
                    ax2.axis('off')
                else:
                    # Show error message if images are not available
                    ax = self.fig.add_subplot(111)
                    ax.text(0.5, 0.5, 'Images not available in results', 
                           ha='center', va='center', transform=ax.transAxes, 
                           fontsize=14, color='red')
                    ax.axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _processing_finished(self):
        """Clean up after processing is complete"""
        self.progress.stop()
        self.process_btn.config(state=tk.NORMAL)
    

def main():
    root = tk.Tk()
    app = MedicalDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()