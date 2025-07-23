
# Cataract and Sclera Spot Detection Using OpenCV

This project presents a lightweight, interpretable tool for early detection of cataracts and scleral abnormalities using classical image processing with OpenCV. Avoiding AI, it ensures transparent processing ideal for resource-limited settings. High-resolution eye images are analyzed through contrast enhancement, grayscale conversion, edge detection, and morphological filtering. Iris regions are detected via Hough Circle Transform and Canny edges, while cataracts are segmented and quantified by coverage over the iris. Scleral health is monitored by tracking dark spots over time. Results are visualized in a simple graphical interface, supporting field-level screening without the need for specialized equipment.


## 🛠️ Tools and Technologies

Python

OpenCV

Matplotlib (for visualization)

Tkinter (GUI Creation)
## ✅ Delivered Outcomes

Identify and highlight cataract-affected regions in the lens using image Processing Techniques

Provide a percentage of cataract spread within the iris.

Detect and monitor the growth of abnormal spots in the sclera

Enable periodic comparison of spread of the spot by analyzing images over time.

Build a user-friendly interface for uploading images and displaying results.
## 🔬Methodology

Preprocess the image using CLAHE and Gaussian blur, Bilateral filters.

Detect iris region using Hough Circles and Contour-Based , Edge-Based Detection..

Used HSV-Based Thresholding , Texture-Based Analysis, Morphological Operations and contour detection to extract abnormal regions.

Calculate spread percentage and track changes across images.
