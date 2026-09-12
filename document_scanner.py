#!/usr/bin/env python3
"""
===============================================================================
Document Scanner & Perspective Warper
Day 08 - 30-Day Computer Vision & Deep Learning Challenge
===============================================================================
Author: Computer Vision & AI Agent
Technologies: OpenCV, NumPy, Contours, Homography, Perspective Transform

Description:
    An enterprise-grade document scanner that takes ANY image containing a document,
    receipt, invoice, or card, automatically detects the 4-corner document contour,
    applies perspective transformation (warping) to flatten it, and enhances text
    legibility using adaptive thresholding and contrast normalization.
===============================================================================
"""

import os
import sys
import glob
import json
import time
import argparse
import cv2
import numpy as np


def order_points(pts):
    """
    Orders 4 quadrilateral points in the following sequence:
    top-left (TL), top-right (TR), bottom-right (BR), bottom-left (BL).
    
    Args:
        pts (np.ndarray): Shape (4, 2) array of (x, y) coordinates.
        
    Returns:
        np.ndarray: Ordered float32 array of shape (4, 2).
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum of coordinates: TL has minimum sum, BR has maximum sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Difference of coordinates: TR has minimum diff (y - x), BL has maximum diff (y - x)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def four_point_transform(image, pts):
    """
    Applies a 4-point perspective transformation to warp a quadrilateral region
    in the image to a top-down, flat rectangular view.
    
    Args:
        image (np.ndarray): Source image.
        pts (np.ndarray): Shape (4, 2) document corner coordinates.
        
    Returns:
        tuple: (warped_image, transform_matrix, (target_width, target_height))
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute maximum width between bottom and top edge
    width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_A), int(width_B))
    
    # Compute maximum height between right and left edge
    height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_A), int(height_B))
    
    # Guard against 0 width/height
    max_width = max(max_width, 100)
    max_height = max(max_height, 100)
    
    # Destination points for target top-down view
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")
    
    # Compute perspective transform matrix and warp image
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    
    return warped, M, (max_width, max_height)


def detect_document_contour(image, max_processing_dim=800):
    """
    Detects the 4-corner document boundary polygon in an input image.
    Uses multi-stage contour detection with automatic fallbacks for maximum robustness.
    
    Args:
        image (np.ndarray): Input BGR image.
        max_processing_dim (int): Rescale dimension for edge detection speed.
        
    Returns:
        tuple: (corners_pts, debug_dict)
            - corners_pts: np.ndarray of shape (4, 2) on original image scale
            - debug_dict: dictionary containing intermediate edge maps for visualization
    """
    orig_h, orig_w = image.shape[:2]
    
    # 1. Resize for fast processing while maintaining aspect ratio
    ratio = orig_h / float(max_processing_dim)
    proc_w = int(orig_w / ratio)
    proc_h = max_processing_dim
    resized = cv2.resize(image, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    
    # 2. Convert to Grayscale & Filter
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    bilateral = cv2.bilateralFilter(blurred, 9, 75, 75)
    
    # 3. Dynamic Canny Edge Detection using Otsu / Median Thresholding
    median_val = np.median(bilateral)
    lower_thresh = int(max(0, (1.0 - 0.33) * median_val))
    upper_thresh = int(min(255, (1.0 + 0.33) * median_val))
    edged = cv2.Canny(bilateral, lower_thresh, upper_thresh)
    
    # 4. Morphological Closing to connect disconnected edge lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    # 5. Find Contours
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    document_cnt = None
    min_area_thresh = 0.03 * (proc_w * proc_h) # At least 3% of processing image area
    
    # Attempt Primary Polygon Approximation
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_thresh:
            continue
            
        peri = cv2.arcLength(c, True)
        
        # Test multiple epsilon scale ratios for approxPolyDP
        for eps_ratio in [0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(c, eps_ratio * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                document_cnt = approx.reshape(4, 2)
                break
        if document_cnt is not None:
            break
            
    # FALLBACK 1: If 4-point convex polygon not found, take Convex Hull of largest contour -> MinAreaRect
    if document_cnt is None and len(contours) > 0 and cv2.contourArea(contours[0]) >= min_area_thresh:
        hull = cv2.convexHull(contours[0])
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        document_cnt = box.astype(np.float32)
        
    # FALLBACK 2: If no suitable document contour found, use 5% inset border of full image
    if document_cnt is None:
        margin_w = proc_w * 0.05
        margin_h = proc_h * 0.05
        document_cnt = np.array([
            [margin_w, margin_h],
            [proc_w - margin_w, margin_h],
            [proc_w - margin_w, proc_h - margin_h],
            [margin_w, proc_h - margin_h]
        ], dtype=np.float32)
        
    # Scale detected points back to original image resolution
    orig_corners = document_cnt * ratio
    
    debug_dict = {
        "resized": resized,
        "gray": gray,
        "blurred": blurred,
        "edged": edged,
        "closed": closed,
        "ratio": ratio
    }
    
    return orig_corners.astype(np.float32), debug_dict


def enhance_scanned_document(warped_bgr, mode="all"):
    """
    Applies professional document enhancement post-processing algorithms.
    
    Args:
        warped_bgr (np.ndarray): Flat warped document BGR image.
        mode (str): Enhancement mode - 'auto', 'bw', 'color', 'gray', or 'all'.
        
    Returns:
        dict: Mapping of mode name to enhanced BGR image.
    """
    results = {}
    
    # 1. Original Unfiltered Warped Image
    results["original_warped"] = warped_bgr.copy()
    
    # Convert to Grayscale
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Clean B&W / CamScanner Mode (Adaptive Thresholding + Light Filter)
    norm_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    bw_adaptive = cv2.adaptiveThreshold(
        norm_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11
    )
    # Median blur to eliminate salt & pepper scan artifacts
    bw_clean = cv2.medianBlur(bw_adaptive, 3)
    results["bw_scanned"] = cv2.cvtColor(bw_clean, cv2.COLOR_GRAY2BGR)
    
    # 3. Enhanced Color Magic Mode (CLAHE on LAB color space + Sharpening)
    lab = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_enhanced = cv2.merge((l_clahe, a, b))
    color_clahe = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # Apply Unsharp Masking kernel for text sharpness
    gaussian_blur = cv2.GaussianBlur(color_clahe, (0, 0), 3.0)
    sharpened_color = cv2.addWeighted(color_clahe, 1.4, gaussian_blur, -0.4, 0)
    results["color_enhanced"] = sharpened_color
    
    # 4. Grayscale Contrast Stretch Mode
    gray_clahe = clahe.apply(gray)
    results["grayscale"] = cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)
    
    if mode in results:
        return {mode: results[mode]}
    return results


def draw_contour_overlay(image, corners, confidence_label="DOCUMENT DETECTED"):
    """
    Draws annotated document polygon outline, 4 corner dots, and corner labels on original frame.
    
    Args:
        image (np.ndarray): Original BGR image.
        corners (np.ndarray): Shape (4, 2) corner points.
        
    Returns:
        np.ndarray: Annotated image.
    """
    vis = image.copy()
    ordered = order_points(corners)
    pts = ordered.astype(np.int32).reshape((-1, 1, 2))
    
    # Draw thick green document polygon outline
    cv2.polylines(vis, [pts], isClosed=True, color=(0, 230, 0), thickness=4, lineType=cv2.LINE_AA)
    
    labels = ["TL (1)", "TR (2)", "BR (3)", "BL (4)"]
    colors = [(0, 0, 255), (0, 255, 255), (0, 165, 255), (255, 0, 255)]
    
    # Draw corner dots and coordinate tags
    for i, (pt, label, col) in enumerate(zip(ordered, labels, colors)):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(vis, (x, y), 10, col, -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, (x, y), 14, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        
        # Position label text offset
        offset_x = -60 if "L" in label else 15
        offset_y = -15 if "T" in label else 30
        cv2.putText(vis, f"{label}: ({x},{y})", (x + offset_x, y + offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, lineType=cv2.LINE_AA)
        cv2.putText(vis, f"{label}: ({x},{y})", (x + offset_x, y + offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        
    # Header Banner
    banner_h = 45
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (vis.shape[1], banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
    cv2.putText(vis, f"OPENCV DOCUMENT SCANNER | {confidence_label}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2, lineType=cv2.LINE_AA)
                
    return vis


def build_comparison_montage(annotated_input, flat_warped, enhanced_bw, target_height=600):
    """
    Creates a 3-panel side-by-side comparison montage image showing:
    [1. Input Image + 4 Corner Detection] | [2. Flat Warped Scan] | [3. Enhanced B&W CamScanner Output]
    
    Args:
        annotated_input (np.ndarray): Original frame with corner annotations.
        flat_warped (np.ndarray): Flat warped document.
        enhanced_bw (np.ndarray): Enhanced B&W document.
        target_height (int): Height in pixels for normalized comparison panel.
        
    Returns:
        np.ndarray: Combined 3-panel montage image.
    """
    def resize_to_h(img, h):
        aspect = img.shape[1] / float(img.shape[0])
        w = int(h * aspect)
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        
    p1 = resize_to_h(annotated_input, target_height)
    p2 = resize_to_h(flat_warped, target_height)
    p3 = resize_to_h(enhanced_bw, target_height)
    
    # Add title headers to each panel
    def add_panel_title(img, text, bg_color=(40, 40, 40)):
        h, w = img.shape[:2]
        header = np.zeros((40, w, 3), dtype=np.uint8)
        header[:] = bg_color
        cv2.putText(header, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        return np.vstack([header, img])
        
    p1_head = add_panel_title(p1, "[1] INPUT & 4-CORNER DETECTOR", (40, 40, 140))
    p2_head = add_panel_title(p2, "[2] PERSPECTIVE WARPED SCAN", (40, 120, 40))
    p3_head = add_panel_title(p3, "[3] CAMSCANNER ENHANCED B&W", (140, 70, 20))
    
    # Separate panels with vertical divider bar
    divider = np.zeros((p1_head.shape[0], 6, 3), dtype=np.uint8)
    divider[:] = (200, 200, 200)
    
    montage = np.hstack([p1_head, divider, p2_head, divider, p3_head])
    return montage


def process_single_image(image_path, output_dir="output", mode="all", save_debug=False, save_json=True):
    """
    Full pipeline to scan a single input image.
    
    Args:
        image_path (str): Path to input image file.
        output_dir (str): Directory where scanned outputs will be saved.
        mode (str): Scan enhancement mode.
        save_debug (str): Save intermediate debug pipeline images.
        save_json (bool): Save detailed scan metadata JSON file.
        
    Returns:
        dict: Scan result statistics and file output paths.
    """
    start_time = time.time()
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
        
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to decode image: {image_path}")
        
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[+] Processing Document: '{os.path.basename(image_path)}' ({image.shape[1]}x{image.shape[0]} px)")
    
    # 1. Detect Document 4-Corner Contour
    corners, debug_dict = detect_document_contour(image)
    ordered_corners = order_points(corners)
    
    # 2. Perform 4-Point Perspective Transformation (Warping)
    warped, M, (target_w, target_h) = four_point_transform(image, corners)
    
    # 3. Apply Scan Post-Processing Enhancements
    enhanced_dict = enhance_scanned_document(warped, mode=mode)
    
    # 4. Draw Corner Annotations on Original Frame
    annotated_input = draw_contour_overlay(image, corners)
    
    # 5. Build 3-Panel Side-by-Side Comparison Montage
    enhanced_bw_bgr = enhanced_dict.get("bw_scanned", list(enhanced_dict.values())[0])
    montage = build_comparison_montage(annotated_input, warped, enhanced_bw_bgr)
    
    # Save Output Files
    output_files = {}
    
    # Save original warped scan
    warped_path = os.path.join(output_dir, f"{base_name}_warped.jpg")
    cv2.imwrite(warped_path, warped)
    output_files["warped"] = warped_path
    
    # Save enhanced B&W scan
    bw_path = os.path.join(output_dir, f"{base_name}_scanned_bw.jpg")
    cv2.imwrite(bw_path, enhanced_bw_bgr)
    output_files["scanned_bw"] = bw_path
    
    # Save 3-panel comparison montage
    montage_path = os.path.join(output_dir, f"{base_name}_comparison.jpg")
    cv2.imwrite(montage_path, montage)
    output_files["comparison"] = montage_path
    
    # Save additional requested enhanced modes
    for enh_key, enh_img in enhanced_dict.items():
        enh_path = os.path.join(output_dir, f"{base_name}_{enh_key}.jpg")
        cv2.imwrite(enh_path, enh_img)
        output_files[enh_key] = enh_path
        
    # Save intermediate debug steps if requested
    if save_debug:
        debug_dir = os.path.join(output_dir, f"{base_name}_debug")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "01_resized.jpg"), debug_dict["resized"])
        cv2.imwrite(os.path.join(debug_dir, "02_gray.jpg"), debug_dict["gray"])
        cv2.imwrite(os.path.join(debug_dir, "03_blurred.jpg"), debug_dict["blurred"])
        cv2.imwrite(os.path.join(debug_dir, "04_edged.jpg"), debug_dict["edged"])
        cv2.imwrite(os.path.join(debug_dir, "05_closed.jpg"), debug_dict["closed"])
        print(f"  - Saved intermediate pipeline debug maps in '{debug_dir}/'")
        
    proc_time = round(time.time() - start_time, 4)
    aspect_ratio = round(target_w / float(target_h), 4)
    
    # Summary JSON report
    report = {
        "filename": os.path.basename(image_path),
        "input_dimensions": {"width": image.shape[1], "height": image.shape[0]},
        "scanned_dimensions": {"width": target_w, "height": target_h},
        "aspect_ratio": aspect_ratio,
        "corners": {
            "top_left": [round(float(ordered_corners[0][0]), 1), round(float(ordered_corners[0][1]), 1)],
            "top_right": [round(float(ordered_corners[1][0]), 1), round(float(ordered_corners[1][1]), 1)],
            "bottom_right": [round(float(ordered_corners[2][0]), 1), round(float(ordered_corners[2][1]), 1)],
            "bottom_left": [round(float(ordered_corners[3][0]), 1), round(float(ordered_corners[3][1]), 1)]
        },
        "processing_time_sec": proc_time,
        "output_files": output_files
    }
    
    if save_json:
        json_path = os.path.join(output_dir, f"{base_name}_scan_report.json")
        with open(json_path, "w") as f:
            json.dump(report, f, indent=4)
        print(f"  - Saved metadata report: '{json_path}'")
        
    print(f"  [OK] Scan completed in {proc_time}s | Target Size: {target_w}x{target_h} px (Aspect: {aspect_ratio})")
    print(f"  - Scanned B&W: '{bw_path}'")
    print(f"  - Comparison Montage: '{montage_path}'")
    
    return report


def run_webcam_scanner(camera_id=0, output_dir="output"):
    """
    Runs a live real-time interactive camera document scanner stream.
    Press 's' or SPACEBAR to scan & warp current frame.
    Press 'q' or ESC to exit.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[!] Error: Could not open camera device ID {camera_id}")
        return
        
    print("\n==========================================================")
    print("  LIVE WEBCAM DOCUMENT SCANNER")
    print("  --------------------------------------------------------")
    print("  Controls:")
    print("    [S] / [SPACE] : Capture & Scan Current Document")
    print("    [Q] / [ESC]   : Exit Scanner")
    print("==========================================================\n")
    
    scan_count = 0
    os.makedirs(output_dir, exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab webcam frame.")
            break
            
        corners, _ = detect_document_contour(frame)
        annotated = draw_contour_overlay(frame, corners, confidence_label="LIVE DETECTOR")
        
        cv2.putText(annotated, "Press 'S' to Save Scan | 'Q' to Quit", (15, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
        cv2.imshow("Document Scanner - Live View", annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('s'), ord('S'), 32]: # 'S' or SPACEBAR
            scan_count += 1
            temp_path = os.path.join(output_dir, f"cam_capture_{scan_count:03d}.jpg")
            cv2.imwrite(temp_path, frame)
            process_single_image(temp_path, output_dir=output_dir, mode="all")
            print(f"[+] Live scan #{scan_count} saved and processed!")
        elif key in [ord('q'), ord('Q'), 27]: # 'Q' or ESC
            break
            
    cap.release()
    cv2.destroyAllWindows()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Document Scanner & Perspective Warper using OpenCV & Contours."
    )
    parser.add_argument(
        "-i", "--input", type=str, default="input",
        help="Path to an image file, a folder containing images, or 'camera'/'0' for live webcam."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output",
        help="Directory to save scanned output images and reports (default: 'output/')."
    )
    parser.add_argument(
        "-m", "--mode", type=str, default="all",
        choices=["all", "auto", "bw", "color", "gray"],
        help="Scan enhancement mode (default: 'all')."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save intermediate pipeline debug images (gray, blur, edge, morph)."
    )
    parser.add_argument(
        "--gen-samples", action="store_true",
        help="Generate synthetic sample test images in 'input/' before scanning."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Generate synthetic samples if requested or if input directory is empty
    if args.gen_samples or (args.input == "input" and not os.path.exists("input")):
        try:
            from generate_demo_samples import generate_all_demo_images
            generate_all_demo_images(output_dir="input")
        except Exception as e:
            print(f"[!] Warning: Could not auto-generate sample images: {e}")
            
    # Handle Live Webcam Mode
    if args.input.lower() in ["camera", "webcam", "0"]:
        run_webcam_scanner(camera_id=0, output_dir=args.output)
        return
        
    # Collect input files (single file or entire directory)
    input_files = []
    if os.path.isfile(args.input):
        input_files.append(args.input)
    elif os.path.isdir(args.input):
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tif", "*.tiff"]
        for ext in extensions:
            input_files.extend(glob.glob(os.path.join(args.input, ext)))
            input_files.extend(glob.glob(os.path.join(args.input, ext.upper())))
        input_files = sorted(list(set(input_files)))
    else:
        print(f"[!] Error: Input path '{args.input}' does not exist.")
        sys.exit(1)
        
    if not input_files:
        print(f"[!] No valid image files found in '{args.input}'. Auto-generating synthetic test samples...")
        from generate_demo_samples import generate_all_demo_images
        generate_all_demo_images(output_dir="input")
        input_files = sorted(glob.glob(os.path.join("input", "*.jpg")))
        
    print("\n==========================================================")
    print("  [DOC] DOCUMENT SCANNER & PERSPECTIVE WARPER")
    print("  --------------------------------------------------------")
    print(f"  Input Source  : {args.input} ({len(input_files)} image(s))")
    print(f"  Output Directory: {args.output}")
    print(f"  Enhancement Mode: {args.mode}")
    print("==========================================================")
    
    reports = []
    for img_path in input_files:
        try:
            rep = process_single_image(
                img_path,
                output_dir=args.output,
                mode=args.mode,
                save_debug=args.debug,
                save_json=True
            )
            reports.append(rep)
        except Exception as e:
            print(f"[!] Error processing '{img_path}': {e}")
            
    print("\n==========================================================")
    print(" [OK] Batch Scanning Complete! Processed {}/{} documents.".format(len(reports), len(input_files)))
    print(f" [OUT] Output files saved in: '{os.path.abspath(args.output)}'")
    print("==========================================================\n")


if __name__ == "__main__":
    main()
