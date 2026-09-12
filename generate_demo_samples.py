import os
import cv2
import numpy as np

def create_sample_receipt(width=600, height=900):
    """Generates a synthetic receipt image with text and graphics."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 248 # off-white receipt paper
    
    # Border & header lines
    cv2.rectangle(img, (20, 20), (width - 20, height - 20), (220, 220, 220), 1)
    
    # Store Header
    cv2.putText(img, "SUPERMARKET EXPRESS", (120, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    cv2.putText(img, "123 Vision Street, Tech City", (150, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
    cv2.putText(img, "TEL: (555) 019-2831", (200, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    
    cv2.line(img, (30, 150), (width - 30, 150), (40, 40, 40), 2)
    
    # Metadata
    cv2.putText(img, "DATE: 2026-09-12    TIME: 14:32:05", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
    cv2.putText(img, "RECEIPT #: 98412    CASHIER: #04", (40, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
    
    cv2.line(img, (30, 225), (width - 30, 225), (100, 100, 100), 1)
    
    # Items
    items = [
        ("01. OpenCV Python Guide", "$29.99"),
        ("02. High-Res Camera Sensor", "$45.50"),
        ("03. Image Processing Manual", "$18.75"),
        ("04. Machine Vision Textbook", "$64.00"),
        ("05. Acrylic Document Frame", "$12.99"),
        ("06. USB-C Vision Cable", "$8.50"),
        ("07. Calibration Target Pattern", "$14.20"),
    ]
    
    y = 260
    cv2.putText(img, "ITEM DESCRIPTION", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)
    cv2.putText(img, "PRICE", (480, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)
    y += 15
    cv2.line(img, (30, y), (width - 30, y), (180, 180, 180), 1)
    
    for desc, price in items:
        y += 35
        cv2.putText(img, desc, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
        cv2.putText(img, price, (480, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
        
    y += 30
    cv2.line(img, (30, y), (width - 30, y), (40, 40, 40), 2)
    
    # Totals
    y += 35
    cv2.putText(img, "SUBTOTAL:", (320, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1)
    cv2.putText(img, "$193.93", (470, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1)
    
    y += 30
    cv2.putText(img, "TAX (8.5%):", (320, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1)
    cv2.putText(img, "$16.48", (470, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1)
    
    y += 35
    cv2.putText(img, "TOTAL AMOUNT:", (280, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
    cv2.putText(img, "$210.41", (460, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
    
    # Barcode representation
    y += 60
    cv2.putText(img, "*98412-2026-CV*", (210, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    y += 15
    for bx in range(100, 500, 5):
        w = np.random.choice([1, 2, 3, 4])
        cv2.rectangle(img, (bx, y), (bx + w, y + 50), (20, 20, 20), -1)
        
    # Footer
    y += 75
    cv2.putText(img, "THANK YOU FOR YOUR BUSINESS!", (160, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    
    return img

def create_sample_invoice(width=700, height=950):
    """Generates a synthetic business document/invoice image."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Blue Header Bar
    cv2.rectangle(img, (0, 0), (width, 100), (120, 60, 20), -1)
    cv2.putText(img, "INVOICE DOCUMENT", (40, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(img, "CONFIDENTIAL", (width - 200, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 1)
    
    # Billing Info Box
    cv2.rectangle(img, (40, 140), (330, 240), (245, 245, 245), -1)
    cv2.rectangle(img, (40, 140), (330, 240), (200, 200, 200), 1)
    cv2.putText(img, "BILLED TO:", (55, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 60, 20), 2)
    cv2.putText(img, "Antigravity Vision Labs Inc.", (55, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    cv2.putText(img, "Dept. of Computer Vision & AI", (55, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1)
    cv2.putText(img, "San Francisco, CA 94105", (55, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1)
    
    # Invoice Details Box
    cv2.rectangle(img, (370, 140), (660, 240), (245, 245, 245), -1)
    cv2.rectangle(img, (370, 140), (660, 240), (200, 200, 200), 1)
    cv2.putText(img, "INVOICE DETAILS:", (385, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 60, 20), 2)
    cv2.putText(img, "Invoice No: INV-2026-008", (385, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    cv2.putText(img, "Issue Date: Sept 12, 2026", (385, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1)
    cv2.putText(img, "Due Date: Oct 12, 2026", (385, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1)
    
    # Project Scope Paragraph
    y = 280
    cv2.putText(img, "PROJECT SUMMARY & SCOPE OF WORK", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
    y += 25
    lines = [
        "This invoice covers technical services rendered for Day 08 of the 30-Day Computer",
        "Vision Challenge. Deliverables include real-time edge detection, contour analysis,",
        "quadrilateral corner extraction, and 4-point perspective transformation algorithms."
    ]
    for line in lines:
        cv2.putText(img, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1)
        y += 20
        
    # Table Header
    y += 20
    cv2.rectangle(img, (40, y), (width - 40, y + 30), (120, 60, 20), -1)
    cv2.putText(img, "MODULE / SERVICE", (55, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(img, "HOURS", (380, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(img, "RATE", (480, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(img, "AMOUNT", (580, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Table Rows
    rows = [
        ("01. OpenCV Contours & Filtering Module", "15.0", "$120.00", "$1,800.00"),
        ("02. Perspective Transform Engine", "12.5", "$120.00", "$1,500.00"),
        ("03. Adaptive Scanned Document Post-Processor", "10.0", "$120.00", "$1,200.00"),
        ("04. Interactive CLI & Batch Runner", "8.0", "$120.00", "$960.00"),
        ("05. Verification & Unit Testing Suite", "6.5", "$120.00", "$780.00"),
    ]
    
    y += 30
    for i, (mod, hrs, rate, amt) in enumerate(rows):
        bg_col = (250, 250, 250) if i % 2 == 0 else (240, 240, 240)
        cv2.rectangle(img, (40, y), (width - 40, y + 35), bg_col, -1)
        cv2.rectangle(img, (40, y), (width - 40, y + 35), (220, 220, 220), 1)
        cv2.putText(img, mod, (55, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)
        cv2.putText(img, hrs, (390, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)
        cv2.putText(img, rate, (485, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)
        cv2.putText(img, amt, (580, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)
        y += 35
        
    y += 30
    cv2.putText(img, "TOTAL BALANCE DUE:", (380, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 2)
    cv2.putText(img, "$6,240.00", (560, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 60, 20), 2)
    
    # Signature block
    y += 70
    cv2.line(img, (40, y), (240, y), (80, 80, 80), 1)
    cv2.line(img, (width - 240, y), (width - 40, y), (80, 80, 80), 1)
    cv2.putText(img, "Authorized Signature", (65, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    cv2.putText(img, "Client Representative", (width - 220, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    
    return img

def create_sample_business_card(width=600, height=350):
    """Generates a synthetic business card document image."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Geometric Background accents
    cv2.rectangle(img, (0, 0), (180, height), (40, 40, 40), -1)
    cv2.rectangle(img, (180, 0), (200, height), (0, 165, 255), -1) # Orange strip
    
    # Logo text on left panel
    cv2.putText(img, "VISION", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    cv2.putText(img, "LABS", (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
    
    # Contact Info on Right Panel
    cv2.putText(img, "Dr. Alex Rivera", (230, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2)
    cv2.putText(img, "Lead Computer Vision Engineer", (230, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 100, 100), 1)
    
    cv2.line(img, (230, 130), (550, 130), (220, 220, 220), 1)
    
    cv2.putText(img, "Email: alex.rivera@visionlabs.ai", (230, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
    cv2.putText(img, "Phone: +1 (555) 382-9901", (230, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
    cv2.putText(img, "Web: www.visionlabs.ai", (230, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
    cv2.putText(img, "Loc: Silicon Valley, CA", (230, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
    
    return img

def embed_document_in_cluttered_scene(doc_img, canvas_w=1280, canvas_h=960, scale=0.75, pos_offset=(100, 80)):
    """
    Places the flat document image onto a complex, textured background canvas 
    with perspective transformation, rotation, and lighting variation to simulate a real photo!
    """
    # Create dark textured wooden/desk background
    bg = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # Generate wood grain texture
    x_coords = np.linspace(0, 10, canvas_w)
    y_coords = np.linspace(0, 10, canvas_h)
    xx, yy = np.meshgrid(x_coords, y_coords)
    grain = (np.sin(xx * 2 + yy * 0.5) * 30 + 60).astype(np.uint8)
    
    bg[:, :, 0] = np.clip(grain + 20, 0, 255) # Blue
    bg[:, :, 1] = np.clip(grain + 40, 0, 255) # Green
    bg[:, :, 2] = np.clip(grain + 70, 0, 255) # Red (Warm wood tone)
    
    # Add subtle random texture noise & clutter lines (pen, coffee cup ring, paperclips)
    np.random.seed(42)
    noise = np.random.randint(-15, 15, (canvas_h, canvas_w, 3), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Draw coffee cup ring on desk
    cv2.circle(bg, (1100, 250), 70, (40, 70, 110), 8, lineType=cv2.LINE_AA)
    cv2.circle(bg, (1100, 250), 74, (30, 50, 90), 3, lineType=cv2.LINE_AA)
    
    # Draw pen on desk
    cv2.line(bg, (100, 750), (350, 900), (30, 30, 30), 14)
    cv2.line(bg, (100, 750), (130, 768), (200, 200, 200), 14) # Pen cap
    
    dh, dw = doc_img.shape[:2]
    
    # Source corners (flat document)
    src_pts = np.float32([
        [0, 0],
        [dw, 0],
        [dw, dh],
        [0, dh]
    ])
    
    # Destination corners with perspective skew and tilt
    cx, cy = canvas_w // 2 + pos_offset[0], canvas_h // 2 + pos_offset[1]
    
    # Define distorted 4-point quadrilateral in camera space
    target_w = dw * scale
    target_h = dh * scale
    
    dst_pts = np.float32([
        [cx - target_w * 0.48 + 40, cy - target_h * 0.45 - 20],  # Top-Left (tilted)
        [cx + target_w * 0.45 + 20, cy - target_h * 0.48 + 50],  # Top-Right
        [cx + target_w * 0.49 - 30, cy + target_h * 0.47 + 10],  # Bottom-Right
        [cx - target_w * 0.44 - 40, cy + target_h * 0.42 - 30]   # Bottom-Left
    ])
    
    # Compute Perspective Matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Warp document onto blank canvas
    warped_doc = cv2.warpPerspective(doc_img, M, (canvas_w, canvas_h))
    
    # Create mask for blending
    mask_src = np.ones((dh, dw), dtype=np.uint8) * 255
    mask_warped = cv2.warpPerspective(mask_src, M, (canvas_w, canvas_h))
    
    # Cast shadow under document
    shadow_M = M.copy()
    shadow_M[0, 2] += 15 # shift shadow X
    shadow_M[1, 2] += 20 # shift shadow Y
    shadow_mask = cv2.warpPerspective(mask_src, shadow_M, (canvas_w, canvas_h))
    
    shadow_blur = cv2.GaussianBlur(shadow_mask, (31, 31), 0)
    shadow_factor = 1.0 - (shadow_blur.astype(np.float32) / 255.0) * 0.4
    
    for c in range(3):
        bg[:, :, c] = np.clip(bg[:, :, c].astype(np.float32) * shadow_factor, 0, 255).astype(np.uint8)
        
    # Overlay document on background using mask
    mask_3c = cv2.merge([mask_warped, mask_warped, mask_warped]) > 128
    composite = np.where(mask_3c, warped_doc, bg)
    
    return composite

def generate_all_demo_images(output_dir="input"):
    """Generates all test dataset images in input/ directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[+] Generating synthetic test images in '{output_dir}/'...")
    
    # 1. Receipt on desk
    receipt_flat = create_sample_receipt()
    receipt_scene = embed_document_in_cluttered_scene(receipt_flat, scale=0.7, pos_offset=(-50, 10))
    cv2.imwrite(os.path.join(output_dir, "sample_receipt_tilted.jpg"), receipt_scene)
    print("  - Saved sample_receipt_tilted.jpg")
    
    # 2. Invoice Document on desk
    invoice_flat = create_sample_invoice()
    invoice_scene = embed_document_in_cluttered_scene(invoice_flat, scale=0.68, pos_offset=(40, -20))
    cv2.imwrite(os.path.join(output_dir, "sample_invoice_tilted.jpg"), invoice_scene)
    print("  - Saved sample_invoice_tilted.jpg")
    
    # 3. Business Card on desk
    card_flat = create_sample_business_card()
    card_scene = embed_document_in_cluttered_scene(card_flat, scale=1.1, pos_offset=(-20, 30))
    cv2.imwrite(os.path.join(output_dir, "sample_business_card_tilted.jpg"), card_scene)
    print("  - Saved sample_business_card_tilted.jpg")
    
    print("[OK] All synthetic sample images successfully generated!")

if __name__ == "__main__":
    generate_all_demo_images()
