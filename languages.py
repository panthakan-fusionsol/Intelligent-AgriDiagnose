# ------------------ Language System ------------------
LANGUAGES = {
    "ไทย": {
        "title": "🌽 ระบบวิเคราะห์โรคข้าวโพด + ผู้ช่วยแชทบอท",
        "subtitle": "อัปโหลดภาพใบข้าวโพดเพื่อวิเคราะห์โรค พร้อมคำแนะนำจากผู้เชี่ยวชาญ (GPT-4o)",
        "select_input": "📥 เลือกวิธีการใส่ภาพ",
        "input_note": "*หมายเหตุ: สามารถใช้ได้ทีละวิธีเท่านั้น*",
        "url_method": "🔗 วิธีที่ 1: ใส่ลิงก์ภาพ",
        "url_placeholder": "https://example.com/image.jpg",
        "url_help": "วางลิงก์ภาพที่ต้องการวิเคราะห์",
        "upload_method": "📤 วิธีที่ 2: อัปโหลดไฟล์ภาพ",
        "upload_help": "อัปโหลดไฟล์ภาพจากคอมพิวเตอร์",
        "leaf_confidence": "ความมั่นใจขั้นต่ำของใบ",
        "disease_confidence": "ความมั่นใจขั้นต่ำของโรค",
        "AUTO_MULTI_DETECT": "🔍 กำลังตรวจจับหลายบริเวณ (Auto Multi-Detect)...",
        "url_label": "URL ของภาพ:",
        "select_image": "เลือกรูปภาพ...",
        "loading_url": "กำลังโหลดภาพจาก URL...",
        "url_load_failed": "❌ ไม่สามารถโหลดภาพจาก URL ได้",
        "url_load_success": "✅ โหลดภาพจาก URL สำเร็จ",
        "upload_success": "✅ อัปโหลดไฟล์สำเร็จ:",
        "mode_select": "โหมดการเลือกพื้นที่",
        # "auto_mode": "โหมด Auto Detect: แสดงโรคที่แตกต่างกัน 1 ใบต่อโรค (สีเขียว=ผ่าน model threshold, ส้ม=ยังไม่ถึง)",
        "manual_mode": "โหมด Manual Crop: ลาก/ปรับกรอบ ภาพจะวิเคราะห์ทุกครั้งที่เปลี่ยน",
        "auto_crop": "⚙️ Auto Crop",
        "manual_crop": "📐 Manual Crop",
        "clear_selection": "🗑️ ล้างการเลือก",
        "analysis_result": "ผลการตรวจสอบพื้นที่ที่เลือก:",
        "leaf_num": "จุดที่พบ ",
        "gradcam": "Grad-CAM",
        "request_advice": "ขอคำแนะนำ",
        "expert_advice": "💬 ขอคำแนะนำจากผู้เชี่ยวชาญ",
        "analyzing": "กำลังวิเคราะห์และให้คำแนะนำ...",
        "ai_advice": "💬 ข้อแนะนำจากผู้เชี่ยวชาญ (AI):",
        "analysis_complete": "วิเคราะห์เสร็จเรียบร้อยแล้ว!",
        "detected_disease": "โรคที่ตรวจพบ:",
        "confidence": "ความมั่นใจ:",
        "clear_image": "✅ ภาพชัดเจนเพียงพอ",
        "low_confidence": "⚠️ ความมั่นใจยังไม่เพียงพอ",
        "unclear_analysis": "❌ ไม่สามารถวิเคราะห์ได้แน่ชัด",
        "low_confidence_tip": """
💡 **คำแนะนำ:**
- กรุณาเลือกพื้นที่ใหม่ที่แสดงอาการของโรคชัดเจนกว่านี้
- หลีกเลี่ยงจุดที่มืด เบลอ หรือไม่มีอาการเด่นชัด
- ลองครอปบริเวณใบที่มีแผลหรือจุดผิดปกติ
""",
        "unclear_analysis_tip": """
❗ **คำแนะนำ:** ลองอัปโหลดภาพใหม่ หรือเลือกบริเวณอื่นที่ชัดเจนขึ้นสำหรับการวิเคราะห์
""",
        "crop_too_small": "⚠️ กรอบเลยรูปภาพ กรุณากดล้างการเลือกแล้วลองใหม่",
        "no_detections": "ภาพต้นฉบับ - ไม่พบกรอบที่ผ่าน threshold",
        "no_auto_boxes": "ไม่มีกรอบ Auto ที่ผ่าน threshold (หรือยังไม่ detect)",
        "original_image": "ภาพต้นฉบับ",
        "cropper_error": "cropper error:",
        "analysis_results": "ผลการวิเคราะห์",
        "full_image_box": "ภาพเต็ม + กรอบ",
        "no_threshold_boxes": "⚠️ ไม่พบกรอบที่ผ่าน threshold ที่ตั้งไว้ — กรุณาลดค่า threshold หรือใช้ Manual Crop",
        "auto_detect_found": "Auto Detect - พบ {count} โรคที่แตกต่างกัน (≥{threshold:.2f} detection prob)",
        "manual_caption": "ภาพเต็ม + กรอบ (Manual)",
        "url_replace_caption": "แทนที่ URL ด้วยลิงก์ภาพที่ต้องการวิเคราะห์",
        "image_restored": "✅ อัปโหลดไฟล์สำเร็จ",
        "auto_detect_failed": "⚠️ Auto Detect ล้มเหลว: {error}",
        "gpt_error": "เกิดข้อผิดพลาดในการเรียก GPT: {error}",
        "device_hint": "{device} บนภาพเพื่อเลือกพื้นที่ที่ต้องการตรวจสอบ",
        "gradcam_expander": "📖 วิธีอ่าน Grad-CAM",
        "gradcam_text": """
Grad-CAM (Gradient-weighted Class Activation Mapping) แสดงพื้นที่ที่โมเดลใช้ในการตัดสินใจ:
สีแดง/ส้ม: พื้นที่ที่โมเดลให้ความสำคัญสูงสุด
สีเหลือง/เขียว: พื้นที่ที่โมเดลให้ความสำคัญปานกลาง
สีน้ำเงิน/ม่วง: พื้นที่ที่โมเดลให้ความสำคัญน้อย

การแปลผล:
หาก heatmap แสดงบริเวณอาการโรคชัดเจน ⇒ มีแนวโน้มว่าการวิเคราะห์ถูกต้อง
หาก heatmap แสดงบริเวณที่ไม่เกี่ยวข้อง ⇒ อาจต้องปรับปรุงข้อมูล/โมเดล/การครอป
""",
    },
    "English": {
        "title": "🌽 Corn Disease Classifier + Chatbot Assistant",
        "subtitle": "Upload corn leaf images for disease analysis with expert advice (GPT-4o)",
        "select_input": "📥 Select Input Method",
        "input_note": "*Note: Only one method can be used at a time*",
        "url_method": "🔗 Method 1: Image URL",
        "url_placeholder": "https://example.com/image.jpg",
        "url_help": "Paste the image URL you want to analyze",
        "upload_method": "📤 Method 2: Upload Image File",
        "upload_help": "Upload image file from your computer",
        "leaf_confidence": "Min leaf confidence",
        "disease_confidence": "Min disease confidence",
        "AUTO_MULTI_DETECT": "🔍 Detecting multiple areas (Auto Multi-Detect)...",
        "url_label": "Image URL:",
        "select_image": "Select image...",
        "loading_url": "Loading image from URL...",
        "url_load_failed": "❌ Unable to load image from URL",
        "url_load_success": "✅ Successfully loaded image from URL",
        "upload_success": "✅ File uploaded successfully:",
        "mode_select": "Area Selection Mode",
        # "auto_mode": "Auto Detect Mode: Shows 1 leaf per different disease (Green=passed model threshold, Orange=below threshold)",
        "manual_mode": "Manual Crop Mode: Drag/adjust frame, image analyzed on every change",
        "auto_crop": "⚙️ Auto Crop",
        "manual_crop": "📐 Manual Crop",
        "clear_selection": "🗑️ Clear Selection",
        "analysis_result": "Analysis Results for Selected Area:",
        "leaf_num": "Detected Point",
        "gradcam": "Grad-CAM",
        "request_advice": "Request Advice",
        "expert_advice": "💬 Request Expert Advice",
        "analyzing": "Analyzing and providing recommendations...",
        "ai_advice": "💬 Expert Advice (AI):",
        "analysis_complete": "Analysis completed successfully!",
        "detected_disease": "Detected Disease:",
        "confidence": "Confidence:",
        "clear_image": "✅ Image clear enough",
        "low_confidence": "⚠️ Confidence not sufficient",
        "unclear_analysis": "❌ Unable to analyze clearly",
        "low_confidence_tip": """
💡 **Tip:**
- Pick another area that shows clearer disease symptoms
- Avoid dark, blurry, or symptom-free regions
- Try cropping around lesions or abnormal spots
""",
        "unclear_analysis_tip": """
❗ **Tip:** Upload a new image or choose a clearer area for analysis
""",
        "crop_too_small": "⚠️ Selection area too small, please drag larger",
        "no_detections": "Original Image - No boxes passed threshold",
        "no_auto_boxes": "No Auto boxes passed threshold (or not detected yet)",
        "original_image": "Original Image",
        "cropper_error": "cropper error:",
        "Auto_MULTI_DETECT": "🔍 Detecting multiple areas (Auto Multi-Detect)...",
        "full_image_box": "Full Image + Box",
        "analysis_results": "Analysis Results",
        "no_threshold_boxes": "⚠️ No boxes passed the set threshold — Please lower threshold or use Manual Crop",
        "auto_detect_found": "Auto Detect - Found {count} different diseases (≥{threshold:.2f} detection prob)",
        "manual_caption": "Full Image + Box (Manual)",
        "url_replace_caption": "Replace URL with the image link you want to analyze",
        "image_restored": "✅ File uploaded successfully",
        "auto_detect_failed": "⚠️ Auto Detect failed: {error}",
        "gpt_error": "Error calling GPT: {error}",
        "device_hint": "{device} on image to select area for analysis",
        "gradcam_expander": "📖 How to Read Grad-CAM",
        "gradcam_text": """
Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions the model focuses on for its decision:

- Red / Orange: Areas with the highest importance to the model  
- Yellow / Green: Areas with moderate importance  
- Blue / Purple: Areas with the least importance

Interpretation: 
- If the heatmap highlights the diseased regions clearly ⇒ the analysis is likely correct  
- If the heatmap focuses on unrelated areas ⇒ the dataset / model / cropping may need improvement
"""
        
    },
    "Tiếng Việt": {
        "title": "🌽 Hệ thống Phân loại Bệnh Ngô + Trợ lý Chatbot",
        "subtitle": "Tải lên hình ảnh lá ngô để phân tích bệnh với lời khuyên chuyên gia (GPT-4o)",
        "select_input": "📥 Chọn Phương thức Nhập",
        "input_note": "*Lưu ý: Chỉ có thể sử dụng một phương thức tại một thời điểm*",
        "url_method": "🔗 Phương thức 1: URL Hình ảnh",
        "url_placeholder": "https://example.com/image.jpg",
        "url_help": "Dán URL hình ảnh bạn muốn phân tích",
        "upload_method": "📤 Phương thức 2: Tải lên Tệp Hình ảnh",
        "upload_help": "Tải lên tệp hình ảnh từ máy tính của bạn",
        "leaf_confidence": "Độ tin cậy lá tối thiểu",
        "disease_confidence": "Độ tin cậy bệnh tối thiểu",
        "AUTO_MULTI_DETECT": "🔍 Đang phát hiện nhiều khu vực (Tự động Đa-Phát hiện)...",
        "url_label": "URL Hình ảnh:",
        "select_image": "Chọn hình ảnh...",
        "loading_url": "Đang tải hình ảnh từ URL...",
        "url_load_failed": "❌ Không thể tải hình ảnh từ URL",
        "url_load_success": "✅ Tải hình ảnh từ URL thành công",
        "upload_success": "✅ Tải tệp lên thành công:",
        "mode_select": "Chế độ Chọn Khu vực",
        "manual_mode": "Chế độ Cắt Thủ công: Kéo/điều chỉnh khung, hình ảnh được phân tích mỗi khi thay đổi",
        "auto_crop": "⚙️ Cắt Tự động",
        "manual_crop": "📐 Cắt Thủ công",
        "clear_selection": "🗑️ Xóa Lựa chọn",
        "analysis_result": "Kết quả Phân tích cho Khu vực Đã chọn:",
        "leaf_num": "Điểm Phát hiện",
        "gradcam": "Grad-CAM",
        "request_advice": "Yêu cầu Lời khuyên",
        "expert_advice": "💬 Yêu cầu Lời khuyên Chuyên gia",
        "analyzing": "Đang phân tích và đưa ra khuyến nghị...",
        "ai_advice": "💬 Lời khuyên Chuyên gia (AI):",
        "analysis_complete": "Phân tích hoàn thành thành công!",
        "detected_disease": "Bệnh Được phát hiện:",
        "confidence": "Độ tin cậy:",
        "clear_image": "✅ Hình ảnh đủ rõ ràng",
        "low_confidence": "⚠️ Độ tin cậy chưa đủ",
        "unclear_analysis": "❌ Không thể phân tích rõ ràng",
    "low_confidence_tip": """
💡 **Gợi ý:**
- Chọn một vùng khác hiển thị triệu chứng bệnh rõ ràng hơn
- Tránh những vùng tối, mờ hoặc không có triệu chứng
- Thử cắt quanh các vết đốm hoặc dấu hiệu bất thường trên lá
""",
    "unclear_analysis_tip": """
❗ **Gợi ý:** Hãy tải ảnh mới hoặc chọn một vùng rõ hơn để phân tích
""",
        "crop_too_small": "⚠️ Khu vực chọn quá nhỏ, vui lòng kéo lớn hơn",
        "no_detections": "Hình ảnh Gốc - Không có hộp nào vượt qua ngưỡng",
        "no_auto_boxes": "Không có hộp Tự động nào vượt qua ngưỡng (hoặc chưa được phát hiện)",
        "original_image": "Hình ảnh Gốc",
        "cropper_error": "lỗi cropper:",
        "Auto_MULTI_DETECT": "🔍 Đang phát hiện nhiều khu vực (Tự động Đa-Phát hiện)...",
        "full_image_box": "Hình ảnh Đầy đủ + Hộp",
        "analysis_results": "Kết quả Phân tích",
        "no_threshold_boxes": "⚠️ Không có hộp nào vượt qua ngưỡng đã đặt — Vui lòng giảm ngưỡng hoặc sử dụng Cắt Thủ công",
        "auto_detect_found": "Tự động Phát hiện - Tìm thấy {count} bệnh khác nhau (≥{threshold:.2f} xác suất phát hiện)",
        "manual_caption": "Hình ảnh Đầy đủ + Hộp (Thủ công)",
        "url_replace_caption": "Thay thế URL bằng liên kết hình ảnh bạn muốn phân tích",
        "image_restored": "✅ Tệp đã tải lên thành công",
        "auto_detect_failed": "⚠️ Tự động Phát hiện thất bại: {error}",
        "gpt_error": "Lỗi gọi GPT: {error}",
        "device_hint": "{device} trên hình ảnh để chọn khu vực phân tích",
        "gradcam_expander": "📖 Cách Đọc Grad-CAM",
        "gradcam_text": """
Grad-CAM (Gradient-weighted Class Activation Mapping) làm nổi bật các vùng mà mô hình tập trung vào để đưa ra quyết định:

- Đỏ / Cam: Các khu vực có tầm quan trọng cao nhất đối với mô hình
- Vàng / Xanh lá: Các khu vực có tầm quan trọng vừa phải
- Xanh dương / Tím: Các khu vực có tầm quan trọng ít nhất

Giải thích:
- Nếu heatmap làm nổi bật các vùng bệnh một cách rõ ràng ⇒ phân tích có khả năng chính xác
- Nếu heatmap tập trung vào các khu vực không liên quan ⇒ dữ liệu / mô hình / cắt có thể cần cải thiện
"""
    },
    "Bahasa Indonesia": {
        "title": "🌽 Sistem Klasifikasi Penyakit Jagung + Asisten Chatbot",
        "subtitle": "Unggah gambar daun jagung untuk analisis penyakit dengan saran ahli (GPT-4o)",
        "select_input": "📥 Pilih Metode Input",
        "input_note": "*Catatan: Hanya satu metode yang dapat digunakan pada satu waktu*",
        "url_method": "🔗 Metode 1: URL Gambar",
        "url_placeholder": "https://example.com/image.jpg",
        "url_help": "Tempelkan URL gambar yang ingin Anda analisis",
        "upload_method": "📤 Metode 2: Unggah File Gambar",
        "upload_help": "Unggah file gambar dari komputer Anda",
        "leaf_confidence": "Kepercayaan daun minimal",
        "disease_confidence": "Kepercayaan penyakit minimal",
        "AUTO_MULTI_DETECT": "🔍 Mendeteksi beberapa area (Multi-Deteksi Otomatis)...",
        "url_label": "URL Gambar:",
        "select_image": "Pilih gambar...",
        "loading_url": "Memuat gambar dari URL...",
        "url_load_failed": "❌ Tidak dapat memuat gambar dari URL",
        "url_load_success": "✅ Berhasil memuat gambar dari URL",
        "upload_success": "✅ File berhasil diunggah:",
        "mode_select": "Mode Pemilihan Area",
        "manual_mode": "Mode Potong Manual: Seret/sesuaikan bingkai, gambar dianalisis setiap kali berubah",
        "auto_crop": "⚙️ Potong Otomatis",
        "manual_crop": "📐 Potong Manual",
        "clear_selection": "🗑️ Hapus Pilihan",
        "analysis_result": "Hasil Analisis untuk Area yang Dipilih:",
        "leaf_num": "Titik Terdeteksi",
        "gradcam": "Grad-CAM",
        "request_advice": "Minta Saran",
        "expert_advice": "💬 Minta Saran Ahli",
        "analyzing": "Menganalisis dan memberikan rekomendasi...",
        "ai_advice": "💬 Saran Ahli (AI):",
        "analysis_complete": "Analisis berhasil diselesaikan!",
        "detected_disease": "Penyakit Terdeteksi:",
        "confidence": "Kepercayaan:",
        "clear_image": "✅ Gambar cukup jelas",
        "low_confidence": "⚠️ Kepercayaan tidak mencukupi",
        "unclear_analysis": "❌ Tidak dapat menganalisis dengan jelas",
    "low_confidence_tip": """
💡 **Saran:**
- Pilih area lain yang menunjukkan gejala penyakit lebih jelas
- Hindari bagian yang gelap, buram, atau tanpa gejala
- Coba potong di sekitar luka atau titik tidak normal pada daun
""",
    "unclear_analysis_tip": """
❗ **Saran:** Unggah gambar baru atau pilih area lain yang lebih jelas untuk analisis
""",
        "crop_too_small": "⚠️ Area pilihan terlalu kecil, silakan seret lebih besar",
        "no_detections": "Gambar Asli - Tidak ada kotak yang melewati ambang batas",
        "no_auto_boxes": "Tidak ada kotak Otomatis yang melewati ambang batas (atau belum terdeteksi)",
        "original_image": "Gambar Asli",
        "cropper_error": "kesalahan cropper:",
        "Auto_MULTI_DETECT": "🔍 Mendeteksi beberapa area (Multi-Deteksi Otomatis)...",
        "full_image_box": "Gambar Lengkap + Kotak",
        "analysis_results": "Hasil Analisis",
        "no_threshold_boxes": "⚠️ Tidak ada kotak yang melewati ambang batas yang ditetapkan — Silakan turunkan ambang batas atau gunakan Potong Manual",
        "auto_detect_found": "Deteksi Otomatis - Ditemukan {count} penyakit berbeda (≥{threshold:.2f} probabilitas deteksi)",
        "manual_caption": "Gambar Lengkap + Kotak (Manual)",
        "url_replace_caption": "Ganti URL dengan tautan gambar yang ingin Anda analisis",
        "image_restored": "✅ File berhasil diunggah",
        "auto_detect_failed": "⚠️ Deteksi Otomatis gagal: {error}",
        "gpt_error": "Kesalahan memanggil GPT: {error}",
        "device_hint": "{device} pada gambar untuk memilih area analisis",
        "gradcam_expander": "📖 Cara Membaca Grad-CAM",
        "gradcam_text": """
Grad-CAM (Gradient-weighted Class Activation Mapping) menyoroti wilayah yang difokuskan model untuk keputusannya:

- Merah / Oranye: Area dengan kepentingan tertinggi bagi model
- Kuning / Hijau: Area dengan kepentingan sedang
- Biru / Ungu: Area dengan kepentingan paling sedikit

Interpretasi:
- Jika heatmap menyoroti wilayah penyakit dengan jelas ⇒ analisis kemungkinan benar
- Jika heatmap fokus pada area yang tidak terkait ⇒ dataset / model / pemotongan mungkin perlu perbaikan
"""
    },
    "Filipino": {
        "title": "🌽 Sistema ng Pag-uri ng Sakit ng Mais + Chatbot Assistant",
        "subtitle": "Mag-upload ng mga larawan ng dahon ng mais para sa pagsusuri ng sakit na may payo ng eksperto (GPT-4o)",
        "select_input": "📥 Pumili ng Paraan ng Pagpasok",
        "input_note": "*Tandaan: Isang paraan lamang ang maaaring gamitin sa isang pagkakataon*",
        "url_method": "🔗 Paraan 1: URL ng Larawan",
        "url_placeholder": "https://example.com/image.jpg",
        "url_help": "I-paste ang URL ng larawan na gusto mong suriin",
        "upload_method": "📤 Paraan 2: Mag-upload ng File ng Larawan",
        "upload_help": "Mag-upload ng file ng larawan mula sa inyong computer",
        "leaf_confidence": "Minimum na kumpiyansa sa dahon",
        "disease_confidence": "Minimum na kumpiyansa sa sakit",
        "AUTO_MULTI_DETECT": "🔍 Tinutukoy ang maraming lugar (Awtomatikong Multi-Detect)...",
        "url_label": "URL ng Larawan:",
        "select_image": "Pumili ng larawan...",
        "loading_url": "Nilo-load ang larawan mula sa URL...",
        "url_load_failed": "❌ Hindi ma-load ang larawan mula sa URL",
        "url_load_success": "✅ Matagumpay na na-load ang larawan mula sa URL",
        "upload_success": "✅ Matagumpay na na-upload ang file:",
        "mode_select": "Mode ng Pagpili ng Lugar",
        "manual_mode": "Manual Crop Mode: I-drag/i-adjust ang frame, sinusuri ang larawan sa bawat pagbabago",
        "auto_crop": "⚙️ Awtomatikong Crop",
        "manual_crop": "📐 Manual na Crop",
        "clear_selection": "🗑️ I-clear ang Pagpili",
        "analysis_result": "Mga Resulta ng Pagsusuri para sa Napiling Lugar:",
        "leaf_num": "Natuklasang Punto",
        "gradcam": "Grad-CAM",
        "request_advice": "Humingi ng Payo",
        "expert_advice": "💬 Humingi ng Payo ng Eksperto",
        "analyzing": "Nagsusuri at nagbibigay ng mga rekomendasyon...",
        "ai_advice": "💬 Payo ng Eksperto (AI):",
        "analysis_complete": "Matagumpay na natapos ang pagsusuri!",
        "detected_disease": "Natuklasang Sakit:",
        "confidence": "Kumpiyansa:",
        "clear_image": "✅ Malinaw na sapat ang larawan",
        "low_confidence": "⚠️ Hindi sapat ang kumpiyansa",
        "unclear_analysis": "❌ Hindi masuri nang malinaw",
    "low_confidence_tip": """
💡 **Paalala:**
- Pumili ng ibang bahagi na mas malinaw ang sintomas ng sakit
- Iwasan ang madilim, malabo, o walang sintomas na lugar
- Subukang i-crop ang mga bahagi ng dahon na may sugat o kakaibang marka
""",
    "unclear_analysis_tip": """
❗ **Paalala:** Mag-upload ng bagong larawan o pumili ng mas malinaw na bahagi para sa pagsusuri
""",
        "crop_too_small": "⚠️ Masyadong maliit ang napiling lugar, paki-drag na mas malaki",
        "no_detections": "Orihinal na Larawan - Walang mga kahon na lumampas sa threshold",
        "no_auto_boxes": "Walang mga Awtomatikong kahon na lumampas sa threshold (o hindi pa natuklasan)",
        "original_image": "Orihinal na Larawan",
        "cropper_error": "error sa cropper:",
        "Auto_MULTI_DETECT": "🔍 Tinutukoy ang maraming lugar (Awtomatikong Multi-Detect)...",
        "full_image_box": "Buong Larawan + Kahon",
        "analysis_results": "Mga Resulta ng Pagsusuri",
        "no_threshold_boxes": "⚠️ Walang mga kahon na lumampas sa itinakdang threshold — Pakibaba ang threshold o gamitin ang Manual Crop",
        "auto_detect_found": "Awtomatikong Detect - Nahanap ang {count} na iba't ibang sakit (≥{threshold:.2f} probabilidad ng pagkakahanap)",
        "manual_caption": "Buong Larawan + Kahon (Manual)",
        "url_replace_caption": "Palitan ang URL ng link ng larawan na gusto mong suriin",
        "image_restored": "✅ Matagumpay na na-upload ang file",
        "auto_detect_failed": "⚠️ Nabigo ang Awtomatikong Detect: {error}",
        "gpt_error": "Error sa pagtawag sa GPT: {error}",
        "device_hint": "{device} sa larawan para pumili ng lugar para sa pagsusuri",
        "gradcam_expander": "📖 Paano Basahin ang Grad-CAM",
        "gradcam_text": """
Ang Grad-CAM (Gradient-weighted Class Activation Mapping) ay nagha-highlight ng mga rehiyon na pinokus ng modelo para sa kanyang desisyon:

- Pula / Orange: Mga lugar na may pinakamataas na kahalagahan sa modelo
- Dilaw / Berde: Mga lugar na may katamtamang kahalagahan
- Asul / Lila: Mga lugar na may pinakakaunting kahalagahan

Interpretasyon:
- Kung ang heatmap ay nangha-highlight ng mga lugar ng sakit nang malinaw ⇒ malamang na tama ang pagsusuri
- Kung ang heatmap ay nakatuon sa mga lugar na hindi kaugnay ⇒ maaaring kailangan ng pagpapabuti ang dataset / modelo / paggupit
"""
    },
    "Bahasa Melayu": {
        "title": "🌽 Sistem Pengelasan Penyakit Jagung + Pembantu Chatbot",
        "subtitle": "Muat naik imej daun jagung untuk analisis penyakit dengan nasihat pakar (GPT-4o)",
        "select_input": "📥 Pilih Kaedah Input",
        "input_note": "*Nota: Hanya satu kaedah boleh digunakan pada satu masa*",
        "url_method": "🔗 Kaedah 1: URL Imej",
        "url_placeholder": "https://example.com/image.jpg",
        "url_help": "Tampal URL imej yang ingin anda analisis",
        "upload_method": "📤 Kaedah 2: Muat Naik Fail Imej",
        "upload_help": "Muat naik fail imej dari komputer anda",
        "leaf_confidence": "Keyakinan daun minimum",
        "disease_confidence": "Keyakinan penyakit minimum",
        "AUTO_MULTI_DETECT": "🔍 Mengesan beberapa kawasan (Auto Multi-Detect)...",
        "url_label": "URL Imej:",
        "select_image": "Pilih imej...",
        "loading_url": "Memuatkan imej dari URL...",
        "url_load_failed": "❌ Tidak dapat memuatkan imej dari URL",
        "url_load_success": "✅ Berjaya memuatkan imej dari URL",
        "upload_success": "✅ Fail berjaya dimuat naik:",
        "mode_select": "Mod Pemilihan Kawasan",
        "manual_mode": "Mod Potong Manual: Seret/laraskan bingkai, imej dianalisis setiap kali berubah",
        "auto_crop": "⚙️ Potong Auto",
        "manual_crop": "📐 Potong Manual",
        "clear_selection": "🗑️ Kosongkan Pilihan",
        "analysis_result": "Keputusan Analisis untuk Kawasan Terpilih:",
        "leaf_num": "Titik Dikesan",
        "gradcam": "Grad-CAM",
        "request_advice": "Minta Nasihat",
        "expert_advice": "💬 Minta Nasihat Pakar",
        "analyzing": "Menganalisis dan memberikan cadangan...",
        "ai_advice": "💬 Nasihat Pakar (AI):",
        "analysis_complete": "Analisis berjaya diselesaikan!",
        "detected_disease": "Penyakit Dikesan:",
        "confidence": "Keyakinan:",
        "clear_image": "✅ Imej cukup jelas",
        "low_confidence": "⚠️ Keyakinan tidak mencukupi",
        "unclear_analysis": "❌ Tidak dapat menganalisis dengan jelas",
    "low_confidence_tip": """
💡 **Nasihat:**
- Pilih kawasan lain yang menunjukkan simptom penyakit dengan lebih jelas
- Elakkan bahagian yang gelap, kabur atau tanpa simptom
- Cuba potong di sekitar luka atau titik tidak normal pada daun
""",
    "unclear_analysis_tip": """
❗ **Nasihat:** Muat naik imej baharu atau pilih kawasan yang lebih jelas untuk analisis
""",
        "crop_too_small": "⚠️ Kawasan pilihan terlalu kecil, sila seret lebih besar",
        "no_detections": "Imej Asal - Tiada kotak yang melepasi ambang",
        "no_auto_boxes": "Tiada kotak Auto yang melepasi ambang (atau belum dikesan)",
        "original_image": "Imej Asal",
        "cropper_error": "ralat cropper:",
        "Auto_MULTI_DETECT": "🔍 Mengesan beberapa kawasan (Auto Multi-Detect)...",
        "full_image_box": "Imej Penuh + Kotak",
        "analysis_results": "Keputusan Analisis",
        "no_threshold_boxes": "⚠️ Tiada kotak yang melepasi ambang yang ditetapkan — Sila kurangkan ambang ataugunakan Potong Manual",
        "auto_detect_found": "Auto Detect - Ditemui {count} penyakit berbeza (≥{threshold:.2f} kebarangkalian pengesanan)",
        "manual_caption": "Imej Penuh + Kotak (Manual)",
        "url_replace_caption": "Gantikan URL dengan pautan imej yang ingin anda analisis",
        "image_restored": "✅ Fail berjaya dimuat naik",
        "auto_detect_failed": "⚠️ Auto Detect gagal: {error}",
        "gpt_error": "Ralat memanggil GPT: {error}",
        "device_hint": "{device} pada imej untuk memilih kawasan untuk analisis",
        "gradcam_expander": "📖 Cara Membaca Grad-CAM",
        "gradcam_text": """
Grad-CAM (Gradient-weighted Class Activation Mapping) menyerlahkan kawasan yang difokuskan oleh model untuk keputusannya:

- Merah / Oren: Kawasan dengan kepentingan tertinggi kepada model
- Kuning / Hijau: Kawasan dengan kepentingan sederhana
- Biru / Ungu: Kawasan dengan kepentingan paling sedikit

Tafsiran:
- Jika heatmap menyerlahkan kawasan penyakit dengan jelas ⇒ analisis berkemungkinan betul
- Jika heatmap memfokuskan pada kawasan yang tidak berkaitan ⇒ dataset / model / pemotongan mungkin perlu penambahbaikan
"""
    },
    "中文 (简体)": {
        "title": "🌽 玉米病害分类系统 + 聊天机器人助手",
        "subtitle": "上传玉米叶片图像进行病害分析，并获得专家建议 (GPT-4o)",
        "select_input": "📥 选择输入方式",
        "input_note": "*注意：一次只能使用一种方式*",
        "url_method": "🔗 方式一：图像URL",
        "url_placeholder": "https://example.com/image.jpg",
        "url_help": "粘贴您想要分析的图像URL",
        "upload_method": "📤 方式二：上传图像文件",
        "upload_help": "从计算机上传图像文件",
        "leaf_confidence": "叶片最小置信度",
        "disease_confidence": "病害最小置信度",
        "AUTO_MULTI_DETECT": "🔍 正在检测多个区域（自动多重检测）...",
        "url_label": "图像URL：",
        "select_image": "选择图像...",
        "loading_url": "正在从URL加载图像...",
        "url_load_failed": "❌ 无法从URL加载图像",
        "url_load_success": "✅ 成功从URL加载图像",
        "upload_success": "✅ 文件上传成功：",
        "mode_select": "区域选择模式",
        "manual_mode": "手动裁剪模式：拖拽/调整框架，图像在每次更改时都会被分析",
        "auto_crop": "⚙️ 自动裁剪",
        "manual_crop": "📐 手动裁剪",
        "clear_selection": "🗑️ 清除选择",
        "analysis_result": "选定区域的分析结果：",
        "leaf_num": "检测点",
        "gradcam": "Grad-CAM",
        "request_advice": "请求建议",
        "expert_advice": "💬 请求专家建议",
        "analyzing": "正在分析并提供建议...",
        "ai_advice": "💬 专家建议 (AI)：",
        "analysis_complete": "分析成功完成！",
        "detected_disease": "检测到的病害：",
        "confidence": "置信度：",
        "clear_image": "✅ 图像足够清晰",
        "low_confidence": "⚠️ 置信度不足",
        "unclear_analysis": "❌ 无法清晰分析",
    "low_confidence_tip": """
💡 **提示：**
- 请选择一个病斑更明显的区域
- 避免光线暗、模糊或没有明显症状的部分
- 尝试裁剪包含病斑或异常斑点的叶片区域
""",
    "unclear_analysis_tip": """
❗ **提示：** 请重新上传更清晰的图像，或选择其他更清楚的区域进行分析
""",
        "crop_too_small": "⚠️ 选择区域太小，请拖拽得更大",
        "no_detections": "原始图像 - 没有框超过阈值",
        "no_auto_boxes": "没有自动检测框超过阈值（或尚未检测到）",
        "original_image": "原始图像",
        "cropper_error": "裁剪器错误：",
        "Auto_MULTI_DETECT": "🔍 正在检测多个区域（自动多重检测）...",
        "full_image_box": "完整图像 + 框",
        "analysis_results": "分析结果",
        "no_threshold_boxes": "⚠️ 没有框超过设定的阈值 — 请降低阈值或使用手动裁剪",
        "auto_detect_found": "自动检测 - 发现 {count} 种不同病害（≥{threshold:.2f} 检测概率）",
        "manual_caption": "完整图像 + 框（手动）",
        "url_replace_caption": "将URL替换为您想要分析的图像链接",
        "image_restored": "✅ 文件上传成功",
        "auto_detect_failed": "⚠️ 自动检测失败：{error}",
        "gpt_error": "调用GPT时出错：{error}",
        "device_hint": "在图像上{device}以选择分析区域",
        "gradcam_expander": "📖 如何阅读Grad-CAM",
        "gradcam_text": """
Grad-CAM（梯度加权类激活映射）突出显示模型用于决策的区域：

- 红色/橙色：模型最重要的区域
- 黄色/绿色：中等重要的区域  
- 蓝色/紫色：最不重要的区域

解释：
- 如果热力图清楚地突出显示病害区域 ⇒ 分析可能是正确的
- 如果热力图聚焦于不相关的区域 ⇒ 数据集/模型/裁剪可能需要改进
"""
    },
    "中文 (繁體)": {
        "title": "🌽 玉米病害分類系統 + 聊天機器人助手",
        "subtitle": "上傳玉米葉片圖像進行病害分析，並獲得專家建議 (GPT-4o)",
        "select_input": "📥 選擇輸入方式",
        "input_note": "*注意：一次只能使用一種方式*",
        "url_method": "🔗 方式一：圖像URL",
        "url_placeholder": "https://example.com/image.jpg",
        "url_help": "貼上您想要分析的圖像URL",
        "upload_method": "📤 方式二：上傳圖像檔案",
        "upload_help": "從電腦上傳圖像檔案",
        "leaf_confidence": "葉片最小信心度",
        "disease_confidence": "病害最小信心度",
        "AUTO_MULTI_DETECT": "🔍 正在檢測多個區域（自動多重檢測）...",
        "url_label": "圖像URL：",
        "select_image": "選擇圖像...",
        "loading_url": "正在從URL載入圖像...",
        "url_load_failed": "❌ 無法從URL載入圖像",
        "url_load_success": "✅ 成功從URL載入圖像",
        "upload_success": "✅ 檔案上傳成功：",
        "mode_select": "區域選擇模式",
        "manual_mode": "手動裁剪模式：拖拽/調整框架，圖像在每次更改時都會被分析",
        "auto_crop": "⚙️ 自動裁剪",
        "manual_crop": "📐 手動裁剪",
        "clear_selection": "🗑️ 清除選擇",
        "analysis_result": "選定區域的分析結果：",
        "leaf_num": "檢測點",
        "gradcam": "Grad-CAM",
        "request_advice": "請求建議",
        "expert_advice": "💬 請求專家建議",
        "analyzing": "正在分析並提供建議...",
        "ai_advice": "💬 專家建議 (AI)：",
        "analysis_complete": "分析成功完成！",
        "detected_disease": "檢測到的病害：",
        "confidence": "信心度：",
        "clear_image": "✅ 圖像足夠清晰",
        "low_confidence": "⚠️ 信心度不足",
        "unclear_analysis": "❌ 無法清晰分析",
    "low_confidence_tip": """
💡 **提示：**
- 請選擇病斑更明顯的區域
- 避免光線昏暗、模糊或沒有明顯症狀的部分
- 嘗試裁剪包含病斑或異常斑點的葉片區域
""",
    "unclear_analysis_tip": """
❗ **提示：** 請重新上傳更清晰的影像，或選擇其他較清楚的區域進行分析
""",
        "crop_too_small": "⚠️ 選擇區域太小，請拖拽得更大",
        "no_detections": "原始圖像 - 沒有框超過閾值",
        "no_auto_boxes": "沒有自動檢測框超過閾值（或尚未檢測到）",
        "original_image": "原始圖像",
        "cropper_error": "裁剪器錯誤：",
        "Auto_MULTI_DETECT": "🔍 正在檢測多個區域（自動多重檢測）...",
        "full_image_box": "完整圖像 + 框",
        "analysis_results": "分析結果",
        "no_threshold_boxes": "⚠️ 沒有框超過設定的閾值 — 請降低閾值或使用手動裁剪",
        "auto_detect_found": "自動檢測 - 發現 {count} 種不同病害（≥{threshold:.2f} 檢測機率）",
        "manual_caption": "完整圖像 + 框（手動）",
        "url_replace_caption": "將URL替換為您想要分析的圖像連結",
        "image_restored": "✅ 檔案上傳成功",
        "auto_detect_failed": "⚠️ 自動檢測失敗：{error}",
        "gpt_error": "呼叫GPT時出錯：{error}",
        "device_hint": "在圖像上{device}以選擇分析區域",
        "gradcam_expander": "📖 如何閱讀Grad-CAM",
        "gradcam_text": """
Grad-CAM（梯度加權類別激活映射）突出顯示模型用於決策的區域：

- 紅色/橙色：模型最重要的區域
- 黃色/綠色：中等重要的區域
- 藍色/紫色：最不重要的區域

解釋：
- 如果熱力圖清楚地突出顯示病害區域 ⇒ 分析可能是正確的
- 如果熱力圖聚焦於不相關的區域 ⇒ 資料集/模型/裁剪可能需要改進
"""
    },
    "မြန်မာ": {
        "title": "🌽 ပြောင်းရောဂါ ခွဲခြားသတ်မှတ်ရေး စနစ် + Chatbot လက်ထောက်",
        "subtitle": "ပြောင်းရွက်ပုံများ အပ်လုဒ်လုပ်ပြီး ရောဂါ ပိုင်းခြားသတ်မှတ်နိုင်ပါ၊ ကျွမ်းကျင်သူ အကြံပေးချက်များဖြင့် (GPT-4o)",
        "select_input": "📥 ထည့်သွင်းနည်းလမ်း ရွေးချယ်ပါ",
        "input_note": "*မှတ်စု - တစ်ကြိမ်လျှင် နည်းလမ်းတစ်ခုသာ အသုံးပြုနိုင်သည်*",
        "url_method": "🔗 နည်းလမ်း ၁ - ပုံ URL",
        "url_placeholder": "https://example.com/image.jpg",
        "url_help": "ပိုင်းခြားသတ်မှတ်လိုသော ပုံ၏ URL ကို ထည့်ပါ",
        "upload_method": "📤 နည်းလမ်း ၂ - ပုံဖိုင် အပ်လုဒ်လုပ်ခြင်း",
        "upload_help": "သင့်ကွန်ပျူတာမှ ပုံဖိုင်ကို အပ်လုဒ်လုပ်ပါ",
        "leaf_confidence": "အနည်းဆုံး ရွက် ယုံကြည်မှု",
        "disease_confidence": "အနည်းဆုံး ရောဂါ ယုံကြည်မှု",
        "AUTO_MULTI_DETECT": "🔍 အများအပြား နေရာများကို ရှာဖွေနေသည် (အလိုအလျောက် Multi-Detect)...",
        "url_label": "ပုံ URL:",
        "select_image": "ပုံရွေးချယ်ပါ...",
        "loading_url": "URL မှ ပုံကို လုဒ်လုပ်နေသည်...",
        "url_load_failed": "❌ URL မှ ပုံကို လုဒ်လုပ်၍မရပါ",
        "url_load_success": "✅ URL မှ ပုံကို အောင်မြင်စွာ လုဒ်လုပ်ပြီးပါပြီ",
        "upload_success": "✅ ဖိုင် အောင်မြင်စွာ အပ်လုဒ်လုပ်ပြီးပါပြီ:",
        "mode_select": "နေရာ ရွေးချယ်မှု မုဒ်",
        "manual_mode": "လက်ဖြင့် ဖြတ်တောက်မှု မုဒ် - ဒရမ်း/ချိန်ညှိ ဘောင်၊ ပြောင်းလဲတိုင်း ပုံကို ပိုင်းခြားသတ်မှတ်ပါမည်",
        "auto_crop": "⚙️ အလိုအလျောက် ဖြတ်တောက်မှု",
        "manual_crop": "📐 လက်ဖြင့် ဖြတ်တောက်မှု",
        "clear_selection": "🗑️ ရွေးချယ်မှု ဖြတ်တောက်မှု",
        "analysis_result": "ရွေးချယ်ထားသော နေရာအတွက် ပိုင်းခြားသတ်မှတ်မှု ရလဒ်များ:",
        "leaf_num": "ရှာဖွေတွေ့ရှိသော နေရာ",
        "gradcam": "Grad-CAM",
        "request_advice": "အကြံပေးချက် တောင်းခံပါ",
        "expert_advice": "💬 ကျွမ်းကျင်သူ အကြံပေးချက် တောင်းခံပါ",
        "analyzing": "ပိုင်းခြားသတ်မှတ်ခြင်းနှင့် အကြံပေးချက်များ ပေးနေပါသည်...",
        "ai_advice": "💬 ကျွမ်းကျင်သူ အကြံပေးချက် (AI):",
        "analysis_complete": "ပိုင်းခြားသတ်မှတ်မှု အောင်မြင်စွာ ပြီးစီးပါပြီ!",
        "detected_disease": "ရှာဖွေတွေ့ရှိသော ရောဂါ:",
        "confidence": "ယုံကြည်မှု:",
    "clear_image": "✅ ပုံ လုံလောက်စွာ ရှင်းလင်းပါသည်",
    "low_confidence": "⚠️ ယုံကြည်မှု လုံလောက်မှု မရှိပါ",
    "unclear_analysis": "❌ ရှင်းလင်းစွာ ပိုင်းခြားသတ်မှတ်၍ မရပါ",
    "low_confidence_tip": """
💡 **အကြံပြုချက်:**
- ရောဂါလက္ခဏာများ ပိုမိုရှင်းလင်းစွာ မြင်ရသော နေရာကို ထပ်မံရွေးချယ်ပါ
- အလင်းမလုံလောက်သော၊ မရှင်းလင်းသော သို့မဟုတ် လက္ခဏာမပေါ်ပေါက်သော ဧရိယာများကို ရှောင်ကြဉ်ပါ
- အနာအပါးများ သို့မဟုတ် ထူးခြားချက်များ ရှိသော ရွက်၏ အစိတ်အပိုင်းတွင် ဖြတ်ယူကြည့်ပါ
""",
    "unclear_analysis_tip": """
❗ **အကြံပြုချက်:** ပိုမိုရှင်းလင်းသော ပုံကို ပြန်လည်အပ်လုဒ်လုပ်ပါ သို့မဟုတ် ပိုရှင်းလင်းသော နေရာတစ်ခုကို ထပ်ရွေးပြီး ခွဲခြားသတ်မှတ်ပါ
""",
        "crop_too_small": "⚠️ ရွေးချယ်ထားသော နေရာ အလွန်သေးငယ်ပါသည်၊ ပိုကြီးအောင် ဒရမ်းပါ",
        "no_detections": "မူရင်းပုံ - သတ်မှတ်ထားသော အဆင့်ကို ကျော်လွန်သော ဘောင်များ မရှိပါ",
        "no_auto_boxes": "သတ်မှတ်ထားသော အဆင့်ကို ကျော်လွန်သော အလိုအလျောက် ဘောင်များ မရှိပါ (သို့မဟုတ် မရှာဖွေရသေးပါ)",
        "original_image": "မူရင်းပုံ",
        "cropper_error": "cropper အမှား:",
        "Auto_MULTI_DETECT": "🔍 အများအပြား နေရာများကို ရှာဖွေနေသည် (အလိုအလျောက် Multi-Detect)...",
        "full_image_box": "ပြည့်စုံသော ပုံ + ဘောင်",
        "analysis_results": "ပိုင်းခြားသတ်မှတ်မှု ရလဒ်များ",
        "no_threshold_boxes": "⚠️ သတ်မှတ်ထားသော အဆင့်ကို ကျော်လွန်သော ဘောင်များ မရှိပါ — အဆင့်ကို နှိမ့်ချပါ သို့မဟုတ် လက်ဖြင့် ဖြတ်တောက်မှုကို အသုံးပြုပါ",
        "auto_detect_found": "အလိုအလျောက် ရှာဖွေမှု - မတူညီသော ရောဂါ {count} ခု တွေ့ရှိပါပြီ (≥{threshold:.2f} ရှာဖွေမှု ဖြစ်နိုင်ခြေ)",
        "manual_caption": "ပြည့်စုံသော ပုံ + ဘောင် (လက်ဖြင့်)",
        "url_replace_caption": "ပိုင်းခြားသတ်မှတ်လိုသော ပုံ လင့်ခ်ဖြင့် URL ကို အစားထိုးပါ",
        "image_restored": "✅ ဖိုင် အောင်မြင်စွာ အပ်လုဒ်လုပ်ပြီးပါပြီ",
        "auto_detect_failed": "⚠️ အလိုအလျောက် ရှာဖွေမှု မအောင်မြင်ပါ: {error}",
        "gpt_error": "GPT ခေါ်ဆိုရာတွင် အမှားရှိပါသည်: {error}",
        "device_hint": "ပိုင်းခြားသတ်မှတ်ရန် နေရာရွေးချယ်ရန်အတွက် ပုံပေါ်တွင် {device}",
        "gradcam_expander": "📖 Grad-CAM ကို ဖတ်နည်း",
        "gradcam_text": """
Grad-CAM (Gradient-weighted Class Activation Mapping) သည် မော်ဒယ်၏ ဆုံးဖြတ်ချက်အတွက် အာရုံစိုက်သော နေရာများကို မီးမောင်းထိုးပြသည်:

- အနီ / လိမ္မော်ရောင် - မော်ဒယ်အတွက် အရေးပါဆုံး နေရာများ
- အဝါ / စိမ်းရောင် - အလယ်အလတ် အရေးပါသော နေရာများ  
- အပြာ / ခရမ်းရောင် - အနည်းဆုံး အရေးပါသော နေရာများ

အဓိပ္ပါယ်ကြည့်ခြင်း:
- heatmap သည် ရောဂါ နေရာများကို ရှင်းလင်းစွာ မီးမောင်းထိုးပြလျှင် ⇒ ပိုင်းခြားသတ်မှတ်မှု မှန်ကန်နိုင်သည်
- heatmap သည် မသက်ဆိုင်သော နေရာများကို အာရုံစိုက်လျှင် ⇒ ဒေတာအစုံ / မော်ဒယ် / ဖြတ်တောက်မှု တိုးတက်ရန် လိုအပ်နိုင်သည်
"""
    },
    "ភាសាខ្មែរ": {
        "title": "🌽 ប្រព័ន្ធប្រភេទជំងឺពោតដំឡូង + ជំនួយការ Chatbot",
        "subtitle": "ផ្ទុកឡើងរូបភាពស្លឹកពោតដំឡូងសម្រាប់ការវិភាគជំងឺជាមួយនឹងការផ្តល់យោបល់ពីអ្នកជំនាញ (GPT-4o)",
        "select_input": "📥 ជ្រើសរើសវិធីសាស្រ្តបញ្ចូល",
        "input_note": "*កំណត់ចំណាំ៖ អាចប្រើបានតែមួយវិធីសាស្រ្តក្នុងពេលតែមួយ*",
        "url_method": "🔗 វិធីសាស្រ្តទី១៖ URL រូបភាព",
        "url_placeholder": "https://example.com/image.jpg",
        "url_help": "បិទភ្ជាប់ URL រូបភាពដែលអ្នកចង់វិភាគ",
        "upload_method": "📤 វិធីសាស្រ្តទី២៖ ផ្ទុកឡើងឯកសាររូបភាព",
        "upload_help": "ផ្ទុកឡើងឯកសាររូបភាពពីកុំព្យូទ័ររបស់អ្នក",
        "leaf_confidence": "ទំនុកចិត្តស្លឹកអប្បបរមា",
        "disease_confidence": "ទំនុកចិត្តជំងឺអប្បបរមា",
        "AUTO_MULTI_DETECT": "🔍 កំពុងរកឃើញតំបន់ជាច្រើន (ការរកឃើញពហុស្វ័យប្រវត្តិ)...",
        "url_label": "URL រូបភាព៖",
        "select_image": "ជ្រើសរើសរូបភាព...",
        "loading_url": "កំពុងផ្ទុករូបភាពពី URL...",
        "url_load_failed": "❌ មិនអាចផ្ទុករូបភាពពី URL បានទេ",
        "url_load_success": "✅ ផ្ទុករូបភាពពី URL បានជោគជ័យ",
        "upload_success": "✅ ផ្ទុកឡើងឯកសារបានជោគជ័យ៖",
        "mode_select": "របៀបជ្រើសរើសតំបន់",
        "manual_mode": "របៀបកាត់ដោយដៃ៖ អូស/កែសម្រួលស៊ុម រូបភាពត្រូវបានវិភាគរៀងរាល់ពេលមានការផ្លាស់ប្តូរ",
        "auto_crop": "⚙️ កាត់ស្វ័យប្រវត្តិ",
        "manual_crop": "📐 កាត់ដោយដៃ",
        "clear_selection": "🗑️ សម្អាតការជ្រើសរើស",
        "analysis_result": "លទ្ធផលការវិភាគសម្រាប់តំបន់ដែលបានជ្រើសរើស៖",
        "leaf_num": "ចំណុចដែលបានរកឃើញ",
        "gradcam": "Grad-CAM",
        "request_advice": "ស្នើសុំការផ្តល់យោបល់",
        "expert_advice": "💬 ស្នើសុំការផ្តល់យោបល់ពីអ្នកជំនាញ",
        "analyzing": "កំពុងវិភាគនិងផ្តល់អនុសាសន៍...",
        "ai_advice": "💬 ការផ្តល់យោបល់ពីអ្នកជំនាញ (AI)៖",
        "analysis_complete": "ការវិភាគបានបញ្ចប់ដោយជោគជ័យ!",
        "detected_disease": "ជំងឺដែលបានរកឃើញ៖",
        "confidence": "ទំនុកចិត្ត៖",
        "clear_image": "✅ រូបភាពច្បាស់គ្រប់គ្រាន់",
        "low_confidence": "⚠️ ទំនុកចិត្តមិនគ្រប់គ្រាន់",
        "unclear_analysis": "❌ មិនអាចវិភាគបានច្បាស់",
    "low_confidence_tip": """
💡 **ព័ត៌មានបន្ថែម:**
- ជ្រើសរើសតំបន់ផ្សេងទៀតដែលបង្ហាញរោគសញ្ញាបានច្បាស់ជាងនេះ
- ជៀសវាងតំបន់ងងឹត ព្រិល ឬគ្មានរោគសញ្ញាច្បាស់លាស់
- សាកល្បងកាត់ជុំវិញតំបន់ដែលមានផ្នែកខូច ឬចំណុចខុសប្រក្រតីលើស្លឹក
""",
    "unclear_analysis_tip": """
❗ **ព័ត៌មានបន្ថែម:** សូមផ្ទុកឡើងរូបភាពថ្មី ឬជ្រើសរើសតំបន់ផ្សេងដែលច្បាស់ជាងសម្រាប់ការវិភាគ
""",
        "crop_too_small": "⚠️ តំបន់ជ្រើសរើសតូចពេក សូមអូសឱ្យធំជាង",
        "no_detections": "រូបភាពដើម - គ្មានប្រអប់ណាមួយឆ្លងកាត់កម្រិត",
        "no_auto_boxes": "គ្មានប្រអប់ស្វ័យប្រវត្តិណាមួយឆ្លងកាត់កម្រិត (ឬមិនទាន់បានរកឃើញ)",
        "original_image": "រូបភាពដើម",
        "cropper_error": "កំហុស cropper៖",
        "Auto_MULTI_DETECT": "🔍 កំពុងរកឃើញតំបន់ជាច្រើន (ការរកឃើញពហុស្វ័យប្រវត្តិ)...",
        "full_image_box": "រូបភាពពេញលេញ + ប្រអប់",
        "analysis_results": "លទ្ធផលការវិភាគ",
        "no_threshold_boxes": "⚠️ គ្មានប្រអប់ណាមួយឆ្លងកាត់កម្រិតដែលបានកំណត់ — សូមបន្ថយកម្រិត ឬប្រើការកាត់ដោយដៃ",
        "auto_detect_found": "ការរកឃើញស្វ័យប្រវត្តិ - រកឃើញជំងឺផ្សេងៗ {count} (≥{threshold:.2f} ប្រូបាប់នៃការរកឃើញ)",
        "manual_caption": "រូបភាពពេញលេញ + ប្រអប់ (ដោយដៃ)",
        "url_replace_caption": "ជំនួស URL ដោយតំណភ្ជាប់រូបភាពដែលអ្នកចង់វិភាគ",
        "image_restored": "✅ ផ្ទុកឡើងឯកសារបានជោគជ័យ",
        "auto_detect_failed": "⚠️ ការរកឃើញស្វ័យប្រវត្តិបរាជ័យ៖ {error}",
        "gpt_error": "កំហុសក្នុងការហៅ GPT៖ {error}",
        "device_hint": "{device} លើរូបភាពដើម្បីជ្រើសរើសតំបន់សម្រាប់ការវិភាគ",
        "gradcam_expander": "📖 របៀបអាន Grad-CAM",
        "gradcam_text": """
Grad-CAM (Gradient-weighted Class Activation Mapping) បន្លិចបញ្ជាក់តំបន់ដែលម៉ូដែលផ្តោតលើការសម្រេចចិត្ត៖

- ក្រហម / ទឹកក្រូច៖ តំបន់ដែលមានសារៈសំខាន់បំផុតចំពោះម៉ូដែល
- លឿង / បៃតង៖ តំបន់ដែលមានសារៈសំខាន់មធ្យម
- ខៀវ / ស្វាយ៖ តំបន់ដែលមានសារៈសំខាន់តិចបំផុត

ការបកស្រាយ៖
- ប្រសិនបើ heatmap បន្លិចបញ្ជាក់តំបន់ជំងឺយ៉ាងច្បាស់ ⇒ ការវិភាគទំនងជាត្រឹមត្រូវ
- ប្រសិនបើ heatmap ផ្តោតលើតំបន់ដែលមិនពាក់ព័ន្ធ ⇒ ទិន្នន័យ / ម៉ូដែល / ការកាត់អាចត្រូវការកែលម្អ
"""
    },
    "ລາວ": {
        "title": "🌽 ລະບົບການຈັດປະເພດໂລກຂ້າວໂພດ + ຜູ້ຊ່ວຍ Chatbot",
        "subtitle": "ອັບໂຫລດຮູບພາບໃບຂ້າວໂພດເພື່ອການວິເຄາະໂລກພ້ອມຄຳແນະນຳຈາກຜູ້ຊ່ຽວຊານ (GPT-4o)",
        "select_input": "📥 ເລືອກວິທີການນຳເຂົ້າ",
        "input_note": "*ໝາຍເຫດ: ສາມາດໃຊ້ໄດ້ພຽງໜຶ່ງວິທີໃນເວລາດຽວ*",
        "url_method": "🔗 ວິທີທີ 1: URL ຮູບພາບ",
        "url_placeholder": "https://example.com/image.jpg",
        "url_help": "ວາງ URL ຮູບພາບທີ່ທ່ານຕ້ອງການວິເຄາະ",
        "upload_method": "📤 ວິທີທີ 2: ອັບໂຫລດໄຟລ໌ຮູບພາບ",
        "upload_help": "ອັບໂຫລດໄຟລ໌ຮູບພາບຈາກຄອມພິວເຕີຂອງທ່ານ",
        "leaf_confidence": "ຄວາມໝັ້ນໃຈຂັ້ນຕ່ຳຂອງໃບ",
        "disease_confidence": "ຄວາມໝັ້ນໃຈຂັ້ນຕ່ຳຂອງໂລກ",
        "AUTO_MULTI_DETECT": "🔍 ກຳລັງກວດພົບຫຼາຍພື້ນທີ່ (ການກວດພົບຫຼາຍແບບອັດຕະໂນມັດ)...",
        "url_label": "URL ຮູບພາບ:",
        "select_image": "ເລືອກຮູບພາບ...",
        "loading_url": "ກຳລັງໂຫລດຮູບພາບຈາກ URL...",
        "url_load_failed": "❌ ບໍ່ສາມາດໂຫລດຮູບພາບຈາກ URL ໄດ້",
        "url_load_success": "✅ ໂຫລດຮູບພາບຈາກ URL ສຳເລັດແລ້ວ",
        "upload_success": "✅ ອັບໂຫລດໄຟລ໌ສຳເລັດແລ້ວ:",
        "mode_select": "ໂໝດການເລືອກພື້ນທີ່",
        "manual_mode": "ໂໝດຕັດດ້ວຍມື: ລາກ/ປັບກອບ, ຮູບພາບຈະຖືກວິເຄາະທຸກຄັ້ງທີ່ມີການປ່ຽນແປງ",
        "auto_crop": "⚙️ ຕັດອັດຕະໂນມັດ",
        "manual_crop": "📐 ຕັດດ້ວຍມື",
        "clear_selection": "🗑️ ລ້າງການເລືອກ",
        "analysis_result": "ຜົນການວິເຄາະສຳລັບພື້ນທີ່ທີ່ເລືອກ:",
        "leaf_num": "ຈຸດທີ່ພົບເຫັນ",
        "gradcam": "Grad-CAM",
        "request_advice": "ຂໍຄຳແນະນຳ",
        "expert_advice": "💬 ຂໍຄຳແນະນຳຈາກຜູ້ຊ່ຽວຊານ",
        "analyzing": "ກຳລັງວິເຄາະແລະໃຫ້ຄຳແນະນຳ...",
        "ai_advice": "💬 ຄຳແນະນຳຈາກຜູ້ຊ່ຽວຊານ (AI):",
        "analysis_complete": "ການວິເຄາະສຳເລັດແລ້ວ!",
        "detected_disease": "ໂລກທີ່ກວດພົບ:",
        "confidence": "ຄວາມໝັ້ນໃຈ:",
        "clear_image": "✅ ຮູບພາບຊັດເຈນພຽງພໍ",
        "low_confidence": "⚠️ ຄວາມໝັ້ນໃຈບໍ່ພຽງພໍ",
        "unclear_analysis": "❌ ບໍ່ສາມາດວິເຄາະໄດ້ຢ່າງຊັດເຈນ",
    "low_confidence_tip": """
💡 **ຄຳແນະນຳ:**
- ເລືອກພື້ນທີ່ອື່ນທີ່ສະແດງອາການໂລກໄດ້ຊັດເຈນກວ່ານີ້
- ເລີກໃຊ້ພື້ນທີ່ທີ່ມືດ ມົວ ຫຼືບໍ່ມີອາການຊັດເຈນ
- ລອງຕັດຮອບບ່ອນທີ່ມີຈຸດຜິດປົກກະຕິ ຫຼືແຜເປື້ອນຢູ່ໃນໃບ
""",
    "unclear_analysis_tip": """
❗ **ຄຳແນະນຳ:** ອັບໂຫລດຮູບໃໝ່ ຫຼືເລືອກພື້ນທີ່ທີ່ຊັດເຈນກວ່າເພື່ອການວິເຄາະ
""",
        "crop_too_small": "⚠️ ພື້ນທີ່ເລືອກນ້ອຍເກີນໄປ, ກາລຸນາລາກໃຫ້ໃຫຍ່ຂຶ້ນ",
        "no_detections": "ຮູບພາບເດີມ - ບໍ່ມີກອບໃດຜ່ານໄລຍະ",
        "no_auto_boxes": "ບໍ່ມີກອບອັດຕະໂນມັດໃດຜ່ານໄລຍະ (ຫຼືຍັງບໍ່ໄດ້ກວດພົບ)",
        "original_image": "ຮູບພາບເດີມ",
        "cropper_error": "ຂໍ້ຜິດພາດ cropper:",
        "Auto_MULTI_DETECT": "🔍 ກຳລັງກວດພົບຫຼາຍພື້ນທີ່ (ການກວດພົບຫຼາຍແບບອັດຕະໂນມັດ)...",
        "full_image_box": "ຮູບພາບເຕັມ + ກອບ",
        "analysis_results": "ຜົນການວິເຄາະ",
        "no_threshold_boxes": "⚠️ ບໍ່ມີກອບໃດຜ່ານໄລຍະທີ່ກໍານົດ — ກາລຸນາຫຼຸດໄລຍະ ຫຼື ໃຊ້ການຕັດດ້ວຍມື",
        "auto_detect_found": "ການກວດພົບອັດຕະໂນມັດ - ພົບໂລກທີ່ແຕກຕ່າງກັນ {count} (≥{threshold:.2f} ຄວາມເປັນໄປໄດ້ການກວດພົບ)",
        "manual_caption": "ຮູບພາບເຕັມ + ກອບ (ດ້ວຍມື)",
        "url_replace_caption": "ປ່ຽນ URL ດ້ວຍລິ້ງຮູບພາບທີ່ທ່ານຕ້ອງການວິເຄາະ",
        "image_restored": "✅ ອັບໂຫລດໄຟລ໌ສຳເລັດແລ້ວ",
        "auto_detect_failed": "⚠️ ການກວດພົບອັດຕະໂນມັດລົ້ມເຫລວ: {error}",
        "gpt_error": "ຂໍ້ຜິດພາດໃນການເອີ້ນ GPT: {error}",
        "device_hint": "{device} ຢູ່ຮູບພາບເພື່ອເລືອກພື້ນທີ່ສຳລັບການວິເຄາະ",
        "gradcam_expander": "📖 ວິທີອ່ານ Grad-CAM",
        "gradcam_text": """
Grad-CAM (Gradient-weighted Class Activation Mapping) ເນັ້ນໃສ່ພື້ນທີ່ທີ່ໂມເດນໃສ່ໃຈສຳລັບການຕັດສິນໃຈ:

- ແດງ / ສົ້ມ: ພື້ນທີ່ທີ່ມີຄວາມສຳຄັນສູງສຸດຕໍ່ໂມເດນ
- ເຫຼືອງ / ຂຽວ: ພື້ນທີ່ທີ່ມີຄວາມສຳຄັນປານກາງ
- ຟ້າ / ມ່ວງ: ພື້ນທີ່ທີ່ມີຄວາມສຳຄັນໜ້ອຍທີ່ສຸດ

ການຕີຄວາມ:
- ຫາກ heatmap ເນັ້ນພື້ນທີ່ທີ່ເປັນໂລກຢ່າງຊັດເຈນ ⇒ ການວິເຄາະອາດຈະຖືກຕ້ອງ
- ຫາກ heatmap ເນັ້ນໃສ່ພື້ນທີ່ທີ່ບໍ່ກ່ຽວຂ້ອງ ⇒ ຊຸດຂໍ້ມູນ / ໂມເດນ / ການຕັດອາດຈະຕ້ອງການປັບປຸງ
"""
    }
}