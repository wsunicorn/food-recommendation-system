 # Hệ thống gợi ý món ăn

Kho chứa này chứa một hệ thống gợi ý món ăn nhỏ cùng giao diện web đơn giản viết bằng Flask. Mục tiêu là prototype chạy tại chỗ để bạn thử nghiệm và mở rộng.

Những tính năng đã triển khai
- Gợi ý dựa trên nội dung (mặc định dùng TF-IDF; có thể dùng SBERT) với Cosine similarity.
- Tải và tiền xử lý dữ liệu: làm sạch HTML/Markdown trong phần mô tả.
- Suy đoán danh mục (dựa trên từ khoá đơn giản) và loại bỏ bản sao theo tiêu đề ở phía server.
- Cài đặt cơ bản Matrix Factorization và helper cho collaborative filtering.
- Ứng dụng Flask gồm:
	- Trang chính (index) có sidebar danh mục, bộ lọc (category/sort/diversify) và phân trang server-side.
	- Trang chi tiết món hiển thị ảnh, mô tả và gợi ý tương tự theo nội dung.
	- Trang "View all" cho từng danh mục và API JSON cho gợi ý nội dung (`/recommend/content/<id>`).

Cấu trúc repository (các file quan trọng)
- `data/Dataset_for_print.csv` — dữ liệu dùng cho ứng dụng (đặt trong thư mục `data/` ở project root).
- `food_recomendation_system/data_loader.py` — load CSV, tạo `title`/`description`, làm sạch text, suy đoán `category`.
- `food_recomendation_system/content_based.py` — mô-đun gợi ý dựa trên nội dung (TF-IDF / SBERT).
- `food_recomendation_system/collaborative.py` — hàm trợ giúp cho CF user/item.
- `food_recomendation_system/matrix_factorization.py` — MF đơn giản (SGD).
- `food_recomendation_system/hybrid.py` — combiner hybrid có trọng số.
- `food_recomendation_system/app.py` — ứng dụng Flask (route + logic view).
- `app.py` (project root) — script chạy: start server bằng `python app.py`.
- `food_recomendation_system/templates/` — template Jinja2 cho index và item.
- `food_recomendation_system/static/styles.css` — stylesheet cơ bản.
- `requirements.txt` — package Python cần cài.

Hướng dẫn chạy (khuyến nghị)
1. Từ project root (nơi chứa `app.py`) tạo và kích hoạt virtualenv, sau đó cài dependencies:

		python -m venv .venv
		.\.venv\Scripts\Activate.ps1
		pip install -r "food_recomendation_system\requirements.txt"

2. Chạy ứng dụng từ project root:

		python app.py

3. Mở http://127.0.0.1:5000/ trên trình duyệt.

Endpoints chính
- `/` — trang index với sidebar danh mục, bộ lọc và phân trang.
- `/category/<name>` — xem toàn bộ (phân trang) các món thuộc một danh mục.
- `/item/<id>` — trang chi tiết món (dùng DataFrame index label làm `<id>`, ví dụ `1827`).
- `/recommend/content/<id>` — API JSON trả về các món tương tự theo nội dung cho món `<id>`.

Ghi chú & cách hoạt động
- Ứng dụng kỳ vọng file CSV `data/Dataset_for_print.csv` nằm ở project root.
- URL của item dùng DataFrame index label. Khi model dùng chỉ số vị trí (positional index), ứng dụng sẽ map qua lại giữa label <-> position nên link như `/item/1827` sẽ hoạt động.
- Mô tả (description) được làm sạch (loại bỏ tag HTML và unescape entity) trước khi hiển thị.
- Danh sách index loại bỏ trùng tiêu đề để hiển thị tên sản phẩm khác nhau; phân trang ngăn việc render hàng ngàn ảnh cùng lúc.

Khắc phục sự cố
- Nếu nhiều ảnh hiển thị lỗi, có thể `link_image_food` trong CSV trỏ tới URL không tồn tại. Ứng dụng chèn đường dẫn ảnh trực tiếp vào `src` của thẻ `img`.
- Nếu app không khởi động được, kiểm tra bạn đang chạy `python app.py` từ project root và các dependency đã được cài trong môi trường Python đang dùng.

Những cải tiến gợi ý
- Giữ các tham số filter/pagination khi chuyển trang (thay đổi nhỏ ở template).
- Thêm lazy-loading cho ảnh để tải trang nhanh hơn khi có nhiều item.
- Cải tiến suy đoán danh mục (phân tích `ingredients` hoặc train classifier) và làm fuzzy dedupe cho các tiêu đề gần giống nhau.
- Thêm CF user-based và lưu model MF ra đĩa để không phải tính lại mỗi lần khởi động.

Nếu bạn muốn, tôi có thể triển khai một trong các cải tiến trên (ví dụ: giữ filter khi phân trang, lazy-load ảnh, hoặc fuzzy dedupe).