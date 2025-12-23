Đây là đồ án môn học Thị giác máy tính, với mục tiêu xây dựng hệ thống quan trắc giao thông tự động và suy luận Macroscopic Fundamental Diagram (MFD) từ video camera giao thông. Hệ thống được thiết kế hướng tới bối cảnh giao thông Hà Nội với đặc trưng mật độ cao, nhiều xe máy, ô tô và làn đường phức tạp.

Do gặp khó khăn trong việc xin cấp quyền truy cập và ghi hình từ hệ thống camera giao thông thật tại Hà Nội (vấn đề thủ tục, bảo mật và thời gian chờ), dữ liệu video dùng trong quá trình phát triển và thử nghiệm hiện tại được lấy từ các video giao thông công khai trên YouTube có đặc điểm tương đối giống điều kiện giao thông Hà Nội (nhiều làn, mật độ xe cao, góc quay từ trên cao).

Mặc dù vậy, toàn bộ kiến trúc, tham số và quy trình xử lý của hệ thống được thiết kế sao cho có thể áp dụng trực tiếp cho dữ liệu camera thật ở Hà Nội chỉ bằng cách thay nguồn video đầu vào và hiệu chỉnh lại tham số hiệu chuẩn (perspective transformation, làn đường, ngưỡng tốc độ, v.v.).

Chức năng chính
Phát hiện phương tiện bằng YOLO (nhiều phiên bản model khác nhau).

Theo dõi đa đối tượng bằng BoT-SORT, gán ID duy nhất cho từng xe.

Phân loại phương tiện (Bike, Car, Truck, Bus) phù hợp bối cảnh giao thông Việt Nam.

Vẽ và quản lý làn đường (lane polygons) trực tiếp trên giao diện.

Ước lượng tốc độ thực bằng perspective transformation.

Tính toán các chỉ số giao thông theo từng làn:

Mật độ (Density)

Lưu lượng (Flow)

Tốc độ trung bình (Speed)

Mức độ chiếm dụng (Occupancy)

Sinh dữ liệu đầu vào cho phân tích MFD và các nghiên cứu giao thông vĩ mô.

Giao diện PyQt5 trực quan: hiển thị video, bounding box, heatmap, bảng metrics theo xe và theo làn.

Xuất dữ liệu ra CSV để phục vụ phân tích/offline hoặc nghiên cứu tiếp theo.

Lưu ý về dữ liệu
Video sử dụng trong repository này là video demo được lấy từ YouTube, dùng cho mục đích nghiên cứu, học thuật và minh họa.

Dữ liệu không quay trực tiếp tại Hà Nội, nhưng được lựa chọn để mô phỏng gần nhất các điều kiện giao thông ở Hà Nội (mật độ, loại phương tiện, góc quay).

Khi triển khai thực tế tại Hà Nội, người dùng cần:

Xin phép và tuân thủ các quy định về sử dụng dữ liệu camera giao thông.

Thay thế đường dẫn video bằng luồng camera thực (RTSP/IP camera).

Hiệu chỉnh lại vùng làn đường và tham số hiệu chuẩn.
https://github.com/user-attachments/assets/b73e30c8-923a-49fb-a900-c5de2684f0dd
