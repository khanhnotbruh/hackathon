# Bài dự thi bảng C

Đây là bài làm siêu gọn nhẹ bằng C/C++ dựa trên [llama.cpp](https://github.com/ggml-org/llama.cpp) , và [simdjson](https://github.com/simdjson/simdjson). Sử dụng model Qwen3.5 9B Q8_0.

---

## 📌 Lời nói đầu

Đây là lần đầu mà em động đến các LLM và parser, bài này đã giúp em học được nhiều thứ hay ho, mới mẻ về thế giới tin học.


---

## ⚡ Những thứ đã được tối giản hóa (Optimizations):

- Để đạt được tốc độ xử lý vượt trội và không làm tràn VRAM, em đã cố gắng loại bỏ toàn bộ các thành phần "thừa thãi" và áp dụng các kỹ thuật tối ưu hóa bộ nhớ nghiêm ngặt:

- <details>
    <summary><b>Không phân mảnh bộ nhớ (Zero Heap Fragmentation)</b></summary>

    Hạn chế tối đa việc lạm dụng <code>malloc</code>/<code>free</code> trong các vòng lặp đệ quy sâu. Toàn bộ các mảng quản lý vị trí con trỏ chuỗi (<code>prompt_pos</code>) và mảng quản lý độ dài (<code>strsize</code>) đều được cấp phát động trực tiếp trên **Stack** thông qua hàm <code>alloca()</code>. 
    
    Cơ chế này giúp con trỏ Stack Pointer (<code>RSP</code>) tự động thu hồi vùng nhớ ngay khi hàm thoát scope (stack unwinding), triệt tiêu hoàn toàn nguy cơ rò rỉ bộ nhớ (Memory Leak) mà không tốn bất kỳ chu kỳ CPU nào cho việc dọn dẹp dữ liệu.
  </details>

- <details>
    <summary><b>Chia để trị đệ quy chống tràn bộ nhớ (OOM Protection)</b></summary>

    Thuật toán tự động tính toán tổng dung lượng token dự kiến (<code>max_token</code>) của phân đoạn câu hỏi hiện tại trước khi nạp vào batch. Nếu con số này vượt quá giới hạn cấu hình phần cứng vật lý (<code>max_batch</code>), hệ thống sẽ kích hoạt giải thuật đệ quy chia đôi mảng câu hỏi (Divide-and-Conquer) để xử lý cô lập trong các batch nhỏ hơn. 
    
    Trong trường hợp phát hiện một câu hỏi đơn lẻ có kích thước quá lớn, vượt rào cấu hình tối đa của hệ thống, bộ xử lý sẽ chủ động **bỏ qua (drop)** câu hỏi đó thay vì cố nạp vào làm sập luồng thực thi, đảm bảo toàn bộ engine chạy liên tục không bao giờ bị dính lỗi <code>OOM</code> (Out of Memory).
  </details>

- <details>
    <summary><b>Xử lý Token Slicing trực tiếp (Zero-Copy Pointer Arithmetic)</b></summary>

    Loại bỏ hoàn toàn các thao tác copy chuỗi thủ công gây thắt nút cổ chai (bottleneck) ở CPU. Độ dài của từng phân đoạn text/token được tính toán trực tiếp bằng **số học con trỏ ngầm (Pointer Arithmetic)** dựa trên khoảng cách giữa các vùng nhớ liên tiếp (<code>prompt_pos[i+1] - prompt_pos[i]</code>). 
    
    Kỹ thuật này giúp tối giản hóa chi phí tính độ dài chuỗi từ độ phức tạp $O(N)$ của hàm <code>strlen</code> truyền thống xuống còn **$O(1)$** tuyệt đối, tối ưu hóa tốc độ chuẩn bị batch lên mức tối đa.
  </details>

- <details>
    <summary><b>Xử lý Batching và Định tuyến Sequence ID Động</b></summary>

    Để tận dụng tối đa sức mạnh xử lý song song của GPU/CPU mà không làm lẫn lộn dữ liệu giữa các câu hỏi khi chia nhỏ đệ quy, engine áp dụng cơ chế định tuyến <code>seq_id</code> (Sequence ID) trực tiếp trên từng token trong batch:
    
    * **Phân rã Sequence độc lập:** Mỗi câu hỏi trong phân đoạn batch được gán một <code>seq_id</code> riêng biệt. Khi nạp vào cấu trúc <code>llama_batch</code>, hệ thống không nhìn nhận dữ liệu dưới dạng một block văn bản thô dài dằng dặc, mà nhìn nhận nó như nhiều luồng xử lý độc lập chạy song song.
    * **Tái sử dụng KV Cache thông minh:** Nhờ việc phân định <code>seq_id</code> tường minh, các lát cắt chung của prompt (như hệ thống tiền tố tùy chọn trong kiến trúc Hamburger) chỉ cần được evaluate một lần duy nhất và lưu vào KV Cache. Các câu hỏi con chỉ cần trỏ chung về vùng cache đó, giúp tiết kiệm bộ nhớ VRAM tối đa và tăng tốc độ xử lý câu hỏi mới lên gấp nhiều lần.
  </details>

---

# Hướng dẫn cài đặt 

## - Sử dụng Docker
Để đảm bảo môi trường chạy đồng nhất, không bị xung đột thư viện hệ thống, project đã được đóng gói hoàn chỉnh trong Docker.
### 1. Tải Image từ Docker Hub

*Lưu ý:* Hãy để các file test đúng tên 'private-test.json' hay 'public-test.json' vào data/

 - Mở terminal lên và kéo image về bằng lệnh:
```bash
docker pull khanhnotbruh/hackathon:latest
```
- Sau đó chạy cái này: (nếu có nvidia gpu thì thêm `--gpus all`)
```bash
docker run --rm khanhnotbruh/hackathon:latest
```

---
​
## ​🛠️Thủ công
### yêu cầu
- Compiler hỗ trợ C++17 trở lên.
-  Có cmake và git.
-  Có python-3 và pip

-  Sao chép repo:
```bash
git clone --depth 1 https://github.com/khanhnotbruh/hackathon.git
```

-  Biên dịch phần mềm
``` bash
cd hackathon&& cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && ./build/engine
```

- Nếu mà trong máy có tải sãn model:
 * hãy để model ở `hackathon/models` và đặt tên là `Qwen3.5-9B-Q4_K_M.gguf`

-  Còn nếu ko có sẵn model thì chạy:
 ``` bash
pip3 install --break-system-packages huggingface_hub
 ```
- Và:
```bash
python3 download.py
```