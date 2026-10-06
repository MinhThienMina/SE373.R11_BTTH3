# SE373 - BTVN#3: Flight Booking Agent

- **Sinh viên:** Đào Minh Thiện
- **MSSV:** 23521477
- **Môn học:** Kỹ thuật xây dựng hệ thống Agentic AI (SE373)
- **Bài tập:** BTVN#3 - Dựng agent đặt vé máy bay bằng LangChain / LangGraph

Project mô phỏng agent đặt vé máy bay bằng Python, LangChain và LangGraph. Bài
tập so sánh ba mẫu thiết kế agent (ReAct, Plan-then-Execute, Hybrid) và đánh giá
vai trò của harness khi agent gọi các công cụ đặt vé.

> Toàn bộ dữ liệu chuyến bay và thao tác đặt vé dùng mock backend chạy cục bộ.
> Project dùng `MockLLM` (seed cố định), không gọi dịch vụ AI bên ngoài và không
> cần API key.

## Bài nộp

| Nội dung                  | Vị trí trong repo                                                                   |
| ------------------------- | ----------------------------------------------------------------------------------- |
| **Báo cáo (PDF)**         | [`BTTH3_23521477_DaoMinhThien.pdf`](BTTH3_23521477_DaoMinhThien.pdf)                |
| Báo cáo (bản Markdown)    | [`REPORT.md`](REPORT.md)                                                            |
| Mã nguồn chính (`.py`)    | [`flight_agent.py`](flight_agent.py): backend, tool, harness, 3 mẫu agent, đánh giá |
| Chạy kiểm thử và demo     | [`main.py`](main.py)                                                                |
| Chạy đánh giá (benchmark) | [`benchmark.py`](benchmark.py)                                                      |

## Đối chiếu với yêu cầu đề bài

| Yêu cầu                                                                                               | Nơi cài đặt                                                                                                      | Cách kiểm chứng                    |
| ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| 1. Đủ các lớp harness: ràng buộc là dữ liệu, tiêu chí hoàn thành kiểm bằng code, kiểm quyền, bàn giao | `Constraints`, `verify_done`, `ToolGuard`, `Handoff` trong `flight_agent.py`; giao diện theo module ở `harness/` | `python main.py test` (8 kiểm tra) |
| 2. Cài đặt agent với 3 mẫu: ReAct, Plan-then-Execute, Lai                                             | `build_react`, `build_plan_graph(replan=False/True)` trong `flight_agent.py`; wrapper ở `agents/`                | `python main.py demo <mẫu>`        |
| 3. Đánh giá hiệu quả 3 mẫu                                                                            | Hàm `evaluate`, `summarize`, `per_scenario`                                                                      | `python benchmark.py`              |

## Kết quả chính

Chạy `python benchmark.py --n 40` (8 kịch bản x 40 seed x 3 mẫu = 960 lượt cho
mỗi chế độ).

**Khi bật harness**

| Mẫu            | Thành công | Không an toàn | LLM calls | Token ước tính |
| -------------- | ---------: | ------------: | --------: | -------------: |
| `react`        |      93.4% |          0.0% |      4.22 |           1702 |
| `plan_execute` |      74.1% |          0.0% |      1.77 |            424 |
| `hybrid`       |      93.8% |          0.0% |      3.17 |            739 |

**Ablation (tắt harness):** tỉ lệ không an toàn tăng lên 33.1% (`react`), 13.8%
(`plan_execute`) và 15.6% (`hybrid`). Phân tích đầy đủ nằm trong báo cáo PDF.

Lưu ý khi đọc số liệu:

- Token là ước tính (độ dài chuỗi chia 4); độ trễ là giá trị mô phỏng.
- Tỉ lệ lỗi của `MockLLM` do người làm đặt tay (`DEFAULT_NOISE`), nên chiều so
  sánh giữa các mẫu đáng tin hơn con số tuyệt đối.
- Kịch bản 1 và 4 chỉ có một chuyến thỏa ràng buộc nên tỉ lệ thành công tối đa
  khoảng 75%.
- `LOOP_DETECTED` và `STALL_DETECTED` chưa lần nào kích hoạt trong 960 lượt;
  hai kiểu dừng này được kiểm bằng `python main.py test`.

## Mục lục

- [Tính năng](#tính-năng)
- [Kiến trúc thư mục](#kiến-trúc-thư-mục)
- [Cài đặt](#cài-đặt)
- [Cách sử dụng](#cách-sử-dụng)
- [Các mẫu agent](#các-mẫu-agent)
- [Safety harness](#safety-harness)
- [Kịch bản đánh giá](#kịch-bản-đánh-giá)

## Tính năng

- Tìm chuyến bay, giữ chỗ, xác nhận đặt vé, giải phóng giữ chỗ và hủy vé qua
  mock tools (LangChain `@tool`).
- Biểu diễn ràng buộc bằng dữ liệu: giá tối đa, số điểm dừng, hãng bay được
  phép, khung giờ khởi hành.
- Kiểm tra trạng thái backend để xác nhận vé đã đặt đúng yêu cầu; agent không
  thể tự tuyên bố hoàn tất thay cho bước kiểm tra này.
- Chặn tool không được cấp quyền, tham số vi phạm ràng buộc, lời gọi lặp và
  chuỗi thao tác không tạo tiến triển.
- Chuyển các trường hợp không tự xử lý được sang trạng thái bàn giao
  (`handoff` / `ESCALATED`).
- Tiêm lỗi vào backend (timeout, hết chỗ, giá nhảy) để thử khả năng chống chịu.
- Kiểm thử harness, demo từng mẫu và benchmark tái lập được theo seed.

## Kiến trúc thư mục

```text
.
├── agents/
│   ├── react_agent.py          # Wrapper/build graph cho ReAct
│   ├── plan_execute_agent.py   # Wrapper/build graph cho Plan-then-Execute
│   └── hybrid_agent.py         # Wrapper/build graph cho Hybrid có replan
├── harness/
│   ├── constraints.py          # Ràng buộc và kiểm tra chuyến bay
│   ├── completion_sensor.py    # Xác minh kết quả đặt vé từ backend
│   ├── permission.py           # Quyền gọi tool và kiểm soát thao tác
│   ├── loop_detector.py        # Phát hiện lặp và bế tắc
│   └── handoff.py              # Thông tin bàn giao khi cần can thiệp
├── tools/
│   └── flight_tools.py         # Giao diện các mock flight tools
├── flight_agent.py             # Backend, MockLLM, graph, harness và benchmark
├── config.py                   # Export cấu hình, mẫu agent và kịch bản
├── main.py                     # CLI cho self-test và demo
├── benchmark.py                # CLI đánh giá và lưu kết quả
├── REPORT.md                   # Báo cáo (Markdown)
├── BTTH3_23521477_DaoMinhThien.pdf  # Báo cáo (PDF) - bản nộp
├── requirements.txt            # Dependency Python
└── .env.example                # Ví dụ biến môi trường; không cần cho MockLLM
```

Các module trong `agents/`, `harness/` và `tools/` cung cấp giao diện theo từng
phần; phần cài đặt chính được tập trung trong `flight_agent.py`.

## Cài đặt

Cần Python 3.10 trở lên và `pip`.

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Cài dependencies:

```bash
python -m pip install -r requirements.txt
```

## Cách sử dụng

Chạy nhanh toàn bộ (sau khi cài đặt):

```bash
python main.py test
python main.py demo hybrid --scenario 1 --seed 3
python benchmark.py --n 40
```

### Chạy self-test cho harness

```bash
python main.py test
```

Chạy 8 kiểm tra: quyền tool, allowlist, ràng buộc chuyến bay, xác minh đặt vé
(chưa có vé / có vé hợp lệ), phát hiện lặp, phát hiện bế tắc và nạp ràng buộc
từ JSON. Khi thành công, chương trình in `selftest: OK` cùng danh sách kiểm tra
đã qua.

### Chạy một demo

```bash
python main.py demo
```

Mặc định demo dùng mẫu `hybrid`, kịch bản `0` và seed `3`. Có thể chọn mẫu,
kịch bản và seed:

```bash
python main.py demo react --scenario 1 --seed 7
python main.py demo plan_execute --scenario 0 --seed 4
python main.py demo hybrid --scenario 1 --seed 3
```

Các mẫu là `react`, `plan_execute` và `hybrid`. `--scenario` nhận chỉ số từ `0`
đến `7`; `--seed` điều khiển dữ liệu và lỗi giả lập để tái lập một lượt chạy.

Hai demo nên xem để thấy sự khác biệt giữa các mẫu:

- `demo hybrid --scenario 1 --seed 3`: guard chặn 2 lần chọn chuyến vi phạm
  ràng buộc, `hybrid` replan rồi đặt vé thành công.
- `demo plan_execute --scenario 0 --seed 4`: gặp `SOLD_OUT`, mẫu này không có
  đường sửa lỗi nên dừng và bàn giao (`ESCALATED`).

Demo in ra mẫu agent, kịch bản, chuỗi tool đã thử, thống kê guard và thông tin
bàn giao.

### Chạy benchmark

```bash
python benchmark.py
```

Mặc định chạy 40 seed cho mỗi tổ hợp mẫu và kịch bản. Đổi số seed:

```bash
python benchmark.py --n 10
```

Benchmark đánh giá mỗi tổ hợp với harness bật và tắt, in thống kê tổng hợp và
tỉ lệ thành công theo kịch bản. Kết quả chi tiết được ghi vào `results.json`
trong thư mục hiện hành (file này bị loại khỏi Git bởi `.gitignore` và được tạo
lại mỗi lần chạy).

## Các mẫu agent

| Mẫu            | Cách hoạt động                                                                                                          | Đặc điểm                                                    |
| -------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `react`        | Chọn hành động tiếp theo dựa trên trạng thái và quan sát hiện tại (`think → act → gate`).                               | Thích nghi theo kết quả tool; cần nhiều lần gọi model hơn.  |
| `plan_execute` | Lập kế hoạch một lần rồi thực thi tuần tự; chỉ tự thử lại khi timeout.                                                  | Ít lần gọi model nhất, nhưng không sửa được lỗi giữa chừng. |
| `hybrid`       | Lập kế hoạch, thêm nút `recover` để sửa tham số, thử lại hoặc chọn chuyến khác khi một bước hỏng (tối đa 3 lần replan). | Cân bằng giữa thực thi có kế hoạch và khả năng thích nghi.  |

Các bước suy luận do `MockLLM` mô phỏng, nhằm tập trung đánh giá luồng điều
khiển, công cụ và harness thay vì chất lượng của một model cụ thể. Muốn dùng
LLM thật chỉ cần cài lại các hàm `plan`, `select`, `react_step`, `recover` của
`MockLLM`.

## Safety harness

Harness đặt các kiểm tra bằng code giữa agent và mock backend:

1. **Ràng buộc (`harness/constraints.py`)**: kiểm tra chuyến bay và giá có thỏa
   yêu cầu của task hay không.
2. **Completion sensor (`harness/completion_sensor.py`)**: xác minh trạng thái
   booking trong backend; chỉ coi là hoàn tất khi vé `CONFIRMED` thỏa yêu cầu.
3. **Permission guard (`harness/permission.py`)**: kiểm tra quyền và allowlist
   trước khi thực thi tool; chặn thao tác không được phép (ví dụ
   `cancel_booking`) hoặc yêu cầu phê duyệt giả lập khi giá vượt hạn mức.
4. **Loop/stall detection (`harness/loop_detector.py`)**: phát hiện lời gọi lặp
   lại (`LOOP_DETECTED`) hoặc nhiều lời gọi hỏng liên tiếp không tạo tiến triển
   (`STALL_DETECTED`).
5. **Handoff (`harness/handoff.py`)**: đóng gói trạng thái, lý do thất bại, các
   thao tác đã thử và hành động tiếp theo khi cần escalation.

Benchmark có chế độ bật/tắt harness để so sánh độ an toàn và hiệu quả trên cùng
tập kịch bản.

## Kịch bản đánh giá

Benchmark gồm 8 kịch bản, bao phủ cả yêu cầu khả thi và không khả thi:

| Chỉ số | Tuyến   | Yêu cầu chính                                          |
| -----: | ------- | ------------------------------------------------------ |
|      0 | SGN–HAN | Ngân sách tối đa 3.000.000 VND, tối đa 1 điểm dừng     |
|      1 | SGN–DAD | Bay thẳng, khởi hành trước 12:00, tối đa 1.800.000 VND |
|      2 | HAN–SGN | Chỉ VN/VJ, tối đa 1 điểm dừng, tối đa 2.600.000 VND    |
|      3 | SGN–PQC | Bay thẳng, khởi hành từ 14:00, tối đa 1.300.000 VND    |
|      4 | SGN–HAN | Ngân sách tối đa 1.700.000 VND, tối đa 1 điểm dừng     |
|      5 | SGN–DAD | Không khả thi: chỉ QH, bay thẳng, tối đa 1.500.000 VND |
|      6 | SGN–HAN | Không khả thi: ngân sách tối đa 300.000 VND            |
|      7 | HAN–SGN | Không khả thi: bay thẳng trước 05:00                   |

Mỗi lượt chạy ghi nhận: kết quả (thành công / fail an toàn / không an toàn), số
lần gọi LLM và tool, token ước tính, độ trễ mô phỏng, số lần guard chặn, lỗi
tool, số lần báo hoàn thành sai bị bắt và số lần replan.
