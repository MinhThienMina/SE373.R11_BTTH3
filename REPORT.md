# Báo cáo BTVN#3 - Agent đặt vé máy bay bằng LangChain/LangGraph

## 1. Đối chiếu yêu cầu slide

Đề bài trên slide yêu cầu: tạo tool mockup, viết harness, cài 3 mẫu agent và đánh giá hiệu quả. Project này đáp ứng bằng các phần sau:

| Yêu cầu | File chính |
|---|---|
| Tool mockup đặt vé máy bay | `tools/flight_tools.py`, logic gốc trong `flight_agent.py` |
| Ràng buộc là dữ liệu | `harness/constraints.py` với `Constraints`, `Task`, `violations()` |
| Tiêu chí hoàn thành kiểm bằng code | `harness/completion_sensor.py` với `verify_done()` đọc trạng thái backend |
| Kiểm quyền trước khi thực thi tool | `harness/permission.py` với `ToolGuard`, allowlist, quyền, approval |
| Phát hiện lặp/đứng tiến triển | `harness/loop_detector.py` với `LoopDetector` và bộ đếm action |
| Bàn giao cho người | `harness/handoff.py` với `Handoff`, `make_handoff()` |
| ReAct | `agents/react_agent.py` |
| Plan-then-Execute | `agents/plan_execute_agent.py` |
| Lai: Plan + recovery/replan | `agents/hybrid_agent.py` |
| Đánh giá 3 mẫu | `benchmark.py`, `results.json` |

## 2. Thiết kế hệ thống

Agent gồm bốn phần theo bài giảng: goal, tools, loop và termination. Goal được biểu diễn bằng `Task`; tools là các hàm mock như `search_flights`, `hold_seat`, `book_flight`; loop được cài bằng `StateGraph`; termination nằm trong harness.

Ranh giới model/harness được tách rõ: `MockLLM` chỉ đề xuất bước kế tiếp, còn code harness parse, kiểm quyền, gọi tool, ghi observation, kiểm điều kiện dừng và bàn giao. Agent không được tự quyết định “đã xong” nếu `verify_done()` không xác nhận có đúng một vé `CONFIRMED` thỏa tuyến, ngày, hành khách, giá, hãng, điểm dừng và khung giờ.

## 3. Ba mẫu thiết kế agent

`react`: mỗi vòng gọi model để chọn hành động dựa trên lịch sử quan sát. Ưu điểm là linh hoạt, nhưng tốn nhiều token hơn và có rủi ro lặp hoặc tuyên bố thành công sớm.

`plan_execute`: model lập kế hoạch một lần, sau đó graph thực thi từng bước. Ưu điểm là ít LLM call, dễ duyệt trước; nhược điểm là kém thích nghi khi timeout, hết chỗ hoặc dữ liệu thay đổi.

`hybrid`: lập kế hoạch như plan-then-execute nhưng có bước recover/replan cục bộ khi lỗi. Mẫu này giữ được chi phí tương đối thấp nhưng thích nghi tốt hơn môi trường biến động.

## 4. Harness và điều kiện dừng

Các điều kiện dừng trong slide được cài như sau:

| Điều kiện dừng | Cài đặt |
|---|---|
| Đạt mục tiêu | `verify_done()` kiểm backend thật |
| Hết ngân sách | `max_steps`, `max_tool_calls`, recursion limit |
| Phát hiện lặp | `ToolGuard.seen` và `LoopDetector` |
| Bế tắc | `ToolGuard`/`LoopDetector` trả `STALL_DETECTED` khi nhiều bước liên tiếp không tạo tiến triển |
| Cần con người/approval | `ToolGuard` chặn destructive tool và hỏi approval giả lập cho `book_flight` |

Khi thất bại hoặc cần người xử lý, `Handoff` ghi: trạng thái, lý do, tool đã thử, việc còn lại, hành động khuyến nghị và cleanup các hold treo.

## 5. Kết quả đánh giá

Dữ liệu trong `results.json` gồm 960 lượt chạy khi harness bật và 960 lượt ablation khi harness tắt. Tóm tắt khi harness bật:

| Mẫu | Thành công | Fail an toàn | Không an toàn | LLM calls TB | Tool calls TB |
|---|---:|---:|---:|---:|---:|
| ReAct | 93.4% | 6.6% | 0.0% | 4.22 | 3.13 |
| Plan-then-Execute | 74.1% | 25.9% | 0.0% | 1.77 | 2.12 |
| Lai | 93.8% | 6.2% | 0.0% | 3.17 | 2.88 |

Khi tắt harness, tỉ lệ không an toàn tăng rõ:

| Mẫu | Thành công | Fail an toàn | Không an toàn |
|---|---:|---:|---:|
| ReAct | 64.7% | 2.2% | 33.1% |
| Plan-then-Execute | 67.5% | 18.8% | 13.8% |
| Lai | 83.1% | 1.2% | 15.6% |

Nhận xét: Hybrid tốt nhất trong bài toán này vì vừa có kế hoạch vừa phục hồi được khi tool lỗi. ReAct gần ngang Hybrid nhưng tốn nhiều LLM call hơn. Plan-then-Execute tiết kiệm chi phí nhất nhưng thất bại an toàn nhiều hơn vì kế hoạch ban đầu dễ lỗi thời.

## 6. Cách chạy

```bash
python main.py test
python main.py demo hybrid --scenario 0 --seed 3
python benchmark.py --n 40
```

`flight_agent.py` vẫn là bản đầy đủ một file. Các folder `agents/`, `harness/`, `tools/` được tách để thể hiện đúng cấu trúc nộp bài.
