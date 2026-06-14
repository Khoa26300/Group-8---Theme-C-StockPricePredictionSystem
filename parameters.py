import os
import time
from tensorflow.keras.layers import LSTM

# 1. Chọn mã cổ phiếu và khung dữ liệu theo yêu cầu của đề.
#    Ở đây ta dùng CBA.AX để đồng bộ với phiên bản v0.1 trong folder stock_prediction.
ticker = "CBA.AX"

# Window size: v0.1 dùng 60 ngày nên ta giữ 60 ngày để dễ so sánh.
N_STEPS = 60

# Lookup step: dự đoán ngày kế tiếp.
LOOKUP_STEP = 1

# Toàn bộ khoảng dữ liệu dùng cho bài làm.
START_DATE = "2020-01-01"
END_DATE = "2024-07-02"

# Các cấu hình tiền xử lý dữ liệu.
SCALE = True
scale_str = f"sc-{int(SCALE)}"
SHUFFLE = True
shuffle_str = f"sh-{int(SHUFFLE)}"

# Bài tập yêu cầu có thể tách train/test theo ngày hoặc ngẫu nhiên.
# Ở đây mặc định là tách theo ngày để phù hợp với chuỗi thời gian.
SPLIT_BY_DATE = True
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"

TEST_SIZE = 0.2
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]

# Cách xử lý NaN và cache dữ liệu cục bộ.
NAN_METHOD = "drop"
SAVE_LOCAL_DATA = True
LOAD_LOCAL_DATA = True
LOCAL_DATA_PATH = os.path.join("data", f"{ticker.replace('.', '_')}_{START_DATE}_{END_DATE}.csv")
date_now = time.strftime("%Y-%m-%d")

### model parameters
N_LAYERS = 2
CELL = LSTM
UNITS = 256
DROPOUT = 0.4
BIDIRECTIONAL = False

### training parameters
# Bạn có thể để Huber_loss hoặc chuyển thành "mean_squared_error" giống v0.1 để dễ so sánh chỉ số lỗi
LOSS = "mean_squared_error"
OPTIMIZER = "adam"
BATCH_SIZE = 64

# SỬA TẠI ĐÂY: v0.1 chạy 25 epochs. P1 cấu hình tận 500 epochs sẽ rất lâu.
# Hãy chỉnh về 25 hoặc 50 epochs để máy bạn (Legion 5) chạy nhanh, lấy kết quả làm báo cáo kịp thời.
EPOCHS = 25

# Khử dấu chấm "." trong tên mã cổ phiếu (CBA.AX -> CBA_AX) 
# để tránh Windows hiểu lầm thư mục lưu trữ Log hoặc File Trọng số (.h5)
ticker_clean = ticker.replace(".", "_")

ticker_data_filename = os.path.join("data", f"{ticker_clean}_{date_now}.csv")

model_name = f"{date_now}_{ticker_clean}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    model_name += "-b"

# ------------------------------
# Task 3 (v0.2) visualization settings
# ------------------------------
# Set to False if you only want numerical evaluation without drawing charts.
ENABLE_TASK3_PLOTS = True

# Candlestick: each candle aggregates n consecutive trading days (n >= 1).
CANDLE_N_DAYS = 5

# Boxplot: each box summarizes a moving window of n consecutive trading days.
BOXPLOT_WINDOW_SIZE = 20
BOXPLOT_STEP = 5
BOXPLOT_VALUE_COLUMN = "adjclose"
BOXPLOT_MAX_WINDOWS = 30