import torch

from collections import OrderedDict

ID_TO_NAMES = {
    0: 'title',
    1: 'plain text',
    2: 'abandon',
    3: 'figure',
    4: 'figure_caption',
    5: 'table',
    6: 'table_caption',
    7: 'table_footnote',
    8: 'isolate_formula',
    9: 'formula_caption'}
"""
ID_TO_NAMES = OrderedDict({
    0: "Caption",
    1: "Footnote",
    2: "Formula",
    3: "List-item",
    4: "Page-footer",
    5: "Page-header",
    6: "Picture",
    7: "Section-header",
    8: "Table",
    9: "Text",
    10: "Title"
})
"""
#MODEL_DIR = r"C:\Users\Mazen\Desktop\CHATBOT\models\DocLayout-YOLO-DocStructBench\doclayout_yolo_docstructbench_imgsz1024.pt"
#MODEL_DIR=r"C:\Users\Mazen\Desktop\CHATBOT\models\Layout\YOLO\yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt"
MODEL_DIR = r"models\LLLAY\doclayout_yolo_ft.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GROQ_API_KEY = "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"
IOU_THRESHOLD = .5
CONF_THRESHOLD = .1
DEBUG_DIR = "Debug"
TEXT = "TEXT_CHUNK"
TABLE = "TABLE_CHUNK"
TEXT_IMAGE = "TEXT_IMAGE"

SUPPRESS_THRESHOLDS  = 0.2