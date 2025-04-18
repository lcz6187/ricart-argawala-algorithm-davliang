from __future__ import annotations

import copy
import enum
import heapq
import json
import math
import random
import sys
import time
from typing import (
    Callable,
    Dict,
    List,
    Tuple,
    Optional,
    Set,
    Union,
    Any,
)
from typing_extensions import TypeAlias

from loguru import logger
import matplotlib

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import (
    Qt,
    QObject,
    QEvent,
    QRegularExpression,
    QRegularExpressionMatchIterator,
    QRegularExpressionMatch,
)
from PySide6.QtGui import (
    QSyntaxHighlighter,
    QTextCharFormat,
    QColor,
    QFont,
    QPalette,
)

from rich.console import Console
from rich.logging import RichHandler

console = Console()

logger.configure(
    handlers=[
        {
            "sink": RichHandler(
                console=console,
                rich_tracebacks=True,
            ),
            "format": "{message}",
            "level": "INFO",
        },
    ]
)

DEFAULT_NUM_NODES: int = 5
DEFAULT_CS_DURATION: int = 30
DEFAULT_MIN_DELAY: int = 5
DEFAULT_MAX_DELAY: int = 15

GRAPH_LAYOUT_SEED: int = 0
NODE_CLICK_TOLERANCE: float = 0.1
NODE_RADIUS_VISUAL: float = 0.05
ARROW_OFFSET_AMOUNT: float = 0.03
ARROW_LENGTH: float = 0.05
ARROW_HEAD_WIDTH: float = 0.035
ARROW_HEAD_LENGTH: float = 0.05

RANDOM_CS_RANGE = (10, 50)
RANDOM_MIN_DELAY_RANGE = (2, 12)
RANDOM_MAX_DELAY_OFFSET = (1, 20)
RANDOM_REQ_MAX_COUNT = 3
RANDOM_REQ_MAX_TIME = 250


class MessageType(enum.Enum):
    REQUEST = 1
    REPLY = 2


class EventType(str, enum.Enum):
    MESSAGE_ARRIVAL = "MSG_ARRIVE"
    CS_ENTER = "CS_ENTER"
    CS_EXIT = "CS_EXIT"
    SCHEDULED_REQUEST = "SCHED_REQ"
    MANUAL_REQUEST = "MANUAL_REQ"
    TIME_ADVANCE = "TIME_ADV"
    INIT = "INIT"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"


class NodeState(enum.Enum):
    IDLE = 0
    WANTED = 1
    HELD = 2

    def __str__(self):
        return self.name[0]


EdgeKey: TypeAlias = Tuple[int, int]

NodeId: TypeAlias = int
CSDurations: TypeAlias = Dict[NodeId, int]
EdgeDelays: TypeAlias = Dict[EdgeKey, Dict[str, int]]
ScheduledRequests: TypeAlias = Dict[NodeId, List[int]]

ClockValue: TypeAlias = int
Timestamp: TypeAlias = Tuple[ClockValue, NodeId]

NodeDict: TypeAlias = Dict[NodeId, "Node"]
NodePositions: TypeAlias = Dict[NodeId, Tuple[float, float]]
HistoryEntry: TypeAlias = Dict[str, Any]
Event: TypeAlias = Tuple[int, int, EventType, Any]
VisCallback: TypeAlias = Callable[
    [nx.Graph, NodeDict, int, NodePositions], None
]


def _edge_key_to_str(edge_key: EdgeKey) -> str:
    u, v = sorted(edge_key)
    return f"{u},{v}"


def _str_to_edge_key(edge_str: str) -> EdgeKey:
    try:
        parts = edge_str.split(",")
        if len(parts) == 2:
            u = int(parts[0].strip())
            v = int(parts[1].strip())
            return tuple(sorted((u, v)))
        else:
            raise ValueError(
                "Edge string must contain two integers separated by a comma."
            )
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid edge string format '{edge_str}': {e}"
        ) from e


APP_STYLESHEET: str = """
QWidget {
    font-family: "Segoe UI", "Cantarell", "Helvetica Neue", sans-serif;
    font-size: 10pt;
    color: #212121;
}

QMainWindow, QDialog {
    background-color: #F5F5F5;
}

QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 6px;
    margin-top: 18px;
    padding: 15px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 15px;
    padding: 2px 8px;
    bottom: -1px;
    background-color: #FFFFFF;
    color: #0277BD;
    font-weight: bold;
    font-size: 11pt;
    border: 1px solid #E0E0E0;
    border-bottom-color: #FFFFFF;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}

QPushButton {
    padding: 5px 12px;
    min-height: 22px;
    min-width: 70px;
    border: 1px solid #CCCCCC;
    background-color: #E8E8E8;
    color: #212121;
    border-radius: 4px;
    outline: none;
}

QPushButton:hover {
    background-color: #DCDCDC;
    border-color: #BBBBBB;
}

QPushButton:pressed {
    background-color: #C8C8C8;
    border-color: #ADADAD;
}

QPushButton:disabled {
    background-color: #EEEEEE;
    color: #BDBDBD;
    border-color: #E0E0E0;
}

QPushButton[default="true"], QPushButton#init_button {
    background-color: #0288D1;
    color: white;
    border: none;
    font-weight: bold;
    padding: 5px 12px;
    min-height: 22px;
    min-width: 80px;
}

QPushButton[default="true"]:hover, QPushButton#init_button:hover {
    background-color: #0277BD;
}

QPushButton[default="true"]:pressed, QPushButton#init_button:pressed {
    background-color: #01579B;
}

QPushButton[default="true"]:disabled, QPushButton#init_button:disabled {
    background-color: #B3E5FC;
    color: #E0F7FA;
    border: none;
    font-weight: bold;
}

QPushButton#NodeRequestButton {
    padding: 3px 6px;
    min-width: 40px;
    min-height: 18px;
    font-size: 9pt;
    border-radius: 3px;
    background-color: #EBF5FF;
    color: #0D47A1;
    border: 1px solid #C5E1F9;
}

QPushButton#NodeRequestButton:hover {
    background-color: #D6EAFD;
    border-color: #ADD6F7;
}

QPushButton#NodeRequestButton:pressed {
    background-color: #C5E1F9;
}

QPushButton#NodeRequestButton:disabled {
    background-color: #EEEEEE;
    color: #BDBDBD;
    border-color: #E0E0E0;
}

QLineEdit, QSpinBox {
    background-color: white;
    border: 1px solid #CCCCCC;
    border-radius: 4px;
    padding: 5px;
    color: #212121;
    min-height: 22px;
}

QLineEdit:focus, QSpinBox:focus {
    border: 1px solid #0288D1;
}

QLineEdit:read-only, QSpinBox:read-only {
    background-color: #F5F5F5;
    color: #757575;
}

QLineEdit:disabled, QSpinBox:disabled {
    background-color: #EEEEEE;
    color: #BDBDBD;
    border-color: #E0E0E0;
}

QSpinBox::up-button, QSpinBox::down-button {
    subcontrol-origin: border;
    background-color: #F5F5F5;
    border: 1px solid #DCDCDC;
    border-radius: 2px;
    width: 18px;
    padding: 0px;
    margin: 0px;
}

QSpinBox::up-button { subcontrol-position: top right; }
QSpinBox::down-button { subcontrol-position: bottom right; }

QSpinBox::up-button:hover, QSpinBox::down-button:hover {
    background-color: #EAEAEA;
    border-color: #C8C8C8;
}

QSpinBox::up-button:pressed, QSpinBox::down-button:pressed {
    background-color: #DDDDDD;
    border-color: #BDBDBD;
}

QSpinBox::up-arrow {
    image: url(":/qt-project.org/styles/commonstyle/images/up-arrow-16.png");
    width: 10px;
    height: 10px;
}

QSpinBox::down-arrow {
    image: url(":/qt-project.org/styles/commonstyle/images/down-arrow-16.png");
    width: 10px;
    height: 10px;
}

QSpinBox::up-arrow:disabled, QSpinBox::up-arrow:off {
    image: url(":/qt-project.org/styles/commonstyle/images/up-arrow-disabled-16.png");
}

QSpinBox::down-arrow:disabled, QSpinBox::down-arrow:off {
    image: url(":/qt-project.org/styles/commonstyle/images/down-arrow-disabled-16.png");
}

QTabWidget::pane {
    border: 1px solid #E0E0E0;
    border-top: none;
    background-color: white;
    border-bottom-left-radius: 4px;
    border-bottom-right-radius: 4px;
}

QTabBar::tab {
    background: #EEEEEE;
    border: 1px solid #E0E0E0;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 8px 18px;
    margin-right: 2px;
    color: #757575;
    min-width: 80px;
}

QTabBar::tab:selected {
    background: white;
    color: #212121;
    border-color: #E0E0E0;
    font-weight: bold;
    margin-bottom: -1px;
    border-bottom: 2px solid #0288D1;
}

QTabBar::tab:!selected:hover {
    background: #F5F5F5;
    color: #424242;
}

QTabBar {
    border-bottom: 1px solid #E0E0E0;
}

QTableWidget {
    border: 1px solid #E0E0E0;
    gridline-color: #EEEEEE;
    background-color: white;
    alternate-background-color: #FAFAFA;
    selection-background-color: #BBDEFB;
    selection-color: #1A237E;
    font-size: 9pt;
}

QHeaderView::section {
    background-color: #F5F5F5;
    color: #424242;
    padding: 8px 5px;
    border-style: none;
    border-bottom: 1px solid #D0D0D0;
    font-weight: bold;
    text-align: left;
}

QHeaderView::section:horizontal {
    border-right: 1px solid #E0E0E0;
}

QHeaderView::section:horizontal:last {
    border-right: none;
}

QTableCornerButton::section {
    background-color: #F5F5F5;
    border: none;
    border-bottom: 1px solid #D0D0D0;
    border-right: 1px solid #E0E0E0;
}

QTextEdit {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 4px;
    padding: 8px;
    color: #333333;
    font-family: "Consolas", "Monaco", "Courier New", monospace;
    font-size: 9pt;
}

QTextEdit:read-only {
    background-color: #FFFFFF;
}

QScrollArea {
    border: 1px solid #E0E0E0;
    border-radius: 4px;
    background-color: white;
}

QScrollBar:vertical {
    border: none;
    background: #F5F5F5;
    width: 10px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #BDBDBD;
    min-height: 25px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #9E9E9E;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
    border: none;
    background: none;
}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}

QScrollBar:horizontal {
    border: none;
    background: #F5F5F5;
    height: 10px;
    margin: 0px;
}

QScrollBar::handle:horizontal {
    background: #BDBDBD;
    min-width: 25px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal:hover {
    background: #9E9E9E;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
    border: none;
    background: none;
}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
}

QSplitter::handle {
    background-color: #E0E0E0;
}

QSplitter::handle:horizontal {
    width: 5px;
    margin: 0 2px;
}

QSplitter::handle:vertical {
    height: 5px;
    margin: 2px 0;
}

QSplitter::handle:hover {
    background-color: #BDBDBD;
}

QSplitter::handle:pressed {
    background-color: #9E9E9E;
}

QStatusBar {
    background-color: #EEEEEE;
    color: #616161;
    border-top: 1px solid #E0E0E0;
    font-size: 9pt;
    padding-left: 8px;
}

QStatusBar::item {
    border: none;
}

QMessageBox {
    background-color: #F5F5F5;
}

QMessageBox QLabel#qt_msgbox_label {
   color: #212121;
   font-size: 10pt;
}

QMessageBox QPushButton {
   min-width: 80px;
   padding: 6px 15px;
}

QFileDialog {
    background-color: #F5F5F5;
}
"""


class AdvancedConfigWindow(QDialog):
    def __init__(
        self,
        parent: QWidget,
        num_nodes: int,
        current_cs_durations: CSDurations,
        current_edge_delays: EdgeDelays,
        current_scheduled_requests: ScheduledRequests,
        default_cs: int,
        default_min: int,
        default_max: int,
    ):
        super().__init__(parent)
        self.setWindowTitle("Advanced Configuration")
        self.setModal(True)

        self.num_nodes: int = num_nodes
        self.result: Optional[Dict[str, Any]] = None

        self.cs_duration_inputs: Dict[NodeId, QSpinBox] = {}
        self.edge_min_delay_inputs: Dict[EdgeKey, QSpinBox] = {}
        self.edge_max_delay_inputs: Dict[EdgeKey, QSpinBox] = {}
        self.scheduled_request_inputs: Dict[NodeId, QLineEdit] = {}

        self.current_cs_durations: CSDurations = copy.deepcopy(
            current_cs_durations
        )
        self.current_edge_delays: EdgeDelays = copy.deepcopy(
            current_edge_delays
        )
        self.current_scheduled_requests: ScheduledRequests = copy.deepcopy(
            current_scheduled_requests
        )
        self.default_cs: int = default_cs
        self.default_min: int = default_min
        self.default_max: int = default_max

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(12)

        cs_group = self._setup_cs_durations_ui()
        edge_group = self._setup_edge_delays_ui()
        req_group = self._setup_scheduled_requests_ui()
        button_layout = self._setup_buttons_ui()

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(cs_group)
        splitter.addWidget(edge_group)
        splitter.addWidget(req_group)

        total_height = 900
        cs_h = int(total_height * 0.25)
        edge_h = int(total_height * 0.40)
        req_h = int(total_height * 0.35)
        splitter.setSizes([cs_h, edge_h, req_h])

        main_layout.addWidget(splitter, 1)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)
        self.resize(700, 900)

    def _create_scrollable_area(
        self, parent_widget: QWidget
    ) -> Tuple[QScrollArea, QGridLayout]:
        scroll_area = QScrollArea(parent_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)

        scroll_content_widget = QWidget()

        scroll_area.setWidget(scroll_content_widget)

        content_layout = QGridLayout(scroll_content_widget)
        content_layout.setContentsMargins(5, 5, 10, 5)
        content_layout.setSpacing(8)
        scroll_content_widget.setLayout(content_layout)

        return scroll_area, content_layout

    def _setup_cs_durations_ui(self) -> QGroupBox:
        cs_group = QGroupBox("CS Durations (per Node)")
        cs_main_layout = QVBoxLayout(cs_group)
        cs_main_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area, grid_layout = self._create_scrollable_area(cs_group)
        cs_main_layout.addWidget(scroll_area)

        scroll_content_widget = scroll_area.widget()
        if scroll_content_widget:
            scroll_content_widget.setUpdatesEnabled(False)
            logger.debug("CS Durations UI updates disabled.")

        try:
            num_cols = 2
            num_rows = (self.num_nodes + num_cols - 1) // num_cols

            for i in range(self.num_nodes):
                row = i % num_rows
                col_base = (i // num_rows) * 2

                node_id = i
                label = QLabel(f"Node {node_id}:")
                spinbox = QSpinBox()
                spinbox.setRange(0, 99999)
                spinbox.setToolTip(
                    f"Time node {node_id} spends in the critical section."
                )
                spinbox.setValue(
                    self.current_cs_durations.get(node_id, self.default_cs)
                )

                grid_layout.addWidget(
                    label, row, col_base + 0, Qt.AlignmentFlag.AlignLeft
                )
                grid_layout.addWidget(spinbox, row, col_base + 1)
                self.cs_duration_inputs[node_id] = spinbox

            grid_layout.setColumnStretch(1, 1)
            grid_layout.setColumnStretch(3, 1)
            grid_layout.setColumnMinimumWidth(0, 60)
            grid_layout.setColumnMinimumWidth(2, 60)

            grid_layout.setRowStretch(grid_layout.rowCount(), 1)

        finally:
            if scroll_content_widget:
                scroll_content_widget.setUpdatesEnabled(True)
                logger.debug("CS Durations UI updates re-enabled.")

        return cs_group

    def _setup_edge_delays_ui(self) -> QGroupBox:
        edge_group = QGroupBox("Edge Delays (Min / Max)")
        edge_main_layout = QVBoxLayout(edge_group)
        edge_main_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area, grid_layout = self._create_scrollable_area(edge_group)

        scroll_content_widget = scroll_area.widget()
        if scroll_content_widget:
            scroll_content_widget.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )

        edge_main_layout.addWidget(scroll_area)

        header_font = QFont()
        header_font.setBold(True)
        headers = ["Edge", "Min", "Max", "", "Edge", "Min", "Max"]
        header_alignments = [
            Qt.AlignmentFlag.AlignLeft,
            Qt.AlignmentFlag.AlignCenter,
            Qt.AlignmentFlag.AlignCenter,
            Qt.AlignmentFlag.AlignCenter,
            Qt.AlignmentFlag.AlignLeft,
            Qt.AlignmentFlag.AlignCenter,
            Qt.AlignmentFlag.AlignCenter,
        ]
        for col, text in enumerate(headers):
            if col == 3:
                continue
            label = QLabel(text)
            label.setFont(header_font)
            grid_layout.addWidget(
                label,
                0,
                col,
                header_alignments[col] | Qt.AlignmentFlag.AlignVCenter,
            )

        all_edges: List[EdgeKey] = sorted(
            [
                tuple(sorted((u, v)))
                for u in range(self.num_nodes)
                for v in range(u + 1, self.num_nodes)
            ]
        )
        num_cols_data = 2
        num_edges = len(all_edges)
        num_rows_data = (
            (num_edges + num_cols_data - 1) // num_cols_data
            if num_cols_data > 0
            else 0
        )
        max_content_row = 0

        if scroll_content_widget:
            scroll_content_widget.setUpdatesEnabled(False)
            logger.debug("Edge Delays UI updates disabled.")

        try:
            for i, edge_key in enumerate(all_edges):
                row = (i % num_rows_data) + 1
                col_group_index = i // num_rows_data
                col_base = col_group_index * 4

                u, v = edge_key
                current_delays = self.current_edge_delays.get(
                    edge_key,
                    {"min": self.default_min, "max": self.default_max},
                )

                label = QLabel(f"({u},{v}):")
                grid_layout.addWidget(
                    label,
                    row,
                    col_base + 0,
                    Qt.AlignmentFlag.AlignRight
                    | Qt.AlignmentFlag.AlignVCenter,
                )

                min_spin = QSpinBox()
                min_spin.setRange(0, 99999)
                min_spin.setToolTip(f"Min delay for edge ({u},{v}).")
                min_spin.setValue(current_delays.get("min", self.default_min))
                grid_layout.addWidget(
                    min_spin, row, col_base + 1, Qt.AlignmentFlag.AlignCenter
                )
                self.edge_min_delay_inputs[edge_key] = min_spin

                max_spin = QSpinBox()
                max_spin.setRange(0, 99999)
                max_spin.setToolTip(f"Max delay for edge ({u},{v}).")
                max_spin.setValue(current_delays.get("max", self.default_max))
                grid_layout.addWidget(
                    max_spin, row, col_base + 2, Qt.AlignmentFlag.AlignCenter
                )
                self.edge_max_delay_inputs[edge_key] = max_spin

                max_content_row = max(max_content_row, row)

            if num_edges > 0 and num_rows_data > 0:
                separator_frame = QFrame()
                separator_frame.setFrameShape(QFrame.Shape.VLine)
                separator_frame.setFrameShadow(QFrame.Shadow.Sunken)
                grid_layout.addWidget(
                    separator_frame, 1, 3, max_content_row, 1
                )

            grid_layout.setColumnStretch(0, 0)
            grid_layout.setColumnStretch(1, 1)
            grid_layout.setColumnStretch(2, 1)
            grid_layout.setColumnStretch(3, 0)
            grid_layout.setColumnStretch(4, 0)
            grid_layout.setColumnStretch(5, 1)
            grid_layout.setColumnStretch(6, 1)

            grid_layout.setRowStretch(max_content_row + 1, 1)

        finally:
            if scroll_content_widget:
                scroll_content_widget.setUpdatesEnabled(True)
                logger.debug("Edge Delays UI updates re-enabled.")

        return edge_group

    def _setup_scheduled_requests_ui(self) -> QGroupBox:
        req_group = QGroupBox("Scheduled CS Request Times (comma-separated)")
        req_main_layout = QVBoxLayout(req_group)
        req_main_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area, grid_layout = self._create_scrollable_area(req_group)
        req_main_layout.addWidget(scroll_area)

        scroll_content_widget = scroll_area.widget()
        if scroll_content_widget:
            scroll_content_widget.setUpdatesEnabled(False)
            logger.debug("Scheduled Requests UI updates disabled.")

        try:
            for i in range(self.num_nodes):
                label = QLabel(f"Node {i}:")
                current_times = self.current_scheduled_requests.get(i, [])
                current_times_str = ",".join(
                    map(str, sorted(list(set(current_times))))
                )

                line_edit = QLineEdit(current_times_str)
                line_edit.setPlaceholderText("e.g., 10, 50, 100")
                line_edit.setToolTip(
                    f"Enter comma-separated times when Node {i} should automatically request the CS."
                )

                grid_layout.addWidget(label, i, 0, Qt.AlignmentFlag.AlignLeft)
                grid_layout.addWidget(line_edit, i, 1)
                self.scheduled_request_inputs[i] = line_edit

            grid_layout.setColumnStretch(1, 1)
            grid_layout.setRowStretch(grid_layout.rowCount(), 1)

        finally:
            if scroll_content_widget:
                scroll_content_widget.setUpdatesEnabled(True)
                logger.debug("Scheduled Requests UI updates re-enabled.")

        return req_group

    def _setup_buttons_ui(self) -> QHBoxLayout:
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 10, 0, 0)
        button_layout.setSpacing(10)

        load_button = QPushButton("Load JSON...")
        load_button.setToolTip("Load configuration from a JSON file.")
        load_button.clicked.connect(self._load_from_json)

        save_button = QPushButton("Save JSON...")
        save_button.setToolTip(
            "Save the current configuration to a JSON file."
        )
        save_button.clicked.connect(self._save_to_json)

        template_button = QPushButton("Template")
        template_button.setToolTip(
            "Show an example JSON configuration structure."
        )
        template_button.clicked.connect(self._show_json_template)

        randomize_button = QPushButton("Randomize")
        randomize_button.setObjectName("RandomizeButton")
        randomize_button.setToolTip(
            "Generate random valid configuration values for all fields."
        )
        randomize_button.clicked.connect(self._randomize_config)

        ok_button = QPushButton("OK")
        ok_button.setToolTip("Apply the configuration and close.")
        ok_button.setDefault(True)
        ok_button.clicked.connect(self._on_ok)

        cancel_button = QPushButton("Cancel")
        cancel_button.setToolTip("Discard changes and close.")
        cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(randomize_button)
        button_layout.addWidget(load_button)
        button_layout.addWidget(save_button)
        button_layout.addWidget(template_button)
        button_layout.addStretch(1)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        return button_layout

    def _on_ok(self) -> None:
        try:
            collected_cs_durations: CSDurations = {}
            for nid, spinbox in self.cs_duration_inputs.items():
                collected_cs_durations[nid] = spinbox.value()

            collected_edge_delays: EdgeDelays = {}
            for edge_key, min_spin in self.edge_min_delay_inputs.items():
                max_spin = self.edge_max_delay_inputs[edge_key]
                min_d = min_spin.value()
                max_d = max_spin.value()
                if min_d < 0 or max_d < 0:
                    raise ValueError(
                        f"Negative delay entered for Edge {edge_key}. Min={min_d}, Max={max_d}"
                    )
                if min_d > max_d:
                    raise ValueError(
                        f"Invalid Delay Range for Edge {edge_key}: "
                        f"Minimum ({min_d}) cannot be greater than Maximum ({max_d})."
                    )
                collected_edge_delays[edge_key] = {"min": min_d, "max": max_d}

            collected_scheduled_requests: ScheduledRequests = {}
            for node_id, line_edit in self.scheduled_request_inputs.items():
                times_str = line_edit.text().strip()
                time_list: List[int] = []
                if times_str:
                    parts = [
                        p.strip() for p in times_str.split(",") if p.strip()
                    ]
                    for part in parts:
                        try:
                            time_val = int(part)
                            if time_val < 0:
                                raise ValueError(
                                    "Request times cannot be negative."
                                )
                            time_list.append(time_val)
                        except ValueError:
                            raise ValueError(
                                f"Invalid non-negative integer found in request times "
                                f"'{part}' for Node {node_id}."
                            )
                collected_scheduled_requests[node_id] = sorted(
                    list(set(time_list))
                )

            self.result = {
                "cs_durations": collected_cs_durations,
                "edge_delays": collected_edge_delays,
                "scheduled_requests": collected_scheduled_requests,
            }
            logger.info("Advanced configuration validated successfully.")
            self.accept()

        except ValueError as e:
            logger.warning(f"Advanced configuration validation failed: {e}")
            QMessageBox.warning(self, "Validation Error", str(e))
            self.result = None

    def _randomize_config(self) -> None:
        logger.info("Randomizing advanced configuration...")
        self.setUpdatesEnabled(False)
        logger.debug("Dialog UI updates disabled for randomization.")
        try:
            if self.cs_duration_inputs:
                logger.debug("- Randomizing CS Durations...")
                for node_id, spinbox in self.cs_duration_inputs.items():
                    rand_cs = random.randint(*RANDOM_CS_RANGE)
                    spinbox.setValue(rand_cs)
            else:
                logger.warning(
                    "CS Duration inputs not found for randomization."
                )

            if self.edge_min_delay_inputs and self.edge_max_delay_inputs:
                logger.debug("- Randomizing Edge Delays...")
                for edge_key, min_spin in self.edge_min_delay_inputs.items():
                    if edge_key not in self.edge_max_delay_inputs:
                        logger.warning(
                            f"Max delay spinbox missing for edge {edge_key}. Skipping."
                        )
                        continue
                    max_spin = self.edge_max_delay_inputs[edge_key]

                    rand_min = random.randint(*RANDOM_MIN_DELAY_RANGE)

                    offset_min = RANDOM_MAX_DELAY_OFFSET[0]
                    offset_max = RANDOM_MAX_DELAY_OFFSET[1]
                    rand_max = rand_min + random.randint(
                        offset_min, offset_max
                    )
                    rand_max = max(rand_min, rand_max)

                    min_spin.setValue(rand_min)
                    max_spin.setValue(rand_max)
            else:
                logger.warning(
                    "Edge delay inputs not found for randomization."
                )

            if self.scheduled_request_inputs:
                logger.debug("- Randomizing Scheduled Requests...")
                for (
                    node_id,
                    line_edit,
                ) in self.scheduled_request_inputs.items():
                    num_requests = random.randint(0, RANDOM_REQ_MAX_COUNT)

                    if num_requests == 0:
                        times_list = []
                    else:
                        possible_times = range(RANDOM_REQ_MAX_TIME + 1)
                        actual_num_requests = min(
                            num_requests, len(possible_times)
                        )
                        if actual_num_requests > 0:
                            times_list = sorted(
                                random.sample(
                                    possible_times, k=actual_num_requests
                                )
                            )
                        else:
                            times_list = []

                    line_edit.setText(",".join(map(str, times_list)))
            else:
                logger.warning(
                    "Scheduled request inputs not found for randomization."
                )

            logger.info(
                "Randomization complete. Review values and click OK to apply."
            )
            QMessageBox.information(
                self,
                "Randomization Complete",
                "Random values generated.\nPlease review the updated configuration.",
            )

        except Exception as e:
            logger.error(f"Error during randomization: {e}", exc_info=True)
            QMessageBox.warning(
                self,
                "Randomization Error",
                f"An error occurred during randomization:\n{e}",
            )
        finally:
            self.setUpdatesEnabled(True)
            logger.debug("Dialog UI updates re-enabled after randomization.")

    def _load_from_json(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Advanced Configuration from JSON",
            "",
            "JSON files (*.json);;All files (*.*)",
        )
        if not filepath:
            logger.debug("JSON load cancelled by user.")
            return

        logger.info(f"Attempting to load configuration from: {filepath}")
        self.setUpdatesEnabled(False)
        logger.debug("Dialog UI updates disabled for loading.")
        load_successful = False
        try:
            with open(filepath, "r") as f:
                loaded_data = json.load(f)

            required_keys = [
                "cs_durations",
                "edge_delays",
                "scheduled_requests",
            ]
            if not all(key in loaded_data for key in required_keys):
                missing = [
                    key for key in required_keys if key not in loaded_data
                ]
                raise ValueError(
                    f"JSON file is missing required top-level key(s): {', '.join(missing)}"
                )

            loaded_cs = loaded_data.get("cs_durations", {})
            if not isinstance(loaded_cs, dict):
                raise ValueError(
                    "'cs_durations' must be a dictionary (object) in JSON."
                )
            for node_id_str, duration in loaded_cs.items():
                try:
                    node_id = int(node_id_str)
                    if node_id in self.cs_duration_inputs:
                        if not isinstance(duration, int) or duration < 0:
                            raise ValueError(
                                f"must be a non-negative integer, got '{duration}'"
                            )
                        self.cs_duration_inputs[node_id].setValue(duration)
                    else:
                        logger.warning(
                            f"Node ID {node_id} from JSON 'cs_durations' not found in current UI ({self.num_nodes} nodes). Skipping."
                        )
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Error processing 'cs_durations' for key '{node_id_str}': {e}"
                    ) from e

            loaded_edges = loaded_data.get("edge_delays", {})
            if not isinstance(loaded_edges, dict):
                raise ValueError(
                    "'edge_delays' must be a dictionary (object) in JSON."
                )
            for edge_str, delays in loaded_edges.items():
                try:
                    edge_key = _str_to_edge_key(edge_str)
                    if edge_key in self.edge_min_delay_inputs:
                        if (
                            not isinstance(delays, dict)
                            or "min" not in delays
                            or "max" not in delays
                        ):
                            raise ValueError(
                                "entry must be an object with 'min' and 'max' keys."
                            )
                        min_d = int(delays["min"])
                        max_d = int(delays["max"])
                        if not (0 <= min_d <= max_d):
                            raise ValueError(
                                f"invalid range min={min_d}, max={max_d} (require 0 <= min <= max)."
                            )
                        self.edge_min_delay_inputs[edge_key].setValue(min_d)
                        self.edge_max_delay_inputs[edge_key].setValue(max_d)
                    else:
                        logger.warning(
                            f"Edge key {edge_key} (from JSON '{edge_str}') not found in current UI. Skipping."
                        )
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Error processing 'edge_delays' for key '{edge_str}': {e}"
                    ) from e

            loaded_reqs = loaded_data.get("scheduled_requests", {})
            if not isinstance(loaded_reqs, dict):
                raise ValueError(
                    "'scheduled_requests' must be a dictionary (object) in JSON."
                )
            for node_id_str, times_list in loaded_reqs.items():
                try:
                    node_id = int(node_id_str)
                    if node_id in self.scheduled_request_inputs:
                        if not isinstance(times_list, list):
                            raise ValueError(
                                "must be a list (array) of times."
                            )
                        valid_times: List[int] = []
                        for t in times_list:
                            if not isinstance(t, int) or t < 0:
                                raise ValueError(
                                    f"contains invalid time '{t}' (must be non-negative integer)."
                                )
                            valid_times.append(t)
                        times_str = ",".join(
                            map(str, sorted(list(set(valid_times))))
                        )
                        self.scheduled_request_inputs[node_id].setText(
                            times_str
                        )
                    else:
                        logger.warning(
                            f"Node ID {node_id} from JSON 'scheduled_requests' not found in current UI. Skipping."
                        )
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Error processing 'scheduled_requests' for key '{node_id_str}': {e}"
                    ) from e

            load_successful = True

        except FileNotFoundError:
            logger.error(f"File not found error during JSON load: {filepath}")
            QMessageBox.critical(
                self, "Load Error", f"File not found:\n{filepath}"
            )
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {filepath}: {e}")
            QMessageBox.critical(
                self, "Load Error", f"Invalid JSON file:\n{e}"
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Data validation error during JSON load: {e}")
            QMessageBox.critical(
                self,
                "Load Error",
                f"Error in JSON data structure or values:\n{e}",
            )
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred loading JSON from {filepath}"
            )
            QMessageBox.critical(
                self, "Load Error", f"An unexpected error occurred:\n{e}"
            )
        finally:
            self.setUpdatesEnabled(True)
            logger.debug("Dialog UI updates re-enabled after load attempt.")

        if load_successful:
            QMessageBox.information(
                self,
                "Load Successful",
                "Configuration loaded successfully from JSON.",
            )
            logger.info(
                f"Successfully loaded and applied configuration from {filepath}"
            )

    def _save_to_json(self) -> None:
        try:
            collected_cs_durations_str_keys: Dict[str, int] = {}
            for nid, spinbox in self.cs_duration_inputs.items():
                collected_cs_durations_str_keys[str(nid)] = spinbox.value()

            collected_edge_delays_str_keys: Dict[str, Dict[str, int]] = {}
            for edge_key, min_spin in self.edge_min_delay_inputs.items():
                max_spin = self.edge_max_delay_inputs[edge_key]
                min_d = min_spin.value()
                max_d = max_spin.value()
                if min_d > max_d:
                    raise ValueError(
                        f"Cannot save: Min delay > Max delay for Edge {edge_key} ({min_d} > {max_d})"
                    )
                edge_str = _edge_key_to_str(edge_key)
                collected_edge_delays_str_keys[edge_str] = {
                    "min": min_d,
                    "max": max_d,
                }

            collected_scheduled_requests_str_keys: Dict[str, List[int]] = {}
            for node_id, line_edit in self.scheduled_request_inputs.items():
                times_str = line_edit.text().strip()
                time_list: List[int] = []
                if times_str:
                    parts = [
                        p.strip() for p in times_str.split(",") if p.strip()
                    ]
                    for part in parts:
                        try:
                            time_val = int(part)
                            if time_val < 0:
                                raise ValueError(
                                    "Request times cannot be negative."
                                )
                            time_list.append(time_val)
                        except ValueError:
                            raise ValueError(
                                f"Invalid request time '{part}' for Node {node_id}."
                            )
                collected_scheduled_requests_str_keys[str(node_id)] = sorted(
                    list(set(time_list))
                )

            json_data = {
                "cs_durations": collected_cs_durations_str_keys,
                "edge_delays": collected_edge_delays_str_keys,
                "scheduled_requests": collected_scheduled_requests_str_keys,
                "metadata": {
                    "num_nodes": self.num_nodes,
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "description": "Ricart-Agrawala Simulation Advanced Configuration",
                },
            }

            default_filename = (
                f"ricart_agrawala_config_nodes_{self.num_nodes}.json"
            )
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "Save Advanced Configuration to JSON",
                default_filename,
                "JSON files (*.json);;All files (*.*)",
            )
            if not filepath:
                logger.debug("JSON save cancelled by user.")
                return

            if filepath and not filepath.lower().endswith(".json"):
                filepath += ".json"

            logger.info(f"Attempting to save configuration to: {filepath}")
            with open(filepath, "w") as f:
                json.dump(json_data, f, indent=4)

            QMessageBox.information(
                self,
                "Save Successful",
                f"Configuration saved successfully to:\n{filepath}",
            )
            logger.info(f"Successfully saved configuration to {filepath}")

        except ValueError as e:
            logger.warning(f"Validation error during save attempt: {e}")
            QMessageBox.warning(
                self,
                "Validation Error",
                f"Cannot save due to invalid input:\n{e}",
            )
        except IOError as e:
            logger.error(
                f"IOError during JSON save to {filepath}: {e}", exc_info=True
            )
            QMessageBox.critical(
                self, "Save Error", f"Could not write to file:\n{e}"
            )
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred saving JSON to {filepath}"
            )
            QMessageBox.critical(
                self,
                "Save Error",
                f"An unexpected error occurred during saving:\n{e}",
            )

    def _show_json_template(self) -> None:
        """Displays a modal dialog showing a complete JSON configuration template."""

        template_cs = {str(i): self.default_cs for i in range(self.num_nodes)}

        template_edges = {}
        for u in range(self.num_nodes):
            for v in range(u + 1, self.num_nodes):
                edge_key_str = _edge_key_to_str((u, v))
                template_edges[edge_key_str] = {
                    "min": self.default_min,
                    "max": self.default_max,
                }

        template_reqs: Dict[str, List[int]] = {
            str(i): [] for i in range(self.num_nodes)
        }

        template_data = {
            "cs_durations": template_cs,
            "edge_delays": template_edges,
            "scheduled_requests": template_reqs,
            "metadata": {
                "num_nodes": self.num_nodes,
                "saved_at": "YYYY-MM-DD HH:MM:SS ZZZ",
                "description": "Complete Configuration Template",
            },
        }
        notes = (
            "# Note: 'edge_delays' keys are strings \"u,v\" with u < v.\n"
            "# 'scheduled_requests' values are lists of non-negative integers (empty list means none).\n"
            "# metadata key does not have to contain any fields, or exist.\n\n"
        )

        try:
            json_string = notes + json.dumps(template_data, indent=4)

            template_dialog = QDialog(self)
            template_dialog.setWindowTitle("JSON Configuration Template")
            template_dialog.setModal(True)
            template_dialog.resize(550, 550)

            layout = QVBoxLayout(template_dialog)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(10)

            text_area = QTextEdit()
            text_area.setPlainText(json_string)
            text_area.setReadOnly(True)
            text_area.setFont(QFont("Consolas, Courier New", 9))

            try:

                class JsonHighlighter(QSyntaxHighlighter):
                    def __init__(self, parent=None):
                        super().__init__(parent)
                        self.highlightingRules: List[
                            Tuple[QRegularExpression, QTextCharFormat]
                        ] = []

                        keywordFormat = QTextCharFormat()
                        keywordFormat.setForeground(QColor(128, 0, 128))
                        keywordFormat.setFontWeight(QFont.Weight.Bold)
                        keywords = ["true", "false", "null"]
                        for word in keywords:
                            pattern = QRegularExpression(f"\\b{word}\\b")
                            self.highlightingRules.append(
                                (pattern, keywordFormat)
                            )

                        stringFormat = QTextCharFormat()
                        stringFormat.setForeground(QColor(0, 128, 0))
                        pattern = QRegularExpression('"([^"\\\\]|\\\\.)*"')
                        self.highlightingRules.append((pattern, stringFormat))

                        numberFormat = QTextCharFormat()
                        numberFormat.setForeground(QColor(0, 0, 255))
                        pattern = QRegularExpression(
                            r"\b-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\b"
                        )
                        self.highlightingRules.append((pattern, numberFormat))

                        keyFormat = QTextCharFormat()
                        keyFormat.setForeground(QColor(218, 112, 214))
                        pattern = QRegularExpression(
                            '"([^"\\\\]|\\\\.)*"(?=\\s*:)'
                        )
                        self.highlightingRules.append((pattern, keyFormat))

                        commentFormat = QTextCharFormat()
                        commentFormat.setForeground(QColor(128, 128, 128))
                        commentFormat.setFontItalic(True)
                        pattern = QRegularExpression("#[^\n]*")
                        self.highlightingRules.append((pattern, commentFormat))

                    def highlightBlock(self, text: str):
                        for pattern, format in self.highlightingRules:
                            iterator: QRegularExpressionMatchIterator = (
                                pattern.globalMatch(text)
                            )
                            while iterator.hasNext():
                                match: QRegularExpressionMatch = (
                                    iterator.next()
                                )
                                start_index = match.capturedStart()
                                length = match.capturedLength()
                                self.setFormat(start_index, length, format)

                self.highlighter = JsonHighlighter(text_area.document())
                logger.debug(
                    "Applied JSON syntax highlighter using QRegularExpression."
                )

            except Exception as highlight_err:
                logger.warning(
                    f"Could not apply JSON syntax highlighting: {highlight_err}",
                    exc_info=True,
                )
                pass

            close_button = QPushButton("Close")
            close_button.setDefault(True)
            close_button.clicked.connect(template_dialog.accept)

            layout.addWidget(text_area)
            button_box = QHBoxLayout()
            button_box.addStretch(1)
            button_box.addWidget(close_button)
            layout.addLayout(button_box)

            template_dialog.setLayout(layout)
            template_dialog.exec()

        except Exception as e:
            logger.exception(
                "Error occurred while trying to show JSON template dialog"
            )
            QMessageBox.critical(
                self, "Template Error", f"Could not display template:\n{e}"
            )


class Node:
    def __init__(
        self,
        node_id: NodeId,
        num_nodes: int,
        cs_duration: Union[int, float, str],
    ):
        self.id: NodeId = node_id
        self.num_nodes: int = num_nodes
        self.cs_duration: int = DEFAULT_CS_DURATION

        try:
            duration_val = int(cs_duration)
            if duration_val < 0:
                logger.warning(
                    f"Node {node_id} initialized with negative CS duration ({cs_duration}). "
                    f"Using 0 instead."
                )
                self.cs_duration = 0
            else:
                self.cs_duration = duration_val
        except (ValueError, TypeError):
            logger.error(
                f"Invalid CS duration type/value '{cs_duration}' for Node {node_id}. "
                f"Using default: {DEFAULT_CS_DURATION}."
            )
            self.cs_duration = DEFAULT_CS_DURATION

        self.state: NodeState = NodeState.IDLE
        self.clock: ClockValue = 0

        self.request_ts: Timestamp = (-1, -1)

        self.outstanding_replies: Set[NodeId] = set()

        self.deferred_queue: List[NodeId] = []

        logger.debug(
            f"Node {self.id} initialized: CS Duration={self.cs_duration}"
        )

    def update_clock(
        self, received_ts: Optional[ClockValue] = None
    ) -> ClockValue:
        if received_ts is not None:
            self.clock = max(self.clock, received_ts) + 1
        else:
            self.clock += 1
        return self.clock

    def __repr__(self) -> str:
        state_char = str(self.state)
        req_ts_str = (
            f"({self.request_ts[0]},{self.request_ts[1]})"
            if self.request_ts != (-1, -1)
            else "(-,-)"
        )
        return (
            f"N(id:{self.id}, S:{state_char}, C:{self.clock}, Req:{req_ts_str}, "
            f"Out:{len(self.outstanding_replies)}, Def:{len(self.deferred_queue)})"
        )


class Simulation:
    def __init__(
        self,
        num_nodes: int,
        node_cs_durations: CSDurations,
        edge_delays: EdgeDelays,
        scheduled_requests: ScheduledRequests,
        vis_callback: Optional[VisCallback] = None,
    ):
        logger.info(f"Initializing Simulation with {num_nodes} nodes.")
        if num_nodes <= 0:
            raise ValueError("Number of nodes must be positive.")

        self.num_nodes: int = num_nodes
        self._node_cs_durations: CSDurations = copy.deepcopy(node_cs_durations)
        self._edge_delays: EdgeDelays = copy.deepcopy(edge_delays)
        self._scheduled_requests: ScheduledRequests = copy.deepcopy(
            scheduled_requests
        )
        self.vis_callback: Optional[VisCallback] = vis_callback

        self.nodes: NodeDict = {}
        for i in range(num_nodes):
            duration = self._node_cs_durations.get(i, DEFAULT_CS_DURATION)
            self.nodes[i] = Node(i, num_nodes, duration)

        self.graph: Optional[nx.Graph] = self._create_graph()
        if self.graph is None:
            logger.error(
                "Graph creation failed. Simulation may not function correctly."
            )

        self.event_queue: List[Event] = []
        heapq.heapify(self.event_queue)

        self.current_time: float = 0.0
        self.event_counter: int = 0
        self.history_data: List[HistoryEntry] = []

        self.node_positions: Optional[NodePositions] = None
        self._generate_layout()

        logger.debug("Initializing scheduled requests...")
        self._initialize_scheduled_requests()

        self._log_state(
            EventType.INIT,
            f"Simulation initialized with {num_nodes} nodes.",
            list(range(num_nodes)),
        )
        logger.info("Simulation Initialization Complete.")
        self._update_visualization()

    def _generate_layout(self) -> None:
        self.node_positions = None
        if self.graph and self.graph.number_of_nodes() > 0:
            try:
                logger.debug(
                    "Generating graph layout using nx.spring_layout..."
                )
                positions_nx = nx.spring_layout(
                    self.graph, seed=GRAPH_LAYOUT_SEED
                )

                self.node_positions = {
                    nid: (float(pos[0]), float(pos[1]))
                    for nid, pos in positions_nx.items()
                    if isinstance(pos, (np.ndarray, list, tuple))
                    and len(pos) == 2
                }
                logger.debug(
                    f"Generated and converted {len(self.node_positions)} node positions."
                )
                if len(self.node_positions) != self.graph.number_of_nodes():
                    logger.warning(
                        "Mismatch between node count and generated positions."
                    )

            except Exception as e:
                logger.error(
                    f"Failed to generate node positions using NetworkX: {e}",
                    exc_info=True,
                )
                self.node_positions = {}
        else:
            logger.warning(
                "Graph is empty or None, cannot generate node positions."
            )
            self.node_positions = {}

    def _get_current_node_snapshots(self) -> Dict[NodeId, Dict[str, Any]]:
        snapshots: Dict[NodeId, Dict[str, Any]] = {}
        for node_id, node in self.nodes.items():
            snapshots[node_id] = {
                "state": node.state,
                "clock": node.clock,
                "request_ts": node.request_ts,
            }
        return snapshots

    def _log_state(
        self,
        event_type: EventType,
        details: str,
        involved_node_ids: List[NodeId],
    ) -> None:
        current_snapshots = self._get_current_node_snapshots()
        log_entry: HistoryEntry = {
            "time": int(round(self.current_time)),
            "type": event_type,
            "details": details,
            "involved": sorted(list(set(involved_node_ids))),
            "node_snapshots": current_snapshots,
        }
        self.history_data.append(log_entry)

    def _initialize_scheduled_requests(self) -> None:
        logger.debug(
            f"Processing scheduled requests config: {self._scheduled_requests}"
        )
        for node_id_str, times in self._scheduled_requests.items():
            try:
                node_id = int(node_id_str)
                if 0 <= node_id < self.num_nodes:
                    if isinstance(times, (list, tuple)):
                        for time_val in times:
                            if isinstance(time_val, int) and time_val >= 0:
                                logger.debug(
                                    f"Scheduling N{node_id} request @ T={time_val}"
                                )
                                self._schedule_event(
                                    delay_or_time=time_val,
                                    event_type=EventType.SCHEDULED_REQUEST,
                                    data=node_id,
                                    is_absolute_time=True,
                                )
                            else:
                                logger.warning(
                                    f"Invalid time value '{time_val}' in scheduled requests for Node {node_id}. Skipping."
                                )
                    elif times:
                        logger.warning(
                            f"Invalid format for scheduled times for Node {node_id}. Expected list/tuple, got {type(times)}. Skipping."
                        )
                else:
                    logger.warning(
                        f"Invalid node_id '{node_id}' found in scheduled requests config. Skipping."
                    )
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Error processing scheduled request key '{node_id_str}': {e}. Skipping."
                )

    def _create_graph(self) -> Optional[nx.Graph]:
        try:
            if self.num_nodes <= 0:
                logger.error("Cannot create graph with 0 or negative nodes.")
                return None

            G = nx.complete_graph(self.num_nodes)
            logger.debug(
                f"Created initial complete graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
            )

            default_delay_dict = {
                "min": DEFAULT_MIN_DELAY,
                "max": DEFAULT_MAX_DELAY,
            }
            for u, v in G.edges():
                edge_key: EdgeKey = tuple(sorted((u, v)))
                delays = self._edge_delays.get(edge_key, default_delay_dict)

                try:
                    min_delay = int(delays.get("min", DEFAULT_MIN_DELAY))
                    max_delay = int(delays.get("max", DEFAULT_MAX_DELAY))

                    if min_delay < 0:
                        min_delay = 0
                    if max_delay < min_delay:
                        max_delay = min_delay

                    G.edges[u, v]["min_delay"] = min_delay
                    G.edges[u, v]["max_delay"] = max_delay
                    G.edges[u, v]["in_transit"] = []

                except (ValueError, TypeError) as e:
                    logger.error(
                        f"Invalid delay configuration for edge {edge_key}: {delays}. Error: {e}. Using defaults.",
                        exc_info=True,
                    )
                    G.edges[u, v]["min_delay"] = DEFAULT_MIN_DELAY
                    G.edges[u, v]["max_delay"] = DEFAULT_MAX_DELAY
                    G.edges[u, v]["in_transit"] = []

            logger.info(
                "Assigned delay attributes and initialized 'in_transit' lists for all graph edges."
            )
            return G

        except Exception as e:
            logger.error(
                f"Unexpected error creating simulation graph: {e}",
                exc_info=True,
            )
            return None

    def _get_delay(self, u: NodeId, v: NodeId) -> int:
        if u == v:
            return 0
        if not self.graph:
            logger.error(
                f"Graph not initialized. Cannot get delay for ({u},{v}). Returning default 1."
            )
            return 1

        try:
            edge_data = self.graph.edges[u, v]
            min_d = edge_data.get("min_delay", DEFAULT_MIN_DELAY)
            max_d = edge_data.get("max_delay", DEFAULT_MAX_DELAY)

            min_d = int(min_d)
            max_d = int(max_d)

            actual_min = max(0, min(min_d, max_d))
            actual_max = max(actual_min, max_d)

            if actual_min == actual_max:
                return actual_min
            else:
                return random.randint(actual_min, actual_max)

        except KeyError:
            logger.warning(
                f"Edge ({u}, {v}) not found in graph for delay lookup. Using default delay 1."
            )
            return 1
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Invalid delay attributes ({min_d}, {max_d}) for edge ({u},{v}). Using min value. Error: {e}"
            )
            try:
                return max(0, min(int(min_d), int(max_d)))
            except Exception:
                return 1

    def _schedule_event(
        self,
        delay_or_time: Union[int, float],
        event_type: EventType,
        data: Any,
        is_absolute_time: bool = False,
    ) -> None:
        current_int_time = int(round(self.current_time))

        if is_absolute_time:
            event_time = int(round(delay_or_time))
        else:
            event_time = int(round(self.current_time + delay_or_time))

        if event_time < current_int_time:
            logger.warning(
                f"Attempted to schedule {event_type} in the past (T={event_time} < Current T={current_int_time}). "
                f"Scheduling for T={current_int_time} instead. Data: {data}"
            )
            event_time = current_int_time

        self.event_counter += 1
        event_tuple: Event = (event_time, self.event_counter, event_type, data)
        heapq.heappush(self.event_queue, event_tuple)

    def _send_message(
        self,
        sender_id: NodeId,
        receiver_id: NodeId,
        msg_type: MessageType,
        msg_ts: ClockValue,
        request_ts: Optional[Timestamp] = None,
    ) -> None:
        if sender_id == receiver_id:
            logger.warning(
                f"Attempt to send message from N{sender_id} to itself. Skipping."
            )
            return
        if not self.graph:
            logger.error(
                f"Cannot send message from N{sender_id} to N{receiver_id}: Graph not initialized."
            )
            return

        delay = self._get_delay(sender_id, receiver_id)
        send_time = self.current_time
        arrival_time = int(round(send_time + delay))

        message_payload = (msg_type, sender_id, msg_ts, request_ts)
        original_edge_tuple = (sender_id, receiver_id)
        event_data = (receiver_id, message_payload, original_edge_tuple)

        log_detail = f"SEND N{sender_id}->N{receiver_id} ({msg_type.name}) MsgTS={msg_ts}"
        if request_ts:
            log_detail += f" ReqTS={request_ts}"
        log_detail += f". Delay:{delay}, ArrivalT:{arrival_time}"
        logger.debug(f"T={int(round(send_time))}: {log_detail}")

        transit_entry = (
            msg_type,
            sender_id,
            receiver_id,
            send_time,
            arrival_time,
        )
        try:
            edge_key = tuple(sorted((sender_id, receiver_id)))
            if edge_key in self.graph.edges:
                transit_list = self.graph.edges[edge_key].get("in_transit")
                if isinstance(transit_list, list):
                    transit_list.append(transit_entry)
                else:
                    logger.error(
                        f"'in_transit' attribute for edge {edge_key} is not a list ({type(transit_list)}). Cannot add message."
                    )
            else:
                logger.error(
                    f"Edge {edge_key} not found when trying to set 'in_transit' for message N{sender_id}->N{receiver_id}."
                )
        except Exception as e:
            logger.error(
                f"Error updating 'in_transit' list for edge between {sender_id} and {receiver_id}: {e}",
                exc_info=True,
            )

        self._schedule_event(delay, EventType.MESSAGE_ARRIVAL, event_data)

    def want_cs(self, node_id: NodeId) -> bool:
        node = self.nodes.get(node_id)
        if not node:
            logger.error(
                f"Cannot initiate CS request: Node {node_id} not found."
            )
            return False

        current_int_time = int(round(self.current_time))
        if node.state != NodeState.IDLE:
            logger.warning(
                f"T={current_int_time}: N{node_id} cannot request CS (State: {node.state.name}). Request ignored."
            )
            return False

        logger.info(f"T={current_int_time}: N{node_id} wants CS.")
        node.state = NodeState.WANTED
        node.clock = node.update_clock()
        node.request_ts = (node.clock, node.id)

        node.outstanding_replies = set(range(self.num_nodes)) - {node_id}
        node.deferred_queue = []

        logger.debug(
            f"  -> N{node_id} state update: State={node.state.name}, Clock={node.clock}, "
            f"ReqTS={node.request_ts}, OutstandingReplies={node.outstanding_replies}"
        )

        if not node.outstanding_replies:
            logger.debug(
                f"  -> N{node_id} requires no replies. Entering CS immediately."
            )
            self._schedule_event(
                delay_or_time=self.current_time,
                event_type=EventType.CS_ENTER,
                data=node_id,
                is_absolute_time=True,
            )
        else:
            num_requests = len(node.outstanding_replies)
            logger.debug(
                f"  -> N{node_id} broadcasting {num_requests} REQUEST(S)."
            )
            send_clock = node.clock
            for other_id in range(self.num_nodes):
                if other_id != node_id:
                    self._send_message(
                        sender_id=node_id,
                        receiver_id=other_id,
                        msg_type=MessageType.REQUEST,
                        msg_ts=send_clock,
                        request_ts=node.request_ts,
                    )

        return True

    def _handle_message_arrival(
        self, event_data: Tuple[NodeId, Tuple, Tuple]
    ) -> List[NodeId]:
        try:
            receiver_id, message_payload, original_edge = event_data
            msg_type, sender_id, msg_ts, request_ts = message_payload
            arrival_time = self.current_time
            arrival_int_time = int(round(arrival_time))

            receiver_node = self.nodes.get(receiver_id)
            if not receiver_node:
                logger.error(
                    f"Receiver node {receiver_id} not found for message arrival. Msg: {message_payload}"
                )
                return []

            involved_nodes = [receiver_id, sender_id]

            if self.graph:
                edge_key = tuple(sorted(original_edge))
                try:
                    transit_list = self.graph.edges[edge_key].get("in_transit")
                    if isinstance(transit_list, list):
                        found_idx = -1
                        for idx, item in enumerate(transit_list):
                            if (
                                len(item) == 5
                                and item[0] == msg_type
                                and item[1] == sender_id
                                and item[2] == receiver_id
                                and item[4] == arrival_int_time
                            ):
                                found_idx = idx
                                break
                        if found_idx != -1:
                            transit_list.pop(found_idx)
                        else:
                            logger.warning(
                                f"Arrived message ({msg_type.name} N{sender_id}->N{receiver_id} @T={arrival_int_time}) not found in transit list for edge {edge_key}. List: {transit_list}"
                            )
                except KeyError:
                    logger.error(
                        f"Edge {edge_key} not found when trying to access 'in_transit' for arrival."
                    )
                except Exception as e:
                    logger.error(
                        f"Error clearing transit list for edge {edge_key}: {e}",
                        exc_info=True,
                    )

            log_detail = f"RECV N{receiver_id}<-N{sender_id} ({msg_type.name}) MsgTS={msg_ts}"
            if request_ts:
                log_detail += f" ReqTS={request_ts}"
            logger.debug(f"T={arrival_int_time}: {log_detail}")

            old_clock = receiver_node.clock
            receiver_node.clock = receiver_node.update_clock(
                received_ts=msg_ts
            )
            if receiver_node.clock != old_clock:
                logger.debug(
                    f"  -> N{receiver_id} Clock updated: {old_clock} -> {receiver_node.clock}"
                )

            if msg_type == MessageType.REQUEST:
                self._handle_request(receiver_node, sender_id, request_ts)
            elif msg_type == MessageType.REPLY:
                self._handle_reply(receiver_node, sender_id)

            details = f"{msg_type.name} N{sender_id}->N{receiver_id} processed by N{receiver_id}."
            self._log_state(EventType.MESSAGE_ARRIVAL, details, involved_nodes)

            return involved_nodes

        except Exception as e:
            logger.exception(f"Error handling message arrival: {event_data}")
            self._log_state(
                EventType.ERROR,
                f"Error handling message arrival: {e}",
                involved_nodes,
            )
            return involved_nodes

    def _handle_scheduled_request(self, node_id: NodeId) -> List[NodeId]:
        current_int_time = int(round(self.current_time))
        logger.info(
            f"T={current_int_time}: Handling scheduled request trigger for N{node_id}."
        )

        success = self.want_cs(node_id)

        details = f"Node {node_id} scheduled request triggered."
        if not success:
            details += " (Request ignored - Node not IDLE)"
            logger.warning(
                f"  -> Scheduled request for N{node_id} failed as node was not IDLE."
            )

        self._log_state(EventType.SCHEDULED_REQUEST, details, [node_id])
        return [node_id]

    def _handle_request(
        self,
        receiver_node: Node,
        sender_id: NodeId,
        sender_request_ts: Timestamp,
    ) -> None:
        should_reply_immediately = False
        receiver_id = receiver_node.id
        my_state = receiver_node.state
        my_req_ts = receiver_node.request_ts

        logger.debug(
            f"  N{receiver_id} processing REQUEST from N{sender_id}. MyState: {my_state.name}, "
            f"MyReqTS: {my_req_ts}, SenderReqTS: {sender_request_ts}"
        )

        if my_state == NodeState.HELD:
            logger.debug(
                f"    -> N{receiver_id} is HELD. Deferring N{sender_id}."
            )
            should_reply_immediately = False
        elif my_state == NodeState.WANTED:
            if sender_request_ts < my_req_ts:
                logger.debug(
                    f"    -> N{receiver_id} is WANTED, N{sender_id}'s request {sender_request_ts} "
                    f"has priority over mine {my_req_ts}. Replying immediately."
                )
                should_reply_immediately = True
            else:
                logger.debug(
                    f"    -> N{receiver_id} is WANTED, my request {my_req_ts} has priority "
                    f"over/equal to N{sender_id}'s {sender_request_ts}. Deferring N{sender_id}."
                )
                should_reply_immediately = False
        else:
            logger.debug(
                f"    -> N{receiver_id} is IDLE. Replying immediately to N{sender_id}."
            )
            should_reply_immediately = True

        if should_reply_immediately:
            reply_ts = receiver_node.update_clock()
            logger.debug(
                f"       -> Sending REPLY to N{sender_id} (Clock now {reply_ts})."
            )
            self._send_message(
                sender_id=receiver_id,
                receiver_id=sender_id,
                msg_type=MessageType.REPLY,
                msg_ts=reply_ts,
                request_ts=None,
            )
        else:
            if sender_id not in receiver_node.deferred_queue:
                receiver_node.deferred_queue.append(sender_id)
                logger.debug(
                    f"       -> Added N{sender_id} to deferred queue: {receiver_node.deferred_queue}"
                )
            else:
                logger.warning(
                    f"       -> N{receiver_id} received REQUEST from N{sender_id}, but it was already in the deferred queue: {receiver_node.deferred_queue}. Ignored duplicate deferral."
                )

    def _handle_reply(self, receiver_node: Node, sender_id: NodeId) -> None:
        receiver_id = receiver_node.id
        logger.debug(
            f"  N{receiver_id} processing REPLY from N{sender_id}. MyState: {receiver_node.state.name}"
        )

        if receiver_node.state == NodeState.WANTED:
            if sender_id in receiver_node.outstanding_replies:
                receiver_node.outstanding_replies.remove(sender_id)
                remaining_count = len(receiver_node.outstanding_replies)
                logger.debug(
                    f"    -> N{receiver_id} received needed REPLY from N{sender_id}. Outstanding remaining: {remaining_count}"
                )

                if not receiver_node.outstanding_replies:
                    logger.debug(
                        f"    -> N{receiver_id} received all replies. Scheduling CS entry."
                    )
                    self._schedule_event(
                        delay_or_time=self.current_time,
                        event_type=EventType.CS_ENTER,
                        data=receiver_id,
                        is_absolute_time=True,
                    )
            else:
                logger.warning(
                    f"    -> N{receiver_id} received unexpected/duplicate REPLY from N{sender_id}. "
                    f"Outstanding set: {sorted(list(receiver_node.outstanding_replies))}. Ignored."
                )
        else:
            logger.warning(
                f"    -> N{receiver_id} received REPLY from N{sender_id} but is not in WANTED state "
                f"(State: {receiver_node.state.name}). Ignored."
            )

    def _enter_cs(self, node_id: NodeId) -> None:
        node = self.nodes.get(node_id)
        current_int_time = int(round(self.current_time))

        if not node:
            logger.error(f"Node {node_id} not found when trying to enter CS.")
            self._log_state(
                EventType.ERROR,
                f"Node {node_id} not found for CS_ENTER event.",
                [node_id],
            )
            return

        if node.state != NodeState.WANTED:
            logger.warning(
                f"T={current_int_time}: N{node_id} entering CS from unexpected state {node.state.name}!"
            )
            node.outstanding_replies.clear()

        node.state = NodeState.HELD
        duration = node.cs_duration
        logger.info(
            f"T={current_int_time}: +++ N{node_id} ENTER CS (Duration: {duration}) +++"
        )
        logger.debug(f"  -> N{node_id} state update: State={node.state.name}")

        details = f"Node {node_id} entered CS. Duration={duration}"
        self._log_state(EventType.CS_ENTER, details, [node_id])

        self._schedule_event(
            delay_or_time=duration, event_type=EventType.CS_EXIT, data=node_id
        )

    def _handle_cs_exit(self, node_id: NodeId) -> List[NodeId]:
        node = self.nodes.get(node_id)
        current_int_time = int(round(self.current_time))
        involved_nodes = [node_id]

        if not node:
            logger.error(f"Node {node_id} not found when trying to exit CS.")
            self._log_state(
                EventType.ERROR,
                f"Node {node_id} not found for CS_EXIT event.",
                involved_nodes,
            )
            return involved_nodes

        if node.state != NodeState.HELD:
            logger.error(
                f"T={current_int_time}: CS_EXIT event for N{node_id} but state is {node.state.name}! Ignoring exit logic."
            )
            details = f"Node {node_id} CS_EXIT event ignored (State was {node.state.name})"
            self._log_state(EventType.ERROR, details, involved_nodes)
            return involved_nodes

        logger.info(f"T={current_int_time}: --- N{node_id} EXIT CS ---")
        node.state = NodeState.IDLE
        node.request_ts = (-1, -1)

        deferred_to_reply = list(node.deferred_queue)
        node.deferred_queue.clear()

        logger.debug(
            f"  -> N{node_id} state update: State={node.state.name}, Request TS cleared. "
            f"Deferred queue cleared (had {len(deferred_to_reply)} entries)."
        )

        details = f"Node {node_id} exited CS."
        if deferred_to_reply:
            details += f" Sending {len(deferred_to_reply)} deferred replies to {sorted(deferred_to_reply)}."
            involved_nodes.extend(deferred_to_reply)
            logger.debug(
                f"  -> Sending deferred REPLYs to: {sorted(deferred_to_reply)}"
            )

            reply_ts = node.update_clock()
            logger.debug(
                f"  -> Clock updated to {reply_ts} for sending deferred replies."
            )

            for waiting_node_id in deferred_to_reply:
                self._send_message(
                    sender_id=node.id,
                    receiver_id=waiting_node_id,
                    msg_type=MessageType.REPLY,
                    msg_ts=reply_ts,
                )
        else:
            details += " No deferred requests to process."
            logger.debug("  -> No deferred requests in queue.")

        self._log_state(EventType.CS_EXIT, details, list(set(involved_nodes)))

        return list(set(involved_nodes))

    def step(self) -> bool:
        if not self.event_queue:
            logger.info("Event queue empty. No step taken.")
            return False

        event_time, _, event_type, event_data = heapq.heappop(self.event_queue)
        event_time_int = int(round(event_time))

        current_int_time = int(round(self.current_time))
        if event_time_int < current_int_time:
            logger.critical(
                f"CRITICAL: Pulled past event! Event T={event_time_int}, Current T={current_int_time}. "
                f"Type={event_type}, Data={event_data}. Potential heap/scheduling error!"
            )
            event_time_int = current_int_time

        if event_time_int > current_int_time:
            logger.debug(
                f"Advancing time from {current_int_time} to {event_time_int}"
            )
            self.current_time = float(event_time_int)

        current_int_time = int(round(self.current_time))
        logger.debug(
            f"\n--- Processing Event: {event_type} at T={current_int_time} ---"
        )
        logger.trace(f"    Event Data: {event_data}")

        involved: List[NodeId] = []
        try:
            if event_type == EventType.MESSAGE_ARRIVAL:
                involved = self._handle_message_arrival(event_data)
            elif event_type == EventType.CS_ENTER:
                self._enter_cs(event_data)
                involved = [event_data]
            elif event_type == EventType.CS_EXIT:
                involved = self._handle_cs_exit(event_data)
            elif event_type == EventType.SCHEDULED_REQUEST:
                involved = self._handle_scheduled_request(event_data)
            else:
                details = f"Unknown or unhandled Event Type {event_type}, Data: {event_data}"
                logger.error(details)
                self._log_state(EventType.UNKNOWN, details, [])

        except Exception as e:
            details = (
                f"Error processing {event_type} for data {event_data}: {e}"
            )
            logger.exception(details)

            involved_guess: List[NodeId] = []
            try:
                if isinstance(event_data, int):
                    involved_guess = [event_data]
                elif isinstance(event_data, tuple) and len(event_data) > 0:
                    if isinstance(event_data[0], int):
                        involved_guess.append(event_data[0])
                    if (
                        len(event_data) > 1
                        and isinstance(event_data[1], tuple)
                        and len(event_data[1]) > 1
                    ):
                        if isinstance(event_data[1][1], int):
                            involved_guess.append(event_data[1][1])
            except Exception:
                pass
            self._log_state(
                EventType.ERROR, details, list(set(involved_guess))
            )

        logger.debug(
            f"--- Finished Event {event_type} processing. Involved: {involved} ---"
        )
        return True

    def advance_time_by(self, amount: Union[int, float]) -> bool:
        if not isinstance(amount, (int, float)) or amount <= 0:
            logger.error(
                f"Advance time amount must be a positive number, got: {amount}."
            )
            return False

        initial_int_time = int(round(self.current_time))
        target_time = initial_int_time + int(math.ceil(amount))

        processed_count = 0
        logger.info(
            f"--- Advancing time from {initial_int_time} up to T={target_time} (Amount: {amount}) ---"
        )

        while self.event_queue and self.event_queue[0][0] <= target_time:
            if self.step():
                processed_count += 1
            else:
                logger.warning(
                    "Simulation step returned False during time advance, stopping advance."
                )
                break

        final_sim_time = max(self.current_time, float(target_time))
        self.current_time = final_sim_time
        final_int_time = int(round(final_sim_time))

        log_detail = f"Advanced time from T={initial_int_time} -> T={final_int_time} (Processed {processed_count} events)."
        logger.info(f"--- Time Advance Done. Current T={final_int_time}. ---")

        self._log_state(EventType.TIME_ADVANCE, log_detail, [])
        return True

    def advance_single_time_unit(self) -> bool:
        return self.advance_time_by(1)

    def _update_visualization(self) -> None:
        if self.vis_callback and callable(self.vis_callback):
            if self.node_positions is None:
                return
            if self.graph is None:
                return

            try:
                self.vis_callback(
                    self.graph,
                    self.nodes,
                    int(round(self.current_time)),
                    self.node_positions,
                )
            except Exception as e:
                logger.error(
                    f"Error during visualization callback: {e}", exc_info=True
                )


class WheelEventFilter(QObject):
    def __init__(self, target_scroll_area: QScrollArea, parent=None):
        super().__init__(parent)
        self._target_scroll_area = target_scroll_area

    def eventFilter(self, obj, event: QEvent) -> bool:
        try:
            target_sa = self._target_scroll_area
            if not target_sa:
                return super().eventFilter(obj, event)

            target_viewport = target_sa.viewport()
            if obj != target_viewport:
                return super().eventFilter(obj, event)

            if event.type() == QEvent.Type.Wheel:
                delta_y = event.angleDelta().y()
                if delta_y != 0:
                    h_bar = target_sa.horizontalScrollBar()
                    if h_bar:
                        new_value = h_bar.value() - (delta_y // 3)
                        h_bar.setValue(new_value)
                        return True

        except RuntimeError as e:
            if "deleted" in str(e).lower():
                logger.debug(
                    "WheelEventFilter: Target widget deleted during event processing."
                )
                return False
            else:
                logger.exception("Unexpected RuntimeError in WheelEventFilter")
                raise
        except Exception as e:
            logger.exception(f"Unexpected error in WheelEventFilter: {e}")
            pass

        return super().eventFilter(obj, event)


class RicartAgrawalaGUI(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(
            "Ricart-Agrawala Simulation (PyQt6 - Modern Style)"
        )
        self.setGeometry(100, 100, 900, 1000)

        self.simulation: Optional[Simulation] = None
        self.log_history_processed_count: int = 0
        self.table_history_processed_count: int = 0
        self.node_cs_durations: CSDurations = {}
        self.edge_delays: EdgeDelays = {}
        self.scheduled_requests: ScheduledRequests = {}

        self.dragged_node: Optional[NodeId] = None
        self.node_positions_cache: NodePositions = {}
        self.wheelEventFilter: Optional[WheelEventFilter] = None

        self.num_nodes_spinbox: Optional[QSpinBox] = None
        self.default_cs_spinbox: Optional[QSpinBox] = None
        self.default_min_delay_spinbox: Optional[QSpinBox] = None
        self.default_max_delay_spinbox: Optional[QSpinBox] = None
        self.init_button: Optional[QPushButton] = None
        self.adv_config_button: Optional[QPushButton] = None
        self.step_event_button: Optional[QPushButton] = None
        self.step_1_time_button: Optional[QPushButton] = None
        self.advance_time_spinbox: Optional[QSpinBox] = None
        self.advance_time_button: Optional[QPushButton] = None
        self.node_scroll_area: Optional[QScrollArea] = None
        self.node_buttons_container: Optional[QWidget] = None
        self.node_buttons_layout: Optional[QHBoxLayout] = None
        self.node_buttons: Dict[NodeId, QPushButton] = {}
        self.notebook: Optional[QTabWidget] = None
        self.fig: Optional[Figure] = None
        self.canvas: Optional[FigureCanvas] = None
        self.ax: Optional[plt.Axes] = None
        self.history_text: Optional[QTextEdit] = None
        self.state_table: Optional[QTableWidget] = None

        self._setup_ui()

        if not self._populate_default_configs():
            QMessageBox.critical(
                self,
                "Init Error",
                "Could not read initial default config values.",
            )
        self._disable_controls()
        self.statusBar().showMessage(
            "Ready. Configure defaults or use Advanced Config, then Initialize / Reset."
        )
        logger.info("RicartAgrawalaGUI initialized.")

    def _setup_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        config_panel = self._setup_config_panel()
        control_panel = self._setup_control_panel()
        top_splitter.addWidget(config_panel)
        top_splitter.addWidget(control_panel)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 0)
        top_splitter.setSizes([500, 500])

        node_request_panel = self._setup_node_request_panel()

        self.notebook = QTabWidget()
        sim_view_tab = self._setup_visualization_tab()
        state_table_tab = self._setup_state_table_tab()
        self.notebook.addTab(sim_view_tab, " Simulation View ")
        self.notebook.addTab(state_table_tab, " State History Table ")

        self.notebook.currentChanged.connect(self._handle_tab_changed)

        main_layout.addWidget(top_splitter)
        main_layout.addWidget(node_request_panel)
        main_layout.addWidget(self.notebook, 1)

        self._setup_status_bar()

    def _setup_config_panel(self) -> QGroupBox:
        group_box = QGroupBox("Configuration")
        outer_layout = QVBoxLayout(group_box)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(8)

        grid_layout = QGridLayout()
        grid_layout.setSpacing(8)

        self.num_nodes_spinbox = QSpinBox()
        self.num_nodes_spinbox.setRange(1, 100)
        self.num_nodes_spinbox.setValue(DEFAULT_NUM_NODES)
        self.num_nodes_spinbox.setToolTip(
            "Total number of nodes in the simulation."
        )
        self.num_nodes_spinbox.editingFinished.connect(
            self._populate_default_configs_on_change
        )

        self.default_cs_spinbox = QSpinBox()
        self.default_cs_spinbox.setRange(0, 99999)
        self.default_cs_spinbox.setValue(DEFAULT_CS_DURATION)
        self.default_cs_spinbox.setToolTip(
            "Default time nodes spend in the critical section."
        )
        self.default_cs_spinbox.editingFinished.connect(
            self._populate_default_configs_on_change
        )

        self.default_min_delay_spinbox = QSpinBox()
        self.default_min_delay_spinbox.setRange(0, 99999)
        self.default_min_delay_spinbox.setValue(DEFAULT_MIN_DELAY)
        self.default_min_delay_spinbox.setToolTip(
            "Default minimum network delay between nodes."
        )
        self.default_min_delay_spinbox.editingFinished.connect(
            self._populate_default_configs_on_change
        )

        self.default_max_delay_spinbox = QSpinBox()
        self.default_max_delay_spinbox.setRange(0, 99999)
        self.default_max_delay_spinbox.setValue(DEFAULT_MAX_DELAY)
        self.default_max_delay_spinbox.setToolTip(
            "Default maximum network delay between nodes."
        )
        self.default_max_delay_spinbox.editingFinished.connect(
            self._populate_default_configs_on_change
        )

        self.init_button = QPushButton("Initialize / Reset")
        self.init_button.setObjectName("init_button")
        self.init_button.setToolTip(
            "Start a new simulation or reset the current one using the current configuration (basic or advanced)."
        )
        self.init_button.clicked.connect(self.initialize_simulation)

        self.adv_config_button = QPushButton("Advanced Config...")
        self.adv_config_button.setToolTip(
            "Set per-node/per-edge parameters and scheduled requests."
        )
        self.adv_config_button.clicked.connect(self.open_advanced_config)

        grid_layout.addWidget(QLabel("Nodes:"), 0, 0)
        grid_layout.addWidget(self.num_nodes_spinbox, 0, 1)
        grid_layout.addWidget(QLabel("Default CS Dur:"), 0, 2)
        grid_layout.addWidget(self.default_cs_spinbox, 0, 3)
        grid_layout.addWidget(QLabel("Default Min Delay:"), 1, 0)
        grid_layout.addWidget(self.default_min_delay_spinbox, 1, 1)
        grid_layout.addWidget(QLabel("Default Max Delay:"), 1, 2)
        grid_layout.addWidget(self.default_max_delay_spinbox, 1, 3)
        grid_layout.setColumnStretch(4, 1)

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 5, 0, 0)
        button_layout.addWidget(self.init_button)
        button_layout.addWidget(self.adv_config_button)

        outer_layout.addLayout(grid_layout)
        outer_layout.addLayout(button_layout)
        outer_layout.addStretch(1)

        return group_box

    def _setup_control_panel(self) -> QGroupBox:
        group_box = QGroupBox("Simulation Control")
        layout = QGridLayout(group_box)
        layout.setSpacing(8)

        self.step_event_button = QPushButton("Process 1 Event")
        self.step_event_button.setToolTip(
            "Process the very next scheduled event in the queue."
        )
        self.step_event_button.clicked.connect(self.step_simulation_event)

        self.step_significant_event_button = QPushButton(
            "Process Next Significant Event"
        )
        self.step_significant_event_button.setToolTip(
            "Run simulation until the next CS entry/exit, request trigger, error, or queue end."
        )
        self.step_significant_event_button.clicked.connect(
            self.step_simulation_significant_event
        )

        self.step_1_time_button = QPushButton("Advance by 1 Time")
        self.step_1_time_button.setToolTip(
            "Advance simulation time by 1 unit, processing any events within."
        )
        self.step_1_time_button.clicked.connect(self.step_simulation_1_time)

        self.advance_time_spinbox = QSpinBox()
        self.advance_time_spinbox.setRange(1, 99999)
        self.advance_time_spinbox.setValue(10)
        self.advance_time_spinbox.setToolTip(
            "Amount of time units to advance."
        )

        self.advance_time_button = QPushButton("Advance Time")
        self.advance_time_button.setToolTip(
            "Advance simulation by the specified time, processing events within."
        )
        self.advance_time_button.clicked.connect(
            self.step_simulation_by_amount
        )

        step_buttons_layout = QHBoxLayout()
        step_buttons_layout.setSpacing(8)
        step_buttons_layout.addWidget(self.step_event_button)
        step_buttons_layout.addWidget(self.step_significant_event_button)
        step_buttons_layout.setStretch(0, 1)
        step_buttons_layout.setStretch(1, 1)

        layout.addLayout(step_buttons_layout, 0, 0, 1, 2)
        layout.addWidget(self.step_1_time_button, 1, 0, 1, 2)
        layout.addWidget(self.advance_time_spinbox, 2, 0)
        layout.addWidget(self.advance_time_button, 2, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(3, 1)

        return group_box

    def _setup_node_request_panel(self) -> QWidget:
        group_box = QGroupBox("Request CS Now:")
        panel_layout = QVBoxLayout(group_box)
        panel_layout.setContentsMargins(0, 5, 0, 0)
        panel_layout.setSpacing(0)

        self.node_scroll_area = QScrollArea()
        self.node_scroll_area.setWidgetResizable(True)
        self.node_scroll_area.setFixedHeight(45)
        self.node_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.node_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        if self.node_scroll_area:
            if self.wheelEventFilter is None:
                self.wheelEventFilter = WheelEventFilter(
                    self.node_scroll_area, self
                )
                self.node_scroll_area.viewport().installEventFilter(
                    self.wheelEventFilter
                )
                logger.debug(
                    "Installed wheel event filter on node request scroll area viewport."
                )

        self.node_buttons_container = QWidget()
        self.node_buttons_container.setAutoFillBackground(True)
        palette = self.node_buttons_container.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("white"))
        self.node_buttons_container.setPalette(palette)

        logger.debug("Attempting to create node_buttons_layout...")
        try:
            container_layout = QHBoxLayout(self.node_buttons_container)
            self.node_buttons_layout = container_layout
            logger.debug(
                f"Node buttons layout CREATED and ASSIGNED: {self.node_buttons_layout}"
            )
        except Exception as e:
            logger.error(
                f"EXCEPTION during QHBoxLayout creation/assignment: {e}",
                exc_info=True,
            )
            self.node_buttons_layout = None

        if self.node_buttons_layout is None:
            logger.error(
                "!!! self.node_buttons_layout is None immediately after creation attempt !!!"
            )
        else:
            self.node_buttons_layout.setContentsMargins(6, 3, 6, 3)
            self.node_buttons_layout.setSpacing(4)

        self.node_scroll_area.setWidget(self.node_buttons_container)

        panel_layout.addWidget(self.node_scroll_area)

        self.node_buttons = {}
        num_nodes_to_create = DEFAULT_NUM_NODES
        if self.num_nodes_spinbox:
            num_nodes_to_create = self.num_nodes_spinbox.value()
        else:
            logger.warning(
                "num_nodes_spinbox was not available when setting up node request panel."
            )

        logger.debug(
            f"Value check BEFORE calling _create_node_request_buttons: self.node_buttons_layout = {self.node_buttons_layout}"
        )
        self._create_node_request_buttons(num_nodes_to_create)

        return group_box

    def _create_node_request_buttons(self, num_nodes: int) -> None:
        logger.debug(
            f"INSIDE _create_node_request_buttons: Checking self.node_buttons_layout which is currently -> {self.node_buttons_layout}"
        )

        layout_obj = self.node_buttons_layout

        if layout_obj is None:
            logger.error(
                "Cannot create node buttons: layout_obj (from self.node_buttons_layout) is None."
            )
            return

        logger.debug(
            f"Clearing node buttons layout (current count: {layout_obj.count()})..."
        )
        while (item := layout_obj.takeAt(0)) is not None:
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.node_buttons.clear()
        logger.debug(
            f"Node buttons layout cleared (count after clear: {layout_obj.count()})."
        )

        logger.debug(f"Creating and adding {num_nodes} node request buttons.")
        for i in range(num_nodes):
            btn = QPushButton(f"N{i}")
            btn.setObjectName("NodeRequestButton")
            btn.setToolTip(
                f"Manually request Critical Section for Node {i} (only enabled if node is IDLE)."
            )
            btn.clicked.connect(
                lambda checked=False, node_id=i: self.request_cs_for_node(
                    node_id
                )
            )
            btn.setEnabled(False)

            layout_obj.addWidget(btn)

            self.node_buttons[i] = btn

        layout_obj.addStretch(1)

        button_count = sum(
            1
            for i in range(layout_obj.count())
            if layout_obj.itemAt(i).widget()
        )
        stretch_count = layout_obj.count() - button_count
        logger.debug(
            f"Layout rebuild complete. Contains {button_count} buttons and {stretch_count} stretch/spacer item(s). Total items: {layout_obj.count()}."
        )

        if self.node_buttons_container:
            self.node_buttons_container.adjustSize()
            self.node_buttons_container.updateGeometry()

    def _setup_visualization_tab(self) -> QWidget:
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Vertical)

        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Network State (Initialize Simulation)", y=0.98)
        self.ax.axis("off")

        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01)

        plot_layout.addWidget(self.canvas)

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        log_group = QGroupBox("Event History Log")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(0, 0, 0, 0)

        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        log_layout.addWidget(self.history_text)

        splitter.addWidget(plot_container)
        splitter.addWidget(log_group)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([600, 200])

        layout.addWidget(splitter)
        return tab_widget

    def _setup_state_table_tab(self) -> QWidget:
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setContentsMargins(5, 5, 5, 5)

        self.state_table = QTableWidget()
        self.state_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.state_table.setAlternatingRowColors(True)
        self.state_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.state_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.state_table.verticalHeader().setVisible(False)
        self.state_table.horizontalHeader().setHighlightSections(False)

        layout.addWidget(self.state_table)

        self._configure_state_table(DEFAULT_NUM_NODES)
        return tab_widget

    def _setup_status_bar(self) -> None:
        status = self.statusBar()

    def _handle_tab_changed(self, index: int) -> None:
        try:
            if self.notebook and self.notebook.widget(index):
                tab_text = self.notebook.tabText(index).strip()
                if tab_text == "Simulation View":
                    logger.debug(
                        f"Switched to Simulation View tab (Index {index}). Triggering redraw if needed."
                    )
                    if self.simulation:
                        self.update_graph_visualization(
                            self.simulation.graph,
                            self.simulation.nodes,
                            int(round(self.simulation.current_time)),
                            self.simulation.node_positions,
                        )
                    else:
                        logger.debug(
                            "Simulation not active, skipping visualization update on tab switch."
                        )
                else:
                    logger.debug(
                        f"Switched to tab '{tab_text}' (Index {index})."
                    )
            else:
                logger.warning(
                    f"Tab widget or widget at index {index} not found in _handle_tab_changed."
                )

        except Exception as e:
            logger.error(
                f"Error handling tab change to index {index}: {e}",
                exc_info=True,
            )

    def _populate_default_configs_on_change(self) -> None:
        sender = self.sender()
        relevant_spinboxes = [
            self.num_nodes_spinbox,
            self.default_cs_spinbox,
            self.default_min_delay_spinbox,
            self.default_max_delay_spinbox,
        ]
        if sender in relevant_spinboxes:
            if self.simulation is None:
                logger.debug(
                    f"Default config spinbox ({sender.objectName()}) changed. Repopulating internal configs."
                )
                if not self._populate_default_configs():
                    QMessageBox.critical(
                        self,
                        "Config Error",
                        "Failed to update internal configs after default value change.",
                    )
                else:
                    if sender == self.num_nodes_spinbox:
                        num_nodes = self.num_nodes_spinbox.value()
                        self._create_node_request_buttons(num_nodes)
                        self._configure_state_table(num_nodes)

            else:
                logger.debug(
                    f"Default config spinbox ({sender.objectName()}) changed, but simulation is active. Ignoring change."
                )

    def initialize_simulation(self) -> None:
        logger.info("=" * 10 + " INITIALIZE / RESET Request " + "=" * 10)
        try:
            num_nodes = self.num_nodes_spinbox.value()
            if num_nodes <= 0:
                raise ValueError("Number of nodes must be positive.")

            if len(self.node_cs_durations) != num_nodes:
                logger.warning(
                    f"Node count in UI ({num_nodes}) mismatches internal config size "
                    f"({len(self.node_cs_durations)}). Repopulating ALL configs with current DEFAULTS."
                )
                if not self._populate_default_configs():
                    raise ValueError(
                        "Failed to repopulate default configs for new node count."
                    )
                if len(self.node_cs_durations) != num_nodes:
                    raise ValueError(
                        "Internal config size mismatch persisted after repopulating defaults."
                    )

            self.simulation = None
            self.dragged_node = None
            self.node_positions_cache = {}
            self.log_history_processed_count = 0
            self.table_history_processed_count = 0

            if self.ax:
                self.ax.clear()
            if self.ax:
                self.ax.set_title("Network State (Initializing...)")
            if self.ax:
                self.ax.axis("off")
            if self.canvas:
                self.canvas.draw_idle()
            if self.history_text:
                self.history_text.clear()
            if self.state_table:
                self.state_table.setRowCount(0)
            self._create_node_request_buttons(num_nodes)
            self._configure_state_table(num_nodes)

            logger.info("Creating new Simulation object...")
            t_start = time.monotonic()

            self.simulation = Simulation(
                num_nodes=num_nodes,
                node_cs_durations=self.node_cs_durations,
                edge_delays=self.edge_delays,
                scheduled_requests=self.scheduled_requests,
                vis_callback=self.update_graph_visualization,
            )
            t_end = time.monotonic()
            logger.info(
                f"Simulation object created in {t_end - t_start:.4f} seconds."
            )

            if self.simulation and self.simulation.node_positions is None:
                logger.warning(
                    "Simulation initialized, but failed to generate node positions."
                )
            elif (
                self.simulation and self.simulation.node_positions is not None
            ):
                self.node_positions_cache = copy.deepcopy(
                    self.simulation.node_positions
                )

            self._update_history_log()
            self._update_state_table()
            self._enable_controls()
            self._update_status_and_buttons()

            if (
                self.simulation
                and self.simulation.graph
                and self.simulation.nodes
                and self.simulation.node_positions
            ):
                logger.debug(
                    "Triggering initial visualization draw from initialize_simulation."
                )
                self.update_graph_visualization(
                    self.simulation.graph,
                    self.simulation.nodes,
                    int(round(self.simulation.current_time)),
                    self.simulation.node_positions,
                )
            else:
                logger.warning(
                    "Could not trigger initial visualization: Simulation missing required data."
                )

        except (ValueError, AttributeError, TypeError) as e:
            logger.error(
                f"Failed to initialize simulation: {e}", exc_info=True
            )
            QMessageBox.critical(
                self,
                "Initialization Error",
                f"Failed to initialize simulation:\n{e}",
            )
            self.statusBar().showMessage("Initialization failed.")
            self._disable_controls()
            self.simulation = None
            if self.ax:
                self.ax.clear()
            if self.ax:
                self.ax.set_title("Network State (Initialization Failed)")
            if self.ax:
                self.ax.axis("off")
            if self.canvas:
                self.canvas.draw_idle()
            if self.history_text:
                self.history_text.clear()
            if self.state_table:
                self.state_table.setRowCount(0)

    def step_simulation_event(self) -> None:
        if not self._check_simulation_ready():
            return
        if not self.simulation.event_queue:
            current_time_int = int(round(self.simulation.current_time))
            self.statusBar().showMessage(
                f"T={current_time_int}. Event queue empty."
            )
            QMessageBox.information(
                self, "Step Event", "Event queue is empty."
            )
            self.step_event_button.setEnabled(False)
            return

        logger.debug("Stepping by next event...")
        processed = self.simulation.step()
        if processed:
            self._post_step_update()

    def step_simulation_significant_event(self) -> None:
        if not self._check_simulation_ready():
            return

        significant_events = {
            EventType.CS_ENTER,
            EventType.CS_EXIT,
            EventType.MANUAL_REQUEST,
            EventType.SCHEDULED_REQUEST,
            EventType.ERROR,
            EventType.UNKNOWN,
        }

        processed_count = 0
        significant_found = False
        max_steps = 10000

        initial_time = int(round(self.simulation.current_time))
        logger.debug(
            f"Stepping until significant event (from T={initial_time})..."
        )

        while self.simulation.event_queue and processed_count < max_steps:
            next_event_type = self.simulation.event_queue[0][2]

            if not self.simulation.step():
                logger.warning(
                    "simulation.step() returned False unexpectedly during significant event search."
                )
                break

            processed_count += 1

            if next_event_type in significant_events:
                logger.debug(
                    f"-> Found significant event ({next_event_type.name}) after {processed_count} step(s)."
                )
                significant_found = True
                break

        final_time = int(round(self.simulation.current_time))
        if processed_count == 0:
            self.statusBar().showMessage(
                f"T={final_time}. Event queue empty. No steps taken."
            )
        elif processed_count >= max_steps:
            logger.warning(
                f"Significant event step hit safety limit ({max_steps} steps). Stopping."
            )
            QMessageBox.warning(
                self,
                "Step Limit Reached",
                f"Processed {max_steps} events without finding a significant state change. Stopped to prevent freezing.",
            )
        elif not significant_found:
            logger.debug(
                f"Processed {processed_count} event(s) until queue end. No further significant events found."
            )
            self.statusBar().showMessage(
                f"T={final_time}. Event queue emptied after {processed_count} step(s)."
            )
        else:
            logger.debug(
                f"Finished processing significant event sequence. Total steps in sequence: {processed_count}. Final T={final_time}"
            )

        self._post_step_update()

    def step_simulation_1_time(self) -> None:
        if not self._check_simulation_ready():
            return
        logger.debug("Advancing by 1 time unit...")
        advanced = self.simulation.advance_single_time_unit()
        if advanced:
            self._post_step_update()

    def step_simulation_by_amount(self) -> None:
        if not self._check_simulation_ready() or not self.advance_time_spinbox:
            return
        amount = self.advance_time_spinbox.value()
        if amount <= 0:
            QMessageBox.warning(
                self, "Input Error", "Advance time amount must be positive."
            )
            return

        logger.debug(f"Advancing by {amount} time units...")
        advanced = self.simulation.advance_time_by(amount)
        if advanced:
            self._post_step_update()

    def request_cs_for_node(self, node_id: NodeId) -> None:
        if not self._check_simulation_ready():
            return

        node = self.simulation.nodes.get(node_id)
        current_int_time = int(round(self.simulation.current_time))

        if node and node.state == NodeState.IDLE:
            logger.info(
                f"T={current_int_time}: Manual CS request for N{node_id} triggered by GUI button."
            )
            self.simulation._log_state(
                EventType.MANUAL_REQUEST,
                f"GUI button pressed to request CS for N{node_id}.",
                [node_id],
            )
            success = self.simulation.want_cs(node_id)
            if not success:
                logger.error(
                    f"Manual req N{node_id} failed unexpectedly after IDLE check."
                )
                QMessageBox.warning(
                    self,
                    "Request Failed",
                    f"Node {node_id} could not initiate request (unexpected error).",
                )
            self._post_step_update()

        elif node:
            QMessageBox.information(
                self,
                "Request Blocked",
                f"Node {node_id} cannot request CS now.\nCurrent State: {node.state.name}",
            )
        else:
            logger.error(
                f"Node {node_id} not found for manual request (button mismatch?)."
            )
            QMessageBox.critical(
                self, "Error", f"Node {node_id} not found in simulation."
            )

    def open_advanced_config(self) -> None:
        try:
            num_nodes = self.num_nodes_spinbox.value()
            if num_nodes <= 0:
                QMessageBox.critical(
                    self, "Error", "Number of nodes must be greater than 0."
                )
                return

            if len(self.node_cs_durations) != num_nodes:
                logger.warning(
                    f"Node count changed ({len(self.node_cs_durations)} -> {num_nodes}) "
                    f"without reset. Repopulating configs with DEFAULTS before opening Advanced window."
                )
                if not self._populate_default_configs():
                    QMessageBox.critical(
                        self,
                        "Config Error",
                        "Failed to update internal configs for new node count.",
                    )
                    return

            default_cs = self.default_cs_spinbox.value()
            default_min = self.default_min_delay_spinbox.value()
            default_max = self.default_max_delay_spinbox.value()

            config_window = AdvancedConfigWindow(
                parent=self,
                num_nodes=num_nodes,
                current_cs_durations=self.node_cs_durations,
                current_edge_delays=self.edge_delays,
                current_scheduled_requests=self.scheduled_requests,
                default_cs=default_cs,
                default_min=default_min,
                default_max=default_max,
            )
            result_code = config_window.exec()

            if (
                result_code == QDialog.DialogCode.Accepted
                and config_window.result
            ):
                logger.info("Applying advanced configuration changes.")
                self.node_cs_durations = config_window.result["cs_durations"]
                self.edge_delays = config_window.result["edge_delays"]
                self.scheduled_requests = config_window.result[
                    "scheduled_requests"
                ]
                self.statusBar().showMessage(
                    "Advanced configuration updated. Click Initialize / Reset to apply."
                )

                if self.simulation:
                    logger.warning(
                        "Advanced config changed while simulation was active. Controls disabled. Reset required."
                    )
                    QMessageBox.warning(
                        self,
                        "Configuration Changed",
                        "Advanced configuration updated.\nPlease click Initialize / Reset to apply the changes.",
                    )
                    self._disable_controls()
                    self.statusBar().showMessage(
                        "Config changed. Initialize / Reset required.", 5000
                    )

            else:
                logger.info(
                    "Advanced configuration cancelled or closed without applying changes."
                )
                self.statusBar().showMessage(
                    "Advanced configuration cancelled or closed.", 3000
                )

        except (ValueError, AttributeError, TypeError) as e:
            logger.error(
                f"Error preparing or opening advanced config: {e}",
                exc_info=True,
            )
            QMessageBox.critical(
                self,
                "Configuration Error",
                f"Error preparing advanced config:\n{e}",
            )

    def _on_press(self, event: MouseEvent) -> None:
        self.dragged_node = None
        if (
            event.inaxes != self.ax
            or event.xdata is None
            or event.ydata is None
            or not self.simulation
            or self.simulation.node_positions is None
        ):
            return

        x, y = event.xdata, event.ydata
        min_dist_sq = NODE_CLICK_TOLERANCE**2
        clicked_node = None

        current_positions = self.simulation.node_positions
        for node_id, pos in current_positions.items():
            if isinstance(pos, (tuple, list, np.ndarray)) and len(pos) == 2:
                try:
                    dist_sq = (float(pos[0]) - x) ** 2 + (
                        float(pos[1]) - y
                    ) ** 2
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        clicked_node = node_id
                except (TypeError, ValueError):
                    logger.warning(
                        f"Invalid position data for node {node_id}: {pos} during click detection."
                    )
                    continue

        if clicked_node is not None:
            self.dragged_node = clicked_node
            logger.debug(f"Dragging Node {self.dragged_node}")

    def _on_motion(self, event: MouseEvent) -> None:
        if (
            self.dragged_node is None
            or event.inaxes != self.ax
            or event.xdata is None
            or event.ydata is None
            or not self.simulation
            or self.simulation.node_positions is None
        ):
            return

        x, y = event.xdata, event.ydata

        if self.dragged_node in self.simulation.node_positions:
            self.simulation.node_positions[self.dragged_node] = (
                float(x),
                float(y),
            )
            if (
                self.simulation.graph
                and self.simulation.nodes
                and self.simulation.node_positions
            ):
                self.update_graph_visualization(
                    self.simulation.graph,
                    self.simulation.nodes,
                    int(round(self.simulation.current_time)),
                    self.simulation.node_positions,
                )
        else:
            logger.warning(
                f"Dragged node {self.dragged_node} not found in simulation positions during motion."
            )
            self.dragged_node = None

    def _on_release(self, event: MouseEvent) -> None:
        if self.dragged_node is not None:
            logger.debug(f"Finished dragging Node {self.dragged_node}")
            if self.simulation and self.simulation.node_positions:
                self.node_positions_cache = copy.deepcopy(
                    self.simulation.node_positions
                )
            if (
                self.simulation
                and self.simulation.graph
                and self.simulation.nodes
                and self.simulation.node_positions
            ):
                self.update_graph_visualization(
                    self.simulation.graph,
                    self.simulation.nodes,
                    int(round(self.simulation.current_time)),
                    self.simulation.node_positions,
                )
            self.dragged_node = None

    def _populate_default_configs(self) -> bool:
        try:
            num_nodes = self.num_nodes_spinbox.value()
            default_cs = self.default_cs_spinbox.value()
            default_min = self.default_min_delay_spinbox.value()
            default_max = self.default_max_delay_spinbox.value()

            if num_nodes <= 0:
                num_nodes = 1
                self.num_nodes_spinbox.setValue(1)
            if default_cs < 0:
                default_cs = 0
                self.default_cs_spinbox.setValue(0)
            if default_min < 0:
                default_min = 0
                self.default_min_delay_spinbox.setValue(0)
            if default_max < default_min:
                default_max = default_min
                self.default_max_delay_spinbox.setValue(default_max)

            self.node_cs_durations = {i: default_cs for i in range(num_nodes)}
            self.edge_delays = {
                tuple(sorted((u, v))): {"min": default_min, "max": default_max}
                for u in range(num_nodes)
                for v in range(u + 1, num_nodes)
            }
            self.scheduled_requests = {i: [] for i in range(num_nodes)}

            logger.debug(
                "Populated internal configs based on current default values."
            )
            return True

        except Exception as e:
            logger.error(
                f"Error reading default config values from UI: {e}",
                exc_info=True,
            )
            return False

    def _enable_controls(self) -> None:
        is_ready = bool(self.simulation)
        queue_has_events = is_ready and bool(self.simulation.event_queue)

        if self.step_event_button:
            self.step_event_button.setEnabled(queue_has_events)
        if self.step_significant_event_button:
            self.step_significant_event_button.setEnabled(queue_has_events)
        if self.step_1_time_button:
            self.step_1_time_button.setEnabled(is_ready)
        if self.advance_time_button:
            self.advance_time_button.setEnabled(is_ready)
        if self.advance_time_spinbox:
            self.advance_time_spinbox.setEnabled(is_ready)

        if is_ready:
            self.update_request_buttons_state()
        else:
            for btn in self.node_buttons.values():
                btn.setEnabled(False)

    def _disable_controls(self) -> None:
        if self.step_event_button:
            self.step_event_button.setEnabled(False)
        if self.step_significant_event_button:
            self.step_significant_event_button.setEnabled(False)
        if self.step_1_time_button:
            self.step_1_time_button.setEnabled(False)
        if self.advance_time_button:
            self.advance_time_button.setEnabled(False)
        if self.advance_time_spinbox:
            self.advance_time_spinbox.setEnabled(False)
        for btn in self.node_buttons.values():
            btn.setEnabled(False)

    def update_graph_visualization(
        self,
        graph: Optional[nx.Graph],
        nodes: Optional[NodeDict],
        current_time: int,
        positions: Optional[NodePositions],
    ) -> None:
        if (
            not self.canvas
            or not self.ax
            or not self.fig
            or not self.canvas.isVisible()
        ):
            return
        if graph is None or nodes is None or positions is None:
            logger.warning(
                "Visualization skipped: Missing graph, nodes, or positions data."
            )
            self.ax.clear()
            self.ax.set_title(
                f"Visualization Error at T={current_time}", color="red"
            )
            self.ax.text(
                0.5,
                0.5,
                "Missing simulation data for visualization.",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                color="red",
            )
            self.ax.axis("off")
            self.canvas.draw_idle()
            return

        try:
            self.ax.clear()

            valid_positions: NodePositions = {}
            for nid, pos in positions.items():
                if isinstance(pos, (tuple, list)) and len(pos) == 2:
                    try:
                        x, y = float(pos[0]), float(pos[1])
                        if math.isfinite(x) and math.isfinite(y):
                            valid_positions[nid] = (x, y)
                        else:
                            logger.warning(
                                f"Non-finite position for node {nid}: ({x},{y}). Skipping."
                            )
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid position value for node {nid}: {pos}. Skipping."
                        )
                else:
                    logger.warning(
                        f"Invalid position structure for node {nid}: {pos}. Skipping."
                    )

            self.node_positions_cache = copy.deepcopy(valid_positions)

            if not valid_positions:
                logger.warning("No valid node positions found to draw.")
                self.ax.set_title(
                    f"No Valid Positions at T={current_time}", color="orange"
                )
                self.ax.axis("off")
                self.canvas.draw_idle()
                return

            valid_node_list = list(valid_positions.keys())
            node_colors: List[str] = []
            node_labels: Dict[NodeId, str] = {}
            held_node_exists = False

            for node_id in valid_node_list:
                node = nodes.get(node_id)
                if node:
                    state = node.state
                    req_ts_str = ""
                    if node.request_ts != (-1, -1) and state in (
                        NodeState.WANTED,
                        NodeState.HELD,
                    ):
                        req_ts_str = (
                            f"\nR:({node.request_ts[0]},{node.request_ts[1]})"
                        )

                    node_labels[node_id] = (
                        f"N{node_id}\nC:{node.clock}\nS:{state.name[0]}{req_ts_str}"
                    )

                    if state == NodeState.IDLE:
                        node_colors.append("lightblue")
                    elif state == NodeState.WANTED:
                        node_colors.append("gold")
                    else:
                        node_colors.append("limegreen")
                        held_node_exists = True
                else:
                    node_labels[node_id] = f"N{node_id}\n(Error!)"
                    node_colors.append("lightcoral")
                    logger.warning(
                        f"Node object missing for ID {node_id} during visualization."
                    )

            common_node_opts = dict(
                ax=self.ax,
                nodelist=valid_node_list,
                node_size=1900,
                edgecolors="dimgray",
                linewidths=1.0,
            )
            nx.draw_networkx_nodes(
                graph,
                valid_positions,
                node_color=node_colors,
                **common_node_opts,
            )
            nx.draw_networkx_labels(
                graph,
                valid_positions,
                labels=node_labels,
                ax=self.ax,
                font_size=7.5,
            )

            valid_edges = [
                (u, v)
                for (u, v) in graph.edges()
                if u in valid_positions and v in valid_positions
            ]
            nx.draw_networkx_edges(
                graph,
                valid_positions,
                edgelist=valid_edges,
                ax=self.ax,
                edge_color="lightgray",
                width=1.0,
                alpha=0.7,
                arrows=False,
            )

            int_current_time = int(round(current_time))
            for u, v, data in graph.edges(data=True):
                if u not in valid_positions or v not in valid_positions:
                    continue

                transit_list = data.get("in_transit", [])
                if isinstance(transit_list, list):
                    for transit_info in transit_list:
                        try:
                            if len(transit_info) != 5:
                                continue
                            (
                                msg_type,
                                sender,
                                receiver,
                                send_time,
                                arrival_time,
                            ) = transit_info
                            if (
                                sender not in valid_positions
                                or receiver not in valid_positions
                            ):
                                continue

                            pos_sender = valid_positions[sender]
                            pos_receiver = valid_positions[receiver]

                            total_duration = float(arrival_time - send_time)
                            progress = (
                                1.0
                                if total_duration <= 1e-6
                                else max(
                                    0.0,
                                    min(
                                        1.0,
                                        (float(current_time) - send_time)
                                        / total_duration,
                                    ),
                                )
                            )

                            dx, dy = (
                                pos_receiver[0] - pos_sender[0],
                                pos_receiver[1] - pos_sender[1],
                            )
                            edge_len = math.hypot(dx, dy)
                            if edge_len < 1e-6:
                                continue

                            ux, uy = dx / edge_len, dy / edge_len

                            start_x = pos_sender[0] + ux * NODE_RADIUS_VISUAL
                            start_y = pos_sender[1] + uy * NODE_RADIUS_VISUAL
                            end_x = pos_receiver[0] - ux * NODE_RADIUS_VISUAL
                            end_y = pos_receiver[1] - uy * NODE_RADIUS_VISUAL

                            vec_x, vec_y = end_x - start_x, end_y - start_y

                            arrow_tip_x = start_x + vec_x * progress
                            arrow_tip_y = start_y + vec_y * progress

                            offset_scale = ARROW_OFFSET_AMOUNT

                            off_x, off_y = (
                                -uy * offset_scale,
                                ux * offset_scale,
                            )

                            head_x = arrow_tip_x + off_x
                            head_y = arrow_tip_y + off_y

                            tail_x = head_x - ux * ARROW_LENGTH
                            tail_y = head_y - uy * ARROW_LENGTH
                            arrow_dx = head_x - tail_x
                            arrow_dy = head_y - tail_y

                            arrow_color = (
                                "mediumblue"
                                if msg_type == MessageType.REQUEST
                                else "darkmagenta"
                            )

                            self.ax.arrow(
                                tail_x,
                                tail_y,
                                arrow_dx,
                                arrow_dy,
                                head_width=ARROW_HEAD_WIDTH,
                                head_length=ARROW_HEAD_LENGTH,
                                length_includes_head=True,
                                fc=arrow_color,
                                ec=arrow_color,
                                lw=1,
                                alpha=0.95,
                                zorder=5,
                            )
                        except Exception as arrow_err:
                            logger.error(
                                f"Error drawing arrow for edge ({u},{v}), transit {transit_info}: {arrow_err}",
                                exc_info=True,
                            )

            title = f"Ricart-Agrawala State at T = {current_time}"
            if held_node_exists:
                title += " (CS HELD)"
            self.ax.set_title(title, fontsize=10)
            self.ax.axis("off")
            self.canvas.draw_idle()

        except Exception as vis_err:
            logger.error(
                f"Error during graph visualization update: {vis_err}",
                exc_info=True,
            )
            try:
                self.ax.clear()
                self.ax.set_title(
                    f"Visualization Error at T={current_time}", color="red"
                )
                self.ax.text(
                    0.5,
                    0.5,
                    f"Error drawing graph:\n{vis_err}",
                    ha="center",
                    va="center",
                    transform=self.ax.transAxes,
                    color="red",
                    wrap=True,
                )
                self.ax.axis("off")
                self.canvas.draw_idle()
            except Exception:
                pass

    def _update_history_log(self) -> None:
        if not self.history_text:
            return

        sim = self.simulation
        history = sim.history_data if sim else []

        if not sim or not history:
            if self.history_text.document().characterCount() > 0:
                logger.debug(
                    "Clearing history log display (no simulation/history)."
                )
                self.history_text.clear()
            if self.log_history_processed_count != 0:
                logger.warning(
                    "Resetting log_history_processed_count counter to 0."
                )
                self.log_history_processed_count = 0
            return

        try:
            total_entries = len(history)
            start_index = self.log_history_processed_count

            if start_index < total_entries:
                num_new = total_entries - start_index
                logger.debug(
                    f"History log: Appending {num_new} entries (indices {start_index} to {total_entries - 1})"
                )
                new_log_lines = []
                for i in range(start_index, total_entries):
                    try:
                        entry = history[i]
                        involved_str = (
                            f"(Inv: {entry.get('involved', [])})"
                            if entry.get("involved")
                            else ""
                        )
                        time_str = f"T={entry.get('time', '?'):<4}"
                        type_str = f"[{str(entry.get('type', 'UNK')): <11}]"
                        details_str = entry.get("details", "N/A")
                        log_str = f"{time_str}: {type_str} {details_str} {involved_str}"
                        new_log_lines.append(log_str)
                    except Exception as fmt_err:
                        logger.error(
                            f"Error formatting history log entry {i}: {fmt_err}"
                        )
                        new_log_lines.append(
                            f"--- Error formatting entry {i} ---"
                        )

                if new_log_lines:
                    prefix = (
                        "\n"
                        if self.history_text.document().characterCount() > 0
                        else ""
                    )
                    self.history_text.append(prefix + "\n".join(new_log_lines))

                self.log_history_processed_count = total_entries

                self.history_text.verticalScrollBar().setValue(
                    self.history_text.verticalScrollBar().maximum()
                )

        except Exception as e:
            logger.error(
                f"Error updating history log incrementally: {e}", exc_info=True
            )

    def _configure_state_table(self, num_nodes: int) -> None:
        if not self.state_table:
            return
        logger.debug(f"Configuring state table for {num_nodes} nodes.")
        self.state_table.setRowCount(0)

        col_ids = ["Time", "Event", "Details"]
        col_tooltips = ["Simulation Time", "Event Type", "Event Details"]
        col_widths = [50, 90, 280]
        col_resize_modes = [
            QHeaderView.ResizeMode.Interactive,
            QHeaderView.ResizeMode.Interactive,
            QHeaderView.ResizeMode.Stretch,
        ]

        node_col_width = 100
        for i in range(num_nodes):
            col_ids.append(f"Node {i}")
            col_tooltips.append(f"State of Node {i} (State[Clock] R(ReqTS))")
            col_widths.append(node_col_width)
            col_resize_modes.append(QHeaderView.ResizeMode.Interactive)

        self.state_table.setColumnCount(len(col_ids))
        self.state_table.setHorizontalHeaderLabels(col_ids)

        header = self.state_table.horizontalHeader()
        for i, tooltip in enumerate(col_tooltips):
            self.state_table.horizontalHeaderItem(i).setToolTip(tooltip)
            self.state_table.setColumnWidth(i, col_widths[i])
            header.setSectionResizeMode(i, col_resize_modes[i])

        logger.debug(f"State table columns configured: {col_ids}")

    def _update_state_table(self) -> None:
        if not self.state_table:
            logger.warning(
                "_update_state_table called but self.state_table is None."
            )
            return

        sim = self.simulation
        history = sim.history_data if sim else []
        num_nodes = sim.num_nodes if sim else 0

        if not sim or not history:
            if self.state_table.rowCount() > 0:
                logger.info(
                    "Clearing state table display (no simulation/history)."
                )
                self.state_table.setRowCount(0)
            if self.table_history_processed_count != 0:
                logger.warning(
                    "Resetting table_history_processed_count counter to 0."
                )
                self.table_history_processed_count = 0
            return

        try:
            total_entries = len(history)
            start_index = self.table_history_processed_count

            logger.debug(
                f"Updating state table. Total history: {total_entries}, "
                f"Table Processed count: {start_index}. "
                f"Current visual rows: {self.state_table.rowCount()}."
            )

            if start_index < total_entries:
                num_new = total_entries - start_index
                logger.info(
                    f"State table: Adding {num_new} new rows (history indices {start_index} to {total_entries - 1})."
                )

                self.state_table.setUpdatesEnabled(False)

                for history_idx in range(start_index, total_entries):
                    try:
                        entry = history[history_idx]
                        new_row_index = self.state_table.rowCount()
                        self.state_table.insertRow(new_row_index)
                        logger.trace(
                            f"  Inserted row at visual index {new_row_index} for history entry {history_idx}"
                        )

                        self.state_table.setItem(
                            new_row_index,
                            0,
                            QTableWidgetItem(str(entry.get("time", "?"))),
                        )
                        self.state_table.setItem(
                            new_row_index,
                            1,
                            QTableWidgetItem(str(entry.get("type", "UNK"))),
                        )
                        self.state_table.setItem(
                            new_row_index,
                            2,
                            QTableWidgetItem(entry.get("details", "N/A")),
                        )
                        snapshots = entry.get("node_snapshots", {})
                        for node_id in range(num_nodes):
                            col_idx = 3 + node_id
                            if col_idx < self.state_table.columnCount():
                                snapshot = snapshots.get(node_id)
                                item_text = "N/A"
                                if snapshot:
                                    state = snapshot.get("state", None)
                                    clock = snapshot.get("clock", -1)
                                    req_ts = snapshot.get(
                                        "request_ts", (-1, -1)
                                    )
                                    state_char = (
                                        str(state)
                                        if isinstance(state, NodeState)
                                        else "?"
                                    )
                                    item_text = f"{state_char}[{clock}]"
                                    if (
                                        req_ts != (-1, -1)
                                        and isinstance(state, NodeState)
                                        and state
                                        in (NodeState.WANTED, NodeState.HELD)
                                    ):
                                        item_text += (
                                            f" R({req_ts[0]},{req_ts[1]})"
                                        )
                                node_item = QTableWidgetItem(item_text)
                                node_item.setTextAlignment(
                                    Qt.AlignmentFlag.AlignCenter
                                )
                                self.state_table.setItem(
                                    new_row_index, col_idx, node_item
                                )

                        logger.trace(
                            f"  Populated row {new_row_index} for T={entry.get('time', '?')}, Type={entry.get('type', 'UNK')}"
                        )

                    except Exception as populate_err:
                        logger.error(
                            f"Error processing history entry {history_idx} for state table row {new_row_index}: {populate_err}",
                            exc_info=True,
                        )
                        if new_row_index < self.state_table.rowCount():
                            self.state_table.setItem(
                                new_row_index,
                                2,
                                QTableWidgetItem(f"ERROR: {populate_err}"),
                            )

                self.table_history_processed_count = total_entries

                self.state_table.setUpdatesEnabled(True)
                self.state_table.scrollToBottom()
                logger.debug(
                    f"Finished state table update. table_history_processed_count={self.table_history_processed_count}. Final visual row count: {self.state_table.rowCount()}"
                )

            else:
                logger.debug(
                    "State table update: No new history entries to add based on processed count."
                )

        except Exception as e:
            logger.error(
                f"Unexpected error during state table incremental update: {e}",
                exc_info=True,
            )
            self.state_table.setUpdatesEnabled(True)

    def _update_status_and_buttons(self) -> None:
        if self.simulation:
            time_str = f"T={int(round(self.simulation.current_time))}"
            queue_empty = not self.simulation.event_queue
            status_msg = f"Simulation Active: {time_str}."

            if queue_empty:
                status_msg += " Event queue empty."
            else:
                next_event_time = self.simulation.event_queue[0][0]
                next_event_type = self.simulation.event_queue[0][2]
                status_msg += f" Next event: {next_event_type} at T={int(round(next_event_time))}."

            self.statusBar().showMessage(status_msg)
            self._enable_controls()
        else:
            self.statusBar().showMessage(
                "Simulation not initialized. Configure and click Initialize / Reset."
            )
            self._disable_controls()

    def update_request_buttons_state(self) -> None:
        if not self.simulation or not self.simulation.nodes:
            self._disable_controls()
            return

        for i, button in self.node_buttons.items():
            node = self.simulation.nodes.get(i)
            can_request = node is not None and node.state == NodeState.IDLE
            button.setEnabled(can_request)

    def _check_simulation_ready(self) -> bool:
        if self.simulation is None:
            QMessageBox.warning(
                self,
                "Simulation Not Running",
                "Simulation not initialized. Please Initialize / Reset.",
            )
            return False
        return True

    def _post_step_update(self) -> None:
        self._update_history_log()
        self._update_state_table()
        self._update_status_and_buttons()

        if (
            self.simulation
            and self.simulation.graph
            and self.simulation.nodes
            and self.simulation.node_positions
        ):
            try:
                self.update_graph_visualization(
                    self.simulation.graph,
                    self.simulation.nodes,
                    int(round(self.simulation.current_time)),
                    self.simulation.node_positions,
                )
            except Exception as e:
                logger.error(
                    f"Error calling update_graph_visualization from _post_step_update: {e}",
                    exc_info=True,
                )
        elif self.simulation:
            logger.warning(
                "Skipping visualization update in _post_step_update: Simulation missing required data (graph/nodes/positions)."
            )

    def closeEvent(self, event) -> None:
        reply = QMessageBox.question(
            self,
            "Confirm Exit",
            "Are you sure you want to quit the simulation?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            logger.info("Exiting application upon user confirmation.")
            event.accept()
        else:
            logger.debug("Application exit cancelled by user.")
            event.ignore()


if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)

    active_stylesheet = None
    style_name = "None"
    try:
        if "APP_STYLESHEET" in globals() and isinstance(APP_STYLESHEET, str):
            active_stylesheet = APP_STYLESHEET
            style_name = "Modern Light"
        else:
            logger.warning(
                "No stylesheet definitions found (APP_STYLESHEET). Using default Qt styles."
            )

        if active_stylesheet:
            app.setStyleSheet(active_stylesheet)
            logger.info(f"Applied '{style_name}' application stylesheet.")

    except Exception as style_err:
        logger.error(
            f"Failed to apply '{style_name}' stylesheet: {style_err}",
            exc_info=True,
        )

    logger.info("PyQt6 Application starting...")

    main_window = None
    try:
        main_window = RicartAgrawalaGUI()
        main_window.show()
        logger.info("Main window created and shown.")
    except Exception as e:
        logger.critical(
            f"FATAL ERROR: Failed to create or show the main application window: {e}",
            exc_info=True,
        )
        try:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Critical)
            msg_box.setWindowTitle("Application Startup Error")
            msg_box.setText(
                f"Fatal error during application startup:\n{e}\n\nSee console log or application log file for details."
            )
            msg_box.exec()
        except Exception as msg_err:
            logger.critical(
                f"FATAL ERROR: {e}\nCould not display error message box: {msg_err}",
                file=sys.stderr,
            )
        sys.exit(1)

    if main_window:
        logger.info("Starting PyQt6 event loop...")
        exit_code = app.exec()
        logger.info(
            f"PyQt6 event loop finished. Exiting with code {exit_code}."
        )
        sys.exit(exit_code)
    else:
        logger.critical("Main window object not created. Exiting.")
        sys.exit(1)
