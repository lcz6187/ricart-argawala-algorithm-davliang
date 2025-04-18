import copy
import enum
import heapq
import json
import math
import random
import time
import tkinter as tk
from tkinter import Event, Toplevel, filedialog, messagebox, scrolledtext, ttk

from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import numpy as np
from rich.console import Console
from rich.logging import RichHandler

try:
    # Themed imports make tkinter look nice.
    # This is optional and not required for the simulation to work.
    from ttkthemes import ThemedTk

    ttk_themes_available = True
except ImportError:
    ttk_themes_available = False
    ThemedTk = tk.Tk  # type: ignore[assignment, misc]

# Setup logger
console = Console()

logger.configure(
    handlers=[
        {
            "sink": RichHandler(console=console),
            "format": "{message}",
        },
    ]
)

# Default number of nodes
DEFAULT_NUM_NODES = 5
# Default time to spend in the critical section
DEFAULT_CS_DURATION = 30
# Default minimum delay for edges between nodes (min network delay)
DEFAULT_MIN_DELAY = 5
# Default maximum delay for edges between nodes (max network delay)
DEFAULT_MAX_DELAY = 15
# Random seed for generating the graph layout. Set to 0 for reproducibility.
GRAPH_LAYOUT_SEED = 0
# Tolerance for clicking on nodes based on their radius
NODE_CLICK_TOLERANCE = 0.1
# Visual radius of nodes in the graph
NODE_RADIUS_VISUAL = 0.05
# Adjascent perpendicular offset for arrows
ARROW_OFFSET_AMOUNT = 0.03
# Length of the arrow
ARROW_LENGTH = 0.05
# Width of the arrow head
ARROW_HEAD_WIDTH = 0.035
# Length of the arrow head
ARROW_HEAD_LENGTH = 0.05


class NodeState(enum.Enum):
    """Node states for the simulation."""

    IDLE = 0
    WANTED = 1
    HELD = 2

    def __str__(self):
        # E.g., IDLE -> I, WANTED -> W, HELD -> H
        return self.name[0]


class MessageType(enum.Enum):
    """
    Message types for the simulation. A requesting process will send a REQUEST and other nodes will receive and send a REPLY.

    MessageType.REQUEST is blue in the visualization.
    MessageType.REPLY is purple in the visualization.
    """

    REQUEST = 1
    REPLY = 2


class EventType(str, enum.Enum):
    """Event types to identify events within the event queue."""

    # A message arrived at a node.
    MESSAGE_ARRIVAL = "MSG_ARRIVE"
    # A node entered the critical section.
    CS_ENTER = "CS_ENTER"
    # A node exited the critical section.
    CS_EXIT = "CS_EXIT"
    # A scheduled request based on the advanced confid was triggered.
    SCHEDULED_REQUEST = "SCHED_REQ"
    # A manual request was triggered by the user.
    MANUAL_REQUEST = "MANUAL_REQ"
    # A meta event to advance the simulation time.
    TIME_ADVANCE = "TIME_ADV"
    # Initialization event for the simulation.
    INIT = "INIT"
    # Unknown event type.
    UNKNOWN = "UNKNOWN"
    # Error event type.
    ERROR = "ERROR"


def _edge_key_to_str(edge_key):
    """Convert an edge key (tuple of two integers) to a string for JSON keys."""

    u, v = sorted(edge_key)
    return f"{u},{v}"


def _str_to_edge_key(edge_str):
    """Convert a string of an edge (e.g., "0,1") to a tuple of integers."""

    try:
        parts = edge_str.split(",")
        if len(parts) == 2:
            u = int(parts[0].strip())
            v = int(parts[1].strip())
            return tuple(sorted((u, v)))
        else:
            raise ValueError(
                "Edge string must have two parts separated by comma."
            )
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid edge string format '{edge_str}': {e}"
        ) from e


class AdvancedConfigWindow(Toplevel):
    """
    Provides a gui for detailed configuration of the simulation parameters
    for the Ricart-Agrawala algorithm.

    Allows the specification of:
    - Critical Section (CS) duration for each node.
    - Minimum and maximum message delay for each communication edge between any two nodes.
    - Specific times at which nodes should initiate a request to enter the critical section.
    """

    def __init__(
        self,
        parent,
        num_nodes,
        current_cs_durations,
        current_edge_delays,
        current_scheduled_requests,
        default_cs,
        default_min,
        default_max,
    ):
        super().__init__(parent)
        self.title("Advanced Configuration")
        self.transient(parent)
        self.grab_set()
        self.num_nodes = num_nodes
        self.result = None

        self.cs_duration_vars = {}
        self.edge_min_delay_vars = {}
        self.edge_max_delay_vars = {}
        self.scheduled_request_vars = {}
        self.widgets_in_tab_order = []

        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # Helper function to create a scrollable frame
        def create_scrollable_frame(parent):
            outer_frame = ttk.Frame(parent, borderwidth=1, relief=tk.SUNKEN)
            outer_frame.rowconfigure(0, weight=1)
            outer_frame.columnconfigure(0, weight=1)

            try:
                frame_bg = ttk.Style().lookup("TFrame", "background")
                canvas_bg = frame_bg if frame_bg else "white"
            except tk.TclError:
                canvas_bg = "white"

            canvas = tk.Canvas(
                outer_frame,
                borderwidth=0,
                highlightthickness=0,
                background=canvas_bg,
            )
            scrollbar = ttk.Scrollbar(
                outer_frame,
                orient="vertical",
                command=canvas.yview,
            )
            inner_frame = ttk.Frame(canvas, padding="10")

            canvas.configure(yscrollcommand=scrollbar.set)
            canvas_window = canvas.create_window(
                (0, 0),
                window=inner_frame,
                anchor="nw",
            )

            def _on_canvas_configure(event: Event):
                canvas_width = event.width
                canvas.itemconfigure(canvas_window, width=canvas_width)
                canvas.configure(scrollregion=canvas.bbox("all"))

            def _on_mousewheel(event: Event):
                if event.num == 4:
                    delta = -1
                elif event.num == 5:
                    delta = 1
                elif event.delta:
                    delta = -1 * int(event.delta / 120)
                else:
                    delta = 0
                if delta:
                    canvas.yview_scroll(delta, "units")

            canvas.bind("<Configure>", _on_canvas_configure)
            for widget in [canvas, inner_frame]:
                widget.bind("<MouseWheel>", _on_mousewheel, add="+")
                widget.bind("<Button-4>", _on_mousewheel, add="+")
                widget.bind("<Button-5>", _on_mousewheel, add="+")

            canvas.grid(row=0, column=0, sticky="nsew")
            scrollbar.grid(row=0, column=1, sticky="ns")

            return outer_frame, inner_frame

        # CS Durations Section
        cs_outer_frame, cs_inner_frame = create_scrollable_frame(main_frame)
        cs_outer_frame.grid(
            row=0, column=0, pady=(0, 5), padx=5, sticky="nsew"
        )
        cs_inner_frame.columnconfigure(0, weight=0, pad=2)
        cs_inner_frame.columnconfigure(1, weight=1, pad=15)
        cs_inner_frame.columnconfigure(2, weight=0, pad=2)
        cs_inner_frame.columnconfigure(3, weight=1, pad=15)

        ttk.Label(
            cs_inner_frame,
            text="CS Durations (per Node)",
            font="TkDefaultFont 9 bold",
        ).grid(
            row=0,
            column=0,
            columnspan=4,
            pady=(0, 10),
            sticky=tk.W,
        )

        num_left_col_nodes = (num_nodes + 1) // 2
        cs_entries_right_col_temp = {}
        for i in range(num_left_col_nodes):
            current_row = i + 1
            node_id_left = i

            ttk.Label(
                cs_inner_frame,
                text=f"Node {node_id_left}:",
            ).grid(
                row=current_row,
                column=0,
                padx=(0, 2),
                pady=4,
                sticky=tk.W,
            )
            var_left = tk.IntVar(
                value=current_cs_durations.get(node_id_left, default_cs)
            )
            entry_left = ttk.Entry(
                cs_inner_frame, textvariable=var_left, width=8
            )
            entry_left.grid(
                row=current_row, column=1, padx=(0, 5), pady=4, sticky="ew"
            )
            self.cs_duration_vars[node_id_left] = var_left
            self.widgets_in_tab_order.append(entry_left)

            node_id_right = i + num_left_col_nodes
            if node_id_right < num_nodes:
                ttk.Label(
                    cs_inner_frame,
                    text=f"Node {node_id_right}:",
                ).grid(
                    row=current_row,
                    column=2,
                    padx=(5, 2),
                    pady=4,
                    sticky=tk.W,
                )
                var_right = tk.IntVar(
                    value=current_cs_durations.get(node_id_right, default_cs)
                )
                entry_right = ttk.Entry(
                    cs_inner_frame,
                    textvariable=var_right,
                    width=8,
                )
                entry_right.grid(
                    row=current_row,
                    column=3,
                    padx=(0, 5),
                    pady=4,
                    sticky="ew",
                )
                self.cs_duration_vars[node_id_right] = var_right
                cs_entries_right_col_temp[i] = entry_right

        # Add right column CS entries to tab order after left column
        for i in range(num_left_col_nodes):
            if i in cs_entries_right_col_temp:
                self.widgets_in_tab_order.append(cs_entries_right_col_temp[i])

        # Edge Delays Section
        edge_outer_frame, edge_inner_frame = create_scrollable_frame(
            main_frame
        )
        edge_outer_frame.grid(row=1, column=0, pady=5, padx=5, sticky="nsew")
        edge_inner_frame.columnconfigure(0, weight=0, pad=2)
        edge_inner_frame.columnconfigure(1, weight=1, pad=5)
        edge_inner_frame.columnconfigure(2, weight=1, pad=15)
        edge_inner_frame.columnconfigure(3, weight=0, pad=2)
        edge_inner_frame.columnconfigure(4, weight=1, pad=5)
        edge_inner_frame.columnconfigure(5, weight=1, pad=15)

        ttk.Label(
            edge_inner_frame,
            text="Edge Delays (Min / Max)",
            font="TkDefaultFont 9 bold",
        ).grid(row=0, column=0, columnspan=6, pady=(0, 10), sticky=tk.W)

        header_font = "TkDefaultFont 8 bold"
        ttk.Label(edge_inner_frame, text="Edge", font=header_font).grid(
            row=1, column=0, sticky=tk.W
        )
        ttk.Label(edge_inner_frame, text="Min", font=header_font).grid(
            row=1, column=1, sticky=tk.W
        )
        ttk.Label(edge_inner_frame, text="Max", font=header_font).grid(
            row=1, column=2, sticky=tk.W
        )
        ttk.Label(edge_inner_frame, text="Edge", font=header_font).grid(
            row=1, column=3, sticky=tk.W, padx=(5, 0)
        )
        ttk.Label(edge_inner_frame, text="Min", font=header_font).grid(
            row=1, column=4, sticky=tk.W
        )
        ttk.Label(edge_inner_frame, text="Max", font=header_font).grid(
            row=1, column=5, sticky=tk.W
        )

        all_edges = [
            tuple(sorted((u, v)))
            for u in range(num_nodes)
            for v in range(u + 1, num_nodes)
        ]
        total_edges = len(all_edges)
        num_left_col_edges = (total_edges + 1) // 2
        edge_widgets_right_col_temp = {}

        for i in range(num_left_col_edges):
            current_row = i + 2
            edge_key_left = all_edges[i]
            u_left, v_left = edge_key_left
            current_delays_left = current_edge_delays.get(
                edge_key_left, {"min": default_min, "max": default_max}
            )

            ttk.Label(edge_inner_frame, text=f"({u_left},{v_left}):").grid(
                row=current_row, column=0, padx=(0, 5), pady=3, sticky=tk.W
            )
            min_var_left = tk.IntVar(
                value=current_delays_left.get("min", default_min)
            )
            min_entry_left = ttk.Entry(
                edge_inner_frame, textvariable=min_var_left, width=6
            )
            min_entry_left.grid(
                row=current_row, column=1, padx=2, pady=1, sticky="ew"
            )
            self.edge_min_delay_vars[edge_key_left] = min_var_left

            max_var_left = tk.IntVar(
                value=current_delays_left.get("max", default_max)
            )
            max_entry_left = ttk.Entry(
                edge_inner_frame, textvariable=max_var_left, width=6
            )
            max_entry_left.grid(
                row=current_row, column=2, padx=2, pady=1, sticky="ew"
            )
            self.edge_max_delay_vars[edge_key_left] = max_var_left
            self.widgets_in_tab_order.append(min_entry_left)
            self.widgets_in_tab_order.append(max_entry_left)

            edge_idx_right = i + num_left_col_edges
            if edge_idx_right < total_edges:
                edge_key_right = all_edges[edge_idx_right]
                u_right, v_right = edge_key_right
                current_delays_right = current_edge_delays.get(
                    edge_key_right, {"min": default_min, "max": default_max}
                )
                ttk.Label(
                    edge_inner_frame, text=f"({u_right},{v_right}):"
                ).grid(
                    row=current_row, column=3, padx=(5, 5), pady=3, sticky=tk.W
                )
                min_var_right = tk.IntVar(
                    value=current_delays_right.get("min", default_min)
                )
                min_entry_right = ttk.Entry(
                    edge_inner_frame, textvariable=min_var_right, width=6
                )
                min_entry_right.grid(
                    row=current_row, column=4, padx=2, pady=1, sticky="ew"
                )
                self.edge_min_delay_vars[edge_key_right] = min_var_right

                max_var_right = tk.IntVar(
                    value=current_delays_right.get("max", default_max)
                )
                max_entry_right = ttk.Entry(
                    edge_inner_frame, textvariable=max_var_right, width=6
                )
                max_entry_right.grid(
                    row=current_row, column=5, padx=2, pady=1, sticky="ew"
                )
                self.edge_max_delay_vars[edge_key_right] = max_var_right
                edge_widgets_right_col_temp[i] = (
                    min_entry_right,
                    max_entry_right,
                )

        # Add right column edge entries to tab order after left column
        for i in range(num_left_col_edges):
            if i in edge_widgets_right_col_temp:
                min_entry_right, max_entry_right = edge_widgets_right_col_temp[
                    i
                ]
                self.widgets_in_tab_order.append(min_entry_right)
                self.widgets_in_tab_order.append(max_entry_right)

        # Scheduled Requests Section
        req_frame = ttk.LabelFrame(
            main_frame,
            text="Scheduled CS Request Times (comma-separated)",
            padding="10",
        )
        req_frame.grid(row=2, column=0, pady=10, padx=5, sticky="ew")
        req_frame.columnconfigure(1, weight=1)

        for i in range(num_nodes):
            ttk.Label(req_frame, text=f"Node {i}:").grid(
                row=i, column=0, padx=(0, 5), pady=2, sticky=tk.W
            )
            current_times_str = ",".join(
                map(str, sorted(current_scheduled_requests.get(i, [])))
            )
            var = tk.StringVar(value=current_times_str)
            entry = ttk.Entry(req_frame, textvariable=var)
            entry.grid(row=i, column=1, padx=5, pady=2, sticky="ew")
            self.scheduled_request_vars[i] = var
            self.widgets_in_tab_order.append(entry)

        # Button Section
        button_frame = ttk.Frame(main_frame, padding=(0, 5, 0, 0))
        button_frame.grid(row=3, column=0, pady=(5, 0), padx=5, sticky="ew")
        button_frame.columnconfigure(0, weight=1)  # Spacer left
        button_frame.columnconfigure(4, weight=0)  # OK button
        button_frame.columnconfigure(5, weight=0)  # Cancel button
        button_frame.columnconfigure(6, weight=1)  # Spacer right

        load_button = ttk.Button(
            button_frame,
            text="Load JSON...",
            command=self._load_from_json,
            width=12,
        )
        load_button.grid(row=0, column=1, padx=3, pady=3)

        save_button = ttk.Button(
            button_frame,
            text="Save JSON...",
            command=self._save_to_json,
            width=12,
        )
        save_button.grid(row=0, column=2, padx=3, pady=3)

        template_button = ttk.Button(
            button_frame,
            text="Template",
            command=self._show_json_template,
            width=10,
        )
        template_button.grid(row=0, column=3, padx=(3, 15), pady=3)

        ok_button = ttk.Button(
            button_frame, text="OK", command=self._on_ok, width=10
        )
        ok_button.grid(row=0, column=4, padx=3)

        cancel_button = ttk.Button(
            button_frame, text="Cancel", command=self.destroy, width=10
        )
        cancel_button.grid(row=0, column=5, padx=3)

        # Add buttons to tab order
        self.widgets_in_tab_order.append(load_button)
        self.widgets_in_tab_order.append(save_button)
        self.widgets_in_tab_order.append(template_button)
        self.widgets_in_tab_order.append(ok_button)
        self.widgets_in_tab_order.append(cancel_button)

        # Final setup: Set tab order and center window
        logger.info(
            f"Setting tab order for {len(self.widgets_in_tab_order)} widgets."
        )
        for widget in self.widgets_in_tab_order:
            if widget and widget.winfo_exists():
                widget.lift()

        self._center_window(parent)

    def _on_ok(self):
        try:
            # Collect and validate CS durations
            collected_cs_durations = {}
            for nid, var in self.cs_duration_vars.items():
                try:
                    dur = var.get()
                    assert dur >= 0
                    collected_cs_durations[nid] = dur
                except (tk.TclError, ValueError, AssertionError):
                    raise ValueError(f"Invalid CS Duration for Node {nid}")

            # Collect and validate edge delays
            collected_edge_delays = {}
            for edge_key, min_var in self.edge_min_delay_vars.items():
                max_var = self.edge_max_delay_vars[edge_key]
                try:
                    min_d = min_var.get()
                    max_d = max_var.get()
                    assert 0 <= min_d <= max_d
                    collected_edge_delays[edge_key] = {
                        "min": min_d,
                        "max": max_d,
                    }
                except (tk.TclError, ValueError, AssertionError):
                    raise ValueError(
                        f"Invalid Delay (Min/Max>=0, Max>=Min) for Edge {edge_key}"
                    )

            # Collect and validate scheduled requests
            collected_scheduled_requests = {}
            for node_id, str_var in self.scheduled_request_vars.items():
                times_str = str_var.get().strip()
                time_list = []
                if times_str:
                    parts = [
                        p.strip() for p in times_str.split(",") if p.strip()
                    ]
                    for part in parts:
                        try:
                            time_val = int(part)
                            assert time_val >= 0
                            time_list.append(time_val)
                        except (ValueError, AssertionError):
                            raise ValueError(
                                f"Invalid/Negative request time '{part}' for Node {node_id}"
                            )
                collected_scheduled_requests[node_id] = sorted(
                    list(set(time_list))
                )

            # Set result and close window if validation passed
            self.result = {
                "cs_durations": collected_cs_durations,
                "edge_delays": collected_edge_delays,
                "scheduled_requests": collected_scheduled_requests,
            }
            self.destroy()

        except (tk.TclError, ValueError) as e:
            # Handle validation errors
            messagebox.showerror("Validation Error", str(e), parent=self)
            self.result = None

    def _center_window(self, parent_window):
        self.update_idletasks()

        # Get parent window geometry
        parent_width = parent_window.winfo_width()
        parent_height = parent_window.winfo_height()
        parent_x = parent_window.winfo_rootx()
        parent_y = parent_window.winfo_rooty()

        # Get self window geometry
        win_width = self.winfo_width()
        win_height = self.winfo_height()

        # Calculate centered position
        x = parent_x + (parent_width - win_width) // 2
        y = parent_y + (parent_height - win_height) // 2

        # Ensure position is non-negative
        x = max(0, x)
        y = max(0, y)

        # Set the geometry
        self.geometry(f"+{x}+{y}")

    def _load_from_json(self):
        # Ask user for file path
        filepath = filedialog.askopenfilename(
            title="Load Advanced Configuration from JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            parent=self,
        )
        if not filepath:
            return  # User cancelled

        try:
            # Load JSON data from file
            with open(filepath, "r") as f:
                loaded_data = json.load(f)

            # Basic validation: check for required top-level keys
            required_keys = [
                "cs_durations",
                "edge_delays",
                "scheduled_requests",
            ]
            if not all(key in loaded_data for key in required_keys):
                raise ValueError(
                    f"JSON file missing one or more required keys: {required_keys}"
                )

            # Process CS durations
            loaded_cs = loaded_data.get("cs_durations", {})
            if not isinstance(loaded_cs, dict):
                raise ValueError("cs_durations must be a dictionary.")
            for node_id_str, duration in loaded_cs.items():
                try:
                    node_id = int(node_id_str)
                    if node_id not in self.cs_duration_vars:
                        logger.warning(
                            f"Node ID {node_id} from JSON not found in current window config. Skipping."
                        )
                        continue
                    if not isinstance(duration, int) or duration < 0:
                        raise ValueError(
                            f"Invalid duration '{duration}' for Node {node_id}."
                        )
                    self.cs_duration_vars[node_id].set(duration)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Error processing CS duration for key '{node_id_str}': {e}"
                    )

            # Process edge delays
            loaded_edges = loaded_data.get("edge_delays", {})
            if not isinstance(loaded_edges, dict):
                raise ValueError("edge_delays must be a dictionary.")
            for edge_str, delays in loaded_edges.items():
                try:
                    edge_key = _str_to_edge_key(edge_str)
                    if edge_key not in self.edge_min_delay_vars:
                        logger.warning(
                            f"Edge key {edge_key} (from '{edge_str}') not found in current window config. Skipping."
                        )
                        continue
                    if (
                        not isinstance(delays, dict)
                        or "min" not in delays
                        or "max" not in delays
                    ):
                        raise ValueError(
                            "Edge delay entry must be a dict with 'min' and 'max'."
                        )
                    min_d = int(delays["min"])
                    max_d = int(delays["max"])
                    if not 0 <= min_d <= max_d:
                        raise ValueError(
                            f"Invalid min/max delay ({min_d}/{max_d}) for edge {edge_key}."
                        )
                    self.edge_min_delay_vars[edge_key].set(min_d)
                    self.edge_max_delay_vars[edge_key].set(max_d)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Error processing edge delay for key '{edge_str}': {e}"
                    )

            # Process scheduled requests
            loaded_reqs = loaded_data.get("scheduled_requests", {})
            if not isinstance(loaded_reqs, dict):
                raise ValueError("scheduled_requests must be a dictionary.")
            for node_id_str, times_list in loaded_reqs.items():
                try:
                    node_id = int(node_id_str)
                    if node_id not in self.scheduled_request_vars:
                        logger.warning(
                            f"Node ID {node_id} for requests not found in current window config. Skipping."
                        )
                        continue
                    if not isinstance(times_list, list):
                        raise ValueError(
                            "Scheduled request times must be a list."
                        )
                    valid_times = []
                    for t in times_list:
                        if not isinstance(t, int) or t < 0:
                            raise ValueError(
                                f"Invalid scheduled time '{t}' for Node {node_id}."
                            )
                        valid_times.append(t)
                    times_str = ",".join(
                        map(str, sorted(list(set(valid_times))))
                    )
                    self.scheduled_request_vars[node_id].set(times_str)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Error processing scheduled requests for key '{node_id_str}': {e}"
                    )

            # Show success message
            messagebox.showinfo(
                "Load Success",
                "Configuration loaded successfully from JSON.",
                parent=self,
            )

        # Handle potential errors during file loading/parsing/validation
        except FileNotFoundError:
            messagebox.showerror(
                "Load Error", f"File not found: {filepath}", parent=self
            )
        except json.JSONDecodeError as e:
            messagebox.showerror(
                "Load Error", f"Invalid JSON file: {e}", parent=self
            )
        except (ValueError, TypeError) as e:
            messagebox.showerror(
                "Load Error",
                f"Error loading or validating data: {e}",
                parent=self,
            )
        except Exception as e:
            messagebox.showerror(
                "Load Error",
                f"An unexpected error occurred during loading: {e}",
                parent=self,
            )

    def _save_to_json(self):
        try:
            # Collect and validate CS durations
            collected_cs_durations = {}
            for nid, var in self.cs_duration_vars.items():
                try:
                    dur = var.get()
                    if not isinstance(dur, int) or dur < 0:
                        raise ValueError("Must be non-negative integer.")
                    collected_cs_durations[str(nid)] = dur
                except (tk.TclError, ValueError) as e:
                    raise ValueError(
                        f"Invalid CS Duration for Node {nid}: {e}"
                    )

            # Collect and validate edge delays
            collected_edge_delays_str_keys = {}
            for edge_key, min_var in self.edge_min_delay_vars.items():
                max_var = self.edge_max_delay_vars[edge_key]
                try:
                    min_d = min_var.get()
                    max_d = max_var.get()
                    if not isinstance(min_d, int) or not isinstance(
                        max_d, int
                    ):
                        raise ValueError("Must be integers.")
                    if not 0 <= min_d <= max_d:
                        raise ValueError("Must have 0 <= Min <= Max.")
                    edge_str = _edge_key_to_str(edge_key)
                    collected_edge_delays_str_keys[edge_str] = {
                        "min": min_d,
                        "max": max_d,
                    }
                except (tk.TclError, ValueError) as e:
                    raise ValueError(f"Invalid Delay for Edge {edge_key}: {e}")

            # Collect and validate scheduled requests
            collected_scheduled_requests = {}
            for node_id, str_var in self.scheduled_request_vars.items():
                times_str = str_var.get().strip()
                time_list = []
                if times_str:
                    parts = [
                        p.strip() for p in times_str.split(",") if p.strip()
                    ]
                    for part in parts:
                        try:
                            time_val = int(part)
                            if time_val < 0:
                                raise ValueError("Times cannot be negative.")
                            time_list.append(time_val)
                        except (ValueError, TypeError):
                            raise ValueError(
                                f"Invalid request time '{part}' for Node {node_id}"
                            )
                collected_scheduled_requests[str(node_id)] = sorted(
                    list(set(time_list))
                )

            # Assemble the final JSON data structure
            json_data = {
                "cs_durations": collected_cs_durations,
                "edge_delays": collected_edge_delays_str_keys,
                "scheduled_requests": collected_scheduled_requests,
                "metadata": {
                    "num_nodes": self.num_nodes,
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            }

            # Ask for file path and save
            filepath = filedialog.asksaveasfilename(
                title="Save Advanced Configuration to JSON",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile=f"config_nodes_{self.num_nodes}.json",
                parent=self,
            )
            if not filepath:
                return  # User cancelled

            with open(filepath, "w") as f:
                json.dump(json_data, f, indent=4)

            messagebox.showinfo(
                "Save Success",
                f"Configuration saved successfully to:\n{filepath}",
                parent=self,
            )

        # Handle potential errors during validation or saving
        except (ValueError, tk.TclError) as e:
            messagebox.showerror(
                "Validation Error",
                f"Cannot save due to invalid input: {e}",
                parent=self,
            )
        except IOError as e:
            messagebox.showerror(
                "Save Error", f"Could not write to file: {e}", parent=self
            )
        except Exception as e:
            messagebox.showerror(
                "Save Error",
                f"An unexpected error occurred during saving: {e}",
                parent=self,
            )

    def _show_json_template(self):
        import datetime

        # Define the template data structure
        template_data = {
            "cs_durations": {
                "0": 100,
                "3": 10,
                "1": 30,
                "4": 50,
                "2": 20,
            },
            "edge_delays": {
                "0,1": {"min": 20, "max": 20},
                "1,3": {"min": 20, "max": 20},
                "0,2": {"min": 20, "max": 20},
                "1,4": {"min": 20, "max": 20},
                "0,3": {"min": 20, "max": 20},
                "2,3": {"min": 20, "max": 20},
                "0,4": {"min": 20, "max": 20},
                "2,4": {"min": 20, "max": 20},
                "1,2": {"min": 20, "max": 20},
                "3,4": {"min": 20, "max": 20},
            },
            "scheduled_requests": {
                "0": [45],
                "1": [5],
                "2": [400],
                "3": [60],
                "4": [10],
            },
            "metadata": {
                "num_nodes": 5,
                "saved_at": datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            },
        }

        try:
            # Create and display the template window
            json_string = json.dumps(template_data, indent=4)

            template_win = Toplevel(self)
            template_win.title("JSON Configuration Template")
            template_win.transient(self)
            template_win.grab_set()
            template_win.geometry("500x500")

            text_area = scrolledtext.ScrolledText(
                template_win, wrap=tk.WORD, font=("Courier New", 9)
            )
            text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            text_area.insert(tk.END, json_string)
            text_area.config(state=tk.DISABLED)

            close_button = ttk.Button(
                template_win, text="Close", command=template_win.destroy
            )
            close_button.pack(pady=5)

            # Center the template window relative to the parent
            self.update_idletasks()
            parent_x = self.winfo_rootx()
            parent_y = self.winfo_rooty()
            parent_w = self.winfo_width()
            parent_h = self.winfo_height()
            win_w = template_win.winfo_reqwidth()
            win_h = template_win.winfo_reqheight()
            x = parent_x + parent_w // 2 - win_w // 2
            y = parent_y + parent_h // 2 - win_h // 2
            template_win.geometry(f"+{max(0, x)}+{max(0, y)}")

        except Exception as e:
            messagebox.showerror(
                "Template Error",
                f"Could not generate or display template: {e}",
                parent=self,
            )


class Node:
    def __init__(self, node_id, num_nodes, cs_duration):
        self.id = node_id
        self.num_nodes = num_nodes
        self.cs_duration = int(cs_duration)
        self.state = NodeState.IDLE
        self.clock = 0
        self.request_ts = -1, -1
        self.outstanding_replies = set()
        self.deferred_queue = []

    def update_clock(self, received_ts=None):
        if received_ts is not None:
            self.clock = max(self.clock, received_ts) + 1
        else:
            self.clock += 1
        return self.clock

    def __repr__(self):
        state_char = str(self.state)
        req_str = (
            f"({self.request_ts[0]},{self.request_ts[1]})"
            if self.request_ts[0] != -1
            else "(-)"
        )
        return f"N({self.id},S:{state_char},C:{self.clock},D:{self.cs_duration},Req:{req_str},Out:{len(self.outstanding_replies)},Def:{len(self.deferred_queue)})"


class Simulation:
    def __init__(
        self,
        num_nodes,
        node_cs_durations,
        edge_delays,
        scheduled_requests,
        vis_callback=None,
    ):
        self.num_nodes = num_nodes
        self._node_cs_durations = copy.deepcopy(node_cs_durations)
        self._edge_delays = copy.deepcopy(edge_delays)
        self._scheduled_requests = copy.deepcopy(scheduled_requests)
        self.vis_callback = vis_callback

        self.nodes = {
            i: Node(
                i,
                num_nodes,
                self._node_cs_durations.get(i, DEFAULT_CS_DURATION),
            )
            for i in range(num_nodes)
        }
        self.graph = self._create_graph()
        self.event_queue = []
        self.current_time = 0
        self.event_counter = 0
        self.history_data = []
        self.node_positions = None

        try:
            if self.graph and self.graph.number_of_nodes() > 0:
                logger.debug("Using spring_layout")
                self.node_positions = nx.spring_layout(
                    self.graph, seed=GRAPH_LAYOUT_SEED
                )
                logger.debug(
                    f"(Sim Init): Type of self.node_positions: {type(self.node_positions)}"
                )
                if (
                    isinstance(self.node_positions, dict)
                    and self.node_positions
                ):
                    logger.debug(
                        f"(Sim Init): Positions data sample: {dict(list(self.node_positions.items())[:3])}"
                    )
                    first_pos_value = list(self.node_positions.values())[0]
                    logger.debug(
                        f"(Sim Init): Type of first position value: {type(first_pos_value)}"
                    )
                    if hasattr(first_pos_value, "dtype"):
                        logger.debug(
                            f"(Sim Init): Dtype of first pos value: {first_pos_value.dtype}"
                        )
                    if (
                        isinstance(first_pos_value, (list, tuple))
                        and len(first_pos_value) == 2
                    ):
                        logger.debug(
                            f"(Sim Init): Types in first pos value: ({type(first_pos_value[0])}, {type(first_pos_value[1])})"
                        )
                logger.info(
                    f"Sim Init: Generated node positions using seed {GRAPH_LAYOUT_SEED}."
                )
            elif self.graph:
                self.node_positions = {}
                logger.info(
                    "Sim Init: Graph has 0 nodes, positions dictionary is empty."
                )
            else:
                self.node_positions = {}
                logger.warning(
                    "Sim Init: Graph object not created, positions dictionary is empty."
                )
        except Exception as e:
            logger.error(
                f"Sim Init: Failed to generate node positions using nx.spring_layout: {e}"
            )
            self.node_positions = {}

        logger.info("Sim Init: Scheduling initial requests...")
        self._initialize_scheduled_requests()
        logger.info("Sim Init: Complete.")
        self._log_state(EventType.INIT, f"Initialized {num_nodes} nodes.", [])
        self._update_visualization()

    def _get_current_node_snapshots(self):
        snapshots = {}
        for node_id, node in self.nodes.items():
            snapshots[node_id] = {
                "state": node.state,
                "clock": node.clock,
                "request_ts": node.request_ts,
            }
        return snapshots

    def _log_state(self, event_type, details, involved_node_ids):
        current_snapshots = self._get_current_node_snapshots()
        log_entry = {
            "time": int(self.current_time),
            "type": event_type,
            "details": details,
            "involved": involved_node_ids,
            "node_snapshots": current_snapshots,
        }
        self.history_data.append(log_entry)

    def _initialize_scheduled_requests(self):
        logger.info(f"  Scheduled Requests Config: {self._scheduled_requests}")
        for node_id, times in self._scheduled_requests.items():
            if 0 <= node_id < self.num_nodes:
                if isinstance(times, (list, tuple)):
                    for time_val in times:
                        if isinstance(time_val, int) and time_val >= 0:
                            logger.info(
                                f"    Scheduling N{node_id} request @ T={time_val}"
                            )
                            self._schedule_event(
                                time_val, EventType.SCHEDULED_REQUEST, node_id
                            )
                        else:
                            logger.warning(
                                f"    Invalid time value '{time_val}' for scheduled request Node {node_id}. Skipping."
                            )
                elif times:
                    logger.warning(
                        f"    Invalid format for scheduled request times for Node {node_id}. Expected list/tuple, got {type(times)}. Skipping."
                    )
            else:
                logger.warning(
                    f"    Invalid node_id {node_id} found in scheduled requests config. Skipping."
                )

    def _create_graph(self):
        try:
            if self.num_nodes <= 0:
                logger.warning(
                    "Graph Creation: Cannot create graph with 0 or negative nodes."
                )
                return nx.Graph()
            G = nx.complete_graph(self.num_nodes)
            logger.info(
                f"Creating Graph ({self.num_nodes} nodes), setting edge attributes..."
            )
            for u, v in G.edges():
                edge_key = tuple(sorted((u, v)))
                delays = self._edge_delays.get(
                    edge_key,
                    {"min": DEFAULT_MIN_DELAY, "max": DEFAULT_MAX_DELAY},
                )
                G.edges[u, v]["min_delay"] = int(
                    delays.get("min", DEFAULT_MIN_DELAY)
                )
                G.edges[u, v]["max_delay"] = int(
                    delays.get("max", DEFAULT_MAX_DELAY)
                )
                G.edges[u, v]["in_transit"] = None
            return G
        except Exception as e:
            logger.error(f"Error creating graph: {e}")
            return None

    def _get_delay(self, u, v):
        if u == v:
            return 0
        try:
            edge_data = self.graph.edges[u, v]
        except KeyError:
            logger.warning(
                f"Edge ({u}, {v}) not found in graph for delay lookup. Using default delay 1."
            )
            return 1
        min_d = edge_data.get("min_delay", DEFAULT_MIN_DELAY)
        max_d = edge_data.get("max_delay", DEFAULT_MAX_DELAY)
        try:
            actual_min = min(min_d, max_d)
            actual_max = max(min_d, max_d)
            if actual_min == actual_max:
                return actual_min
            else:
                return random.randint(actual_min, actual_max)
        except ValueError:
            logger.warning(
                f"Invalid delay values ({min_d}, {max_d}) for edge ({u},{v}). Using min value."
            )
            return min(min_d, max_d)

    def _schedule_event(self, delay_or_time, event_type, data):
        event_time = (
            int(delay_or_time)
            if self.current_time == 0
            else self.current_time + int(delay_or_time)
        )
        if event_time < self.current_time:
            logger.warning(
                f"Attempted to schedule {event_type} in the past (T={event_time} < Current T={self.current_time}). Skipping."
            )
            return
        self.event_counter += 1
        heapq.heappush(
            self.event_queue,
            (event_time, self.event_counter, event_type, data),
        )

    def _send_message(
        self, sender_id, receiver_id, msg_type, msg_ts, request_ts=None
    ):
        if sender_id == receiver_id:
            logger.warning(
                f"Attempt to send message from N{sender_id} to itself. Skipping."
            )
            return

        delay = self._get_delay(sender_id, receiver_id)
        arrival_time = self.current_time + delay
        message_data = msg_type, sender_id, msg_ts, request_ts
        event_data = receiver_id, message_data, (sender_id, receiver_id)

        req_ts_str = f", ReqTS={request_ts}" if request_ts else ""
        logger.info(
            f"T={self.current_time}: SEND N{sender_id}->N{receiver_id} ({msg_type.name}) TS={msg_ts}, Delay:{delay}, Arrival:{arrival_time}{req_ts_str}"
        )

        transit_entry = (
            msg_type,
            sender_id,
            receiver_id,
            self.current_time,
            arrival_time,
        )
        edge_tuple = sender_id, receiver_id

        try:
            current_transit_list = self.graph.edges[edge_tuple].get(
                "in_transit"
            )
            if current_transit_list is None:
                self.graph.edges[edge_tuple]["in_transit"] = [transit_entry]
            elif isinstance(current_transit_list, list):
                current_transit_list.append(transit_entry)
            else:
                logger.warning(
                    f"Corrupted 'in_transit' data on edge {edge_tuple} was not None or list ({type(current_transit_list)}). Overwriting."
                )
                self.graph.edges[edge_tuple]["in_transit"] = [transit_entry]
        except KeyError:
            logger.error(
                f"Edge {edge_tuple} not found when trying to set in_transit."
            )

        self._schedule_event(delay, EventType.MESSAGE_ARRIVAL, event_data)
        self._update_visualization()

    def want_cs(self, node_id):
        node = self.nodes.get(node_id)
        if not node:
            logger.error(f"Node {node_id} not found in want_cs.")
            return False
        if node.state != NodeState.IDLE:
            logger.info(
                f"T={self.current_time}: N{node_id} cannot want CS (State: {node.state.name})."
            )
            return False

        logger.info(f"T={self.current_time}: N{node_id} wants CS.")
        node.state = NodeState.WANTED
        node.clock = node.update_clock()
        node.request_ts = node.clock, node.id
        node.outstanding_replies = set(range(self.num_nodes)) - {node_id}
        node.deferred_queue = []

        if not node.outstanding_replies:
            logger.info(
                f"  -> N{node_id} is the only node, entering CS immediately."
            )
            self._enter_cs(node_id)
        else:
            logger.info(
                f"  -> N{node_id} broadcasting REQUESTS (ReqTS: {node.request_ts})."
            )
            for other_id in range(self.num_nodes):
                if other_id != node_id:
                    self._send_message(
                        sender_id=node_id,
                        receiver_id=other_id,
                        msg_type=MessageType.REQUEST,
                        msg_ts=node.clock,
                        request_ts=node.request_ts,
                    )

        self._update_visualization()
        return True

    def _handle_message_arrival(self, event_data):
        receiver_id, message, original_edge = event_data
        msg_type, sender_id, msg_ts, request_ts = message
        arrival_time = self.current_time
        receiver_node = self.nodes.get(receiver_id)
        edge_tuple = original_edge

        # Remove message from in-transit visualization list
        try:
            transit_list = self.graph.edges[edge_tuple].get("in_transit")
            if transit_list and isinstance(transit_list, list):
                found_idx = -1
                for idx, item in enumerate(transit_list):
                    if (
                        len(item) == 5
                        and item[0] == msg_type
                        and item[1] == sender_id
                        and item[2] == receiver_id
                        and item[4] == arrival_time
                    ):
                        found_idx = idx
                        break
                if found_idx != -1:
                    transit_list.pop(found_idx)
                else:
                    logger.warning(
                        f"Arrived message ({msg_type.name} N{sender_id}->N{receiver_id} @T={arrival_time}) not found in transit list for edge {edge_tuple}. List: {transit_list}"
                    )
            elif transit_list:
                logger.error(
                    f"'in_transit' for edge {edge_tuple} is not a list: {transit_list}. Clearing."
                )
                self.graph.edges[edge_tuple]["in_transit"] = None
        except KeyError:
            logger.error(
                f"Edge {edge_tuple} not found when trying to access in_transit list for arrival."
            )
        except Exception as e:
            logger.error(
                f"Unexpected error clearing transit list for edge {edge_tuple}: {e}"
            )

        if not receiver_node:
            logger.error(
                f"Receiver node {receiver_id} not found for message arrival. Message details: {message}"
            )
            self._update_visualization()
            return []

        req_ts_str = f" ReqTS:{request_ts}" if request_ts else ""
        logger.info(
            f"T={self.current_time}: RECV N{receiver_id}<-N{sender_id} ({msg_type.name}) MsgTS:{msg_ts}{req_ts_str}"
        )
        receiver_node.clock = receiver_node.update_clock(received_ts=msg_ts)
        logger.info(
            f"  -> N{receiver_id} Clock updated to {receiver_node.clock}"
        )

        if msg_type == MessageType.REQUEST:
            self._handle_request(receiver_node, sender_id, request_ts)
        elif msg_type == MessageType.REPLY:
            self._handle_reply(receiver_node, sender_id)

        details = f"{msg_type.name} N{sender_id}->N{receiver_id} processed."
        involved = [receiver_id, sender_id]
        self._log_state(EventType.MESSAGE_ARRIVAL, details, involved)
        self._update_visualization()
        return involved

    def _handle_scheduled_request(self, node_id):
        logger.info(
            f"T={self.current_time}: Handling scheduled request for N{node_id}."
        )
        success = self.want_cs(node_id)
        details = f"Node {node_id} scheduled request triggered." + (
            " (Initiated)" if success else " (Failed - Not IDLE)"
        )
        self._log_state(EventType.SCHEDULED_REQUEST, details, [node_id])
        return [node_id]

    def _handle_request(self, receiver_node, sender_id, sender_request_ts):
        should_reply_immediately = False
        receiver_id = receiver_node.id
        my_req_ts = receiver_node.request_ts

        if receiver_node.state == NodeState.HELD:
            logger.info(
                f"  -> N{receiver_id} is HELD, deferring N{sender_id}."
            )
        elif receiver_node.state == NodeState.WANTED:
            if sender_request_ts < my_req_ts:
                logger.info(
                    f"  -> N{receiver_id} is WANTED, N{sender_id}'s request {sender_request_ts} has priority over mine {my_req_ts}. Replying."
                )
                should_reply_immediately = True
            else:
                logger.info(
                    f"  -> N{receiver_id} is WANTED, my request {my_req_ts} has priority over N{sender_id}'s {sender_request_ts}. Deferring."
                )
        else:  # NodeState.IDLE
            logger.info(
                f"  -> N{receiver_id} is IDLE. Replying to N{sender_id}."
            )
            should_reply_immediately = True

        if should_reply_immediately:
            reply_ts = receiver_node.update_clock()
            logger.info(
                f"    -> N{receiver_id} sending REPLY to N{sender_id} (Clock now {reply_ts})."
            )
            self._send_message(
                sender_id=receiver_id,
                receiver_id=sender_id,
                msg_type=MessageType.REPLY,
                msg_ts=reply_ts,
            )
        elif sender_id not in receiver_node.deferred_queue:
            receiver_node.deferred_queue.append(sender_id)
            logger.info(
                f"  -> N{receiver_id} added N{sender_id} to deferred queue: {receiver_node.deferred_queue}"
            )
        else:
            logger.warning(
                f"  -> N{receiver_id} N{sender_id} already in deferred queue. {receiver_node.deferred_queue}"
            )

    def _handle_reply(self, receiver_node, sender_id):
        receiver_id = receiver_node.id
        if receiver_node.state == NodeState.WANTED:
            if sender_id in receiver_node.outstanding_replies:
                receiver_node.outstanding_replies.remove(sender_id)
                remaining_count = len(receiver_node.outstanding_replies)
                logger.info(
                    f"  -> N{receiver_id} got needed REPLY from N{sender_id}. Remaining replies needed: {remaining_count}"
                )
                if not receiver_node.outstanding_replies:
                    self._enter_cs(receiver_id)
            else:
                logger.warning(
                    f"  -> N{receiver_id} got unexpected/duplicate REPLY from N{sender_id}. Outstanding set: {receiver_node.outstanding_replies}. Ignored."
                )
        else:
            logger.warning(
                f"  -> N{receiver_id} got REPLY from N{sender_id} but is not in WANTED state (State: {receiver_node.state.name}). Ignored."
            )

    def _enter_cs(self, node_id):
        node = self.nodes.get(node_id)
        if not node:
            logger.error(f"Node {node_id} not found when trying to enter CS.")
            return
        if node.state != NodeState.WANTED:
            logger.warning(
                f"N{node_id} entering CS from unexpected state {node.state.name}!"
            )

        node.state = NodeState.HELD
        duration = node.cs_duration
        logger.info(
            f"T={self.current_time}: +++ N{node_id} ENTER CS (Duration: {duration}) +++"
        )
        details = f"Node {node_id} entered CS (Duration: {duration})"
        self._log_state(EventType.CS_ENTER, details, [node_id])
        self._schedule_event(duration, EventType.CS_EXIT, node_id)
        self._update_visualization()

    def _handle_cs_exit(self, node_id):
        node = self.nodes.get(node_id)
        if not node:
            logger.error(f"Node {node_id} not found when trying to exit CS.")
            return []
        if node.state != NodeState.HELD:
            logger.error(
                f"CS_EXIT event for N{node_id} but state is {node.state.name}! Ignoring exit."
            )
            details = f"Node {node_id} CS_EXIT event ignored (State was {node.state.name})"
            self._log_state(EventType.ERROR, details, [node_id])
            self._update_visualization()
            return [node_id]

        logger.info(f"T={self.current_time}: --- N{node_id} EXIT CS ---")
        node.state = NodeState.IDLE
        involved_nodes = [node_id]
        deferred_copy = list(node.deferred_queue)
        node.deferred_queue = []
        details = f"Node {node_id} exited CS."

        if deferred_copy:
            details += f" Sending {len(deferred_copy)} deferred replies to {deferred_copy}."
            involved_nodes.extend(deferred_copy)
            logger.info(
                f"  -> N{node_id} sending deferred REPLYs to: {deferred_copy}"
            )
            reply_ts = node.update_clock()
            logger.info(
                f"  -> N{node_id} Clock updated to {reply_ts} for deferred replies."
            )
            for waiting_node_id in deferred_copy:
                self._send_message(
                    sender_id=node.id,
                    receiver_id=waiting_node_id,
                    msg_type=MessageType.REPLY,
                    msg_ts=reply_ts,
                )
        else:
            details += " No deferred requests."

        node.request_ts = -1, -1
        self._log_state(EventType.CS_EXIT, details, involved_nodes)
        self._update_visualization()
        return involved_nodes

    def step(self):
        if not self.event_queue:
            logger.info("Event queue empty. No step taken.")
            return False

        event_time, _, event_type, event_data = heapq.heappop(self.event_queue)

        if event_time < self.current_time:
            logger.warning(
                f"Skipping past event {event_type} scheduled for T={event_time} (Current T={self.current_time})"
            )
            return True

        if event_time > self.current_time:
            self.current_time = event_time

        logger.info(
            f"--- Processing Event {event_type} at T={self.current_time} ---"
        )
        try:
            if event_type == EventType.MESSAGE_ARRIVAL:
                self._handle_message_arrival(event_data)
            elif event_type == EventType.CS_EXIT:
                self._handle_cs_exit(event_data)
            elif event_type == EventType.SCHEDULED_REQUEST:
                self._handle_scheduled_request(event_data)
            else:
                details = (
                    f"Unknown Event Type {event_type}, Data: {event_data}"
                )
                logger.error(details)
                self._log_state(EventType.UNKNOWN, details, [])
        except Exception as e:
            details = (
                f"Error processing {event_type} for data {event_data}: {e}"
            )
            logger.error(details)
            involved_guess = []
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
            self._log_state(
                EventType.ERROR, details, list(set(involved_guess))
            )

        logger.info(f"--- Finished Event {event_type} ---")
        return True

    def advance_time_by(self, amount):
        if not isinstance(amount, int) or amount <= 0:
            logger.warning(
                f"Advance time amount must be a positive integer, got: {amount}."
            )
            return False

        target_time = self.current_time + amount
        processed_count = 0
        initial_time = self.current_time
        logger.info(
            f"--- Advancing time from {initial_time} up to {target_time} (Amount: {amount}) ---"
        )

        while self.event_queue and self.event_queue[0][0] <= target_time:
            if self.step():
                processed_count += 1
            else:
                logger.warning("Simulation step failed during time advance.")
                break

        log_detail = f"Advanced time from T={int(initial_time)} to T={int(self.current_time)} (processed {processed_count} events)."
        if self.current_time < target_time:
            logger.info(
                f"--- Jumping time from {self.current_time} to target {target_time} ---"
            )
            self.current_time = target_time
            log_detail = f"Advanced time from T={int(initial_time)} to T={int(target_time)} (processed {processed_count} events)."

        logger.info(
            f"--- Advancing Done. Current T={self.current_time}. Processed {processed_count} events. ---"
        )
        self._log_state(EventType.TIME_ADVANCE, log_detail, [])
        self._update_visualization()
        return True

    def advance_single_time_unit(self):
        return self.advance_time_by(1)

    def _update_visualization(self):
        if self.vis_callback:
            try:
                if self.node_positions is None:
                    logger.warning(
                        "_update_visualization called but node_positions is None. Skipping callback."
                    )
                    return
                self.vis_callback(
                    self.graph,
                    self.nodes,
                    int(self.current_time),
                    self.node_positions,
                )
            except Exception as e:
                logger.error(f"Error during visualization callback: {e}")


class RicartAgrawalaGUI:
    def __init__(self, master):
        self.master = master
        master.title("Ricart-Agrawala Simulation")
        master.geometry("1200x1200")
        master.minsize(width=800, height=600)

        # Simulation state variables
        self.simulation = None
        self.node_cs_durations = {}
        self.edge_delays = {}
        self.scheduled_requests = {}
        self.dragged_node = None
        self.node_positions_cache = {}

        # Top Paned Window (Config + Control)
        top_pane = tk.PanedWindow(
            master,
            orient=tk.HORIZONTAL,
            sashrelief=tk.RAISED,
            bd=2,
            sashwidth=6,
        )
        top_pane.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(5, 2))

        # Configuration Frame (Left side of top pane)
        config_frame = ttk.LabelFrame(
            top_pane, text="Configuration", padding="10"
        )
        top_pane.add(config_frame, stretch="always", minsize=350)

        config_grid_frame = ttk.Frame(config_frame)
        config_grid_frame.pack(fill=tk.X, expand=True)

        ttk.Label(config_grid_frame, text="Nodes:").grid(
            row=0, column=0, padx=3, pady=3, sticky=tk.W
        )
        self.num_nodes_var = tk.IntVar(value=DEFAULT_NUM_NODES)
        self.num_nodes_entry = ttk.Entry(
            config_grid_frame, textvariable=self.num_nodes_var, width=5
        )
        self.num_nodes_entry.grid(row=0, column=1, padx=3, pady=3, sticky=tk.W)

        ttk.Label(config_grid_frame, text="Default CS Dur:").grid(
            row=0, column=2, padx=(10, 3), pady=3, sticky=tk.W
        )
        self.default_cs_var = tk.IntVar(value=DEFAULT_CS_DURATION)
        self.default_cs_entry = ttk.Entry(
            config_grid_frame, textvariable=self.default_cs_var, width=7
        )
        self.default_cs_entry.grid(
            row=0, column=3, padx=3, pady=3, sticky=tk.W
        )

        ttk.Label(config_grid_frame, text="Default Min Delay:").grid(
            row=1, column=0, padx=3, pady=3, sticky=tk.W
        )
        self.default_min_delay_var = tk.IntVar(value=DEFAULT_MIN_DELAY)
        self.default_min_delay_entry = ttk.Entry(
            config_grid_frame, textvariable=self.default_min_delay_var, width=5
        )
        self.default_min_delay_entry.grid(
            row=1, column=1, padx=3, pady=3, sticky=tk.W
        )

        ttk.Label(config_grid_frame, text="Default Max Delay:").grid(
            row=1, column=2, padx=(10, 3), pady=3, sticky=tk.W
        )
        self.default_max_delay_var = tk.IntVar(value=DEFAULT_MAX_DELAY)
        self.default_max_delay_entry = ttk.Entry(
            config_grid_frame, textvariable=self.default_max_delay_var, width=7
        )
        self.default_max_delay_entry.grid(
            row=1, column=3, padx=3, pady=3, sticky=tk.W
        )

        config_grid_frame.columnconfigure(1, pad=10)
        config_grid_frame.columnconfigure(3, pad=10)

        config_button_frame = ttk.Frame(config_frame)
        config_button_frame.pack(fill=tk.X, pady=(8, 0), padx=0)

        self.init_button = ttk.Button(
            config_button_frame,
            text="Initialize / Reset",
            command=self.initialize_simulation,
        )
        self.init_button.pack(side=tk.LEFT, padx=(0, 5))

        self.adv_config_button = ttk.Button(
            config_button_frame,
            text="Advanced Config...",
            command=self.open_advanced_config,
        )
        self.adv_config_button.pack(side=tk.LEFT, padx=5)

        # Simulation Control Frame (Right side of top pane)
        step_control_frame = ttk.LabelFrame(
            top_pane, text="Simulation Control", padding="10"
        )
        top_pane.add(step_control_frame, stretch="never", minsize=220)

        self.step_event_button = ttk.Button(
            step_control_frame,
            text="Step -> (Next Event)",
            command=self.step_simulation_event,
            state=tk.DISABLED,
        )
        self.step_event_button.grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=3
        )

        self.step_1_time_button = ttk.Button(
            step_control_frame,
            text="Advance by 1 Time",
            command=self.step_simulation_1_time,
            state=tk.DISABLED,
        )
        self.step_1_time_button.grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=3
        )

        self.advance_time_var = tk.IntVar(value=10)
        self.advance_time_entry = ttk.Entry(
            step_control_frame, textvariable=self.advance_time_var, width=6
        )
        self.advance_time_entry.grid(
            row=2, column=0, sticky="ew", padx=5, pady=3
        )
        self.advance_time_entry.config(state=tk.DISABLED)

        self.advance_time_button = ttk.Button(
            step_control_frame,
            text="Advance Time",
            command=self.step_simulation_by_amount,
            state=tk.DISABLED,
        )
        self.advance_time_button.grid(
            row=2, column=1, sticky="ew", padx=5, pady=3
        )

        step_control_frame.columnconfigure(0, weight=1)
        step_control_frame.columnconfigure(1, weight=1)

        # Node Request Frame (Below top pane)
        self.node_frame = ttk.Frame(master, padding="5")
        self.node_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(2, 2))

        ttk.Label(self.node_frame, text="Request CS Now:").pack(
            side=tk.LEFT, padx=(0, 2)
        )

        self.node_canvas = tk.Canvas(
            self.node_frame, height=35, borderwidth=0, highlightthickness=0
        )
        try:
            canvas_bg = ttk.Style().lookup("TFrame", "background")
            self.node_canvas.config(background=canvas_bg)
        except tk.TclError:
            pass

        self.node_scrollbar = ttk.Scrollbar(
            self.node_frame,
            orient=tk.HORIZONTAL,
            command=self.node_canvas.xview,
        )
        self.node_canvas.configure(xscrollcommand=self.node_scrollbar.set)
        self.node_scrollbar.pack(side=tk.BOTTOM, fill=tk.X, padx=0)
        self.node_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.node_buttons_frame = ttk.Frame(self.node_canvas, padding=0)
        self.canvas_window_id = self.node_canvas.create_window(
            (0, 0), window=self.node_buttons_frame, anchor="nw"
        )

        # --- Canvas scroll configuration ---
        def _configure_canvas_scrollregion(event):
            bbox = self.node_canvas.bbox("all")
            if bbox:
                self.node_canvas.configure(scrollregion=bbox)

        def _configure_canvas_window(event):
            self.node_canvas.itemconfig(
                self.canvas_window_id, height=event.height
            )

        self.node_buttons_frame.bind(
            "<Configure>", _configure_canvas_scrollregion
        )
        self.node_canvas.bind("<Configure>", _configure_canvas_window)

        def _on_mousewheel_horizontal(event):
            if event.delta:
                delta = -1 * int(event.delta / 120)
            elif event.num == 4:
                delta = -1
            elif event.num == 5:
                delta = 1
            else:
                delta = 0
            if delta:
                self.node_canvas.xview_scroll(delta, "units")

        for widget in [self.node_canvas, self.node_buttons_frame]:
            widget.bind("<MouseWheel>", _on_mousewheel_horizontal, add="+")
            widget.bind(
                "<Button-4>",
                lambda e: self.node_canvas.xview_scroll(-1, "units"),
                add="+",
            )
            widget.bind(
                "<Button-5>",
                lambda e: self.node_canvas.xview_scroll(1, "units"),
                add="+",
            )
        # --- End Canvas scroll configuration ---

        self.node_buttons = {}

        # Main Notebook (Simulation View, State Table)
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(
            side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=(0, 5)
        )

        # Tab 1: Simulation View and Log
        tab1_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab1_frame, text=" Simulation View ", padding=2)

        vis_log_pane = tk.PanedWindow(
            tab1_frame,
            orient=tk.VERTICAL,
            sashrelief=tk.RAISED,
            bd=2,
            sashwidth=6,
        )
        vis_log_pane.pack(fill=tk.BOTH, expand=True)

        # Visualization Frame (Top of Tab 1)
        vis_frame = ttk.Frame(vis_log_pane)
        vis_log_pane.add(vis_frame, stretch="always", minsize=300, height=450)

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01)
        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.ax.set_title("Network State (Initialize Simulation)")
        self.ax.axis("off")

        # Connect matplotlib events for dragging nodes
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        # Log Frame (Bottom of Tab 1)
        log_frame = ttk.LabelFrame(
            vis_log_pane, text="Event History Log", padding="5"
        )
        vis_log_pane.add(log_frame, stretch="never", minsize=100, height=200)

        self.history_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            height=10,
            state=tk.DISABLED,
            font=("Courier New", 9),
        )
        self.history_text.pack(fill=tk.BOTH, expand=True)

        # Tab 2: State History Table
        tab2_frame = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(tab2_frame, text=" State History Table ", padding=2)

        style = ttk.Style()
        style.configure("Treeview.Heading", font=("TkDefaultFont", 9, "bold"))
        self.state_table = ttk.Treeview(tab2_frame, show="headings")

        vsb = ttk.Scrollbar(
            tab2_frame, orient="vertical", command=self.state_table.yview
        )
        hsb = ttk.Scrollbar(
            tab2_frame, orient="horizontal", command=self.state_table.xview
        )
        self.state_table.configure(
            yscrollcommand=vsb.set, xscrollcommand=hsb.set
        )

        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.state_table.pack(side="left", fill="both", expand=True)

        # Define initial columns for the state table
        self.state_table["columns"] = "Time", "Event", "Details"
        self.state_table.heading("Time", text="Time")
        self.state_table.column(
            "Time", width=50, anchor="center", stretch=False
        )
        self.state_table.heading("Event", text="Event")
        self.state_table.column("Event", width=100, anchor="w", stretch=False)
        self.state_table.heading("Details", text="Details")
        self.state_table.column("Details", width=250, anchor="w", stretch=True)

        # Status Bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(
            master,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(5, 2),
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var.set(
            "Ready. Configure defaults or use Advanced Config, then Initialize / Reset."
        )

        # Initial setup
        self._populate_default_configs()
        self._disable_controls()

    def update_graph_visualization(
        self, graph, nodes, current_time, positions
    ):
        if not self.canvas_widget.winfo_exists():
            return

        if not isinstance(positions, dict):
            logger.warning(
                f"Invalid or missing node positions (expected dict, got {type(positions)}), skipping visualization."
            )
            self.ax.clear()
            self.ax.set_title(
                f"Error: Invalid Positions Data at T={current_time}"
            )
            self.ax.text(
                0.5,
                0.5,
                "Invalid node positions data received.",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                color="red",
            )
            self.ax.axis("off")
            try:
                self.canvas.draw_idle()
            except Exception:
                pass
            return

        self.ax.clear()
        logger.debug(f"(GUI Vis): Received positions type: {type(positions)}")
        if isinstance(positions, dict) and positions:
            logger.debug(
                f"(GUI Vis): Received positions data sample: {dict(list(positions.items())[:3])}"
            )
            first_pos_value = list(positions.values())[0]
            logger.debug(
                f"(GUI Vis): Type of first received position value: {type(first_pos_value)}"
            )
            if hasattr(first_pos_value, "dtype"):
                logger.debug(
                    f"(GUI Vis): Dtype of first received pos value: {first_pos_value.dtype}"
                )
            if (
                isinstance(first_pos_value, (list, tuple))
                and len(first_pos_value) == 2
            ):
                logger.debug(
                    f"(GUI Vis): Types in first received pos value: ({type(first_pos_value[0])}, {type(first_pos_value[1])})"
                )

        # Filter for valid positions before drawing
        valid_positions = {}
        logger.debug(
            f"(GUI Vis): Filtering {len(positions)} incoming positions..."
        )
        for nid, pos in positions.items():
            is_valid = False
            pos_tuple = None
            try:
                if isinstance(pos, (list, tuple)) and len(pos) == 2:
                    if (
                        isinstance(pos[0], (int, float))
                        and np.isfinite(pos[0])
                        and isinstance(pos[1], (int, float))
                        and np.isfinite(pos[1])
                    ):
                        is_valid = True
                        pos_tuple = tuple(pos)
                elif isinstance(pos, np.ndarray) and pos.shape == (2,):
                    if (
                        np.issubdtype(pos.dtype, np.number)
                        and np.isfinite(pos).all()
                    ):
                        is_valid = True
                        pos_tuple = tuple(pos)
            except Exception as e:
                logger.debug(
                    f"Error checking position for node {nid}: {pos} - {e}"
                )
                is_valid = False

            if is_valid:
                valid_positions[nid] = pos_tuple
            else:
                logger.debug(
                    f"Filtering out invalid position for node {nid}: {pos} (type: {type(pos)})"
                )

        self.node_positions_cache = valid_positions

        if not valid_positions and graph and graph.nodes():
            logger.warning(
                "No valid node positions found to draw, although graph has nodes."
            )
            self.ax.set_title(f"Error: No Valid Positions at T={current_time}")
            self.ax.text(
                0.5,
                0.5,
                "No valid node positions found.",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                color="red",
            )
            self.ax.axis("off")
            try:
                self.canvas.draw_idle()
            except Exception:
                pass
            return

        valid_node_list = list(valid_positions.keys())

        # Prepare node colors and labels based on state
        node_colors = []
        node_labels = {}
        held_node_exists = False
        for node_id in valid_node_list:
            node = nodes.get(node_id)
            if node and isinstance(node, Node):
                state = node.state
                req_ts_str = ""
                if (
                    isinstance(node.request_ts, tuple)
                    and len(node.request_ts) == 2
                    and state in (NodeState.WANTED, NodeState.HELD)
                    and node.request_ts[0] != -1
                ):
                    req_ts_str = (
                        f"\nR:({node.request_ts[0]},{node.request_ts[1]})"
                    )
                node_labels[node_id] = (
                    f"N{node_id}\nC:{node.clock}\nD:{node.cs_duration}{req_ts_str}"
                )
                if state == NodeState.IDLE:
                    node_colors.append("lightblue")
                elif state == NodeState.WANTED:
                    node_colors.append("yellow")
                else:
                    node_colors.append("limegreen")
                    held_node_exists = True
            else:
                node_labels[node_id] = f"N{node_id}\n(Error!)"
                node_colors.append("red")
                logger.warning(
                    f"Node object missing or invalid for ID {node_id} with valid position during visualization."
                )

        # Draw nodes
        nx.draw_networkx_nodes(
            graph,
            valid_positions,
            nodelist=valid_node_list,
            ax=self.ax,
            node_color=node_colors,
            node_size=1800,
            edgecolors="black",
            linewidths=1.0,
        )

        # Draw node labels
        nx.draw_networkx_labels(
            graph,
            valid_positions,
            labels=node_labels,
            ax=self.ax,
            font_size=7,
            font_weight="normal",
        )

        # Draw edges
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

        # Draw arrows for messages in transit
        for u, v, data in graph.edges(data=True):
            if u not in valid_positions or v not in valid_positions:
                continue

            transit_list = data.get("in_transit")
            if transit_list and isinstance(transit_list, list):
                for message_index, transit_info in enumerate(transit_list):
                    try:
                        if (
                            not isinstance(transit_info, tuple)
                            or len(transit_info) != 5
                        ):
                            logger.warning(
                                f"Invalid transit_info format on edge ({u},{v}): {transit_info}"
                            )
                            continue

                        (
                            msg_type,
                            sender_id,
                            receiver_id,
                            send_time,
                            arrival_time,
                        ) = transit_info

                        if (
                            sender_id not in valid_positions
                            or receiver_id not in valid_positions
                        ):
                            continue

                        pos_sender = valid_positions[sender_id]
                        pos_receiver = valid_positions[receiver_id]

                        # Calculate arrow position based on progress
                        total_duration = float(arrival_time - send_time)
                        progress = (
                            1.0
                            if total_duration <= 0
                            else max(
                                0.0,
                                min(
                                    1.0,
                                    float(current_time - send_time)
                                    / total_duration,
                                ),
                            )
                        )

                        # Calculate arrow geometry
                        dx = pos_receiver[0] - pos_sender[0]
                        dy = pos_receiver[1] - pos_sender[1]
                        edge_len = math.hypot(dx, dy)
                        if edge_len < 1e-06:
                            continue
                        ux, uy = dx / edge_len, dy / edge_len

                        # Adjust start/end points to be outside node radius
                        start_base_x = pos_sender[0] + ux * NODE_RADIUS_VISUAL
                        start_base_y = pos_sender[1] + uy * NODE_RADIUS_VISUAL
                        end_base_x = pos_receiver[0] - ux * NODE_RADIUS_VISUAL
                        end_base_y = pos_receiver[1] - uy * NODE_RADIUS_VISUAL

                        # Calculate position along the shortened edge segment
                        vec_short_x = end_base_x - start_base_x
                        vec_short_y = end_base_y - start_base_y
                        pos_on_segment_x = (
                            start_base_x + vec_short_x * progress
                        )
                        pos_on_segment_y = (
                            start_base_y + vec_short_y * progress
                        )

                        # Offset arrow perpendicularly for visibility
                        perp_dx, perp_dy = -uy, ux
                        offset_x = perp_dx * ARROW_OFFSET_AMOUNT
                        offset_y = perp_dy * ARROW_OFFSET_AMOUNT
                        final_arrow_head_x = pos_on_segment_x + offset_x
                        final_arrow_head_y = pos_on_segment_y + offset_y

                        # Calculate tail position based on head and direction
                        final_arrow_tail_x = (
                            final_arrow_head_x - ux * ARROW_LENGTH
                        )
                        final_arrow_tail_y = (
                            final_arrow_head_y - uy * ARROW_LENGTH
                        )

                        # Draw the arrow
                        arrow_draw_dx = ux * ARROW_LENGTH
                        arrow_draw_dy = uy * ARROW_LENGTH
                        arrow_color = (
                            "mediumblue"
                            if msg_type == MessageType.REQUEST
                            else "darkmagenta"
                        )
                        self.ax.arrow(
                            final_arrow_tail_x,
                            final_arrow_tail_y,
                            arrow_draw_dx,
                            arrow_draw_dy,
                            head_width=ARROW_HEAD_WIDTH,
                            head_length=ARROW_HEAD_LENGTH,
                            length_includes_head=True,
                            fc=arrow_color,
                            ec=arrow_color,
                            lw=1,
                            alpha=0.95,
                        )
                    except KeyError as e:
                        logger.warning(
                            f"Error drawing arrow edge ({u},{v}). Missing position data for node {e}?"
                        )
                    except Exception as e:
                        logger.error(
                            f"Unexpected error drawing arrow edge ({u},{v}), message {message_index}: {e}"
                        )

        # Draw edge delay labels
        try:
            edge_lbls = {
                (u, v): f"{d.get('min_delay', '?')}-{d.get('max_delay', '?')}"
                for (u, v, d) in graph.edges(data=True)
                if u in valid_positions and v in valid_positions
            }
            if edge_lbls:
                nx.draw_networkx_edge_labels(
                    graph,
                    valid_positions,
                    edge_labels=edge_lbls,
                    ax=self.ax,
                    font_size=6,
                    font_color="dimgrey",
                    label_pos=0.3,
                    bbox=dict(fc="white", alpha=0.4, ec="none", pad=0.1),
                    rotate=False,
                )
        except Exception as e:
            logger.warning(f"Could not draw edge labels: {e}")

        # Set title and finalize plot
        title = f"Ricart-Agrawala State at T = {current_time}"
        if held_node_exists:
            title += " (CS Held)"
        if self.dragged_node is not None:
            title += f" (Dragging N{self.dragged_node})"
        self.ax.set_title(title, fontsize=10)
        self.ax.axis("off")
        self.ax.autoscale_view()

        try:
            self.canvas.draw_idle()
        except Exception as e:
            logger.error(f"Canvas draw_idle error: {e}")

    def _on_press(self, event):
        # Check if click is within the axes and simulation is running
        if (
            event.inaxes != self.ax
            or not self.simulation
            or not self.node_positions_cache
            or not event.xdata
            or not event.ydata
        ):
            self.dragged_node = None
            return

        x, y = event.xdata, event.ydata
        min_dist_sq = NODE_CLICK_TOLERANCE**2
        clicked_node = None

        # Find the closest node within tolerance
        for node_id, pos in self.node_positions_cache.items():
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                dist_sq = (pos[0] - x) ** 2 + (pos[1] - y) ** 2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    clicked_node = node_id

        # If a node is clicked, start dragging
        if clicked_node is not None:
            self.dragged_node = clicked_node
            logger.debug(f"Dragging Node {self.dragged_node}")
            # Update visualization to show dragging state in title
            self.update_graph_visualization(
                self.simulation.graph,
                self.simulation.nodes,
                int(self.simulation.current_time),
                self.simulation.node_positions,
            )
        else:
            self.dragged_node = None

    def _on_motion(self, event):
        # Check if a node is being dragged and mouse is in axes
        if (
            self.dragged_node is None
            or event.inaxes != self.ax
            or not event.xdata
            or not event.ydata
        ):
            return

        x, y = event.xdata, event.ydata

        # Update the position of the dragged node in the simulation's position data
        if self.simulation and hasattr(self.simulation, "node_positions"):
            if self.dragged_node in self.simulation.node_positions:
                self.simulation.node_positions[self.dragged_node] = x, y
                # Redraw the graph with the updated position
                self.update_graph_visualization(
                    self.simulation.graph,
                    self.simulation.nodes,
                    int(self.simulation.current_time),
                    self.simulation.node_positions,
                )
            else:
                logger.warning(
                    f"Node {self.dragged_node} being dragged is no longer in simulation positions. Stopping drag."
                )
                self.dragged_node = None
        else:
            logger.warning(
                "Simulation object missing during drag motion. Stopping drag."
            )
            self.dragged_node = None

    def _on_release(self, event):
        # If a node was being dragged, stop dragging
        if self.dragged_node is not None:
            logger.debug(f"Finished dragging Node {self.dragged_node}")
            self.dragged_node = None
            # Redraw to remove the "Dragging" text from the title
            if self.simulation:
                self.update_graph_visualization(
                    self.simulation.graph,
                    self.simulation.nodes,
                    int(self.simulation.current_time),
                    self.simulation.node_positions,
                )

    def _populate_default_configs(self):
        # Reads default values from GUI entries and populates internal config dicts
        try:
            num_nodes = self.num_nodes_var.get()
            default_cs = self.default_cs_var.get()
            default_min = self.default_min_delay_var.get()
            default_max = self.default_max_delay_var.get()

            # Basic validation and clamping
            if num_nodes <= 0:
                num_nodes = 1
            if default_cs < 0:
                default_cs = 0
            if default_min < 0:
                default_min = 0
            if default_max < default_min:
                default_max = default_min

            # Update GUI vars if clamped
            self.num_nodes_var.set(num_nodes)
            self.default_cs_var.set(default_cs)
            self.default_min_delay_var.set(default_min)
            self.default_max_delay_var.set(default_max)

            # Populate internal dictionaries
            self.node_cs_durations = {i: default_cs for i in range(num_nodes)}
            self.edge_delays = {
                tuple(sorted((u, v))): {"min": default_min, "max": default_max}
                for u in range(num_nodes)
                for v in range(num_nodes)
                if u < v
            }
            self.scheduled_requests = {i: [] for i in range(num_nodes)}

            logger.info(
                "Populated internal configs based on current default values."
            )
            return True
        except (tk.TclError, ValueError) as e:
            messagebox.showerror(
                "Configuration Error", f"Invalid default value entered: {e}"
            )
            logger.error(
                f"Failed to read or validate default config values: {e}"
            )
            return False

    def _enable_controls(self):
        # Enables simulation control buttons and node request buttons
        self.step_event_button.config(state=tk.NORMAL)
        self.step_1_time_button.config(state=tk.NORMAL)
        self.advance_time_button.config(state=tk.NORMAL)
        self.advance_time_entry.config(state=tk.NORMAL)
        self.update_request_buttons_state()  # Enables relevant node buttons

    def _disable_controls(self):
        # Disables simulation control buttons and all node request buttons
        self.step_event_button.config(state=tk.DISABLED)
        self.step_1_time_button.config(state=tk.DISABLED)
        self.advance_time_button.config(state=tk.DISABLED)
        self.advance_time_entry.config(state=tk.DISABLED)
        for btn in self.node_buttons.values():
            if btn.winfo_exists():
                btn.config(state=tk.DISABLED)

    def open_advanced_config(self):
        try:
            # Get current number of nodes from GUI
            num_nodes = self.num_nodes_var.get()
            if num_nodes <= 0:
                messagebox.showerror(
                    "Error",
                    "Number of nodes must be greater than 0.",
                    parent=self.master,
                )
                return

            # Check if node count changed since last config population
            current_config_nodes = len(self.node_cs_durations)
            if current_config_nodes != num_nodes:
                logger.info(
                    f"Node count changed ({current_config_nodes} -> {num_nodes}). Repopulating configs before Advanced window."
                )
                # Repopulate internal configs using current default values
                if not self._populate_default_configs():
                    messagebox.showerror(
                        "Config Error",
                        "Failed to update internal configs for new node count based on default values.",
                        parent=self.master,
                    )
                    return
                # Double-check repopulation worked
                if len(self.node_cs_durations) != num_nodes:
                    raise ValueError(
                        "Internal configuration size mismatch persisted after repopulation."
                    )

            # Get default values for the advanced window
            default_cs = self.default_cs_var.get()
            default_min = self.default_min_delay_var.get()
            default_max = self.default_max_delay_var.get()

            # Create and show the advanced config window
            config_window = AdvancedConfigWindow(
                self.master,
                num_nodes,
                copy.deepcopy(self.node_cs_durations),
                copy.deepcopy(self.edge_delays),
                copy.deepcopy(self.scheduled_requests),
                default_cs,
                default_min,
                default_max,
            )
            self.master.wait_window(config_window)  # Wait for it to close

            # Apply results if OK was clicked
            if config_window.result:
                self.node_cs_durations = config_window.result["cs_durations"]
                self.edge_delays = config_window.result["edge_delays"]
                self.scheduled_requests = config_window.result[
                    "scheduled_requests"
                ]
                self.status_var.set(
                    "Advanced configuration updated. Click Initialize / Reset."
                )
                logger.info("Advanced configuration applied.")
            else:
                self.status_var.set(
                    "Advanced configuration cancelled or closed."
                )
                logger.info("Advanced configuration cancelled or closed.")

        except (tk.TclError, ValueError) as e:
            messagebox.showerror(
                "Configuration Error",
                f"Error preparing advanced config: {e}",
                parent=self.master,
            )

    def initialize_simulation(self):
        logger.info("\n" + "=" * 10 + " INITIALIZE / RESET " + "=" * 10 + "\n")
        try:
            # Get node count and validate
            num_nodes = self.num_nodes_var.get()
            if num_nodes <= 0:
                raise ValueError("Number of nodes must be greater than 0.")

            # Final check: ensure internal configs match GUI node count
            if len(self.node_cs_durations) != num_nodes:
                logger.info(
                    f"Node count mismatch on Init ({len(self.node_cs_durations)} vs {num_nodes}). Repopulating configs."
                )
                if not self._populate_default_configs():
                    raise ValueError(
                        "Failed to populate configs based on defaults during final init check."
                    )
                if len(self.node_cs_durations) != num_nodes:
                    raise ValueError(
                        "Config mismatch persisted after repopulation during Init."
                    )

            # Reset state
            self.simulation = None
            self.dragged_node = None
            self.node_positions_cache = {}

            # Clear visualization
            if hasattr(self, "ax") and self.ax:
                self.ax.clear()
                self.ax.set_title("Network State (Initializing...)")
                self.ax.axis("off")
            if hasattr(self, "canvas") and self.canvas:
                try:
                    self.canvas.draw_idle()
                except Exception as e:
                    logger.warning(
                        f"Ignoring error during init canvas clear: {e}"
                    )

            # Clear history log
            if (
                hasattr(self, "history_text")
                and self.history_text.winfo_exists()
            ):
                self.history_text.config(state=tk.NORMAL)
                self.history_text.delete("1.0", tk.END)
                self.history_text.config(state=tk.DISABLED)

            # Clear state table
            if (
                hasattr(self, "state_table")
                and self.state_table.winfo_exists()
            ):
                for item in self.state_table.get_children():
                    self.state_table.delete(item)

            # Create the new Simulation instance
            logger.info("Creating new Simulation object...")
            self.simulation = Simulation(
                num_nodes=num_nodes,
                node_cs_durations=copy.deepcopy(self.node_cs_durations),
                edge_delays=copy.deepcopy(self.edge_delays),
                scheduled_requests=copy.deepcopy(self.scheduled_requests),
                vis_callback=self.update_graph_visualization,
            )
            logger.info("Simulation object created.")
            if not self.simulation.node_positions:
                logger.warning(
                    "Simulation initialized but failed to generate node positions."
                )

            # Recreate node request buttons
            if (
                hasattr(self, "node_buttons_frame")
                and self.node_buttons_frame.winfo_exists()
            ):
                # Destroy old buttons first
                for widget in self.node_buttons_frame.winfo_children():
                    if isinstance(widget, ttk.Button):
                        widget.destroy()
                self.node_buttons = {}
                # Create new buttons
                for i in range(num_nodes):
                    btn = ttk.Button(
                        self.node_buttons_frame,
                        text=f"N{i}",
                        width=4,
                        command=lambda node_id=i: self.request_cs_for_node(
                            node_id
                        ),
                    )
                    btn.pack(side=tk.LEFT, padx=2, pady=2)
                    self.node_buttons[i] = btn
                # Update canvas scroll region after adding buttons
                self.master.update_idletasks()
                bbox = self.node_canvas.bbox("all")
                if bbox:
                    self.node_canvas.configure(scrollregion=bbox)
                self.node_canvas.xview_moveto(0)
            else:
                logger.error(
                    "Node button frame container (inside canvas) does not exist."
                )

            # Configure the state table columns for the new number of nodes
            self._configure_state_table(num_nodes)

            # Enable controls and update UI
            self._enable_controls()
            self.status_var.set(
                f"Initialized {num_nodes} nodes at T=0. Ready."
            )
            self._update_history_log()
            self._update_state_table()
            self._update_status_and_buttons()  # Redundant call, but ensures consistency

        except (tk.TclError, ValueError, AttributeError) as e:
            # Handle initialization errors
            messagebox.showerror(
                "Initialization Error",
                f"Failed to initialize simulation: {e}",
                parent=self.master,
            )
            self.status_var.set("Initialization failed.")
            self._disable_controls()
            self.simulation = None

            # Reset UI elements to reflect failed state
            if hasattr(self, "ax") and self.ax:
                self.ax.clear()
                self.ax.set_title("Network State (Initialization Failed)")
                self.ax.axis("off")
            if hasattr(self, "canvas") and self.canvas:
                try:
                    self.canvas.draw_idle()
                except Exception:
                    pass
            if (
                hasattr(self, "history_text")
                and self.history_text.winfo_exists()
            ):
                self.history_text.config(state=tk.NORMAL)
                self.history_text.delete("1.0", tk.END)
                self.history_text.config(state=tk.DISABLED)
            if (
                hasattr(self, "state_table")
                and self.state_table.winfo_exists()
            ):
                for item in self.state_table.get_children():
                    self.state_table.delete(item)

            # Log traceback for debugging
            import traceback

            logger.error("--- Initialization Error Traceback ---")
            logger.error(traceback.format_exc())
            logger.error("------------------------------------")

    def _configure_state_table(self, num_nodes):
        logger.debug(f"Configuring state table for {num_nodes} nodes.")
        if (
            not hasattr(self, "state_table")
            or not self.state_table.winfo_exists()
        ):
            logger.debug("State table widget not found, cannot configure.")
            return

        # Clear existing rows
        try:
            items = self.state_table.get_children()
            if items:
                self.state_table.delete(*items)
        except Exception as e:
            logger.debug(f"Error clearing table items: {e}")

        # Clear existing columns (important for re-initialization)
        try:
            current_cols = self.state_table["columns"]
            if current_cols:
                logger.debug(f"Clearing existing columns: {current_cols}")
                # Resetting columns directly is safer than trying to modify headings of non-existent columns
                self.state_table["columns"] = ()
        except Exception as e:
            logger.debug(f"Error during column clearing: {e}")

        # Define new column identifiers
        col_ids = ["#time", "#event", "#details"]
        for i in range(num_nodes):
            col_ids.append(f"#node{i}")
        logger.debug(f"Setting table columns to: {col_ids}")

        # Set columns and configure headings/widths
        try:
            self.state_table["columns"] = tuple(col_ids)

            # Configure standard columns
            self.state_table.heading("#time", text="Time")
            self.state_table.column(
                "#time", width=40, minwidth=30, anchor="e", stretch=False
            )
            self.state_table.heading("#event", text="Event")
            self.state_table.column(
                "#event", width=80, minwidth=60, anchor="w", stretch=False
            )
            self.state_table.heading("#details", text="Details")
            self.state_table.column(
                "#details", width=200, minwidth=100, anchor="w", stretch=True
            )

            # Configure node-specific columns
            node_col_width = 80
            for i in range(num_nodes):
                col_id = f"#node{i}"
                self.state_table.heading(col_id, text=f"Node {i}")
                self.state_table.column(
                    col_id,
                    width=node_col_width,
                    minwidth=60,
                    anchor="center",
                    stretch=False,
                )
            logger.debug("State table columns configured successfully.")
        except Exception as e:
            logger.error(f"Error setting/configuring columns: {e}")

    def _update_history_log(self):
        # Updates the scrolled text widget with simulation history
        if (
            not hasattr(self, "history_text")
            or not self.history_text.winfo_exists()
        ):
            return

        try:
            self.history_text.config(state=tk.NORMAL)
            self.history_text.delete("1.0", tk.END)

            if self.simulation and self.simulation.history_data:
                for entry in self.simulation.history_data:
                    involved_str = (
                        f"(Inv: {entry['involved']})"
                        if entry["involved"]
                        else ""
                    )
                    log_str = f"T={entry['time']:<4}: [{str(entry['type']):<10}] {entry['details']} {involved_str}\n"
                    self.history_text.insert(tk.END, log_str)
                self.history_text.see(tk.END)  # Scroll to the end

        except Exception as e:
            logger.error(f"Error updating text log: {e}")
        finally:
            # Ensure text widget is disabled after update
            if (
                hasattr(self, "history_text")
                and self.history_text.winfo_exists()
            ):
                self.history_text.config(state=tk.DISABLED)

    def _update_state_table(self):
        logger.debug("Updating state table...")
        if (
            not hasattr(self, "state_table")
            or not self.state_table.winfo_exists()
        ):
            logger.debug("State table widget does not exist. Skipping update.")
            return

        try:
            # Clear existing rows
            items = self.state_table.get_children()
            if items:
                self.state_table.delete(*items)
            logger.debug(f"Cleared {len(items)} rows.")

            if self.simulation and self.simulation.history_data:
                num_nodes = self.simulation.num_nodes
                history_len = len(self.simulation.history_data)
                logger.debug(f"Simulation has {history_len} history entries.")

                # Get the expected columns from the table itself
                cols = list(self.state_table["columns"])
                logger.debug(f"Expected table columns: {cols}")

                # Iterate through history and add rows
                for entry_idx, entry in enumerate(
                    self.simulation.history_data
                ):
                    # Prepare basic values
                    values = [
                        entry["time"],
                        str(entry["type"]),
                        entry["details"],
                    ]

                    # Get node snapshots for this entry
                    snapshots = entry.get("node_snapshots", {})

                    # Append node state strings for each node column
                    for i in range(num_nodes):
                        node_col_id = f"#node{i}"
                        if node_col_id in cols:  # Check if column exists
                            snapshot = snapshots.get(i)
                            if snapshot:
                                state = snapshot.get("state", "?")
                                clock = snapshot.get("clock", -1)
                                req_ts = snapshot.get("request_ts", (-1, -1))

                                # Format the state string
                                state_str = f"{state}[{clock}]"
                                if (
                                    isinstance(req_ts, tuple)
                                    and len(req_ts) == 2
                                    and state
                                    in (NodeState.WANTED, NodeState.HELD)
                                    and req_ts[0] != -1
                                ):
                                    state_str += f" R({req_ts[0]},{req_ts[1]})"
                                values.append(state_str)
                            else:
                                values.append("N/A")  # Node snapshot missing
                        # If column doesn't exist (shouldn't happen if configured correctly),
                        # we implicitly skip it to avoid errors, but log a warning below.

                    # Debugging output for the first row
                    if entry_idx == 0:
                        logger.debug(f"First row values prepared: {values}")
                        logger.debug(
                            f"Length of values: {len(values)}, Expected columns: {len(cols)}"
                        )

                    # Insert row if value count matches column count
                    if len(values) == len(cols):
                        tag = "evenrow" if entry_idx % 2 == 0 else "oddrow"
                        if entry_idx == 0:
                            logger.debug("Attempting to insert first row...")
                        self.state_table.insert(
                            "", tk.END, values=tuple(values), tags=(tag,)
                        )
                        if entry_idx == 0:
                            logger.debug("First row insertion call completed.")
                    else:
                        # Log mismatch error
                        logger.warning(
                            f"Column count mismatch row {entry_idx}. Expected {len(cols)}, Got {len(values)}. Row skipped."
                        )
                        logger.warning(f"  Columns: {cols}")
                        logger.warning(f"  Values: {values}")

                # Apply alternating row colors after inserting all rows
                self.master.update_idletasks()  # Ensure table is updated before tagging
                logger.debug(f"Finished processing {history_len} entries.")
                self.state_table.tag_configure("oddrow", background="white")
                self.state_table.tag_configure("evenrow", background="#f0f0f0")

            else:
                logger.debug("No simulation or no history data found.")

        except Exception as e:
            logger.error(f"ERROR during updating state table: {e}")
            import traceback

            logger.error("--- State Table Update Traceback ---")
            logger.error(traceback.format_exc())
            logger.error("------------------------------------")

    def _update_status_and_buttons(self):
        # Updates the status bar text and enables/disables control buttons
        if self.simulation:
            time_str = f"T={int(self.simulation.current_time)}"
            queue_empty = not self.simulation.event_queue
            status_msg = f"Sim Active: {time_str}."

            # Update status based on event queue
            if queue_empty:
                status_msg += " Event queue empty."
                self.step_event_button.config(state=tk.DISABLED)
            else:
                next_event_time = self.simulation.event_queue[0][0]
                status_msg += f" Next event at T={next_event_time}."
                self.step_event_button.config(state=tk.NORMAL)

            # Enable time advance controls
            self.step_1_time_button.config(state=tk.NORMAL)
            self.advance_time_button.config(state=tk.NORMAL)
            self.advance_time_entry.config(state=tk.NORMAL)

            # Set status bar text
            self.status_var.set(status_msg)

            # Update node request button states
            self.update_request_buttons_state()
        else:
            # Simulation not running state
            self.status_var.set(
                "Simulation not initialized. Click Initialize / Reset."
            )
            self._disable_controls()

    def step_simulation_event(self):
        # Executes the next event in the simulation queue
        if not self.simulation:
            messagebox.showerror(
                "Error", "Simulation not initialized.", parent=self.master
            )
            return

        if not self.simulation.event_queue:
            self.status_var.set(
                f"T={int(self.simulation.current_time)}. No events in queue to process."
            )
            messagebox.showinfo(
                "Step Event", "Event queue is empty.", parent=self.master
            )
            return

        logger.info("Stepping by next event...")
        self.simulation.step()

        # Update UI after step
        self._update_history_log()
        self._update_state_table()
        self._update_status_and_buttons()

    def step_simulation_1_time(self):
        # Advances simulation time by 1 unit, processing any events within that unit
        if not self.simulation:
            messagebox.showerror(
                "Error", "Simulation not initialized.", parent=self.master
            )
            return

        logger.info("Advancing by 1 time unit...")
        self.simulation.advance_single_time_unit()

        # Update UI after advance
        self._update_history_log()
        self._update_state_table()
        self._update_status_and_buttons()

    def step_simulation_by_amount(self):
        # Advances simulation time by a user-specified amount
        if not self.simulation:
            messagebox.showerror(
                "Error", "Simulation not initialized.", parent=self.master
            )
            return

        # Get and validate amount from entry widget
        try:
            amount = self.advance_time_var.get()
            if amount <= 0:
                messagebox.showwarning(
                    "Input Error",
                    "Advance time amount must be a positive integer.",
                    parent=self.master,
                )
                return
        except tk.TclError:
            messagebox.showerror(
                "Input Error",
                "Invalid advance time amount entered.",
                parent=self.master,
            )
            return

        logger.info(f"Advancing by {amount} time units...")
        self.simulation.advance_time_by(amount)

        # Update UI after advance
        self._update_history_log()
        self._update_state_table()
        self._update_status_and_buttons()

    def request_cs_for_node(self, node_id):
        # Handles clicks on the "Request CS Now" buttons for individual nodes
        if not self.simulation:
            messagebox.showerror(
                "Error", "Simulation not initialized.", parent=self.master
            )
            return

        node = self.simulation.nodes.get(node_id)

        # Check if node exists and is IDLE
        if node and node.state == NodeState.IDLE:
            logger.info(
                f"Manual CS request for N{node_id} initiated by GUI button."
            )
            # Log the manual request event
            self.simulation._log_state(
                EventType.MANUAL_REQUEST,
                f"Manual CS request button for N{node_id} pressed.",
                [node_id],
            )
            # Initiate the want_cs process in the simulation
            success = self.simulation.want_cs(node_id)
            if not success:
                # This shouldn't happen if the state check passed, but handle defensively
                logger.warning(
                    f"Manual req N{node_id} failed unexpectedly after IDLE check."
                )
                messagebox.showwarning(
                    "Request Failed",
                    f"Node {node_id} could not initiate request unexpectedly.",
                    parent=self.master,
                )
            # Update UI after request attempt
            self._update_history_log()
            self._update_state_table()
            self._update_status_and_buttons()
        elif node:
            # Node is not IDLE, show info message
            messagebox.showinfo(
                "Request Blocked",
                f"Node {node_id} cannot request CS now.\nCurrent State: {node.state.name}",
                parent=self.master,
            )
        else:
            # Node ID not found (shouldn't happen with correct initialization)
            messagebox.showerror(
                "Error",
                f"Node {node_id} not found in simulation.",
                parent=self.master,
            )

        # Ensure button states are correct after the action
        self._update_status_and_buttons()

    def update_request_buttons_state(self):
        # Enables/disables individual node request buttons based on node state
        if not self.simulation or not hasattr(self.simulation, "nodes"):
            # Disable all if simulation isn't running
            for i, button in self.node_buttons.items():
                if button.winfo_exists():
                    button.config(state=tk.DISABLED)
            return

        # Check each node's state
        for i, button in self.node_buttons.items():
            if not button.winfo_exists():
                continue
            node = self.simulation.nodes.get(i)
            # Enable button only if node exists and is IDLE
            can_request = node is not None and node.state == NodeState.IDLE
            button.config(state=tk.NORMAL if can_request else tk.DISABLED)


if __name__ == "__main__":
    root = None
    logger.info("Initializing GUI...")
    try:
        # Try using themed Tkinter if available
        if ttk_themes_available:
            root = ThemedTk(theme="arc")
            logger.info("Using ttkthemes theme 'arc'.")
        else:
            logger.info("ttkthemes not found, using standard tkinter styling.")
            root = tk.Tk()
    except Exception as e:
        # Fallback to basic Tkinter if themed fails
        logger.error(
            f"ERROR initializing Tkinter root window (themed or standard): {e}"
        )
        logger.info("Falling back to basic tk.Tk()...")
        try:
            root = tk.Tk()
        except Exception as final_e:
            # Fatal error if even basic Tkinter fails
            logger.critical(
                f"FATAL: Failed to create even basic tk.Tk() root window: {final_e}"
            )
            import sys

            sys.exit("Could not initialize GUI.")

    # If root window was created successfully, start the app
    if root:
        app = RicartAgrawalaGUI(root)

        # Define closing behavior
        def on_closing():
            if messagebox.askokcancel(
                "Quit", "Do you want to quit the simulation?"
            ):
                logger.info("Exiting application.")
                root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)
        logger.info("Starting Tkinter mainloop...")
        root.mainloop()
        logger.info("Tkinter mainloop finished.")
    else:
        logger.critical(
            "Failed to create Tk root window. Application cannot start."
        )
