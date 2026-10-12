
import io
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload


APP_TITLE = "Google Drive Road Image Encoder"
DEFAULT_SCHEMA_FILE = "Attributes_Schema.json"

# Change these if needed
SOURCE_FOLDER_ID = ""
ENCODED_FOLDER_ID = ""

ANNOTATIONS_FILENAME = "annotations.json"
MAX_IMAGE_SIZE = (700, 400)

SCOPES = ["https://www.googleapis.com/auth/drive"]


@dataclass
class DriveImageItem:
    file_id: str
    name: str
    mime_type: str


class DriveAnnotationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1300x750")

        self.base_dir = os.getcwd()
        self.schema_path = os.path.join(self.base_dir, DEFAULT_SCHEMA_FILE)

        self.schema = self.load_or_select_schema()
        self.fields = self.schema.get("fields", [])

        self.drive_service = self.get_drive_service()

        self.annotations_file_id: Optional[str] = None
        self.annotations = self.load_annotations_from_drive()

        self.source_images: List[DriveImageItem] = []
        self.current_index = -1
        self.current_item: Optional[DriveImageItem] = None
        self.current_photo = None
        self.widgets: Dict[str, Dict] = {}
        self.focusable_widgets = []

        self.build_ui()
        self.refresh_source_images()

        if self.source_images:
            self.load_random_image(exclude_current=False)
        else:
            messagebox.showinfo(
                "No images found",
                "No image files were found in the source Google Drive folder."
            )

    def get_drive_service(self):
        creds = None

        if os.path.exists("token.json"):
            try:
                creds = Credentials.from_authorized_user_file("token.json", SCOPES)
            except Exception:
                try:
                    os.remove("token.json")
                except OSError:
                    pass
                creds = None

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists("credentials.json"):
                    raise FileNotFoundError(
                        "credentials.json not found. Place your Google OAuth desktop credentials beside this script."
                    )
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                creds = flow.run_local_server(port=0)

            with open("token.json", "w", encoding="utf-8") as token_file:
                token_file.write(creds.to_json())

        return build("drive", "v3", credentials=creds)

    def list_images_in_folder(self, folder_id: str) -> List[DriveImageItem]:
        items: List[DriveImageItem] = []
        page_token = None

        query = (
            f"'{folder_id}' in parents and trashed = false and "
            "("
            "mimeType contains 'image/' or "
            "name contains '.jpg' or name contains '.jpeg' or name contains '.png' or "
            "name contains '.webp' or name contains '.bmp' or name contains '.tif' or name contains '.tiff'"
            ")"
        )

        while True:
            response = self.drive_service.files().list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType)",
                pageSize=1000,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()

            for f in response.get("files", []):
                items.append(DriveImageItem(
                    file_id=f["id"],
                    name=f["name"],
                    mime_type=f.get("mimeType", "application/octet-stream"),
                ))

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        items.sort(key=lambda x: x.name.lower())
        return items

    def find_file_in_folder(self, folder_id: str, filename: str) -> Optional[dict]:
        response = self.drive_service.files().list(
            q=f"'{folder_id}' in parents and trashed = false and name = '{filename}'",
            spaces="drive",
            fields="files(id, name, mimeType)",
            pageSize=10,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        files = response.get("files", [])
        return files[0] if files else None

    def download_file_bytes(self, file_id: str) -> bytes:
        request = self.drive_service.files().get_media(fileId=file_id, supportsAllDrives=True)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            _, done = downloader.next_chunk()

        return fh.getvalue()

    def upload_bytes_to_folder(self, folder_id: str, filename: str, data: bytes, mime_type: str) -> str:
        metadata = {"name": filename, "parents": [folder_id]}
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime_type, resumable=True)

        created = self.drive_service.files().create(
            body=metadata,
            media_body=media,
            fields="id,name",
            supportsAllDrives=True
        ).execute()

        return created["id"]

    def update_file_bytes(self, file_id: str, data: bytes, mime_type: str):
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime_type, resumable=True)
        self.drive_service.files().update(
            fileId=file_id,
            media_body=media,
            supportsAllDrives=True
        ).execute()

    def move_file_to_folder(self, file_id: str, new_folder_id: str):
        file_meta = self.drive_service.files().get(
            fileId=file_id,
            fields="parents",
            supportsAllDrives=True
        ).execute()

        previous_parents = ",".join(file_meta.get("parents", []))

        self.drive_service.files().update(
            fileId=file_id,
            addParents=new_folder_id,
            removeParents=previous_parents,
            fields="id, parents",
            supportsAllDrives=True
        ).execute()

    def trash_file(self, file_id: str):
        self.drive_service.files().update(
            fileId=file_id,
            body={"trashed": True},
            supportsAllDrives=True
        ).execute()

    def load_or_select_schema(self):
        if os.path.exists(self.schema_path):
            return self.read_json_file(self.schema_path)

        messagebox.showinfo(
            "Schema not found",
            f"Could not find {DEFAULT_SCHEMA_FILE} in:\n{self.base_dir}\n\nPlease choose your schema JSON file."
        )
        chosen = filedialog.askopenfilename(
            title="Select schema JSON",
            filetypes=[("JSON files", "*.json")]
        )
        if not chosen:
            raise SystemExit("No schema file selected.")

        self.schema_path = chosen
        self.base_dir = os.path.dirname(chosen)
        return self.read_json_file(chosen)

    def read_json_file(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_annotations_from_drive(self):
        remote = self.find_file_in_folder(ENCODED_FOLDER_ID, ANNOTATIONS_FILENAME)
        if remote:
            self.annotations_file_id = remote["id"]
            data = self.download_file_bytes(remote["id"]).decode("utf-8")
            annotations = json.loads(data)
            annotations.setdefault("schema_name", self.schema.get("schema_name", "unknown"))
            annotations.setdefault("schema_version", self.schema.get("schema_version", "unknown"))
            annotations.setdefault("annotations", {})
            return annotations

        self.annotations_file_id = None
        return {
            "schema_name": self.schema.get("schema_name", "unknown"),
            "schema_version": self.schema.get("schema_version", "unknown"),
            "annotations": {}
        }

    def sync_annotations_to_drive(self):
        data = json.dumps(self.annotations, indent=2, ensure_ascii=False).encode("utf-8")
        mime_type = "application/json"

        if self.annotations_file_id:
            self.update_file_bytes(self.annotations_file_id, data, mime_type)
        else:
            self.annotations_file_id = self.upload_bytes_to_folder(
                ENCODED_FOLDER_ID,
                ANNOTATIONS_FILENAME,
                data,
                mime_type
            )

    def build_ui(self):
        top_bar = ttk.Frame(self.root, padding=8)
        top_bar.pack(fill="x")

        ttk.Button(top_bar, text="Refresh Drive Images (F5)", command=self.refresh_and_reload).pack(side="left", padx=4)
        ttk.Button(top_bar, text="Random Previous (Ctrl+Left)", command=self.prev_image).pack(side="left", padx=4)
        ttk.Button(top_bar, text="Random Next (Ctrl+Right)", command=self.next_image).pack(side="left", padx=4)
        ttk.Button(top_bar, text="Save + Move (Ctrl+S)", command=self.save_and_advance).pack(side="left", padx=4)
        ttk.Button(top_bar, text="Delete Image (Ctrl+D)", command=self.delete_current_image).pack(side="left", padx=4)

        self.image_label_var = tk.StringVar(value="No image loaded")
        ttk.Label(top_bar, textvariable=self.image_label_var).pack(side="left", padx=12)

        self.count_label_var = tk.StringVar(value="0 / 0")
        ttk.Label(top_bar, textvariable=self.count_label_var).pack(side="right", padx=8)

        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill="both", expand=True)

        left_frame = ttk.Frame(main, padding=8)
        main.add(left_frame, weight=2)

        self.image_canvas = tk.Canvas(left_frame, bg="#222222", height=550, width=600)
        self.image_canvas.pack(fill="both", expand=True)

        right_frame = ttk.Frame(main, padding=8)
        main.add(right_frame, weight=3)

        self.form_canvas = tk.Canvas(right_frame)
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=self.form_canvas.yview)
        self.form_container = ttk.Frame(self.form_canvas)

        self.form_container.bind(
            "<Configure>",
            lambda e: self.form_canvas.configure(scrollregion=self.form_canvas.bbox("all"))
        )

        self.canvas_window = self.form_canvas.create_window((0, 0), window=self.form_container, anchor="nw")
        self.form_canvas.configure(yscrollcommand=scrollbar.set)

        self.form_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.form_canvas.bind("<Configure>", lambda e: self.form_canvas.itemconfig(self.canvas_window, width=e.width))
        self.form_canvas.bind("<Enter>", lambda e: self._bind_to_mousewheel())
        self.form_canvas.bind("<Leave>", lambda e: self._unbind_from_mousewheel())

        self.build_form()
        self.bind_keyboard_shortcuts()

    def _on_mousewheel(self, event):
        self.form_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.form_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.form_canvas.yview_scroll(1, "units")

    def _bind_to_mousewheel(self):
        self.form_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.form_canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.form_canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _unbind_from_mousewheel(self):
        self.form_canvas.unbind_all("<MouseWheel>")
        self.form_canvas.unbind_all("<Button-4>")
        self.form_canvas.unbind_all("<Button-5>")

    def build_form(self):
        self.widgets = {}
        self.focusable_widgets = []

        info_frame = ttk.LabelFrame(self.form_container, text="Sample Metadata", padding=10)
        info_frame.pack(fill="x", padx=6, pady=6)

        self.sample_id_var = tk.StringVar()
        self.image_name_var = tk.StringVar()
        self.country_var = tk.StringVar()

        self.add_labeled_entry(info_frame, "image_id", self.sample_id_var, 0)
        self.add_labeled_entry(info_frame, "image_name", self.image_name_var, 1, readonly=True)

        self.category_frames = {}
        self.category_rows = {}

        for field in self.fields:
            field_id = field.get("id")
            if field_id in {"image_id", "image_path", "country"}:
                continue

            category = field.get("category", "other")
            if category not in self.category_frames:
                lf = ttk.LabelFrame(self.form_container, text=category.replace("_", " ").title(), padding=10)
                lf.pack(fill="x", padx=6, pady=6)
                self.category_frames[category] = lf
                self.category_rows[category] = 0

            row = self.category_rows[category]
            row_increment = self.create_field_widget(self.category_frames[category], field, row)
            self.category_rows[category] += row_increment

    def add_labeled_entry(self, parent, label_text, variable, row, readonly=False):
        ttk.Label(parent, text=label_text, width=32).grid(row=row, column=0, sticky="w", padx=5, pady=4)
        entry = ttk.Entry(parent, textvariable=variable)
        if readonly:
            entry.state(["readonly"])
        entry.grid(row=row, column=1, sticky="ew", padx=5, pady=4)
        if not readonly:
            self.register_focusable_widget(entry)
        parent.columnconfigure(1, weight=1)

    def create_field_widget(self, parent, field, row):
        field_id = field["id"]
        field_type = field.get("type", "string")
        observable = field.get("observable_from_image", "")
        description = field.get("description", "")

        label_text = field_id
        ttk.Label(parent, text=label_text, width=20).grid(row=row, column=0, sticky="nw", padx=5, pady=4)

        if field_type == "boolean":
            var = tk.StringVar(value="")
            combo = ttk.Combobox(parent, textvariable=var, state="readonly", values=["", "true", "false"])
            combo.grid(row=row, column=1, sticky="ew", padx=5, pady=4)
            self.widgets[field_id] = {"type": "boolean", "var": var, "field": field, "widget": combo}
            self.register_focusable_widget(combo, is_combo=True)
        elif field_type == "single_choice":
            var = tk.StringVar(value="")
            choices = self.get_choice_labels(field.get("choices", []))
            combo = ttk.Combobox(parent, textvariable=var, state="readonly", values=[""] + choices)
            combo.grid(row=row, column=1, sticky="ew", padx=5, pady=4)
            self.widgets[field_id] = {"type": "single_choice", "var": var, "field": field, "widget": combo}
            self.register_focusable_widget(combo, is_combo=True)
        else:
            var = tk.StringVar(value="")
            entry = ttk.Entry(parent, textvariable=var)
            entry.grid(row=row, column=1, sticky="ew", padx=5, pady=4)
            self.widgets[field_id] = {"type": field_type, "var": var, "field": field, "widget": entry}
            self.register_focusable_widget(entry)

        row_used = 1
        if description:
            ttk.Label(
                parent,
                text=description,
                foreground="#555555",
                wraplength=500
            ).grid(row=row + 1, column=1, sticky="w", padx=5, pady=(0, 6))
            row_used = 2

        parent.columnconfigure(1, weight=1)
        return row_used


    def register_focusable_widget(self, widget, is_combo=False):
        self.focusable_widgets.append(widget)
        widget.bind("<FocusIn>", lambda event: self.ensure_widget_visible(event.widget))
        widget.bind("<Down>", lambda event: self.focus_next_field(event.widget))
        widget.bind("<Up>", lambda event: self.focus_previous_field(event.widget))
        widget.bind("<Return>", lambda event: self.focus_next_field(event.widget))
        widget.bind("<KP_Enter>", lambda event: self.focus_next_field(event.widget))

        if is_combo:
            widget.bind("<Right>", lambda event: self.cycle_combobox(event.widget, 1))
            widget.bind("<Left>", lambda event: self.cycle_combobox(event.widget, -1))
            widget.bind("<space>", lambda event: self.cycle_combobox(event.widget, 1))

    def bind_keyboard_shortcuts(self):
        self.root.bind_all("<Control-s>", lambda event: self.save_and_advance())
        self.root.bind_all("<Control-S>", lambda event: self.save_and_advance())
        self.root.bind_all("<Control-Right>", lambda event: self.next_image())
        self.root.bind_all("<Control-Left>", lambda event: self.prev_image())
        self.root.bind_all("<Control-d>", lambda event: self.delete_current_image())
        self.root.bind_all("<Control-D>", lambda event: self.delete_current_image())
        self.root.bind_all("<F5>", lambda event: self.refresh_and_reload())
        self.root.bind_all("<Prior>", lambda event: self.scroll_form(-1))  # PageUp
        self.root.bind_all("<Next>", lambda event: self.scroll_form(1))    # PageDown
        self.root.bind_all("<Home>", lambda event: self.scroll_to_top())
        self.root.bind_all("<End>", lambda event: self.scroll_to_bottom())

    def focus_next_field(self, widget):
        return self.focus_field_relative(widget, 1)

    def focus_previous_field(self, widget):
        return self.focus_field_relative(widget, -1)

    def focus_field_relative(self, widget, offset):
        if not self.focusable_widgets:
            return "break"

        try:
            idx = self.focusable_widgets.index(widget)
        except ValueError:
            idx = -1 if offset > 0 else 0

        next_idx = max(0, min(len(self.focusable_widgets) - 1, idx + offset))
        next_widget = self.focusable_widgets[next_idx]
        next_widget.focus_set()
        self.ensure_widget_visible(next_widget)
        return "break"

    def cycle_combobox(self, combo, direction):
        values = list(combo.cget("values"))
        if not values:
            return "break"

        current = combo.get()
        try:
            current_index = values.index(current)
        except ValueError:
            current_index = 0

        new_index = (current_index + direction) % len(values)
        combo.set(values[new_index])
        return "break"

    def ensure_widget_visible(self, widget):
        try:
            self.root.update_idletasks()
            widget_top = widget.winfo_rooty() - self.form_canvas.winfo_rooty()
            widget_bottom = widget_top + widget.winfo_height()
            canvas_height = self.form_canvas.winfo_height()

            if widget_top < 20:
                self.form_canvas.yview_scroll(-3, "units")
            elif widget_bottom > canvas_height - 20:
                self.form_canvas.yview_scroll(3, "units")
        except tk.TclError:
            pass

    def scroll_form(self, direction):
        self.form_canvas.yview_scroll(direction * 6, "units")
        return "break"

    def scroll_to_top(self):
        self.form_canvas.yview_moveto(0)
        return "break"

    def scroll_to_bottom(self):
        self.form_canvas.yview_moveto(1)
        return "break"

    def focus_first_attribute(self):
        if self.focusable_widgets:
            self.focusable_widgets[0].focus_set()
            self.ensure_widget_visible(self.focusable_widgets[0])

    def get_choice_labels(self, choices):
        labels = []
        for choice in choices:
            if isinstance(choice, dict):
                label = choice.get("label", "")
                code = choice.get("code")
                if code is not None:
                    labels.append(f"{label} (code: {code})")
                else:
                    labels.append(label)
            else:
                labels.append(str(choice))
        return labels

    def refresh_source_images(self):
        self.source_images = self.list_images_in_folder(SOURCE_FOLDER_ID)
        random.shuffle(self.source_images)

        if self.current_index >= len(self.source_images):
            self.current_index = len(self.source_images) - 1

    def load_random_image(self, exclude_current=True):
        if not self.source_images:
            return

        available_indices = list(range(len(self.source_images)))

        if (
            exclude_current
            and self.current_index in available_indices
            and len(available_indices) > 1
        ):
            available_indices.remove(self.current_index)

        random_index = random.choice(available_indices)
        self.load_image_by_index(random_index)

    def refresh_and_reload(self):
        current_id = self.current_item.file_id if self.current_item else None
        self.refresh_source_images()

        if not self.source_images:
            self.current_index = -1
            self.current_item = None
            self.clear_form(keep_metadata=False)
            self.image_canvas.delete("all")
            self.image_label_var.set("No image loaded")
            self.count_label_var.set("0 / 0")
            messagebox.showinfo("No images", "No source images remain in the source folder.")
            return

        if current_id:
            for idx, item in enumerate(self.source_images):
                if item.file_id == current_id:
                    self.load_image_by_index(idx)
                    return

        self.load_random_image(exclude_current=False)

    def load_image_by_index(self, index: int):
        if index < 0 or index >= len(self.source_images):
            return

        self.current_index = index
        self.current_item = self.source_images[index]

        self.image_label_var.set(self.current_item.name)
        self.count_label_var.set(f"{index + 1} / {len(self.source_images)}")
        self.image_name_var.set(self.current_item.name)

        filename_id = os.path.splitext(self.current_item.name)[0]
        self.sample_id_var.set(filename_id)

        self.clear_form(keep_metadata=True)
        self.display_current_drive_image()
        self.load_existing_annotation_for_current()
        self.root.after(100, self.focus_first_attribute)

    def display_current_drive_image(self):
        if not self.current_item:
            return

        try:
            data = self.download_file_bytes(self.current_item.file_id)
            img = Image.open(io.BytesIO(data))
            img.thumbnail(MAX_IMAGE_SIZE)
            self.current_photo = ImageTk.PhotoImage(img)

            self.image_canvas.delete("all")
            canvas_w = max(self.image_canvas.winfo_width(), MAX_IMAGE_SIZE[0])
            canvas_h = max(self.image_canvas.winfo_height(), MAX_IMAGE_SIZE[1])
            x = canvas_w // 2
            y = canvas_h // 2
            self.image_canvas.create_image(x, y, image=self.current_photo, anchor="center")
        except Exception as e:
            messagebox.showerror("Image error", f"Could not load image from Drive:\n{e}")

    def prev_image(self):
        # Randomised retrieval is intentional so multiple encoder instances are less likely
        # to work on the same next image.
        self.load_random_image()

    def next_image(self):
        # Randomised retrieval is intentional so multiple encoder instances are less likely
        # to work on the same next image.
        self.load_random_image()

    def clear_form(self, keep_metadata=False):
        for widget_info in self.widgets.values():
            widget_info["var"].set("")
        if not keep_metadata:
            self.sample_id_var.set("")
            self.image_name_var.set("")
            self.country_var.set("")

    def current_annotation_key(self) -> Optional[str]:
        return self.current_item.file_id if self.current_item else None

    def load_existing_annotation_for_current(self):
        key = self.current_annotation_key()
        if not key:
            return

        record = self.annotations.get("annotations", {}).get(key)
        if not record:
            return

        self.sample_id_var.set(record.get("image_id", self.sample_id_var.get()))
        self.country_var.set(record.get("country") or "")

        values = record.get("attributes", {})
        for field_id, widget_info in self.widgets.items():
            value = values.get(field_id, "")
            widget_info["var"].set("" if value is None else str(value))

    def collect_attributes(self) -> Dict[str, Optional[str]]:
        attributes = {}
        for field_id, widget_info in self.widgets.items():
            raw = widget_info["var"].get().strip()
            attributes[field_id] = None if raw == "" else raw
        return attributes

    def save_and_advance(self):
        if not self.current_item:
            messagebox.showwarning("No image", "No current image is loaded.")
            return

        image_id = self.sample_id_var.get().strip()
        image_name = self.image_name_var.get().strip()
        country = self.country_var.get().strip()

        if not image_id:
            messagebox.showwarning("Missing image_id", "Please enter an image_id.")
            return

        record = {
            "drive_file_id": self.current_item.file_id,
            "image_id": image_id,
            "image_name": image_name,
            "country": country if country else None,
            "source_folder_id": SOURCE_FOLDER_ID,
            "encoded_folder_id": ENCODED_FOLDER_ID,
            "attributes": self.collect_attributes(),
        }

        self.annotations["annotations"][self.current_item.file_id] = record

        try:
            self.sync_annotations_to_drive()
            self.move_file_to_folder(self.current_item.file_id, ENCODED_FOLDER_ID)

            moved_name = self.current_item.name

            self.refresh_source_images()

            if not self.source_images:
                self.current_index = -1
                self.current_item = None
                self.clear_form(keep_metadata=False)
                self.image_canvas.delete("all")
                self.image_label_var.set("No image loaded")
                self.count_label_var.set("0 / 0")
                messagebox.showinfo(
                    "Saved",
                    f"Saved annotation and moved '{moved_name}' to the encoded folder.\n\nNo more source images remain."
                )
                return

            self.load_random_image(exclude_current=False)

            messagebox.showinfo(
                "Saved",
                f"Saved annotation and moved '{moved_name}' to the encoded folder."
            )
        except Exception as e:
            messagebox.showerror("Save error", f"Could not save/move image:\n{e}")

    def delete_current_image(self):
        if not self.current_item:
            messagebox.showwarning("No image", "No current image is loaded.")
            return

        confirm = messagebox.askyesno(
            "Delete image",
            f"Move '{self.current_item.name}' to Google Drive Trash?\n\nUse this if the angle doesn't provide enough information."
        )
        if not confirm:
            return

        try:
            deleted_name = self.current_item.name

            self.trash_file(self.current_item.file_id)
            self.refresh_source_images()

            if not self.source_images:
                self.current_index = -1
                self.current_item = None
                self.clear_form(keep_metadata=False)
                self.image_canvas.delete("all")
                self.image_label_var.set("No image loaded")
                self.count_label_var.set("0 / 0")
                messagebox.showinfo(
                    "Deleted",
                    f"Moved '{deleted_name}' to Google Drive Trash.\n\nNo more source images remain."
                )
                return

            self.load_random_image(exclude_current=False)

            messagebox.showinfo("Deleted", f"Moved '{deleted_name}' to Google Drive Trash.")
        except Exception as e:
            messagebox.showerror("Delete error", f"Could not delete image:\n{e}")


def main():
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    app = DriveAnnotationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
