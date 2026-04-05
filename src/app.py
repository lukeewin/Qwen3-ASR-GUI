# !/usr/bin/env python
# _*_ coding utf-8 _*_
# @Time: 2026/4/4 17:46
# @Author: Luke Ewin
# @Blog: https://blog.lukeewin.top
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dotenv import load_dotenv
import torch
from qwen_asr import Qwen3ASRModel
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "D:/Works/Python/Qwen3_ASR_GUI/models/Qwen3-ASR-1.7B")
DEVICE = os.getenv("DEVICE", "cuda")
MAX_INFERENCE_BATCH_SIZE = os.getenv("MAX_INFERENCE_BATCH_SIZE", 8)
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 256))

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("音视频转写工具")
        self.root.geometry("650x480")
        self.root.resizable(False, False)

        # 存储用户选择的文件列表和输出目录
        self.input_files = []
        self.output_dir = ""

        # 模型对象，初始为None
        self.model = None
        self.model_loading = False

        # 创建界面组件
        self.create_widgets()

    def create_widgets(self):
        """创建UI组件：使用三列布局实现整体居中，同时保证按钮左边缘对齐"""
        # 配置三列：左、中、右，左右两列权重为1，中间列权重为0，实现内容居中
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.columnconfigure(2, weight=1)

        # 中间列（第1列）放置所有内容，内部再使用grid或Frame保证按钮对齐
        main_frame = tk.Frame(self.root)
        main_frame.grid(row=0, column=1, sticky="n")

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("模型未加载，请先点击「加载模型」")
        status_label = tk.Label(main_frame, textvariable=self.status_var, fg="gray", anchor="center")
        status_label.grid(row=0, column=0, columnspan=2, pady=(10, 5), sticky="ew")

        # 第1行：加载模型按钮
        self.load_btn = tk.Button(main_frame, text="加载模型", command=self.load_model, width=25, bg="#4CAF50", fg="white")
        self.load_btn.grid(row=1, column=0, pady=10, sticky="w")

        # 第2行：选择音视频文件按钮 + 文件数量（两个组件放在同一行，按钮左对齐）
        self.file_btn = tk.Button(main_frame, text="选择音视频文件 (可多选)", command=self.select_files, width=25, state="disabled")
        self.file_btn.grid(row=2, column=0, pady=5, sticky="w")
        self.file_count_var = tk.StringVar(value="未选择任何文件")
        file_count_label = tk.Label(main_frame, textvariable=self.file_count_var, anchor="w")
        file_count_label.grid(row=2, column=1, pady=5, padx=(10, 0), sticky="w")

        # 第3行：选择输出目录按钮 + 路径显示
        self.output_btn = tk.Button(main_frame, text="选择输出目录", command=self.select_output_dir, width=25, state="disabled")
        self.output_btn.grid(row=3, column=0, pady=5, sticky="w")
        self.output_dir_var = tk.StringVar(value="未选择输出目录")
        output_dir_label = tk.Label(main_frame, textvariable=self.output_dir_var, anchor="w", wraplength=350)
        output_dir_label.grid(row=3, column=1, pady=5, padx=(10, 0), sticky="w")

        # 第4行：开始转写按钮
        self.start_btn = tk.Button(main_frame, text="开始转写", command=self.start_transcription, width=25, state="disabled")
        self.start_btn.grid(row=4, column=0, pady=10, sticky="w")

        # 进度条和百分比（单独占一行，跨两列，居中显示）
        progress_frame = tk.Frame(main_frame)
        progress_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")
        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", length=400, mode="determinate")
        self.progress.pack()
        self.progress_percent = tk.StringVar(value="0%")
        percent_label = tk.Label(progress_frame, textvariable=self.progress_percent, anchor="center")
        percent_label.pack()

        # 当前处理文件提示
        self.current_file_var = tk.StringVar(value="")
        current_label = tk.Label(main_frame, textvariable=self.current_file_var, anchor="center", fg="blue")
        current_label.grid(row=6, column=0, columnspan=2, pady=5)

        # 让第0列和第1列的内容左对齐，且第1列随窗口拉伸（如果窗口变宽，标签可以变长）
        main_frame.columnconfigure(0, weight=0)
        main_frame.columnconfigure(1, weight=1)

    def load_model(self):
        if self.model is not None:
            messagebox.showinfo("提示", "模型已加载，无需重复加载")
            return
        if self.model_loading:
            messagebox.showinfo("提示", "模型正在加载中，请稍候...")
            return

        self.load_btn.config(state="disabled", text="加载中...")
        self.model_loading = True
        self.status_var.set("正在加载模型，请稍候...（界面不会卡顿）")

        thread = threading.Thread(target=self._load_model_thread, daemon=True)
        thread.start()

    def _load_model_thread(self):
        try:
            model = Qwen3ASRModel.from_pretrained(
                MODEL_PATH,
                dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
                device_map=DEVICE,
                max_inference_batch_size=MAX_INFERENCE_BATCH_SIZE,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            self.model = model
            self.root.after(0, self._on_model_loaded_success)
        except Exception as e:
            self.root.after(0, self._on_model_loaded_failed, str(e))

    def _on_model_loaded_success(self):
        self.model_loading = False
        self.status_var.set("模型加载成功！现在可以选择文件并开始转写。")
        self.load_btn.config(text="模型已加载", state="disabled", bg="gray")
        self.file_btn.config(state="normal")
        self.output_btn.config(state="normal")
        self.update_start_button_state()

    def _on_model_loaded_failed(self, error_msg):
        self.model_loading = False
        self.status_var.set(f"模型加载失败: {error_msg}")
        self.load_btn.config(state="normal", text="加载模型", bg="#4CAF50")
        messagebox.showerror("模型错误", f"无法加载模型：{error_msg}\n请检查路径 {MODEL_PATH} 是否正确。")

    def select_files(self):
        file_types = [
            ("音视频文件", "*.wav *.mp3 *.flac *.m4a *.mp4 *.avi *.mkv *.wav"),
            ("所有文件", "*.*")
        ]
        files = filedialog.askopenfilenames(title="选择音视频文件", filetypes=file_types)
        if files:
            self.input_files = list(files)
            self.file_count_var.set(f"已选择 {len(self.input_files)} 个文件")
        else:
            self.input_files = []
            self.file_count_var.set("未选择任何文件")
        self.update_start_button_state()

    def select_output_dir(self):
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.output_dir = directory
            self.output_dir_var.set(directory)
        else:
            self.output_dir = ""
            self.output_dir_var.set("未选择输出目录")
        self.update_start_button_state()

    def update_start_button_state(self):
        if self.model is not None and self.input_files and self.output_dir and os.path.isdir(self.output_dir):
            self.start_btn.config(state="normal")
        else:
            self.start_btn.config(state="disabled")

    def start_transcription(self):
        if self.model is None:
            messagebox.showwarning("模型未加载", "请先点击「加载模型」按钮，等待加载完成后再开始转写。")
            return
        if not self.input_files:
            messagebox.showwarning("无文件", "请先选择音视频文件。")
            return
        if not self.output_dir or not os.path.isdir(self.output_dir):
            messagebox.showwarning("无输出目录", "请选择有效的输出目录。")
            return

        self.start_btn.config(state="disabled")
        self.file_btn.config(state="disabled")
        self.output_btn.config(state="disabled")

        self.progress["value"] = 0
        self.progress_percent.set("0%")
        self.current_file_var.set("")
        self.status_var.set("转写进行中...")

        thread = threading.Thread(target=self._transcribe_all, daemon=True)
        thread.start()

    def _split_sentences(self, text: str) -> str:
        """根据标点符号（。？！.!?）将文本分割为多行，每行一个句子，保留标点"""
        # 定义中英文标点
        punctuation = '。？！.!?'
        sentences = []
        current = []
        for ch in text:
            current.append(ch)
            if ch in punctuation:
                sentences.append(''.join(current).strip())
                current = []
        if current:
            sentences.append(''.join(current).strip())
        return '\n'.join(sentences)

    def _transcribe_all(self):
        total = len(self.input_files)
        for idx, file_path in enumerate(self.input_files):
            self.root.after(0, self._update_current_file, os.path.basename(file_path))

            try:
                results = self.model.transcribe(audio=file_path, language=None)
                text = results[0].text if results else ""
                torch.cuda.empty_cache()
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                output_file = os.path.join(self.output_dir, f"{base_name}.txt")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(self._split_sentences(text))

            except Exception as e:
                error_msg = f"转写失败: {str(e)}"
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                output_file = os.path.join(self.output_dir, f"{base_name}_error.txt")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(error_msg)
                self.root.after(0, self._update_status, f"错误: {os.path.basename(file_path)} - {str(e)}")

            progress_value = int((idx + 1) / total * 100)
            self.root.after(0, self._update_progress, progress_value)

        self.root.after(0, self._on_transcription_finished)

    def _update_current_file(self, filename):
        self.current_file_var.set(f"正在处理: {filename}")

    def _update_status(self, msg):
        self.status_var.set(msg)

    def _update_progress(self, value):
        self.progress["value"] = value
        self.progress_percent.set(f"{value}%")

    def _on_transcription_finished(self):
        self.current_file_var.set("")
        self.status_var.set("转写完成！")
        messagebox.showinfo("完成", f"所有文件转写完毕，结果保存在 {self.output_dir}")
        self.start_btn.config(state="normal")
        self.file_btn.config(state="normal")
        self.output_btn.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()
