# Author: Edward Romero, OSTEM Intern, Spring 2025, NASA Kennedy Space Center
# This is a computer model that evaluates the efficacy of microgravity simulation devices

import os
from tkinter import messagebox, filedialog
import tkinter as tk
import tkinter.ttk as ttk
import webbrowser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageTk
from dateutil import parser
from dataCompile import DataProcessor, PathVisualization  

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


class GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Computer Model - NASA")
        self.master.configure(bg="#f1f1f1")

        self._setup_gui_elements()
        self._setup_plot_frames()

    def _setup_gui_elements(self):
        self._load_images()
        self.master.iconphoto(False, self.favicon)
        self._create_custom_theme()

        font_style = ("Calibri", 12)
        title_font_style = ("Calibri", 20, "bold")
        category_font_style = ("Calibri", 14, "bold")

        input_frame = tk.Frame(self.master, padx=1, pady=1, bg="#f1f1f1")
        input_frame.pack(side=tk.TOP, anchor=tk.CENTER)

        self._create_title_frame(input_frame, title_font_style)
        center_frame = tk.Frame(input_frame, bg="#f1f1f1")
        center_frame.pack()

        self._create_mode_frame(center_frame, font_style, category_font_style)
        self._create_operating_frame(center_frame, font_style, category_font_style)
        self._create_duration_frame(center_frame, font_style, category_font_style)
        self._create_analysis_frame(center_frame, font_style, category_font_style)
        self._create_analysis_frame_exp(center_frame, font_style, category_font_style)
        self._create_submit_button(center_frame, font_style)
        self._create_accelerometer_frame(center_frame, font_style, category_font_style)

    def _load_images(self):
        nasa_image_path = os.path.join(SCRIPT_DIR, 'images', 'NASA_logo.png')
        nasa_image = Image.open(nasa_image_path).resize((69, 58), Image.LANCZOS)
        self.nasa_logo = ImageTk.PhotoImage(nasa_image)

        mssf_image_path = os.path.join(SCRIPT_DIR, 'images', 'MSSF_logo.png')
        mssf_image = Image.open(mssf_image_path).resize((65, 58), Image.LANCZOS)
        self.mssf_logo = ImageTk.PhotoImage(mssf_image)

        favicon_path = os.path.join(SCRIPT_DIR, 'images', 'favicon.ico')
        self.favicon = ImageTk.PhotoImage(file=favicon_path)

    def _create_title_frame(self, parent, font_style):
        title_frame = tk.Frame(parent)
        title_frame.pack(pady=(10, 0))

        nasa_label = tk.Label(title_frame, image=self.nasa_logo)
        nasa_label.pack(side=tk.LEFT, padx=1)
        nasa_label.bind("<Button-1>", lambda e: self._open_url("https://www.nasa.gov/"))

        title_label = tk.Label(title_frame, text="Computer Model", font=font_style)
        title_label.pack(side=tk.LEFT, padx=1)

        mssf_label = tk.Label(title_frame, image=self.mssf_logo)
        mssf_label.pack(side=tk.LEFT, padx=1)
        mssf_label.bind("<Button-1>", lambda e: self._open_url("https://public.ksc.nasa.gov/partnerships/capabilities-and-testing/testing-and-labs/microgravity-simulation-support-facility/"))

    def _create_mode_frame(self, parent, font_style, category_font_style):
        mode_frame = tk.Frame(parent, padx=1, pady=1)
        mode_frame.grid(row=0, column=0, padx=30)

        tk.Label(mode_frame, text="Mode", font=category_font_style).pack()
        self.mode_var = tk.StringVar(value="Theoretical")
        self.mode_menu = tk.OptionMenu(mode_frame, self.mode_var, "Theoretical", "Experimental", command=self._switch_mode)
        self.mode_menu.config(font=font_style, bg="#aeb0b5", activebackground="#d6d7d9")
        self.mode_menu["menu"].config(font=("Calibri", 10), bg="#d6d7d9")
        self.mode_menu.pack()

    def _create_operating_frame(self, parent, font_style, category_font_style):
        self.operating_frame = tk.Frame(parent, padx=1, pady=1)
        self.operating_frame.grid(row=0, column=1, padx=30)

        tk.Label(self.operating_frame, text="Frame Velocities (rpm)", font=category_font_style).pack()
        operating_input_frame = tk.Frame(self.operating_frame)
        operating_input_frame.pack()

        self.inner_v_label = tk.Label(operating_input_frame, text="Inner:", font=font_style)
        self.inner_v_label.pack(side=tk.LEFT)
        self.inner_v_entry = tk.Entry(operating_input_frame, font=font_style, width=10)
        self.inner_v_entry.pack(side=tk.LEFT)

        self.outer_v_label = tk.Label(operating_input_frame, text="Outer:", font=font_style)
        self.outer_v_label.pack(side=tk.LEFT, padx=(10, 0))
        self.outer_v_entry = tk.Entry(operating_input_frame, font=font_style, width=10)
        self.outer_v_entry.pack(side=tk.LEFT)

    def _create_duration_frame(self, parent, font_style, category_font_style):
        self.duration_frame = tk.Frame(parent, padx=1, pady=1)
        self.duration_frame.grid(row=0, column=2, padx=30)

        tk.Label(self.duration_frame, text="Simulation Duration (hours)", font=category_font_style).pack()
        self.max_seg_entry = tk.Entry(self.duration_frame, font=font_style)
        self.max_seg_entry.pack()

    def _create_analysis_frame(self, parent, font_style, category_font_style):
        self.analysis_frame = tk.Frame(parent, padx=1, pady=1)
        self.analysis_frame.grid(row=0, column=3, padx=30)

        tk.Label(self.analysis_frame, text="Time Period of Analysis (hours)", font=category_font_style).pack()
        analysis_period_frame = tk.Frame(self.analysis_frame)
        analysis_period_frame.pack()

        self.start_analysis_entry = tk.Entry(analysis_period_frame, font=font_style, width=10)
        self.start_analysis_entry.pack(side=tk.LEFT)
        tk.Label(analysis_period_frame, text="-", font=font_style).pack(side=tk.LEFT)
        self.end_analysis_entry = tk.Entry(analysis_period_frame, font=font_style, width=10)
        self.end_analysis_entry.pack(side=tk.LEFT)

    def _create_analysis_frame_exp(self, parent, font_style, category_font_style):
        self.analysis_frame_exp = tk.Frame(parent, padx=1, pady=1)
        self.analysis_frame_exp.grid(row=0, column=4, padx=30)
        self.analysis_frame_exp.grid_remove()

        tk.Label(self.analysis_frame_exp, text="Time Period of Analysis (hours)", font=category_font_style).pack()
        analysis_period_frame_exp = tk.Frame(self.analysis_frame_exp)
        analysis_period_frame_exp.pack()

        self.start_analysis_entry_exp = tk.Entry(analysis_period_frame_exp, font=font_style, width=10)
        self.start_analysis_entry_exp.pack(side=tk.LEFT)
        tk.Label(analysis_period_frame_exp, text="-", font=font_style).pack(side=tk.LEFT)
        self.end_analysis_entry_exp = tk.Entry(analysis_period_frame_exp, font=font_style, width=10)
        self.end_analysis_entry_exp.pack(side=tk.LEFT)

    def _create_submit_button(self, parent, font_style):
        self.submit_button = tk.Button(parent, text="Start", command=self._submit, font=font_style, bg="#0066b2", fg="#ffffff", activebackground="#3380cc", activeforeground="#ffffff")
        self.submit_button.grid(row=1, column=0, columnspan=4, pady=(10, 5))

    def _create_accelerometer_frame(self, parent, font_style, category_font_style):
        self.accelerometer_frame = tk.Frame(parent, padx=1, pady=1)
        tk.Label(self.accelerometer_frame, text="Acceleration Data", font=category_font_style).pack()
        self.import_button = tk.Button(self.accelerometer_frame, text="Upload File (CSV)", command=self._import_data, font=font_style, bg="#aeb0b5", activebackground="#d6d7d9")
        self.import_button.pack()

    def _setup_plot_frames(self):
        plot_frame = tk.Frame(self.master, padx=5, pady=5, bg="#f1f1f1")
        plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=(5, 5), pady=(0, 5))

        notebook = ttk.Notebook(plot_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.magnitude_frame = tk.Frame(notebook, borderwidth=1, relief=tk.SOLID)
        self.vector_components_frame = tk.Frame(notebook, borderwidth=1, relief=tk.SOLID)
        self.path_frame = tk.Frame(notebook, borderwidth=0, relief=tk.SOLID)

        notebook.add(self.magnitude_frame, text="Resultant Vector")
        notebook.add(self.vector_components_frame, text="Vector Components")
        notebook.add(self.path_frame, text="Vector Path")

        rcParams['font.family'] = 'Calibri'
        rcParams['font.size'] = 10

        self._setup_magnitude_plot()
        self._setup_path_plots()
        self._setup_components_plot()
        self._clear_plots()

    def _setup_magnitude_plot(self):
        self.figure = plt.Figure()
        self.ax = self.figure.add_subplot(1, 1, 1)
        self.ax.set_yscale('log')
        self.ax.set_title("Resultant Acceleration Vector")
        self.ax.set_xlabel('Time (hours)')
        self.ax.set_ylabel('Magnitude (g)')
        self.canvas = FigureCanvasTkAgg(self.figure, self.magnitude_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.magnitude_frame)
        self.toolbar.update()

    def _setup_path_plots(self):
        self.path_figure = plt.Figure()
        self.path_ax = self.path_figure.add_subplot(1, 1, 1, projection='3d')
        self._configure_3d_axes(self.path_ax, "Acceleration Vector Path (Full Duration)")
        self.path_frame_left = tk.Frame(self.path_frame, borderwidth=1, relief=tk.SOLID)
        self.path_canvas = FigureCanvasTkAgg(self.path_figure, self.path_frame_left)
        self.path_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.path_toolbar = NavigationToolbar2Tk(self.path_canvas, self.path_frame_left)
        self.path_toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.path_frame_left.grid(row=0, column=0, sticky="nsew")

        self.path_figure_analysis = plt.Figure()
        self.path_ax_analysis = self.path_figure_analysis.add_subplot(1, 1, 1, projection='3d')
        self._configure_3d_axes(self.path_ax_analysis, "Acceleration Vector Path (Analysis Period)")
        self.path_frame_right = tk.Frame(self.path_frame, borderwidth=1, relief=tk.SOLID)
        self.path_canvas_analysis = FigureCanvasTkAgg(self.path_figure_analysis, self.path_frame_right)
        self.path_canvas_analysis.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.path_toolbar_analysis = NavigationToolbar2Tk(self.path_canvas_analysis, self.path_frame_right)
        self.path_toolbar_analysis.pack(side=tk.BOTTOM, fill=tk.X)
        self.path_frame_right.grid(row=0, column=1, sticky="nsew")

        self.path_frame.grid_columnconfigure(0, weight=1)
        self.path_frame.grid_columnconfigure(1, weight=1)
        self.path_frame.grid_rowconfigure(0, weight=1)

    def _setup_components_plot(self):
        self.components_figure = plt.Figure()
        self.components_ax = self.components_figure.add_subplot(1, 1, 1)
        self.components_ax.set_title("Acceleration Vector Components")
        self.components_ax.set_xlabel('Time (hours)')
        self.components_ax.set_ylabel('Magnitude (g)')
        self.components_canvas = FigureCanvasTkAgg(self.components_figure, self.vector_components_frame)
        self.components_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.components_toolbar = NavigationToolbar2Tk(self.components_canvas, self.vector_components_frame)
        self.components_toolbar.update()

    def _configure_3d_axes(self, ax, title):
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ticks = np.arange(-1.0, 1.5, 0.5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_zticks(ticks)
        ax.set_title(title)

    def _create_custom_theme(self):
        style = ttk.Style()
        style.theme_create("yummy", parent="alt", settings={
            "TNotebook": {"configure": {"tabmargins": [1, 0, 0, 0], "background": "#f1f1f1"}},
            "TNotebook.Tab": {
                "configure": {"padding": [5, 1], "background": "#aeb0b5", "font": ("Calibri", 12), "focuscolor": ""},
                "map": {"background": [("selected", "#d6d7d9")], "expand": [("selected", [1, 1, 1, 0])]}
            }
        })
        style.theme_use("yummy")

    def _switch_mode(self, mode):
        if mode == "Theoretical":
            self._show_theoretical_inputs()
        else:
            self._show_experimental_inputs()

    def _show_theoretical_inputs(self):
        self.operating_frame.grid()
        self.duration_frame.grid()
        self.analysis_frame.grid()
        self.analysis_frame_exp.grid_remove()
        self.accelerometer_frame.grid_remove()
        self.submit_button.grid(row=1, column=0, columnspan=4, pady=(10, 5))
        self._clear_plots()

    def _show_experimental_inputs(self):
        self.operating_frame.grid_remove()
        self.duration_frame.grid_remove()
        self.analysis_frame.grid_remove()
        self.analysis_frame_exp.grid(row=0, column=2, padx=30)
        self.accelerometer_frame.grid(row=0, column=1, padx=30)
        self.submit_button.grid(row=1, column=0, columnspan=4, pady=(10, 5))
        self._clear_plots()

    def _clear_plots(self):
        self.ax.clear()
        self.ax.set_yscale('log')
        self.ax.set_title("Resultant Acceleration Vector")
        self.ax.set_xlabel('Time (hours)')
        self.ax.set_ylabel('Magnitude (g)')
        self.ax.set_yticks([10**(-i) for i in range(0, 17, 2)])
        self.ax.set_ylim(10**-17, 10**0)
        self.canvas.draw()

        self.path_ax.clear()
        self._configure_3d_axes(self.path_ax, "Acceleration Vector Path (Full Duration)")
        self.path_canvas.draw()

        self.path_ax_analysis.clear()
        self._configure_3d_axes(self.path_ax_analysis, "Acceleration Vector Path (Analysis Period)")
        self.path_canvas_analysis.draw()

        self.components_ax.clear()
        self.components_ax.set_title("Acceleration Vector Components")
        self.components_ax.set_xlabel('Time (hours)')
        self.components_ax.set_ylabel('Magnitude (g)')
        self.components_canvas.draw()

    def _import_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    main_array = file.read().replace("   ", " ").replace('\t', ' ').replace('\n', ' ').replace(',', ' ').split(' ')
                self.experimental_data = main_array
                messagebox.showinfo("Success", "CSV file uploaded successfully.")
            except FileNotFoundError:
                messagebox.showerror("File Error", f"File not found: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def _process_experimental_data(self, main_array, start_analysis, end_analysis):
        datetime_str = []
        x, y, z = [], [], []

        for k in range(0, len(main_array) - 4, 5):
            try:
                dt = parser.parse(main_array[k] + " " + main_array[k + 1])
            except ValueError:
                dt = parser.parse(main_array[k + 1] + " " + main_array[k])
            datetime_str.append(dt)
            x.append(float(main_array[k + 2]))
            y.append(float(main_array[k + 3]))
            z.append(float(main_array[k + 4]))

        time_in_seconds = [(dt - datetime_str[0]).total_seconds() for dt in datetime_str]
        time_in_hours = [t / 3600 for t in time_in_seconds]

        path_vis = PathVisualization("experimental", x, y, z)
        distribution_score = path_vis.get_distribution()

        self._update_experimental_plots(x, y, z, time_in_hours, start_analysis, end_analysis, distribution_score)

    def _update_experimental_plots(self, x, y, z, time_in_hours, start_analysis, end_analysis, distribution_score):
        rcParams['font.family'] = 'Calibri'
        self.ax.clear()
        self.ax.set_yscale('log')
        self.ax.set_title("Resultant Acceleration Vector")

        x_time_avg = np.cumsum(x) / np.arange(1, len(x) + 1)
        y_time_avg = np.cumsum(y) / np.arange(1, len(y) + 1)
        z_time_avg = np.cumsum(z) / np.arange(1, len(z) + 1)
        magnitude = np.sqrt(x_time_avg**2 + y_time_avg**2 + z_time_avg**2)
        avg_mag_full = np.mean(magnitude)

        self.ax.plot(time_in_hours, magnitude, color='#0066b2', label=f"Time-Averaged Magnitude: {avg_mag_full:.3g}")
        if start_analysis is not None and end_analysis is not None:
            start_seg = next(i for i, t in enumerate(time_in_hours) if t >= start_analysis)
            end_seg = next(i for i, t in enumerate(time_in_hours) if t >= end_analysis)
            self.ax.axvline(x=start_analysis, color='#ec1c24', linestyle='--')
            self.ax.axvline(x=end_analysis, color='#ec1c24', linestyle='--')
            avg_mag_analysis = np.mean(magnitude[start_seg:end_seg])
            self.ax.plot(time_in_hours[start_seg:end_seg], magnitude[start_seg:end_seg], color='#ec1c24', label=f"Time-Averaged Magnitude: {avg_mag_analysis:.3g}")

        self.ax.legend()
        self.ax.set_xlabel('Time (hours)')
        self.ax.set_ylabel('Magnitude (g)')
        self.canvas.draw()

        self.path_ax.clear()
        self.path_ax.plot(x, y, z, color='#0066b2', linewidth=1)
        self._configure_3d_axes(self.path_ax, "Acceleration Vector Path (Full Duration)")
        self.path_ax.legend([f"Distribution: {distribution_score}"])
        self.path_canvas.draw()

        self._create_time_avg_fig(x_time_avg, y_time_avg, z_time_avg, time_in_hours)

        self.path_ax_analysis.clear()
        if start_analysis is not None and end_analysis is not None:
            self.path_ax_analysis.plot(x[start_seg:end_seg], y[start_seg:end_seg], z[start_seg:end_seg], color='#ec1c24', linewidth=1)
            self._configure_3d_axes(self.path_ax_analysis, "Acceleration Vector Path (Analysis Period)")
            path_vis_analysis = PathVisualization("experimental", x[start_seg:end_seg], y[start_seg:end_seg], z[start_seg:end_seg])
            distribution_score_analysis = path_vis_analysis.get_distribution()
            self.path_ax_analysis.legend([f"Distribution: {distribution_score_analysis}"])
        else:
            self._configure_3d_axes(self.path_ax_analysis, "Acceleration Vector Path (Analysis Period)")
        self.path_canvas_analysis.draw()

    def _submit(self):
        try:
            if self.mode_var.get() == "Theoretical":
                self._process_theoretical_data()
            else:
                self._process_experimental_data_submission()
        except ValueError as ve:
            messagebox.showerror("Input Error", str(ve))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _process_theoretical_data(self):
        if not all([self.inner_v_entry.get(), self.outer_v_entry.get(), self.max_seg_entry.get()]):
            raise ValueError("Set frame velocities and simulation duration.")

        inner_v = float(self.inner_v_entry.get())
        outer_v = float(self.outer_v_entry.get())
        max_seg = float(self.max_seg_entry.get())
        start_analysis = self.start_analysis_entry.get()
        end_analysis = self.end_analysis_entry.get()

        if max_seg <= 0:
            raise ValueError("Simulation duration must be positive.")

        start_analysis = float(start_analysis) if start_analysis else None
        end_analysis = float(end_analysis) if end_analysis else None
        if start_analysis and end_analysis:
            if start_analysis < 0 or end_analysis < 0:
                raise ValueError("Time values must be positive.")
            if end_analysis <= start_analysis:
                raise ValueError("Upper bound for analysis period must be greater than the lower bound.")
            if end_analysis > max_seg:
                raise ValueError("Upper bound for analysis period must be less than or equal to the simulation duration.")

        analysis = DataProcessor(inner_v, outer_v, max_seg, start_analysis, end_analysis)
        path_vis = PathVisualization(inner_v, analysis.x, analysis.y, analysis.z)
        x_time_avg, y_time_avg, z_time_avg = analysis._get_time_avg()
        magnitude = analysis._get_magnitude(x_time_avg, y_time_avg, z_time_avg)
        avg_mag_seg, avg_mag_analysis = analysis._get_mag_seg(magnitude)
        dis_score = analysis.get_distribution()
        self._update_plot(analysis, magnitude, start_analysis, end_analysis, avg_mag_seg, avg_mag_analysis, inner_v, outer_v, dis_score, path_vis)

    def _process_experimental_data_submission(self):
        if not hasattr(self, 'experimental_data') or not self.experimental_data:
            raise ValueError("Upload a CSV file.")

        start_analysis = self.start_analysis_entry_exp.get()
        end_analysis = self.end_analysis_entry_exp.get()

        start_analysis = float(start_analysis) if start_analysis else None
        end_analysis = float(end_analysis) if end_analysis else None
        if start_analysis and end_analysis:
            if start_analysis < 0 or end_analysis < 0:
                raise ValueError("Time values must be positive.")
            if end_analysis <= start_analysis:
                raise ValueError("Upper bound for analysis period must be greater than the lower bound.")
            datetime_str = []
            for k in range(0, len(self.experimental_data) - 4, 5):
                try:
                    dt = parser.parse(self.experimental_data[k] + " " + self.experimental_data[k + 1])
                except ValueError:
                    dt = parser.parse(self.experimental_data[k + 1] + " " + self.experimental_data[k])
                datetime_str.append(dt)
            time_in_hours = [(dt - datetime_str[0]).total_seconds() / 3600 for dt in datetime_str]
            if end_analysis > max(time_in_hours):
                raise ValueError("Upper bound for analysis period exceeds the final timestamp in the CSV.")

        self._process_experimental_data(self.experimental_data, start_analysis, end_analysis)

    def _update_plot(self, analysis, magnitude, start_analysis, end_analysis, avg_mag_seg, avg_mag_analysis, inner_v, outer_v, dis_score, path_vis):
        rcParams['font.family'] = 'Calibri'
        self.ax.clear()
        f_time = path_vis.format_time(analysis.time)

        self.ax.set_yscale('log')
        self.ax.set_title("Resultant Acceleration Vector")
        self.ax.plot(f_time, magnitude, color='#0066b2', label=f"Time-Averaged Magnitude: {avg_mag_seg:.3g}")

        if start_analysis is not None and end_analysis is not None:
            start_index = next(i for i, t in enumerate(f_time) if t >= start_analysis)
            end_index = next(i for i, t in enumerate(f_time) if t >= end_analysis)
            self.ax.axvline(x=start_analysis, color='#ec1c24', linestyle='--')
            self.ax.axvline(x=end_analysis, color='#ec1c24', linestyle='--')
            self.ax.plot(f_time[start_index:end_index], magnitude[start_index:end_index], color='#ec1c24', label=f"Time-Averaged Magnitude: {avg_mag_analysis:.3g}")

        self.ax.legend()
        self.ax.set_xlabel('Time (hours)')
        self.ax.set_ylabel('Magnitude (g)')
        self.canvas.draw()

        self.path_ax.clear()
        self.path_ax.plot(analysis.x, analysis.y, analysis.z, color='#0066b2', linewidth=1)
        self._configure_3d_axes(self.path_ax, "Acceleration Vector Path (Full Duration)")
        self.path_ax.legend([f"Distribution: {dis_score}"])
        self.path_canvas.draw()

        x_time_avg, y_time_avg, z_time_avg = analysis._get_time_avg()
        self._create_time_avg_fig(x_time_avg, y_time_avg, z_time_avg, analysis.time)

        self.path_ax_analysis.clear()
        if start_analysis is not None and end_analysis is not None:
            self.path_ax_analysis.plot(analysis.x[start_index:end_index], analysis.y[start_index:end_index], analysis.z[start_index:end_index], color='#ec1c24', linewidth=1)
            self._configure_3d_axes(self.path_ax_analysis, "Acceleration Vector Path (Analysis Period)")
            path_vis_analysis = PathVisualization("experimental", analysis.x[start_index:end_index], analysis.y[start_index:end_index], analysis.z[start_index:end_index])
            distribution_score_analysis = path_vis_analysis.get_distribution()
            self.path_ax_analysis.legend([f"Distribution: {distribution_score_analysis}"])
        else:
            self._configure_3d_axes(self.path_ax_analysis, "Acceleration Vector Path (Analysis Period)")
        self.path_canvas_analysis.draw()

    def _create_time_avg_fig(self, x_time_avg, y_time_avg, z_time_avg, time_data, legend=True, title=True):
        if self.mode_var.get() == "Theoretical":
            time_in_hours = [t / 3600 for t in time_data]  
        else:
            time_in_hours = time_data 

        self.components_ax.clear()
        if title:
            self.components_ax.set_title('Acceleration Vector Components')

        self.components_ax.plot(time_in_hours, x_time_avg, label='X-Component', color='#0066b2')
        self.components_ax.plot(time_in_hours, y_time_avg, label='Y-Component', color='#ec1c24')
        self.components_ax.plot(time_in_hours, z_time_avg, label='Z-Component', color='#aeb0b5')
        self.components_ax.set_xlabel('Time (hours)')
        self.components_ax.set_ylabel('Magnitude (g)')
        if legend:
            self.components_ax.legend()
        self.components_canvas.draw()

    def _open_url(self, url):
        webbrowser.open_new(url)


if __name__ == "__main__":
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()