# utils/viz/display.py

import tkinter as tk
from tkinter import ttk
import pandas as pd
from tabulate import tabulate
import os


class DisplayUtils:
    """
    Utility class for displaying pandas DataFrames and summaries in a GUI or console.
    """

    @staticmethod
    def show_dataframe_popup(df: pd.DataFrame, title="Data Preview", max_rows=100):
        if not isinstance(df, pd.DataFrame) or df.empty:
            print("The DataFrame is empty or invalid.")
            return

        df_to_show = df.copy()
        if len(df_to_show) > max_rows:
            df_to_show = df_to_show.sample(max_rows)

        root = tk.Tk()
        root.title(title)
        root.geometry("1200x400")

        frame = ttk.Frame(root)
        frame.pack(fill="both", expand=True)

        tree = ttk.Treeview(frame, columns=list(
            df_to_show.columns), show="headings")

        for col in df_to_show.columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor="center")

        for _, row in df_to_show.iterrows():
            tree.insert("", "end", values=list(row))

        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)

        root.mainloop()

    @staticmethod
    def show_dataframe_notebook(df: pd.DataFrame, max_rows=100):
        """
        Displays a styled preview of a pandas DataFrame within a Jupyter Notebook.

        This method shows up to `max_rows` of the DataFrame with enhanced formatting 
        to improve readability, including:
        - Centered text alignment
        - Highlighting of missing values
        - Header styling with background color and bold font

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame to display.
        max_rows : int, optional (default=100)
            The maximum number of rows to display from the DataFrame.

        Notes:
        ------
        This method is intended for use in Jupyter Notebooks only and does not open any GUI window.
        """
        from IPython.display import display
        if not isinstance(df, pd.DataFrame) or df.empty:
            print("The DataFrame is empty or invalid.")
            return

        df_to_show = df.sample(min(len(df), max_rows))
        display(df_to_show.style
                .highlight_null(color='lightcoral')
                .set_properties(**{'text-align': 'center'})
                .set_table_styles(
                    [{'selector': 'th', 'props': [
                        ('background-color', '#f2f2f2'), ('color', 'black'), ('font-weight', 'bold')]}]
                ))

    @staticmethod
    def show_summary_console(summary: dict):
        print("\n=== Dataset Summary ===")
        print(summary["info"])
        print(f"\nDuplicate Count: {summary['duplicate_count']}")

        print("\nMissing Values:")
        print(tabulate(summary["missing_values"].items(),
              headers=["Feature", "Count"]))

        print("\nMissing Percentages:")
        print(tabulate(
            [(k, f"{v:.2f}%")
             for k, v in summary["missing_percentage"].items()],
            headers=["Feature", "Percentage"]
        ))

        print("\nUnique Values:")
        print(tabulate(summary["unique_values"].items(),
              headers=["Feature", "Count"]))

    @staticmethod
    def show_summary_popup(summary: dict, title="Dataset Summary"):
        try:
            root = tk.Tk()
        except tk.TclError:
            print("GUI not supported. Falling back to console output.")
            DisplayUtils.show_summary_console(summary)
            return

        root.title(title)
        root.geometry("1000x800")

        main_canvas = tk.Canvas(root)
        scrollbar = ttk.Scrollbar(
            root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(
                scrollregion=main_canvas.bbox("all"))
        )

        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)

        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def sort_treeview_column(treeview, col, reverse):
            items = [(treeview.set(k, col), k)
                     for k in treeview.get_children('')]
            try:
                items.sort(key=lambda t: float(
                    t[0].strip('%')), reverse=reverse)
            except ValueError:
                items.sort(reverse=reverse)
            for index, (val, k) in enumerate(items):
                treeview.move(k, '', index)
            treeview.heading(col, command=lambda: sort_treeview_column(
                treeview, col, not reverse))

        def add_table(frame, title, data, headers):
            label = ttk.Label(frame, text=title, font=("Arial", 12, "bold"))
            label.pack(anchor="w", pady=(10, 2))

            # Dynamically set height to avoid scroll if possible
            height = max(len(data), 5)
            tree = ttk.Treeview(frame, columns=headers,
                                show="headings", height=height)
            for h in headers:
                tree.heading(
                    h, text=h, command=lambda _h=h: sort_treeview_column(tree, _h, False))
                tree.column(h, width=200, anchor="center")
            for row in data:
                tree.insert("", "end", values=row)
            tree.pack(fill="x", padx=10)

        # Info section
        ttk.Label(scrollable_frame, text="Dataset Info:", font=(
            "Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 2))
        info_box = tk.Text(scrollable_frame, height=6, wrap="word")
        info_box.insert("1.0", summary["info"])
        info_box.configure(state="disabled")
        info_box.pack(fill="x", padx=10)

        ttk.Label(scrollable_frame, text=f"Duplicate Count: {summary['duplicate_count']}", font=(
            "Arial", 11)).pack(anchor="w", padx=10, pady=(5, 5))

        add_table(scrollable_frame, "Missing Values", list(
            summary["missing_values"].items()), ["Feature", "Count"])
        add_table(scrollable_frame, "Missing Percentage", [
                  (k, f"{v:.2f}%") for k, v in summary["missing_percentage"].items()], ["Feature", "Percentage"])
        add_table(scrollable_frame, "Unique Values", list(
            summary["unique_values"].items()), ["Feature", "Count"])

        root.mainloop()

    @staticmethod
    def show_summary(summary: dict):
        """
        Displays the summary in GUI if supported, else falls back to console.
        """
        try:
            # Windows usually allows GUI
            if os.environ.get("DISPLAY") is not None or os.name == "nt":
                DisplayUtils.show_summary_popup(summary)
            else:
                raise RuntimeError("No GUI environment")
        except Exception as e:
            print(f"[Fallback Notice] {e}")
            DisplayUtils.show_summary_console(summary)

    @staticmethod
    def show_feature_summary_popup(feature_name: str, summary: dict, title="Feature Summary"):
        try:
            root = tk.Tk()
        except tk.TclError:
            print("GUI not supported.")
            return

        root.title(f"{title} - {feature_name}")
        root.geometry("1000x800")

        main_canvas = tk.Canvas(root)
        scrollbar = ttk.Scrollbar(
            root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(
                scrollregion=main_canvas.bbox("all"))
        )

        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)

        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def add_table(frame, title, data, headers):
            label = ttk.Label(frame, text=title, font=("Arial", 12, "bold"))
            label.pack(anchor="w", pady=(10, 2), padx=10)

            tree = ttk.Treeview(frame, columns=headers, show="headings")
            for h in headers:
                tree.heading(
                    h, text=h, command=lambda _h=h: sort_column(tree, _h, False))
                tree.column(h, anchor="center", width=200)

            for row in data:
                tree.insert("", "end", values=row)
            tree.pack(fill="x", padx=10)

        def sort_column(treeview, col, reverse):
            data_list = [(treeview.set(k, col), k)
                         for k in treeview.get_children('')]
            try:
                data_list.sort(key=lambda t: float(t[0]), reverse=reverse)
            except ValueError:
                data_list.sort(key=lambda t: t[0], reverse=reverse)
            for index, (val, k) in enumerate(data_list):
                treeview.move(k, '', index)
            treeview.heading(col, command=lambda: sort_column(
                treeview, col, not reverse))

        # Basic statistics
        numeric_summary = [(k, f"{v:.4f}" if isinstance(v, float) else v)
                           for k, v in summary.items()
                           if isinstance(v, (int, float)) and not isinstance(v, bool)]
        add_table(scrollable_frame, "Basic Statistics",
                  numeric_summary, ["Metric", "Value"])

        # Frequency statistics
        for group_name in ["most_frequent", "largest_values", "smallest_values", "categories"]:
            if group_name in summary and isinstance(summary[group_name], dict):
                data = [(k, v) for k, v in summary[group_name].items()]
                add_table(scrollable_frame, group_name.replace(
                    '_', ' ').title(), data, ["Value", "Count"])

        root.mainloop()

    @staticmethod
    def show_all_feature_summaries_popup(feature_summaries: dict, title="All Feature Summaries"):
        try:
            root = tk.Tk()
        except tk.TclError:
            print("GUI not supported.")
            return

        root.title(title)
        root.geometry("1000x800")

        selected_feature = tk.StringVar()
        feature_names = list(feature_summaries.keys())

        # Dropdown to select feature
        dropdown = ttk.Combobox(
            root, textvariable=selected_feature, values=feature_names, state="readonly")
        dropdown.pack(pady=10)
        selected_feature.set(feature_names[0])

        # Frame for scrollable content
        main_canvas = tk.Canvas(root)
        scrollbar = ttk.Scrollbar(
            root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)

        scrollable_frame.bind("<Configure>", lambda e: main_canvas.configure(
            scrollregion=main_canvas.bbox("all")))

        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)

        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def add_table(frame, title, data, headers):
            label = ttk.Label(frame, text=title, font=("Arial", 12, "bold"))
            label.pack(anchor="w", pady=(10, 2), padx=10)

            tree = ttk.Treeview(frame, columns=headers, show="headings")
            for h in headers:
                tree.heading(
                    h, text=h, command=lambda _h=h: sort_column(tree, _h, False))
                tree.column(h, anchor="center", width=200)

            for row in data:
                tree.insert("", "end", values=row)
            tree.pack(fill="x", padx=10)

        def sort_column(treeview, col, reverse):
            data_list = [(treeview.set(k, col), k)
                         for k in treeview.get_children('')]
            try:
                data_list.sort(key=lambda t: float(t[0]), reverse=reverse)
            except ValueError:
                data_list.sort(key=lambda t: t[0], reverse=reverse)
            for index, (val, k) in enumerate(data_list):
                treeview.move(k, '', index)
            treeview.heading(col, command=lambda: sort_column(
                treeview, col, not reverse))

        def update_summary(*args):
            for widget in scrollable_frame.winfo_children():
                widget.destroy()
            summary = feature_summaries[selected_feature.get()]

            numeric_summary = [(k, f"{v:.4f}" if isinstance(v, float) else v)
                               for k, v in summary.items()
                               if isinstance(v, (int, float)) and not isinstance(v, bool)]
            add_table(scrollable_frame, "Basic Statistics",
                      numeric_summary, ["Metric", "Value"])

            for group_name in ["most_frequent", "largest_values", "smallest_values", "categories"]:
                if group_name in summary and isinstance(summary[group_name], dict):
                    data = [(k, v) for k, v in summary[group_name].items()]
                    add_table(scrollable_frame, group_name.replace(
                        '_', ' ').title(), data, ["Value", "Count"])

        dropdown.bind("<<ComboboxSelected>>", update_summary)
        update_summary()

        root.mainloop()

    @staticmethod
    def print_feature_summary(feature, summary):
        """
        Prints a detailed summary of a single feature to the console.
        """
        print(f"\n=== Feature: {feature} ===")

        # Basic stats (excluding nested dicts)
        basic_stats = {k: v for k,
                       v in summary.items() if not isinstance(v, dict)}
        table = [[k, f"{v:.2f}" if isinstance(
            v, float) else v] for k, v in basic_stats.items()]
        print(tabulate(table, headers=[
              "Statistic", "Value"], tablefmt="pretty"))

        # Dict-based sections
        def print_dict_section(title, section_key):
            if section_key in summary:
                print(f"\n{title}:")
                section = summary[section_key]
                section_table = [[k, v] for k, v in section.items()]
                print(tabulate(section_table, headers=[
                      "Value", "Count"], tablefmt="pretty"))

        print_dict_section("Most Frequent Values", "most_frequent")
        print_dict_section("Largest Values", "largest_values")
        print_dict_section("Smallest Values", "smallest_values")

    @staticmethod
    def print_high_level_summary(all_summaries):
        """
        Prints a high-level summary of all features in a tabular format to the console.
        Only shows basic (non-dict) stats.
        """
        # Determine common stats (excluding nested dicts)
        first_feature = next(iter(all_summaries))
        base_stats = [
            k for k, v in all_summaries[first_feature].items() if not isinstance(v, dict)]

        table_data = []
        headers = ["Statistic"] + list(all_summaries.keys())
        for stat in base_stats:
            row = [stat]
            for feature in all_summaries:
                val = all_summaries[feature].get(stat, "N/A")
                row.append(f"{val:.2f}" if isinstance(val, float) else val)
            table_data.append(row)

        print("\n=== High-Level Feature Summary ===")
        print(tabulate(table_data, headers=headers, tablefmt="pretty"))
