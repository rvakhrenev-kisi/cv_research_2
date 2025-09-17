#!/usr/bin/env python3
"""
Simple line marking interface for tailgating detection.
Based on the reference code but simplified for the people-counting repository.
"""

import os
import cv2
import gradio as gr
import numpy as np
import yaml
import json
from pathlib import Path

class SimpleLineMarker:
    def __init__(self):
        self.line_points = [None, None]  # [start_point, end_point]
        self.current_frame = None
        self.video_path = None
        self.cap = None
        self.frame_index = 0
        self.total_frames = 0
        self.fps = 30  # Default FPS
        self.current_dataset = "cisco"  # Default dataset
        self.config_dir = Path("configs")
        self.config_dir.mkdir(exist_ok=True)
    
    def load_video(self, video_file):
        """Load video and extract first frame for line marking."""
        if video_file is None:
            return None, "No video selected", gr.update(maximum=100, value=0)
        
        try:
            self.video_path = video_file.name
            self.cap = cv2.VideoCapture(self.video_path)
            
            if not self.cap.isOpened():
                return None, f"Error: Could not open video '{self.video_path}'", gr.update(maximum=100, value=0)
            
            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.frame_index = 0
            
            # Read first frame
            ret, frame = self.cap.read()
            if not ret:
                return None, "Error reading frame from video", gr.update(maximum=100, value=0)
            
            self.current_frame = frame.copy()
            
            # Reset line points
            self.line_points = [None, None]
            
            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            return frame_rgb, f"Video loaded: {os.path.basename(self.video_path)} ({self.total_frames} frames). Use slider to select frame, then click to mark line points.", gr.update(maximum=self.total_frames-1, value=0)
            
        except Exception as e:
            return None, f"Error loading video: {str(e)}", gr.update(maximum=100, value=0)
    
    def update_frame(self, frame_slider):
        """Update the displayed frame based on slider position."""
        if self.cap is None:
            return None, "No video loaded"
        
        # Convert slider value to frame index
        self.frame_index = min(int(frame_slider), self.total_frames - 1)
        
        # Set the video position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
        ret, frame = self.cap.read()
        
        if not ret:
            return None, f"Error reading frame {self.frame_index}"
        
        self.current_frame = frame.copy()
        
        # Draw existing line if available
        frame_display = self.draw_line_on_frame()
        
        return frame_display, f"Frame: {self.frame_index}/{self.total_frames}"
    
    def on_image_click(self, evt: gr.SelectData):
        """Handle image click for line marking."""
        if self.current_frame is None:
            return None, "No video loaded"
        
        if evt.index is None:
            return None, "Click on the image to mark line points"
        
        x, y = evt.index[0], evt.index[1]
        
        if self.line_points[0] is None:
            # First point
            self.line_points[0] = (x, y)
            return self.draw_line_on_frame(), f"First point set at ({x}, {y}). Click for second point.", x, y, None, None
        else:
            # Second point
            self.line_points[1] = (x, y)
            return self.draw_line_on_frame(), f"Line marked: ({self.line_points[0][0]},{self.line_points[0][1]}) -> ({x},{y})", self.line_points[0][0], self.line_points[0][1], x, y
    
    def draw_line_on_frame(self):
        """Draw line on current frame and return RGB image."""
        if self.current_frame is None:
            return None
        
        # Create a copy of the frame
        frame_with_line = self.current_frame.copy()
        
        # Draw start point if available
        if self.line_points[0]:
            cv2.circle(frame_with_line, self.line_points[0], 8, (0, 0, 255), -1)
        
        # Draw line if both points are available
        if self.line_points[0] and self.line_points[1]:
            cv2.line(frame_with_line, self.line_points[0], self.line_points[1], (0, 255, 0), 3)
            cv2.circle(frame_with_line, self.line_points[1], 8, (0, 0, 255), -1)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_with_line, cv2.COLOR_BGR2RGB)
        return frame_rgb
    
    def reset_line(self):
        """Reset the line points."""
        if self.current_frame is None:
            return None, "No video loaded"
        
        self.line_points = [None, None]
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        return frame_rgb, "Line reset. Click on the image to draw a new line.", 0, 0, 100, 100
    
    def get_line_coordinates(self):
        """Get current line coordinates."""
        if self.line_points[0] and self.line_points[1]:
            return self.line_points[0][0], self.line_points[0][1], self.line_points[1][0], self.line_points[1][1]
        return 0, 0, 100, 100
    
    def generate_command(self, model_type="yolov8", model_size="n", confidence=0.3):
        """Generate the people counter command with current line coordinates."""
        if not self.line_points[0] or not self.line_points[1]:
            return "Please draw a line first"
        
        if not self.video_path:
            return "Please load a video first"
        
        x1, y1, x2, y2 = self.get_line_coordinates()
        video_name = os.path.basename(self.video_path)
        
        command = f"""python people_counter.py \\
  --video {video_name} \\
  --model models/{model_type}{model_size}.pt \\
  --model-type {model_type} \\
  --line-start {x1} {y1} \\
  --line-end {x2} {y2} \\
  --confidence {confidence} \\
  --output output/{video_name.replace('.mp4', '_counting.mp4')} \\
  --show"""
        
        return command
    
    def save_line_config(self, dataset, x1, y1, x2, y2):
        """Save line configuration for a dataset."""
        try:
            config = {
                "line": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                },
                "dataset": dataset,
                "saved_at": str(np.datetime64('now'))
            }
            
            config_file = self.config_dir / f"{dataset}_line_config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            return f"✅ Line configuration saved for {dataset} dataset"
        except Exception as e:
            return f"❌ Error saving configuration: {str(e)}"
    
    def load_line_config(self, dataset):
        """Load line configuration for a dataset."""
        try:
            config_file = self.config_dir / f"{dataset}_line_config.json"
            if not config_file.exists():
                return None, "No saved configuration found for this dataset"
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            line = config.get("line", {})
            x1 = line.get("x1", 0)
            y1 = line.get("y1", 0)
            x2 = line.get("x2", 100)
            y2 = line.get("y2", 100)
            
            # Update line points
            self.line_points = [(x1, y1), (x2, y2)]
            
            # Update current frame display
            if self.current_frame is not None:
                frame_display = self.draw_line_on_frame()
                return frame_display, f"✅ Loaded line configuration for {dataset}: ({x1},{y1}) -> ({x2},{y2})", x1, y1, x2, y2
            else:
                return None, f"✅ Loaded line configuration for {dataset}: ({x1},{y1}) -> ({x2},{y2})", x1, y1, x2, y2
                
        except Exception as e:
            return None, f"❌ Error loading configuration: {str(e)}", 0, 0, 100, 100
    
    def set_dataset(self, dataset):
        """Set the current dataset."""
        self.current_dataset = dataset
        return f"Dataset set to: {dataset}"

def create_line_marking_interface():
    """Create the Gradio interface for line marking."""
    
    line_marker = SimpleLineMarker()
    
    with gr.Blocks(title="Tailgating Line Marker") as demo:
        gr.Markdown("# 🎯 Tailgating Line Marker")
        gr.Markdown("Upload a video and mark the entrance line for people counting.")
        
        with gr.Row():
            with gr.Column():
                # Dataset selection
                gr.Markdown("### 📁 Dataset Selection")
                dataset_dropdown = gr.Dropdown(
                    choices=["cisco", "vortex", "courtyard"],
                    value="cisco",
                    label="Dataset",
                    info="Choose which dataset configuration to use"
                )
                
                video_input = gr.File(
                    label="Video File",
                    file_types=[".mp4", ".avi", ".mov"]
                )
                
                load_btn = gr.Button("Load Video", variant="primary")
                
                # Frame selection
                gr.Markdown("### Frame Selection")
                frame_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    label="Frame Selection",
                    info="Drag to select different frames from the video"
                )
                
                gr.Markdown("### Instructions")
                gr.Markdown("1. Load a video")
                gr.Markdown("2. Use the slider to select the best frame")
                gr.Markdown("3. Click once to set line start point")
                gr.Markdown("4. Click again to set line end point")
                gr.Markdown("5. Use 'Reset Line' to start over")
                
                reset_btn = gr.Button("Reset Line", variant="secondary")
                
                # Configuration management
                gr.Markdown("### 💾 Configuration Management")
                with gr.Row():
                    load_config_btn = gr.Button("Load Saved Line", variant="secondary")
                    save_config_btn = gr.Button("Save Line Config", variant="primary")
                
                config_status = gr.Textbox(
                    label="Configuration Status",
                    interactive=False,
                    value="Select a dataset and draw a line to save configuration"
                )
                
            with gr.Column():
                preview_image = gr.Image(
                    label="Video Preview - Click to mark line points",
                    type="numpy",
                    interactive=True
                )
                
                status_text = gr.Textbox(
                    label="Status",
                    value="Select a video to begin",
                    interactive=False
                )
        
        gr.Markdown("### 📏 Line Coordinates")
        gr.Markdown("Current line coordinates (in pixels):")
        
        with gr.Row():
            x1_input = gr.Number(label="X1", value=0, precision=0)
            y1_input = gr.Number(label="Y1", value=0, precision=0)
            x2_input = gr.Number(label="X2", value=100, precision=0)
            y2_input = gr.Number(label="Y2", value=100, precision=0)
        
        gr.Markdown("### 🚀 Generate People Counter Command")
        gr.Markdown("Configure the detection parameters and generate the command to run people counting:")
        
        with gr.Row():
            model_type = gr.Dropdown(
                choices=["yolov8", "yolov10", "yolov11"],
                value="yolov8",
                label="Model Type"
            )
            model_size = gr.Dropdown(
                choices=["n", "s", "m", "l", "x"],
                value="n",
                label="Model Size"
            )
            confidence = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.3,
                step=0.05,
                label="Confidence Threshold"
            )
        
        generate_btn = gr.Button("Generate Command", variant="primary")
        command_output = gr.Textbox(
            label="People Counter Command",
            lines=8,
            interactive=False,
            placeholder="Draw a line and click 'Generate Command' to see the command"
        )
        
        # Event handlers
        load_btn.click(
            fn=line_marker.load_video,
            inputs=[video_input],
            outputs=[preview_image, status_text, frame_slider]
        )
        
        # Handle frame slider changes
        frame_slider.change(
            fn=line_marker.update_frame,
            inputs=[frame_slider],
            outputs=[preview_image, status_text]
        )
        
        # Handle image clicks
        preview_image.select(
            fn=line_marker.on_image_click,
            inputs=[],
            outputs=[preview_image, status_text, x1_input, y1_input, x2_input, y2_input]
        )
        
        reset_btn.click(
            fn=line_marker.reset_line,
            inputs=[],
            outputs=[preview_image, status_text, x1_input, y1_input, x2_input, y2_input]
        )
        
        generate_btn.click(
            fn=line_marker.generate_command,
            inputs=[model_type, model_size, confidence],
            outputs=[command_output]
        )
        
        # Dataset selection handler
        dataset_dropdown.change(
            fn=line_marker.set_dataset,
            inputs=[dataset_dropdown],
            outputs=[config_status]
        )
        
        # Configuration management handlers
        load_config_btn.click(
            fn=line_marker.load_line_config,
            inputs=[dataset_dropdown],
            outputs=[preview_image, config_status, x1_input, y1_input, x2_input, y2_input]
        )
        
        save_config_btn.click(
            fn=line_marker.save_line_config,
            inputs=[dataset_dropdown, x1_input, y1_input, x2_input, y2_input],
            outputs=[config_status]
        )
    
    return demo

if __name__ == "__main__":
    print("🚀 Starting Tailgating Line Marker...")
    demo = create_line_marking_interface()
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
