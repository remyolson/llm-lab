"""
Real-time Training Dashboard for Fine-Tuning

This module provides a real-time visualization dashboard for monitoring
fine-tuning progress. It displays loss curves, learning rate schedules,
gradient statistics, and other training metrics with live updates.

The dashboard supports both Streamlit (default) and Gradio backends.
"""

import os
import json
import time
import threading
import queue
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import deque, defaultdict

# Visualization imports
try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and manages training metrics for visualization."""
    
    def __init__(self, max_history: int = 10000):
        """Initialize metrics collector.
        
        Args:
            max_history: Maximum number of data points to keep in memory
        """
        self.max_history = max_history
        self.metrics_history = defaultdict(lambda: deque(maxlen=max_history))
        self.current_epoch = 0
        self.current_step = 0
        self.start_time = time.time()
        self.recipe_config = {}
        self._lock = threading.Lock()
    
    def update_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Update metrics with new values.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (auto-increments if not provided)
        """
        with self._lock:
            if step is None:
                step = self.current_step
                self.current_step += 1
            else:
                self.current_step = step
            
            timestamp = time.time()
            
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float, np.number)):
                    self.metrics_history[metric_name].append({
                        'step': step,
                        'value': float(value),
                        'timestamp': timestamp,
                        'epoch': self.current_epoch
                    })
    
    def set_epoch(self, epoch: int):
        """Update current epoch."""
        with self._lock:
            self.current_epoch = epoch
    
    def get_metric_history(self, metric_name: str) -> pd.DataFrame:
        """Get history for a specific metric as DataFrame."""
        with self._lock:
            if metric_name not in self.metrics_history:
                return pd.DataFrame()
            
            data = list(self.metrics_history[metric_name])
            if not data:
                return pd.DataFrame()
            
            return pd.DataFrame(data)
    
    def get_all_metrics(self) -> Dict[str, pd.DataFrame]:
        """Get all metrics as DataFrames."""
        with self._lock:
            return {
                name: self.get_metric_history(name)
                for name in self.metrics_history.keys()
            }
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get the latest value for each metric."""
        with self._lock:
            latest = {}
            for name, history in self.metrics_history.items():
                if history:
                    latest[name] = history[-1]['value']
            return latest
    
    def estimate_time_remaining(self, total_steps: int) -> timedelta:
        """Estimate time remaining based on current progress."""
        if self.current_step == 0:
            return timedelta(seconds=0)
        
        elapsed = time.time() - self.start_time
        steps_per_second = self.current_step / elapsed
        remaining_steps = max(0, total_steps - self.current_step)
        remaining_seconds = remaining_steps / steps_per_second
        
        return timedelta(seconds=int(remaining_seconds))
    
    def export_metrics(self, output_path: str, format: str = 'json'):
        """Export metrics to file.
        
        Args:
            output_path: Path to save metrics
            format: Export format ('json', 'csv', 'parquet')
        """
        with self._lock:
            all_metrics = self.get_all_metrics()
            
            if format == 'json':
                # Convert DataFrames to JSON-serializable format
                export_data = {
                    'metadata': {
                        'start_time': self.start_time,
                        'current_step': self.current_step,
                        'current_epoch': self.current_epoch,
                        'recipe_config': self.recipe_config
                    },
                    'metrics': {}
                }
                
                for name, df in all_metrics.items():
                    if not df.empty:
                        export_data['metrics'][name] = df.to_dict('records')
                
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            
            elif format == 'csv':
                # Export each metric to a separate CSV
                output_dir = Path(output_path).parent
                base_name = Path(output_path).stem
                
                for name, df in all_metrics.items():
                    if not df.empty:
                        csv_path = output_dir / f"{base_name}_{name}.csv"
                        df.to_csv(csv_path, index=False)
            
            elif format == 'parquet':
                # Export all metrics as parquet files
                output_dir = Path(output_path).parent
                base_name = Path(output_path).stem
                
                for name, df in all_metrics.items():
                    if not df.empty:
                        parquet_path = output_dir / f"{base_name}_{name}.parquet"
                        df.to_parquet(parquet_path, index=False)


class TrainingDashboard:
    """Real-time training dashboard for fine-tuning visualization."""
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        recipe_config: Optional[Dict[str, Any]] = None,
        update_interval: float = 1.0,
        backend: str = 'streamlit'
    ):
        """Initialize training dashboard.
        
        Args:
            metrics_collector: MetricsCollector instance
            recipe_config: Recipe configuration for customization
            update_interval: Update interval in seconds
            backend: Dashboard backend ('streamlit' or 'gradio')
        """
        self.metrics_collector = metrics_collector
        self.recipe_config = recipe_config or {}
        self.update_interval = update_interval
        self.backend = backend
        
        # Configure dashboard layout based on recipe
        self.configure_layout()
        
        # WebSocket/SSE queue for live updates
        self.update_queue = queue.Queue()
        self.is_running = False
        self.update_thread = None
    
    def configure_layout(self):
        """Configure dashboard layout based on recipe."""
        # Default metrics to display
        self.primary_metrics = ['loss', 'eval_loss', 'learning_rate']
        self.secondary_metrics = ['gradient_norm', 'loss_scale', 'perplexity']
        
        # Customize based on recipe
        if self.recipe_config:
            training_config = self.recipe_config.get('training', {})
            
            # Add LoRA-specific metrics if using LoRA
            if training_config.get('use_lora', False):
                self.secondary_metrics.extend(['lora_loss', 'lora_alpha_effective'])
            
            # Add custom metrics from recipe
            eval_config = self.recipe_config.get('evaluation', {})
            custom_metrics = eval_config.get('metrics', [])
            self.secondary_metrics.extend(custom_metrics)
    
    def create_loss_chart(self) -> go.Figure:
        """Create loss curves chart."""
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=('Training and Validation Loss',)
        )
        
        # Training loss
        train_loss_df = self.metrics_collector.get_metric_history('loss')
        if not train_loss_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=train_loss_df['step'],
                    y=train_loss_df['value'],
                    mode='lines',
                    name='Training Loss',
                    line=dict(color='blue', width=2)
                )
            )
        
        # Validation loss
        eval_loss_df = self.metrics_collector.get_metric_history('eval_loss')
        if not eval_loss_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=eval_loss_df['step'],
                    y=eval_loss_df['value'],
                    mode='lines+markers',
                    name='Validation Loss',
                    line=dict(color='orange', width=2),
                    marker=dict(size=6)
                )
            )
        
        fig.update_layout(
            height=400,
            xaxis_title='Step',
            yaxis_title='Loss',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def create_learning_rate_chart(self) -> go.Figure:
        """Create learning rate schedule chart."""
        fig = go.Figure()
        
        lr_df = self.metrics_collector.get_metric_history('learning_rate')
        if not lr_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=lr_df['step'],
                    y=lr_df['value'],
                    mode='lines',
                    name='Learning Rate',
                    line=dict(color='green', width=2)
                )
            )
        
        fig.update_layout(
            height=300,
            title='Learning Rate Schedule',
            xaxis_title='Step',
            yaxis_title='Learning Rate',
            yaxis_type='log'
        )
        
        return fig
    
    def create_gradient_chart(self) -> go.Figure:
        """Create gradient statistics chart."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Gradient Norm', 'Gradient Clipping'),
            vertical_spacing=0.15
        )
        
        # Gradient norm
        grad_norm_df = self.metrics_collector.get_metric_history('gradient_norm')
        if not grad_norm_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=grad_norm_df['step'],
                    y=grad_norm_df['value'],
                    mode='lines',
                    name='Gradient Norm',
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )
        
        # Gradient clipping events
        clip_df = self.metrics_collector.get_metric_history('gradient_clipped')
        if not clip_df.empty:
            fig.add_trace(
                go.Bar(
                    x=clip_df['step'],
                    y=clip_df['value'],
                    name='Clipped',
                    marker_color='purple'
                ),
                row=2, col=1
            )
        
        fig.update_xaxes(title_text='Step', row=2, col=1)
        fig.update_yaxes(title_text='Norm', row=1, col=1)
        fig.update_yaxes(title_text='Clipped', row=2, col=1)
        
        fig.update_layout(height=500, showlegend=False)
        
        return fig
    
    def create_metrics_table(self) -> pd.DataFrame:
        """Create current metrics summary table."""
        latest_metrics = self.metrics_collector.get_latest_metrics()
        
        # Format metrics for display
        formatted_metrics = []
        for name, value in latest_metrics.items():
            formatted_metrics.append({
                'Metric': name.replace('_', ' ').title(),
                'Value': f"{value:.6f}" if isinstance(value, float) else str(value)
            })
        
        return pd.DataFrame(formatted_metrics)
    
    def create_progress_indicators(self, total_steps: int) -> Dict[str, Any]:
        """Create progress indicators."""
        current_step = self.metrics_collector.current_step
        current_epoch = self.metrics_collector.current_epoch
        
        # Calculate progress
        progress = min(100, (current_step / total_steps) * 100) if total_steps > 0 else 0
        
        # Estimate time remaining
        time_remaining = self.metrics_collector.estimate_time_remaining(total_steps)
        
        # Calculate throughput
        elapsed = time.time() - self.metrics_collector.start_time
        throughput = current_step / elapsed if elapsed > 0 else 0
        
        return {
            'progress': progress,
            'current_step': current_step,
            'total_steps': total_steps,
            'current_epoch': current_epoch,
            'time_remaining': str(time_remaining).split('.')[0],
            'throughput': f"{throughput:.2f} steps/sec"
        }
    
    def run_streamlit(self):
        """Run Streamlit dashboard."""
        st.set_page_config(
            page_title="Fine-Tuning Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🚀 Fine-Tuning Training Dashboard")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("Configuration")
            
            # Recipe info
            if self.recipe_config:
                st.subheader("Recipe")
                st.text(self.recipe_config.get('name', 'Unknown'))
                st.text(self.recipe_config.get('description', ''))
            
            # Update settings
            auto_refresh = st.checkbox("Auto Refresh", value=True)
            refresh_rate = st.slider("Refresh Rate (seconds)", 1, 10, 2)
            
            # Export options
            st.subheader("Export Metrics")
            export_format = st.selectbox("Format", ["JSON", "CSV", "Parquet"])
            if st.button("Export"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = f"metrics_export_{timestamp}.{export_format.lower()}"
                self.metrics_collector.export_metrics(export_path, export_format.lower())
                st.success(f"Exported to {export_path}")
        
        # Main content
        col1, col2, col3, col4 = st.columns(4)
        
        # Progress indicators
        total_steps = st.session_state.get('total_steps', 10000)
        progress_info = self.create_progress_indicators(total_steps)
        
        with col1:
            st.metric("Progress", f"{progress_info['progress']:.1f}%")
        with col2:
            st.metric("Current Step", f"{progress_info['current_step']:,}")
        with col3:
            st.metric("Time Remaining", progress_info['time_remaining'])
        with col4:
            st.metric("Throughput", progress_info['throughput'])
        
        # Progress bar
        st.progress(progress_info['progress'] / 100)
        
        # Charts
        tab1, tab2, tab3, tab4 = st.tabs(["Loss", "Learning Rate", "Gradients", "Metrics"])
        
        with tab1:
            st.plotly_chart(self.create_loss_chart(), use_container_width=True)
        
        with tab2:
            st.plotly_chart(self.create_learning_rate_chart(), use_container_width=True)
        
        with tab3:
            st.plotly_chart(self.create_gradient_chart(), use_container_width=True)
        
        with tab4:
            metrics_df = self.create_metrics_table()
            if not metrics_df.empty:
                st.dataframe(metrics_df, use_container_width=True)
            else:
                st.info("No metrics available yet")
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(refresh_rate)
            st.rerun()
    
    def run_gradio(self):
        """Run Gradio dashboard."""
        def update_dashboard():
            """Update function for Gradio."""
            total_steps = 10000  # Default, should be configurable
            progress_info = self.create_progress_indicators(total_steps)
            
            # Create plots
            loss_fig = self.create_loss_chart()
            lr_fig = self.create_learning_rate_chart()
            grad_fig = self.create_gradient_chart()
            metrics_df = self.create_metrics_table()
            
            return (
                loss_fig,
                lr_fig,
                grad_fig,
                metrics_df,
                f"Progress: {progress_info['progress']:.1f}%",
                f"Step: {progress_info['current_step']:,}",
                f"Time Remaining: {progress_info['time_remaining']}",
                f"Throughput: {progress_info['throughput']}"
            )
        
        with gr.Blocks(title="Fine-Tuning Dashboard") as interface:
            gr.Markdown("# 🚀 Fine-Tuning Training Dashboard")
            
            with gr.Row():
                progress_text = gr.Textbox(label="Progress", interactive=False)
                step_text = gr.Textbox(label="Current Step", interactive=False)
                time_text = gr.Textbox(label="Time Remaining", interactive=False)
                throughput_text = gr.Textbox(label="Throughput", interactive=False)
            
            with gr.Tabs():
                with gr.TabItem("Loss"):
                    loss_plot = gr.Plot()
                
                with gr.TabItem("Learning Rate"):
                    lr_plot = gr.Plot()
                
                with gr.TabItem("Gradients"):
                    grad_plot = gr.Plot()
                
                with gr.TabItem("Metrics"):
                    metrics_table = gr.DataFrame()
            
            # Auto-update
            interface.load(
                update_dashboard,
                outputs=[
                    loss_plot, lr_plot, grad_plot, metrics_table,
                    progress_text, step_text, time_text, throughput_text
                ],
                every=self.update_interval
            )
        
        interface.launch(server_name="0.0.0.0", server_port=7860)
    
    def run(self):
        """Run the dashboard with the configured backend."""
        if self.backend == 'streamlit' and STREAMLIT_AVAILABLE:
            self.run_streamlit()
        elif self.backend == 'gradio' and GRADIO_AVAILABLE:
            self.run_gradio()
        else:
            logger.error(f"Backend '{self.backend}' not available. Please install required packages.")
            raise RuntimeError(f"Dashboard backend '{self.backend}' not available")
    
    def update_from_trainer(self, logs: Dict[str, Any]):
        """Update metrics from trainer logs (for integration with HuggingFace Trainer).
        
        Args:
            logs: Training logs from trainer
        """
        # Extract relevant metrics
        metrics = {}
        
        # Standard metrics
        for key in ['loss', 'eval_loss', 'learning_rate', 'epoch']:
            if key in logs:
                metrics[key] = logs[key]
        
        # Gradient metrics
        if 'grad_norm' in logs:
            metrics['gradient_norm'] = logs['grad_norm']
        
        # Custom metrics
        for key, value in logs.items():
            if key not in metrics and isinstance(value, (int, float, np.number)):
                metrics[key] = value
        
        # Update collector
        step = logs.get('global_step', None)
        self.metrics_collector.update_metrics(metrics, step)
        
        # Update epoch if changed
        if 'epoch' in logs:
            self.metrics_collector.set_epoch(int(logs['epoch']))


# Callback for HuggingFace Trainer integration
class DashboardCallback:
    """Callback for integrating dashboard with HuggingFace Trainer."""
    
    def __init__(self, dashboard: TrainingDashboard):
        """Initialize callback.
        
        Args:
            dashboard: TrainingDashboard instance
        """
        self.dashboard = dashboard
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs metrics."""
        if logs:
            self.dashboard.update_from_trainer(logs)


def create_dashboard(
    recipe_config: Optional[Dict[str, Any]] = None,
    backend: str = 'streamlit',
    max_history: int = 10000
) -> Tuple[TrainingDashboard, MetricsCollector]:
    """Create a training dashboard instance.
    
    Args:
        recipe_config: Recipe configuration
        backend: Dashboard backend ('streamlit' or 'gradio')
        max_history: Maximum metric history to keep
        
    Returns:
        Tuple of (dashboard, metrics_collector)
    """
    metrics_collector = MetricsCollector(max_history=max_history)
    dashboard = TrainingDashboard(
        metrics_collector=metrics_collector,
        recipe_config=recipe_config,
        backend=backend
    )
    
    return dashboard, metrics_collector


# Example usage
if __name__ == "__main__":
    # Example recipe config
    recipe_config = {
        "name": "llama2_lora_finetune",
        "description": "LoRA fine-tuning for Llama 2",
        "training": {
            "use_lora": True,
            "num_epochs": 3
        },
        "evaluation": {
            "metrics": ["perplexity", "bleu", "rouge"]
        }
    }
    
    # Create dashboard
    dashboard, collector = create_dashboard(recipe_config=recipe_config)
    
    # Simulate some training data
    import random
    for step in range(100):
        collector.update_metrics({
            'loss': 2.5 * np.exp(-step / 50) + random.random() * 0.1,
            'eval_loss': 2.8 * np.exp(-step / 50) + random.random() * 0.15,
            'learning_rate': 2e-5 * (1 - step / 100),
            'gradient_norm': random.random() * 2 + 0.5
        })
        time.sleep(0.1)
    
    # Run dashboard
    # dashboard.run()