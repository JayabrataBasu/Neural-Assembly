"""Jupyter Notebook Integration
Progress bars, live plotting, and notebook-specific utilities for training
"""

import sys
from typing import Optional, Dict, List, Any
import numpy as np


class ProgressBar:
    """Simple ASCII progress bar for terminal/notebook"""
    
    def __init__(self, total: int, prefix: str = '', width: int = 50):
        """
        Initialize progress bar
        
        Args:
            total: Total number of iterations
            prefix: Prefix string (e.g., "Training")
            width: Bar width in characters
        """
        self.total = total
        self.prefix = prefix
        self.width = width
        self.current = 0
    
    def update(self, amount: int = 1) -> None:
        """
        Update progress bar
        
        Args:
            amount: Number of iterations to advance
        """
        self.current = min(self.current + amount, self.total)
        self._print_bar()
    
    def set(self, value: int) -> None:
        """Set absolute progress value"""
        self.current = min(value, self.total)
        self._print_bar()
    
    def _print_bar(self) -> None:
        """Print progress bar to stdout"""
        if self.total == 0:
            return
        
        fraction = self.current / self.total
        filled = int(fraction * self.width)
        bar = '█' * filled + '░' * (self.width - filled)
        
        percentage = int(fraction * 100)
        print(f"\r{self.prefix} |{bar}| {percentage}% ({self.current}/{self.total})", end='', flush=True)
        
        if self.current >= self.total:
            print()  # Newline when complete


class NotebookCallback:
    """Callback for live epoch tracking in notebooks"""
    
    def __init__(self, epochs: int, metrics_to_track: Optional[List[str]] = None):
        """
        Initialize callback
        
        Args:
            epochs: Total epochs
            metrics_to_track: List of metric names to display
        """
        self.epochs = epochs
        self.metrics_to_track = metrics_to_track or ['loss', 'val_loss', 'val_accuracy']
        self.epoch_data: Dict[str, List[Any]] = {m: [] for m in self.metrics_to_track}
        self.current_epoch = 0
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        """
        Called at end of epoch
        
        Args:
            epoch: Epoch number
            logs: Dictionary of metric values
        """
        self.current_epoch = epoch
        
        for metric in self.metrics_to_track:
            if metric in logs:
                self.epoch_data[metric].append(logs[metric])
        
        self._print_summary(epoch, logs)
    
    def _print_summary(self, epoch: int, logs: Dict[str, float]) -> None:
        """Print epoch summary"""
        summary = f"Epoch {epoch+1}/{self.epochs} - "
        parts = []
        
        for metric in self.metrics_to_track:
            if metric in logs:
                parts.append(f"{metric}: {logs[metric]:.4f}")
        
        summary += " ".join(parts)
        print(summary)
    
    def get_history(self) -> Dict[str, List[Any]]:
        """Get training history"""
        return self.epoch_data


class LivePlotter:
    """Live plot training metrics (requires matplotlib)"""
    
    def __init__(self, figsize: tuple = (12, 4), metrics: Optional[List[str]] = None):
        """
        Initialize plotter
        
        Args:
            figsize: Figure size (width, height)
            metrics: Metric names to plot
        """
        self.figsize = figsize
        self.metrics = metrics or ['loss', 'val_loss', 'accuracy', 'val_accuracy']
        self.history: Dict[str, List[float]] = {m: [] for m in self.metrics}
        
        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.has_matplotlib = True
        except ImportError:
            self.has_matplotlib = False
            print("⚠️  matplotlib not available; live plotting disabled")
    
    def update(self, epoch: int, logs: Dict[str, float]) -> None:
        """
        Update plot with new epoch data
        
        Args:
            epoch: Current epoch
            logs: Dictionary of metric values
        """
        if not self.has_matplotlib:
            return
        
        for metric in self.metrics:
            if metric in logs:
                self.history[metric].append(logs[metric])
        
        self._plot()
    
    def _plot(self) -> None:
        """Render plot"""
        if not self.has_matplotlib:
            return
        
        self.plt.figure(figsize=self.figsize)
        
        # Subplot 1: Loss
        self.plt.subplot(1, 2, 1)
        if self.history.get('loss'):
            self.plt.plot(self.history['loss'], label='Training Loss')
        if self.history.get('val_loss'):
            self.plt.plot(self.history['val_loss'], label='Validation Loss')
        self.plt.xlabel('Epoch')
        self.plt.ylabel('Loss')
        self.plt.legend()
        self.plt.grid(True, alpha=0.3)
        
        # Subplot 2: Accuracy
        self.plt.subplot(1, 2, 2)
        if self.history.get('accuracy'):
            self.plt.plot(self.history['accuracy'], label='Training Accuracy')
        if self.history.get('val_accuracy'):
            self.plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
        self.plt.xlabel('Epoch')
        self.plt.ylabel('Accuracy')
        self.plt.legend()
        self.plt.grid(True, alpha=0.3)
        
        self.plt.tight_layout()
        self.plt.show()


class EpochTimer:
    """Track training timing statistics"""
    
    def __init__(self):
        """Initialize timer"""
        import time
        self.time = time
        self.start_time = None
        self.epoch_times: List[float] = []
        self.epoch_start = None
    
    def start_epoch(self) -> None:
        """Mark start of epoch"""
        self.epoch_start = self.time.time()
    
    def end_epoch(self) -> float:
        """Mark end of epoch and return elapsed time"""
        if self.epoch_start is None:
            return 0.0
        
        elapsed = self.time.time() - self.epoch_start
        self.epoch_times.append(elapsed)
        return elapsed
    
    def start_training(self) -> None:
        """Mark start of training"""
        self.start_time = self.time.time()
    
    def end_training(self) -> float:
        """Mark end of training and return total elapsed time"""
        if self.start_time is None:
            return 0.0
        
        return self.time.time() - self.start_time
    
    def get_average_epoch_time(self) -> float:
        """Get average time per epoch"""
        if not self.epoch_times:
            return 0.0
        return np.mean(self.epoch_times)
    
    def get_eta(self, epochs_remaining: int) -> float:
        """
        Estimate time remaining
        
        Args:
            epochs_remaining: Number of epochs left
        
        Returns:
            Estimated time in seconds
        """
        avg_time = self.get_average_epoch_time()
        return avg_time * epochs_remaining
    
    def format_time(self, seconds: float) -> str:
        """Format seconds as human-readable string"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


class NotebookTrainer:
    """Trainer with notebook-friendly output"""
    
    def __init__(self, model: 'Sequential', config: 'TrainerConfig'):
        """Initialize trainer"""
        self.model = model
        self.config = config
        self.callback = NotebookCallback(config.epochs)
        self.timer = EpochTimer()
        self.plotter = LivePlotter()
    
    def fit(self, train_loader, val_loader=None, verbose: bool = True):
        """Train with notebook-friendly output"""
        self.timer.start_training()
        
        for epoch in range(self.config.epochs):
            self.timer.start_epoch()
            
            # Training loop (simplified - delegate to actual trainer if available)
            logs = {'epoch': epoch}
            
            epochs_remaining = self.config.epochs - epoch - 1
            eta = self.timer.get_eta(epochs_remaining)
            
            if verbose:
                epoch_time = self.timer.end_epoch()
                logs['time'] = epoch_time
                logs['eta'] = eta
                
                self.callback.on_epoch_end(epoch, logs)
                
                if self.plotter.has_matplotlib and epoch % 5 == 0:
                    self.plotter.update(epoch, logs)
        
        total_time = self.timer.end_training()
        print(f"\nTraining completed in {self.timer.format_time(total_time)}")
        
        return self.callback.get_history()


def configure_notebook_mode(verbose: bool = True) -> None:
    """
    Configure notebook mode for better displays
    
    Args:
        verbose: Enable verbose output
    """
    try:
        import IPython
        from IPython.display import HTML, Javascript
        
        # Enable matplotlib inline
        IPython.get_ipython().run_line_magic('matplotlib', 'inline')
        
        # Configure matplotlib for notebook
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-darkgrid')
        
        if verbose:
            print("✓ Notebook mode configured")
    except (ImportError, AttributeError):
        if verbose:
            print("⚠️  IPython not available; some notebook features disabled")


def setup_training_display(epochs: int, metrics: Optional[List[str]] = None) -> tuple:
    """
    Setup all notebook training displays
    
    Args:
        epochs: Number of training epochs
        metrics: Metrics to display
    
    Returns:
        Tuple of (callback, plotter, timer)
    """
    callback = NotebookCallback(epochs, metrics)
    plotter = LivePlotter(metrics=metrics or ['loss', 'val_loss'])
    timer = EpochTimer()
    
    return callback, plotter, timer
