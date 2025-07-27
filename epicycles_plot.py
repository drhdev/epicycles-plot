#!/usr/bin/env python3
"""
Name: epicycles_plot.py
Author: drhdev
Date: 2025-07-27
Version: 1.0.0
Description: Generate and visualize epicycles with winding number coloring. Reads all parameters from a JSON file, logs all steps, and supports reproducibility.
License: G
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, Optional, Any, Dict
from matplotlib.path import Path
from matplotlib import colors
import argparse
import logging

# ========== Logging Setup ==========
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "epicycles_plot.log")
LOGGER = None

def setup_logger(verbose: bool = False):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except Exception as e:
        print(f"[FATAL] Could not create log directory '{LOG_DIR}': {e}")
        sys.exit(10)
    logger = logging.getLogger("epicycles_plot")
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    try:
        fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    except Exception as e:
        print(f"[FATAL] Could not open log file '{LOG_FILE}': {e}")
        sys.exit(11)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    if verbose:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

# ========== Config & Calculated Classes ==========
class Config:
    """Hält alle Benutzereingaben für die Epizykel-Grafik."""
    REQUIRED_FIELDS = [
        "N_TERMS", "DURATION", "POINTS", "SAVE_DIR", "RADIUS_RANGE", "FREQ_RANGE", "PHASE_RANGE",
        "IMG_WIDTH", "IMG_HEIGHT", "BG_COLOR", "CURVE_COLOR", "ODD_COLOR", "EVEN_COLOR"
    ]
    def __init__(self, d: Dict[str, Any]):
        for field in self.REQUIRED_FIELDS:
            if field not in d:
                raise ValueError(f"Missing required config field: {field}")
        try:
            self.N_TERMS: int = int(d["N_TERMS"])
            self.DURATION: float = float(d["DURATION"])
            self.POINTS: int = int(d["POINTS"])
            self.SAVE_DIR: str = str(d["SAVE_DIR"])
            self.RADIUS_RANGE: Tuple[float, float] = tuple(map(float, d["RADIUS_RANGE"]))
            self.FREQ_RANGE: Tuple[int, int] = tuple(map(int, d["FREQ_RANGE"]))
            self.PHASE_RANGE: Tuple[float, float] = tuple(map(float, d["PHASE_RANGE"]))
            self.IMG_WIDTH: int = int(d["IMG_WIDTH"])
            self.IMG_HEIGHT: int = int(d["IMG_HEIGHT"])
            self.BG_COLOR: str = d["BG_COLOR"]
            self.CURVE_COLOR: str = d["CURVE_COLOR"]
            self.ODD_COLOR: str = d["ODD_COLOR"]
            self.EVEN_COLOR: str = d["EVEN_COLOR"]
            self.RANDOM_SEED: Optional[int] = int(d["RANDOM_SEED"]) if d.get("RANDOM_SEED") is not None else None
        except Exception as e:
            raise ValueError(f"Invalid config value or type: {e}")
        # Additional value checks
        if self.N_TERMS < 1:
            raise ValueError("N_TERMS must be >= 1")
        if self.POINTS < 2:
            raise ValueError("POINTS must be >= 2")
        if self.IMG_WIDTH < 10 or self.IMG_HEIGHT < 10:
            raise ValueError("IMG_WIDTH and IMG_HEIGHT must be >= 10")
        if self.RADIUS_RANGE[0] < 0 or self.RADIUS_RANGE[1] < self.RADIUS_RANGE[0]:
            raise ValueError("RADIUS_RANGE must be [min >= 0, max >= min]")
        if self.FREQ_RANGE[0] < 1 or self.FREQ_RANGE[1] < self.FREQ_RANGE[0]:
            raise ValueError("FREQ_RANGE must be [min >= 1, max >= min]")
        if self.PHASE_RANGE[1] < self.PHASE_RANGE[0]:
            raise ValueError("PHASE_RANGE must be [min, max >= min]")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "N_TERMS": self.N_TERMS,
            "DURATION": self.DURATION,
            "POINTS": self.POINTS,
            "SAVE_DIR": self.SAVE_DIR,
            "RADIUS_RANGE": list(self.RADIUS_RANGE),
            "FREQ_RANGE": list(self.FREQ_RANGE),
            "PHASE_RANGE": list(self.PHASE_RANGE),
            "IMG_WIDTH": self.IMG_WIDTH,
            "IMG_HEIGHT": self.IMG_HEIGHT,
            "BG_COLOR": self.BG_COLOR,
            "CURVE_COLOR": self.CURVE_COLOR,
            "ODD_COLOR": self.ODD_COLOR,
            "EVEN_COLOR": self.EVEN_COLOR,
            "RANDOM_SEED": self.RANDOM_SEED
        }

class Calculated:
    """
    Hält alle berechneten und zufällig generierten Werte, sodass ein Bild exakt reproduzierbar ist.
    """
    def __init__(self, cfg: Config, d: Dict[str, Any] = None):
        if d is not None:
            try:
                self.t = np.array(d["t"])
                self.radii = np.array(d["radii"])
                self.freqs = np.array(d["freqs"])
                self.phases = np.array(d["phases"])
                self.x = np.array(d["x"])
                self.y = np.array(d["y"])
                self.x_norm = np.array(d["x_norm"])
                self.y_norm = np.array(d["y_norm"])
                self.x_img = np.array(d["x_img"])
                self.y_img = np.array(d["y_img"])
                self.curve_points = np.array(d["curve_points"])
            except Exception as e:
                raise ValueError(f"Invalid calculated values in JSON: {e}")
        else:
            try:
                if cfg.RANDOM_SEED is not None:
                    np.random.seed(cfg.RANDOM_SEED)
                else:
                    np.random.seed()
                self.t = np.linspace(0, cfg.DURATION, cfg.POINTS)
                self.radii = np.random.uniform(*cfg.RADIUS_RANGE, cfg.N_TERMS)
                self.freqs = np.random.randint(*cfg.FREQ_RANGE, cfg.N_TERMS)
                self.phases = np.random.uniform(*cfg.PHASE_RANGE, cfg.N_TERMS)
                self.x = np.zeros_like(self.t)
                self.y = np.zeros_like(self.t)
                for r, f, p in zip(self.radii, self.freqs, self.phases):
                    self.x += r * np.cos(f * self.t + p)
                    self.y += r * np.sin(f * self.t + p)
                self.x_norm = (self.x - self.x.min()) / (self.x.max() - self.x.min())
                self.y_norm = (self.y - self.y.min()) / (self.y.max() - self.y.min())
                self.x_img = self.x_norm * (cfg.IMG_WIDTH - 1)
                self.y_img = self.y_norm * (cfg.IMG_HEIGHT - 1)
                self.curve_points = np.vstack([self.x_img, self.y_img]).T
            except Exception as e:
                raise RuntimeError(f"Error during calculation: {e}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": self.t.tolist(),
            "radii": self.radii.tolist(),
            "freqs": self.freqs.tolist(),
            "phases": self.phases.tolist(),
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "x_norm": self.x_norm.tolist(),
            "y_norm": self.y_norm.tolist(),
            "x_img": self.x_img.tolist(),
            "y_img": self.y_img.tolist(),
            "curve_points": self.curve_points.tolist()
        }

# ========== Utility Functions ==========
def validate_json(json_path: str, logger) -> Dict[str, Any]:
    if not os.path.exists(json_path):
        logger.error(f"Input-JSON {json_path} not found.")
        raise FileNotFoundError(f"Input-JSON {json_path} not found.")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Could not parse JSON file {json_path}: {e}")
        raise ValueError(f"Could not parse JSON file {json_path}: {e}")
    if "config" in data:
        config_data = data["config"]
    else:
        config_data = data
    # Check required fields
    missing = [field for field in Config.REQUIRED_FIELDS if field not in config_data]
    if missing:
        logger.error(f"Missing required config fields in {json_path}: {missing}")
        raise ValueError(f"Missing required config fields in {json_path}: {missing}")
    logger.info(f"Input JSON {json_path} successfully validated.")
    return data

def save_epicycle_plot(calc: Calculated, cfg: Config, basename: str, logger) -> str:
    try:
        os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    except Exception as e:
        logger.error(f"Could not create output directory '{cfg.SAVE_DIR}': {e}")
        raise
    try:
        if isinstance(cfg.BG_COLOR, str):
            bg_rgb = np.array(colors.to_rgb(cfg.BG_COLOR))
        else:
            bg_rgb = np.array(cfg.BG_COLOR)
        if isinstance(cfg.ODD_COLOR, str):
            odd_rgb = np.array(colors.to_rgb(cfg.ODD_COLOR))
        else:
            odd_rgb = np.array(cfg.ODD_COLOR)
        if isinstance(cfg.EVEN_COLOR, str):
            even_rgb = np.array(colors.to_rgb(cfg.EVEN_COLOR))
        else:
            even_rgb = np.array(cfg.EVEN_COLOR)
        img = np.ones((cfg.IMG_HEIGHT, cfg.IMG_WIDTH, 3), dtype=np.float32) * bg_rgb
        path = Path(calc.curve_points)
        yy, xx = np.mgrid[0:cfg.IMG_HEIGHT, 0:cfg.IMG_WIDTH]
        points = np.vstack((xx.ravel(), yy.ravel())).T
        wn = path.contains_points(points, radius=-1e-9)
        wn = wn.reshape((cfg.IMG_HEIGHT, cfg.IMG_WIDTH))
        # Set even/odd coloring
        img[~wn] = even_rgb
        img[wn] = odd_rgb
        fig, ax = plt.subplots(figsize=(cfg.IMG_WIDTH/100, cfg.IMG_HEIGHT/100), dpi=100)
        ax.imshow(img, origin='lower')
        ax.plot(calc.x_img, calc.y_img, linewidth=2, color=cfg.CURVE_COLOR)
        ax.set_aspect('equal')
        ax.axis('off')
        filename = f"{basename}.png"
        filepath = os.path.join(cfg.SAVE_DIR, filename)
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        logger.info(f"Image saved: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error during image creation or saving: {e}")
        raise

def save_json(cfg: Config, calc: Calculated, basename: str, logger) -> str:
    data = {
        "config": cfg.to_dict(),
        "calculated": calc.to_dict()
    }
    json_path = os.path.join(cfg.SAVE_DIR, f"{basename}.json")
    try:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Parameter JSON saved: {json_path}")
        return json_path
    except Exception as e:
        logger.error(f"Error writing output JSON file '{json_path}': {e}")
        raise

def parse_args():
    parser = argparse.ArgumentParser(
        description="Epicycles Plotter: Generate and visualize epicycles with winding number coloring.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-jsoninput", type=str, default="default-input.json",
        help="Path to JSON input file (default: default-input.json)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose output (log to screen as well as logfile)"
    )
    parser.epilog = (
        "Examples:\n"
        "  python epicycles_plot.py\n"
        "  python epicycles_plot.py -jsoninput myparams.json\n"
        "  python epicycles_plot.py -jsoninput output/epicycle_YYYYMMDD_HHMMSS.json\n"
        "  python epicycles_plot.py -v\n"
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    global LOGGER
    LOGGER = setup_logger(verbose=args.verbose)
    try:
        data = validate_json(args.jsoninput, LOGGER)
        if "config" in data:
            cfg = Config(data["config"])
            if "calculated" in data:
                calc = Calculated(cfg, data["calculated"])
                LOGGER.info("Loaded calculated values from input JSON (repro mode).")
            else:
                calc = Calculated(cfg)
                LOGGER.info("Generated new calculated values from config.")
        else:
            cfg = Config(data)
            calc = Calculated(cfg)
            LOGGER.info("Generated new calculated values from config.")
    except Exception as e:
        LOGGER.error(f"Script aborted due to error: {e}")
        print(f"[ERROR] {e}")
        sys.exit(1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    basename = f"epicycle_{timestamp}"
    try:
        img_path = save_epicycle_plot(calc, cfg, basename, LOGGER)
        json_path = save_json(cfg, calc, basename, LOGGER)
        LOGGER.info("Script completed successfully. All files written.")
        print(f"Grafik gespeichert unter: {img_path}")
        print(f"Parameter gespeichert unter: {json_path}")
    except Exception as e:
        LOGGER.error(f"Script failed during file writing: {e}")
        print(f"[ERROR] {e}")
        sys.exit(2)

if __name__ == "__main__":
    main() 