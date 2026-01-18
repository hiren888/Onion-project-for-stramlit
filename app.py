import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags
import pandas as pd
from roboflow import Roboflow
import gc

# --- CONFIGURATION ---
GRADE_STANDARDS = {
    "Small": (0, 50.8),
    "Medium": (50.8, 76.2),
    "Large": (76.2, 95.0),
    "Colossal": (95.0, 1000)
}

GRADE_COLORS = {
    "Small": (255, 200, 0),    
    "Medium": (0, 255, 150),   
    "Large": (0, 150, 255),    
    "Colossal": (255, 0, 255
