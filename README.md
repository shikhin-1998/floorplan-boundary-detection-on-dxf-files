# floorplan-boundary-detection-on-dxf-files



This tool processes a DXF file, renders it to an image, detects the outermost contour using OpenCV, maps the contour back to DXF coordinates, and overlays it onto the original DXF file.

## Features

- Converts DXF to PNG using `ezdxf` and `matplotlib`
- Detects the outer boundary contour from the rendered image
- Maps the image-based contour to DXF coordinates
- Draws the detected contour back onto the DXF file
- Saves the new DXF with the overlaid contour

## Requirements

- Python 3.7+
- OpenCV
- ezdxf
- matplotlib

Install dependencies using:

```bash
pip install -r requirements.txt
