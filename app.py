
import cv2
import ezdxf
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend suitable for server/headless rendering
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import matplotlib as draw_matplotlib
from ezdxf.addons.drawing import RenderContext, Frontend

# Function to convert DXF to PNG image
def dxf_to_image(dxf_path, png_path, dpi=300):
    # Load DXF document
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # Calculate bounding box of the DXF elements
    min_x, min_y, max_x, max_y = get_dxf_extents(msp)
    print(f"DXF Bounding Box: ({min_x}, {min_y}) to ({max_x}, {max_y})")

    # Calculate width and height of the DXF content
    dxf_width = max_x - min_x
    dxf_height = max_y - min_y

    # Set up rendering figure with adjusted size to remove whitespace
    # The aspect ratio is preserved by matching the bounding box size to the figure size
    fig = plt.figure(figsize=(dxf_width / dpi, dxf_height / dpi))  # Adjust figure size based on DXF content
    ax = fig.add_axes([0, 0, 1, 1])  # Full size figure
    ax.axis("off")  # Hide axis for a clean image

    # Setup rendering context and backend for drawing DXF
    ctx = RenderContext(doc)
    backend = draw_matplotlib.MatplotlibBackend(ax)
    frontend = Frontend(ctx, backend)
    frontend.draw_layout(msp)

    # Save the rendered image as PNG with tight bounding box
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    print(f"Saved rendered DXF image to {png_path}")
    return doc, msp, (min_x, min_y, max_x, max_y)

# Function to calculate DXF bounding box manually (in case bbox() isn't available)
def get_dxf_extents(msp):
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    for entity in msp:
        if hasattr(entity.dxf, 'start') and hasattr(entity.dxf, 'end'):
            min_x = min(min_x, entity.dxf.start.x, entity.dxf.end.x)
            min_y = min(min_y, entity.dxf.start.y, entity.dxf.end.y)
            max_x = max(max_x, entity.dxf.start.x, entity.dxf.end.x)
            max_y = max(max_y, entity.dxf.start.y, entity.dxf.end.y)

    return min_x, min_y, max_x, max_y



def find_outer_contour(image_path, output_path="contour_output.png"):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contour found")

    biggest = max(contours, key=cv2.contourArea)

    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_img, [biggest], -1, (0, 0, 255), 2)

    # Save the image with contour
    cv2.imwrite(output_path, output_img)

    # Optionally show the image
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.title("Detected Boundary")
    plt.axis("off")
    plt.show()

    return biggest, img.shape


def map_image_to_dxf_coords(contour, image_shape, dxf_bounds):
    img_h, img_w = image_shape[:2]
    min_x, min_y, max_x, max_y = dxf_bounds

    dxf_width = max_x - min_x
    dxf_height = max_y - min_y

    mapped_points = []
    for pt in contour:
        x_img = pt[0][0]
        y_img = pt[0][1]

        # Map image pixel coordinates to DXF units (accounting for Y-axis flip)
        x_dxf = min_x + (x_img / img_w) * dxf_width
        y_dxf = min_y + ((img_h - y_img) / img_h) * dxf_height

        mapped_points.append((x_dxf, y_dxf))

    return mapped_points


def get_dxf_bounds(msp):
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    for entity in msp:
        if hasattr(entity.dxf, 'start') and hasattr(entity.dxf, 'end'):
            min_x = min(min_x, entity.dxf.start.x, entity.dxf.end.x)
            min_y = min(min_y, entity.dxf.start.y, entity.dxf.end.y)
            max_x = max(max_x, entity.dxf.start.x, entity.dxf.end.x)
            max_y = max(max_y, entity.dxf.start.y, entity.dxf.end.y)

    return min_x, min_y, max_x, max_y


def draw_contour_on_dxf(doc, msp, mapped_points, output_path):
    for i in range(len(mapped_points) - 1):
        msp.add_line(mapped_points[i], mapped_points[i + 1])

    # Close the loop
    if len(mapped_points) > 2:
        msp.add_line(mapped_points[-1], mapped_points[0])

    doc.saveas(output_path)
    print(f"Saved DXF with contour to {output_path}")


def full_pipeline(dxf_path, output_dxf_path, temp_image_path="rendered.png"):
    doc, msp, _ = dxf_to_image(dxf_path, temp_image_path)
    contour, image_shape = find_outer_contour(temp_image_path)
    dxf_bounds = get_dxf_bounds(msp)
    mapped_points = map_image_to_dxf_coords(contour, image_shape, dxf_bounds)
    draw_contour_on_dxf(doc, msp, mapped_points, output_dxf_path)


# === Run the pipeline ===
full_pipeline(
    dxf_path="845 Modified.dxf",
    output_dxf_path="845-Modified-with-contour.dxf"
)
