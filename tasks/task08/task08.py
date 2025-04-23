import numpy as np
import matplotlib.colors as mcolors
import imageio
import os


def mandelbrot(c, max_iter):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter


def julia(z, c, max_iter):
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter


def create_fractal(width, height, xmin, xmax, ymin, ymax, max_iter=256, fractal_type='mandelbrot', c=None):
    img = np.zeros((height, width))
    x_coords = np.linspace(xmin, xmax, width)
    y_coords = np.linspace(ymin, ymax, height)

    for i in range(height):
        for j in range(width):
            z_point = complex(x_coords[j], y_coords[i])
            if fractal_type == 'mandelbrot':
                img[i, j] = mandelbrot(z_point, max_iter)
            elif fractal_type == 'julia':
                img[i, j] = julia(z_point, c, max_iter)

    return img


def create_high_quality_animation(width, height, center, initial_zoom, zoom_factor, num_frames, max_iter=256,
                                  fractal_type='mandelbrot', c=None, output_path='output.gif'):
    # Zajistíme, že výstupní cesta existuje
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Vytvořen adresář: {output_dir}")

    x_center, y_center = center

    colors_ = [
        (0, 0, 0.2),  # Deep blue-black
        (0, 0, 0.5),  # Dark blue
        (0, 0.3, 0.8),  # Blue
        (0, 0.7, 0.9),  # Cyan
        (0.2, 0.9, 0.8),  # Teal
        (0.8, 1, 0.5),  # Light green-yellow
        (1, 0.8, 0),  # Orange
        (1, 0.3, 0)  # Red
    ]

    cmap = mcolors.LinearSegmentedColormap.from_list('enhanced', colors_)

    frames = []

    print("Generuji úvodní statický snímek...")
    xmin_init = x_center - initial_zoom
    xmax_init = x_center + initial_zoom
    ymin_init = y_center - initial_zoom
    ymax_init = y_center + initial_zoom

    first_frame_img = create_fractal(width, height, xmin_init, xmax_init, ymin_init, ymax_init,
                                     max_iter, fractal_type, c)

    with np.errstate(divide='ignore'):
        log_img_init = np.log(first_frame_img + 1)
        norm_init = mcolors.Normalize(vmin=log_img_init.min(), vmax=log_img_init.max())

    static_image_filename = os.path.splitext(output_path)[0] + ".png"
    colored_frame_init = cmap(norm_init(log_img_init))
    image_data_init = (colored_frame_init[:, :, :3] * 255).astype(np.uint8)
    imageio.imwrite(static_image_filename, image_data_init)
    print(f"Úvodní snímek uložen do {static_image_filename}")

    frames.append((log_img_init, norm_init))

    for frame in range(1, num_frames):
        current_zoom = initial_zoom * (zoom_factor ** frame)
        xmin = x_center - current_zoom
        xmax = x_center + current_zoom
        ymin = y_center - current_zoom
        ymax = y_center + current_zoom

        print(f"Generuji snímek {frame + 1}/{num_frames} pro animaci (zoom: {current_zoom:.6f})...")
        frame_img = create_fractal(width, height, xmin, xmax, ymin, ymax,
                                   max_iter, fractal_type, c)

        with np.errstate(divide='ignore'):
            log_img = np.log(frame_img + 1)  # +1 pro zabránění log(0)
            norm = mcolors.Normalize(vmin=log_img.min(), vmax=log_img.max())
            frames.append((log_img, norm))

    print("Vytvářím GIF animaci...")
    with imageio.get_writer(output_path, mode='I', duration=0.05) as writer:
        for log_img, norm in frames:
            colored_frame = cmap(norm(log_img))
            writer.append_data((colored_frame[:, :, :3] * 255).astype(np.uint8))

    print(f"Animace uložena do {output_path}")


if __name__ == '__main__':
    width, height = 800, 800
    max_iter = 256
    num_frames = 125

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print("Vytvářím vylepšený Mandelbrotův zoom...")
    create_high_quality_animation(
        width, height,
        center=(-0.75, 0.1),
        initial_zoom=1.5,
        zoom_factor=0.95,
        num_frames=num_frames,
        max_iter=max_iter,
        fractal_type='mandelbrot',
        output_path=os.path.join(results_dir, 'mandelbrot.gif')
    )

    print("Vytvářím vylepšený Juliův zoom...")
    create_high_quality_animation(
        width, height,
        center=(0.0, 0.0),
        initial_zoom=1.5,
        zoom_factor=0.95,
        num_frames=num_frames // 2,
        max_iter=max_iter,
        fractal_type='julia',
        c=complex(-0.7, 0.27015),
        output_path=os.path.join(results_dir, 'julia.gif')
    )
