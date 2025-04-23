import numpy as np
import matplotlib.colors as mcolors
import imageio
import os


def mandelbrot(c, max_iter):
    """
    Vypočítá počet iterací pro bod c v Mandelbrotově množině.
    Vrací počet iterací, než |z| > 2, nebo max_iter, pokud bod patří do množiny.
    """
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:  # Bod unikl z kruhu s poloměrem 2
            return n
        z = z * z + c
    return max_iter  # Bod patří do množiny


def julia(z, c, max_iter):
    """
    Vypočítá počet iterací pro bod z v Juliově množině s konstantou c.
    Vrací počet iterací, než |z| > 2, nebo max_iter.
    """
    for n in range(max_iter):
        if abs(z) > 2:  # Bod unikl a nepatří do množiny
            return n
        z = z * z + c
    return max_iter  # Bod patří do množiny


def create_fractal(width, height, xmin, xmax, ymin, ymax, max_iter=256, fractal_type='mandelbrot', c=None):
    """
    Generuje 2D pole reprezentující fraktál (Mandelbrotův nebo Juliův).
    Každý prvek pole obsahuje počet iterací pro daný bod komplexní roviny.
    """
    img = np.zeros((height, width))  # Inicializace pole pro obrázek
    # Mapování pixelových souřadnic na komplexní čísla
    x_coords = np.linspace(xmin, xmax, width)
    y_coords = np.linspace(ymin, ymax, height)

    # Iterace přes každý pixel (bod v komplexní rovině)
    for i in range(height):
        for j in range(width):
            z_point = complex(x_coords[j], y_coords[i])  # Komplexní číslo odpovídající pixelu
            if fractal_type == 'mandelbrot':
                img[i, j] = mandelbrot(z_point, max_iter)
            elif fractal_type == 'julia':
                # Pro Juliovu množinu je 'c' konstanta, 'z' je počáteční bod
                img[i, j] = julia(z_point, c, max_iter)

    return img


def create_high_quality_animation(width, height, center, initial_zoom, zoom_factor, num_frames, max_iter=256,
                                  fractal_type='mandelbrot', c=None, output_path='output.gif'):
    """Vytváří animaci (GIF) zoomování do fraktálu a ukládá úvodní snímek jako PNG."""
    # Zajistíme, že výstupní adresář existuje
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Vytvořen adresář: {output_dir}")

    x_center, y_center = center  # Střed zoomu

    # Definice barevné mapy pro vizualizaci
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
    # Výpočet hranic pro první snímek
    xmin_init = x_center - initial_zoom
    xmax_init = x_center + initial_zoom
    ymin_init = y_center - initial_zoom
    ymax_init = y_center + initial_zoom

    # Generování dat pro první snímek
    first_frame_img = create_fractal(width, height, xmin_init, xmax_init, ymin_init, ymax_init,
                                     max_iter, fractal_type, c)

    # Zpracování a uložení prvního snímku jako PNG
    with np.errstate(divide='ignore'):  # Ignorování varování při logaritmování nuly
        log_img_init = np.log(first_frame_img + 1)  # Logaritmování pro lepší vizuální kontrast (+1 zabraňuje log(0))
        # Normalizace hodnot pro mapování na barvy
        norm_init = mcolors.Normalize(vmin=log_img_init.min(), vmax=log_img_init.max())

    static_image_filename = os.path.splitext(output_path)[0] + ".png"
    colored_frame_init = cmap(norm_init(log_img_init))  # Aplikace barevné mapy
    image_data_init = (colored_frame_init[:, :, :3] * 255).astype(np.uint8)  # Převod na 8-bit RGB
    imageio.imwrite(static_image_filename, image_data_init)
    print(f"Úvodní snímek uložen do {static_image_filename}")

    # Uložení zpracovaného prvního snímku pro GIF
    frames.append((log_img_init, norm_init))

    # Generování zbývajících snímků pro animaci
    for frame in range(1, num_frames):
        # Výpočet aktuálního zoomu a hranic
        current_zoom = initial_zoom * (zoom_factor ** frame)
        xmin = x_center - current_zoom
        xmax = x_center + current_zoom
        ymin = y_center - current_zoom
        ymax = y_center + current_zoom

        print(f"Generuji snímek {frame + 1}/{num_frames} pro animaci (zoom: {current_zoom:.6f})...")
        # Generování dat fraktálu pro aktuální snímek
        frame_img = create_fractal(width, height, xmin, xmax, ymin, ymax,
                                   max_iter, fractal_type, c)

        # Zpracování snímku (logaritmování a normalizace)
        with np.errstate(divide='ignore'):
            log_img = np.log(frame_img + 1)
            norm = mcolors.Normalize(vmin=log_img.min(), vmax=log_img.max())
            frames.append((log_img, norm))  # Uložení zpracovaného snímku

    print("Vytvářím GIF animaci...")
    # Sestavení GIFu z jednotlivých snímků
    with imageio.get_writer(output_path, mode='I', duration=0.05) as writer:  # duration = doba zobrazení jednoho snímku
        for log_img, norm in frames:
            colored_frame = cmap(norm(log_img))  # Aplikace barevné mapy
            # Přidání obarveného snímku (převedeného na 8-bit RGB) do GIFu
            writer.append_data((colored_frame[:, :, :3] * 255).astype(np.uint8))

    print(f"Animace uložena do {output_path}")


if __name__ == '__main__':
    width, height = 800, 800  # Rozměry obrázku/animace
    max_iter = 256  # Maximální počet iterací pro výpočet fraktálu
    num_frames = 125  # Počet snímků v animaci

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print("Vytvářím vylepšený Mandelbrotův zoom...")
    create_high_quality_animation(
        width, height,
        center=(-0.75, 0.1),  # Střed zoomu
        initial_zoom=1.5,  # Počáteční "velikost" pohledu
        zoom_factor=0.95,  # Faktor přiblížení (menší než 1 pro zoom in)
        num_frames=num_frames,
        max_iter=max_iter,
        fractal_type='mandelbrot',
        output_path=os.path.join(results_dir, 'mandelbrot.gif')  # Cesta pro uložení GIFu
    )

    print("Vytvářím vylepšený Juliův zoom...")
    create_high_quality_animation(
        width, height,
        center=(0.0, 0.0),  # Střed zoomu
        initial_zoom=1.5,
        zoom_factor=0.95,
        num_frames=num_frames // 2,  # Kratší animace pro Julii
        max_iter=max_iter,
        fractal_type='julia',
        c=complex(-0.7, 0.27015),  # Konstanta 'c' specifická pro tuto Juliovu množinu
        output_path=os.path.join(results_dir, 'julia.gif')
    )
