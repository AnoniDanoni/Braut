import argparse
import json
import os
from collections.abc import Iterable
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path.home() / "Desktop" / "Braut imagens"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEFAULT_CHUNK_SIZE = 250_000
DEFAULT_THICKNESS = 0.25
MAX_PIXELS = 4096
MAX_IMAGE_CELLS = 16_000_000
MAX_BATCH_SLICES = 8
DEFAULT_MIN_AREA = 20
MAX_DETECTED_ELEMENTS = 500


def log(message: str) -> None:
    print(f"[LOG] {message}", flush=True)


def lower_process_priority() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        below_normal = 0x00004000
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), below_normal)
    except Exception:
        pass


def prepare_output_dir() -> Path:
    log(f"Preparando pasta de saida: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(exist_ok=True)
    deleted = 0
    for file in OUTPUT_DIR.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            file.unlink()
            deleted += 1
    log(f"Imagens antigas apagadas: {deleted}")
    return OUTPUT_DIR


def output_path(input_path: Path, name: Path | None, height: float, total: int) -> Path:
    if name is None or total > 1:
        safe_height = f"{height:.2f}".replace("-", "menos_").replace(".", "_")
        return OUTPUT_DIR / f"{input_path.stem}_planta_h{safe_height}.png"
    if name.suffix.lower() not in IMAGE_EXTENSIONS:
        name = name.with_suffix(".png")
    return OUTPUT_DIR / name.name


def metadata_path(input_path: Path) -> Path:
    return OUTPUT_DIR / f"{input_path.stem}_fatias.json"


def resolve_input(path: Path) -> Path:
    path = path.expanduser()
    if not path.exists():
        raise SystemExit(f"Arquivo de entrada nao encontrado: {path}")
    log(f"Arquivo de entrada encontrado: {path}")
    return path


def iter_points(path: Path, chunk_size: int) -> Iterable[np.ndarray]:
    suffix = path.suffix.lower()
    log(f"Lendo arquivo {suffix}: {path}")
    if suffix in {".e57", ".e47"}:
        yield from iter_e57(path, chunk_size)
        return
    if suffix == ".ply":
        yield from iter_ascii_ply(path, chunk_size)
        return

    yield from iter_text_points(path, chunk_size)


def iter_e57(path: Path, chunk_size: int) -> Iterable[np.ndarray]:
    try:
        import pye57
        from pye57.e57 import COORDINATE_SYSTEMS, SUPPORTED_CARTESIAN_POINT_FIELDS, SUPPORTED_SPHERICAL_POINT_FIELDS
        from pye57.e57 import convert_spherical_to_cartesian
    except ImportError as exc:
        raise SystemExit("Instale o suporte E57 com: pip install -r requirements.txt") from exc

    log("Abrindo arquivo E57")
    e57 = pye57.E57(str(path))
    try:
        log(f"Scans encontrados: {e57.scan_count}")
        for scan_index in range(e57.scan_count):
            header = e57.get_header(scan_index)
            coordinate_system = header.get_coordinate_system(COORDINATE_SYSTEMS)
            if coordinate_system is COORDINATE_SYSTEMS.CARTESIAN:
                fields = list(SUPPORTED_CARTESIAN_POINT_FIELDS.keys())
                valid_state = "cartesianInvalidState"
            else:
                fields = list(SUPPORTED_SPHERICAL_POINT_FIELDS.keys())
                valid_state = "sphericalInvalidState"

            if valid_state in header.point_fields:
                fields.append(valid_state)

            log(f"Lendo scan {scan_index + 1}/{e57.scan_count} em blocos")
            data, buffers = e57.make_buffers(fields, chunk_size)
            reader = header.points.reader(buffers)
            read_total = 0
            try:
                while True:
                    read_count = reader.read()
                    if read_count == 0:
                        break
                    chunk_data = {field: values[:read_count] for field, values in data.items()}
                    if valid_state in chunk_data:
                        valid = ~chunk_data[valid_state].astype("?")
                        chunk_data = {
                            field: values[valid] for field, values in chunk_data.items() if field != valid_state
                        }

                    if coordinate_system is COORDINATE_SYSTEMS.CARTESIAN:
                        xyz = np.column_stack(
                            (chunk_data["cartesianX"], chunk_data["cartesianY"], chunk_data["cartesianZ"])
                        )
                    else:
                        rae = np.column_stack(
                            (
                                chunk_data["sphericalRange"],
                                chunk_data["sphericalAzimuth"],
                                chunk_data["sphericalElevation"],
                            )
                        )
                        xyz = convert_spherical_to_cartesian(rae)
                    if header.has_pose():
                        xyz = e57.to_global(xyz, header.rotation, header.translation)
                    xyz = xyz[~np.isnan(xyz).any(axis=1)]
                    read_total += len(xyz)
                    yield xyz
            finally:
                reader.close()
            log(f"Scan {scan_index + 1}: {read_total} pontos validos")
    finally:
        e57.close()


def iter_ascii_ply(path: Path, chunk_size: int) -> Iterable[np.ndarray]:
    vertex_count = None
    header_end = None
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        if file.readline().strip() != "ply":
            raise SystemExit("PLY invalido")
        for i, line in enumerate(file, start=1):
            parts = line.split()
            if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
            if line.strip() == "end_header":
                header_end = i + 1
                break

    if vertex_count is None or header_end is None:
        raise SystemExit("PLY ASCII sem cabecalho de vertices")

    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for _ in range(header_end):
            next(file, None)
        rows = []
        read = 0
        for line in file:
            if read >= vertex_count:
                break
            rows.append(line.split()[:3])
            read += 1
            if len(rows) >= chunk_size:
                yield valid_points(rows)
                rows = []
        if rows:
            yield valid_points(rows)


def iter_text_points(path: Path, chunk_size: int) -> Iterable[np.ndarray]:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.replace(",", " ").split()[:3])
            if len(rows) >= chunk_size:
                yield valid_points(rows)
                rows = []
    if rows:
        yield valid_points(rows)


def valid_points(rows: list[list[str]]) -> np.ndarray:
    try:
        points = np.asarray(rows, dtype=float)
    except ValueError as exc:
        raise SystemExit(f"Arquivo precisa ter colunas numericas x y z: {exc}") from exc
    if points.ndim != 2 or points.shape[1] < 3:
        raise SystemExit("Arquivo precisa ter pelo menos 3 colunas: x y z")
    points = points[:, :3]
    return points[~np.isnan(points).any(axis=1)]


def e57_bounds(path: Path) -> tuple[float, float, float, float, float, float] | None:
    if path.suffix.lower() not in {".e57", ".e47"}:
        return None
    try:
        import pye57
    except ImportError as exc:
        raise SystemExit("Instale o suporte E57 com: pip install -r requirements.txt") from exc

    e57 = pye57.E57(str(path))
    try:
        bounds = []
        for scan_index in range(e57.scan_count):
            header = e57.get_header(scan_index)
            try:
                corners = np.array(
                    [
                        [x, y, z]
                        for x in (header.xMinimum, header.xMaximum)
                        for y in (header.yMinimum, header.yMaximum)
                        for z in (header.zMinimum, header.zMaximum)
                    ],
                    dtype=float,
                )
            except Exception:
                return None
            bounds.append(
                (
                    corners[:, 0].min(),
                    corners[:, 0].max(),
                    corners[:, 1].min(),
                    corners[:, 1].max(),
                    corners[:, 2].min(),
                    corners[:, 2].max(),
                )
            )
        if not bounds:
            return None
        values = np.asarray(bounds)
        return (
            float(values[:, 0].min()),
            float(values[:, 1].max()),
            float(values[:, 2].min()),
            float(values[:, 3].max()),
            float(values[:, 4].min()),
            float(values[:, 5].max()),
        )
    finally:
        e57.close()


def scan_bounds(path: Path, chunk_size: int) -> tuple[float, float, float, float, float, float] | None:
    bounds = e57_bounds(path)
    if bounds is not None:
        log("Limites da nuvem lidos do cabecalho E57")
        return bounds

    min_x = min_y = min_z = np.inf
    max_x = max_y = max_z = -np.inf
    total = 0
    for chunk in iter_points(path, chunk_size):
        if len(chunk) == 0:
            continue
        total += len(chunk)
        min_x = min(min_x, chunk[:, 0].min())
        max_x = max(max_x, chunk[:, 0].max())
        min_y = min(min_y, chunk[:, 1].min())
        max_y = max(max_y, chunk[:, 1].max())
        min_z = min(min_z, chunk[:, 2].min())
        max_z = max(max_z, chunk[:, 2].max())
    log(f"Total de pontos validos varridos: {total}")
    if total == 0:
        return None
    return min_x, max_x, min_y, max_y, min_z, max_z


def histogram_sizes(pixels: int, min_x: float, max_x: float, min_y: float, max_y: float) -> tuple[int, int]:
    width, height = max_x - min_x, max_y - min_y
    bins_x = max(1, pixels)
    bins_y = max(1, int(pixels * (height / width))) if width else pixels
    if bins_x * bins_y > MAX_IMAGE_CELLS:
        scale = (MAX_IMAGE_CELLS / (bins_x * bins_y)) ** 0.5
        bins_x = max(1, int(bins_x * scale))
        bins_y = max(1, int(bins_y * scale))
        log(f"Resolucao reduzida para proteger memoria: {bins_x}x{bins_y}")
    return bins_x, bins_y


def build_plan_slices(
    path: Path,
    chunk_size: int,
    slices: list[tuple[float, float, Path]],
    bins_x: int,
    bins_y: int,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    detect_shapes: bool,
    min_area: float,
) -> tuple[list[dict], float | None, float | None]:
    images = [np.zeros((bins_y, bins_x), dtype=np.uint32) for _ in slices]
    counts = [0] * len(slices)
    actual_min_z = None
    actual_max_z = None
    x_span = max(max_x - min_x, np.finfo(float).eps)
    y_span = max(max_y - min_y, np.finfo(float).eps)
    for chunk in iter_points(path, chunk_size):
        if len(chunk) == 0:
            continue
        chunk_min_z = chunk[:, 2].min()
        chunk_max_z = chunk[:, 2].max()
        actual_min_z = chunk_min_z if actual_min_z is None else min(actual_min_z, chunk_min_z)
        actual_max_z = chunk_max_z if actual_max_z is None else max(actual_max_z, chunk_max_z)
        for index, (z0, z1, _) in enumerate(slices):
            slice_points = chunk[(chunk[:, 2] >= z0) & (chunk[:, 2] <= z1)]
            if len(slice_points) == 0:
                continue
            counts[index] += len(slice_points)
            xs = np.minimum(((slice_points[:, 0] - min_x) / x_span * bins_x).astype(np.int32), bins_x - 1)
            ys = np.minimum(((slice_points[:, 1] - min_y) / y_span * bins_y).astype(np.int32), bins_y - 1)
            np.add.at(images[index], (ys, xs), 1)

    records = []
    for image, count, (z0, z1, output) in zip(images, counts, slices):
        record = {
            "arquivo": str(output),
            "gerada": count > 0,
            "pontos": count,
            "z_inicial": float(z0),
            "z_final": float(z1),
            "x_min": float(min_x),
            "x_max": float(max_x),
            "y_min": float(min_y),
            "y_max": float(max_y),
            "pixels_x": int(bins_x),
            "pixels_y": int(bins_y),
        }
        if count == 0:
            log(f"Nenhum ponto entre z={z0:.4f} e z={z1:.4f}")
            records.append(record)
            continue
        if detect_shapes:
            record["elementos"] = detect_geometric_elements(image, min_area, min_x, max_x, min_y, max_y)
        plt.imsave(output, np.log1p(image), cmap="gray", origin="lower")
        records.append(record)
    return records, actual_min_z, actual_max_z


def point_to_world(
    x: float,
    y: float,
    bins_x: int,
    bins_y: int,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> tuple[float, float]:
    world_x = min_x + (x / max(bins_x - 1, 1)) * (max_x - min_x)
    world_y = min_y + (y / max(bins_y - 1, 1)) * (max_y - min_y)
    return float(world_x), float(world_y)


def classify_shape(vertices: int, area: float, perimeter: float, bbox: tuple[int, int, int, int]) -> str:
    if vertices == 3:
        return "triangulo"
    if vertices == 4:
        _, _, width, height = bbox
        ratio = width / max(height, 1)
        return "quadrado" if 0.9 <= ratio <= 1.1 else "retangulo"
    circularity = 4 * np.pi * area / max(perimeter * perimeter, np.finfo(float).eps)
    if circularity > 0.72:
        return "circulo"
    return "poligono"


def detect_geometric_elements(
    image: np.ndarray,
    min_area: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> list[dict]:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Instale OpenCV com: pip install opencv-python") from exc

    normalized = np.log1p(image)
    if normalized.max() > 0:
        normalized = (normalized / normalized.max() * 255).astype(np.uint8)
    else:
        normalized = normalized.astype(np.uint8)

    blurred = cv2.GaussianBlur(normalized, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bins_y, bins_x = image.shape
    elements = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        bbox = tuple(int(value) for value in cv2.boundingRect(approx))
        moments = cv2.moments(contour)
        if moments["m00"]:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            x, y, width, height = bbox
            cx, cy = x + width / 2, y + height / 2
        world_cx, world_cy = point_to_world(cx, cy, bins_x, bins_y, min_x, max_x, min_y, max_y)
        x, y, width, height = bbox
        world_x0, world_y0 = point_to_world(x, y, bins_x, bins_y, min_x, max_x, min_y, max_y)
        world_x1, world_y1 = point_to_world(x + width, y + height, bins_x, bins_y, min_x, max_x, min_y, max_y)
        elements.append(
            {
                "tipo": classify_shape(len(approx), area, perimeter, bbox),
                "vertices": int(len(approx)),
                "area_px": area,
                "perimetro_px": perimeter,
                "centro_px": [float(cx), float(cy)],
                "centro": [world_cx, world_cy],
                "bbox_px": [x, y, width, height],
                "bbox": [world_x0, world_y0, world_x1, world_y1],
            }
        )
        if len(elements) >= MAX_DETECTED_ELEMENTS:
            break
    return elements


def parse_heights(args: argparse.Namespace, z_min: float, z_max: float) -> list[float]:
    if args.alturas:
        return [float(value.strip().replace(",", ".")) for value in args.alturas.split(";") if value.strip()]
    if args.inicio is not None and args.fim is not None:
        step = args.passo or args.espessura
        if step <= 0:
            raise SystemExit("--passo precisa ser maior que zero")
        count = int(np.floor((args.fim - args.inicio) / step)) + 1
        return [args.inicio + i * step for i in range(max(0, count))]
    if args.altura is not None:
        return [args.altura]
    step = args.passo or args.espessura
    if step <= 0:
        raise SystemExit("--passo precisa ser maior que zero")
    start = z_min if args.z_absoluto else 0.0
    end = z_max if args.z_absoluto else z_max - z_min
    count = int(np.floor((end - start) / step)) + 1
    return [start + i * step for i in range(max(0, count))]


def main() -> None:
    lower_process_priority()
    parser = argparse.ArgumentParser(description="Gera vistas de planta por fatias horizontais da nuvem 3D.")
    parser.add_argument("entrada", type=Path, nargs="?", help="Arquivo .e57/.e47, .xyz, .txt, .csv ou .ply ASCII")
    parser.add_argument("saida", type=Path, nargs="?", help="Imagem de saida opcional .png, .jpg ou .jpeg")
    parser.add_argument("--altura", type=float, help="Altura da fatia a partir do ponto mais baixo")
    parser.add_argument("--z-absoluto", action="store_true", help="Usa altura como coordenada Z absoluta")
    parser.add_argument("--alturas", help="Alturas separadas por ponto e virgula. Ex: 1;2;3.5")
    parser.add_argument("--inicio", type=float, help="Primeira altura para gerar varias fatias")
    parser.add_argument("--fim", type=float, help="Ultima altura para gerar varias fatias")
    parser.add_argument("--passo", type=float, help="Intervalo entre fatias")
    parser.add_argument("--espessura", type=float, default=DEFAULT_THICKNESS, help="Espessura da fatia")
    parser.add_argument("--pixels", type=int, default=1600, help="Resolucao do maior lado da imagem")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Pontos lidos por bloco")
    parser.add_argument("--batch-size", type=int, default=MAX_BATCH_SLICES, help="Fatias processadas por varredura")
    parser.add_argument("--sem-deteccao", action="store_true", help="Nao identifica formas com OpenCV")
    parser.add_argument("--area-minima", type=float, default=DEFAULT_MIN_AREA, help="Area minima em pixels para detectar")
    args = parser.parse_args()

    if args.entrada is None:
        args.entrada = Path(input("Arquivo de entrada: ").strip().strip('"'))
    args.entrada = resolve_input(args.entrada)
    prepare_output_dir()
    args.chunk_size = max(1_000, args.chunk_size)
    args.batch_size = max(1, min(args.batch_size, MAX_BATCH_SLICES))
    args.pixels = max(1, min(args.pixels, MAX_PIXELS))

    bounds = scan_bounds(args.entrada, args.chunk_size)
    if bounds is None:
        raise SystemExit("Nenhum ponto encontrado")
    min_x, max_x, min_y, max_y, z_min, z_max = bounds
    log(f"Intervalo Z usado como referencia: {z_min:.4f}..{z_max:.4f}")

    heights = parse_heights(args, z_min, z_max)
    if not heights:
        raise SystemExit("Nenhuma altura informada")
    log(f"Fatias: {len(heights)} | Espessura: {args.espessura} | Pixels: {args.pixels}")

    bins_x, bins_y = histogram_sizes(args.pixels, min_x, max_x, min_y, max_y)
    base_z = 0 if args.z_absoluto else z_min
    slices = [
        (base_z + height, base_z + height + args.espessura, output_path(args.entrada, args.saida, height, len(heights)))
        for height in heights
    ]
    log("Gerando vista de planta")
    records = []
    actual_min_z = actual_max_z = None
    for start in range(0, len(slices), args.batch_size):
        batch = slices[start : start + args.batch_size]
        log(f"Processando fatias {start + 1}..{start + len(batch)} de {len(slices)}")
        batch_records, batch_min_z, batch_max_z = build_plan_slices(
            args.entrada,
            args.chunk_size,
            batch,
            bins_x,
            bins_y,
            min_x,
            max_x,
            min_y,
            max_y,
            not args.sem_deteccao,
            max(1, args.area_minima),
        )
        records.extend(batch_records)
        if batch_min_z is not None:
            actual_min_z = batch_min_z if actual_min_z is None else min(actual_min_z, batch_min_z)
        if batch_max_z is not None:
            actual_max_z = batch_max_z if actual_max_z is None else max(actual_max_z, batch_max_z)
    log("Processo finalizado")
    metadata = {
        "entrada": str(args.entrada),
        "z_min": float(actual_min_z if actual_min_z is not None else z_min),
        "z_max": float(actual_max_z if actual_max_z is not None else z_max),
        "espessura": float(args.espessura),
        "passo": float(args.passo or args.espessura),
        "z_absoluto": bool(args.z_absoluto),
        "fatias": records,
    }
    json_output = metadata_path(args.entrada)
    json_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    saved = [record for record in records if record["gerada"]]
    if not saved:
        if actual_min_z is not None and actual_max_z is not None:
            raise SystemExit(f"Nenhuma imagem gerada. Z real encontrado: {actual_min_z:.4f}..{actual_max_z:.4f}")
        raise SystemExit("Nenhuma imagem gerada: fatias sem pontos")
    for record in saved:
        print(
            f"Imagem salva: {record['arquivo']} "
            f"({record['pontos']} pontos, z {record['z_inicial']:.4f}..{record['z_final']:.4f})"
        )
    print(f"JSON salvo: {json_output}")


if __name__ == "__main__":
    main()
