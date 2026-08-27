import json
import math
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import numpy as np


SAIDA_JSON = Path(__file__).with_name("formas_detectadas.json")
EXTENSOES_IMAGEM = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def eh_linha(cnt, w, h):
    if max(w, h) < 20:
        return False
    if min(w, h) <= 4 and max(w, h) / max(1, min(w, h)) > 5:
        return True

    pts = cnt.reshape(-1, 2).astype("float32")
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    dist = np.abs(vy * (pts[:, 0] - x0) - vx * (pts[:, 1] - y0))
    return np.percentile(dist, 90) <= max(4, max(w, h) * 0.08)


def eh_retangulo_aberto(cnt, x, y, w, h):
    if min(w, h) < 30:
        return False

    roi = np.zeros((h + 4, w + 4), dtype=np.uint8)
    local = cnt - [x - 2, y - 2]
    cv2.drawContours(roi, [local], -1, 255, 2)
    linhas = cv2.HoughLinesP(roi, 1, np.pi / 180, 25, minLineLength=min(w, h) * 0.35, maxLineGap=8)
    if linhas is None:
        return False

    horizontais = verticais = 0
    for linha in linhas:
        x1, y1, x2, y2 = linha.flatten()
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx > dy * 4:
            horizontais += 1
        elif dy > dx * 4:
            verticais += 1

    return horizontais >= 1 and verticais >= 1 and horizontais + verticais >= 3


def preparar_imagem(caminho_imagem):
    img = cv2.imread(str(caminho_imagem), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Imagem nao encontrada: {caminho_imagem}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) if len(img.shape) == 3 and img.shape[2] == 4 else img
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY) if len(gray.shape) == 3 else gray
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(mask) > mask.size * 0.65:
        mask = cv2.bitwise_not(mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return gray, mask


def tem_degrade(gray, cnt):
    mascara = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mascara, [cnt], -1, 255, -1)
    pixels = gray[mascara == 255]
    if pixels.size < 30:
        return False

    p10, p90 = np.percentile(pixels, [10, 90])
    return pixels.std() >= 18 and p90 - p10 >= 35


def classificar_forma(cnt, gray):
    area = cv2.contourArea(cnt)
    if area <= 80:
        return None

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    pontas = len(approx)
    x, y, w, h = cv2.boundingRect(cnt)
    preenchimento = area / float(w * h)
    circularidade = 4 * math.pi * area / (peri * peri) if peri else 0
    aspect_ratio = w / float(h)
    fechada = "FECHADA" if preenchimento > 0.35 else "ABERTA"
    degrade = tem_degrade(gray, cnt)

    if eh_linha(cnt, w, h):
        forma = "LINHA"
        fechada = "ABERTA"
    elif degrade and 0.65 <= circularidade <= 1.20 and 0.75 <= aspect_ratio <= 1.25:
        forma = "ESFERA"
        fechada = "FECHADA"
    elif degrade and pontas in (3, 4) and preenchimento <= 0.85:
        forma = "RAMPA"
    elif pontas == 3:
        forma = "TRIANGULO"
    elif pontas == 4 or preenchimento > 0.85:
        forma = "QUADRADO" if 0.90 <= aspect_ratio <= 1.10 else "RETANGULO"
    elif eh_retangulo_aberto(cnt, x, y, w, h):
        forma = "QUADRADO" if 0.90 <= aspect_ratio <= 1.10 else "RETANGULO"
        fechada = "ABERTA"
    elif 0.65 <= circularidade <= 1.15 and 0.75 <= aspect_ratio <= 1.25 and 0.55 <= preenchimento <= 0.85:
        forma = "CIRCULO"
    else:
        forma = "INDEFINIDA"

    return {
        "forma": forma,
        "fechada": fechada,
        "degrade": bool(degrade),
        "coordenada": {
            "x": int(x),
            "y": int(y),
            "largura": int(w),
            "altura": int(h),
            "centro_x": int(x + w / 2),
            "centro_y": int(y + h / 2),
        },
        "pontos": approx.reshape(-1, 2).astype(int).tolist(),
    }


def analisar_imagem(caminho_imagem):
    gray, mask = preparar_imagem(caminho_imagem)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    formas = []
    for cnt in contours:
        forma = classificar_forma(cnt, gray)
        if forma:
            formas.append(forma)

    return {
        "imagem": str(caminho_imagem),
        "formas": formas,
    }


def analisar_imagens(caminhos_imagens, caminho_json=SAIDA_JSON):
    resultado = [analisar_imagem(caminho) for caminho in caminhos_imagens]

    with open(caminho_json, "w", encoding="utf-8") as arquivo:
        json.dump(resultado, arquivo, ensure_ascii=False, indent=2)

    return resultado


def analisar_pasta(caminho_pasta, caminho_json=SAIDA_JSON):
    pasta = Path(caminho_pasta)
    imagens = sorted(
        arquivo for arquivo in pasta.iterdir()
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_IMAGEM
    )
    return analisar_imagens(imagens, caminho_json)


if __name__ == "__main__":
    janela = Tk()
    janela.withdraw()
    pasta = filedialog.askdirectory(title="Selecione a pasta com as imagens")
    janela.destroy()

    if pasta:
        resultado = analisar_pasta(pasta)
        print(f"{len(resultado)} imagens analisadas.")
        print(f"JSON salvo em: {SAIDA_JSON}")
    else:
        print("Nenhuma pasta selecionada.")
