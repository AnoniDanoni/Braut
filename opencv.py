import json
import math
import re
from collections import Counter
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import numpy as np


SAIDA_JSON = Path(__file__).with_name("formas_detectadas.json")
SAIDA_JSON_3D = Path(__file__).with_name("modelo_3d.json")
EXTENSOES_IMAGEM = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MIN_FATIAS_3D = 3
MIN_FATIAS_LINHA = 5
MIN_CONFIANCA_TUBULACAO = 0.7


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


def tem_degrade(gray, cnt, sensivel=False):
    mascara = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mascara, [cnt], -1, 255, -1)
    ys, xs = np.where(mascara == 255)
    pixels = gray[ys, xs]
    if pixels.size < 30:
        return False

    regioes = [pixels]
    if sensivel:
        cx, cy = np.median(xs), np.median(ys)
        regioes += [pixels[xs <= cx], pixels[xs > cx], pixels[ys <= cy], pixels[ys > cy]]
    limite_desvio, limite_amplitude = (16, 30) if sensivel else (14, 28)
    return any(
        regiao.size >= 20 and regiao.std() >= limite_desvio
        and np.ptp(np.percentile(regiao, [10, 90])) >= limite_amplitude
        for regiao in regioes
    )


def classificar_forma(cnt, gray, forma_forcada=None):
    area = cv2.contourArea(cnt)
    if area <= 80:
        return None

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    pontas = len(approx)
    x, y, w, h = cv2.boundingRect(cnt)
    momentos = cv2.moments(cnt)
    centroide_x = momentos["m10"] / momentos["m00"] if momentos["m00"] else x + w / 2
    centroide_y = momentos["m01"] / momentos["m00"] if momentos["m00"] else y + h / 2
    (circulo_x, circulo_y), _ = cv2.minEnclosingCircle(cnt)
    centros = [[x + w / 2, y + h / 2], [centroide_x, centroide_y], [circulo_x, circulo_y]]
    centro_medio = np.mean(centros, axis=0)
    preenchimento = area / float(w * h)
    circularidade = 4 * math.pi * area / (peri * peri) if peri else 0
    aspect_ratio = w / float(h)
    fechada = "FECHADA" if preenchimento > 0.35 else "ABERTA"
    degrade = tem_degrade(gray, cnt, forma_forcada == "CIRCULO")

    if forma_forcada:
        forma = forma_forcada
    elif eh_linha(cnt, w, h):
        forma = "LINHA"
        fechada = "ABERTA"
    elif degrade and 0.65 <= circularidade <= 1.20 and 0.75 <= aspect_ratio <= 1.25:
        forma = "MEIA_ESFERA"
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
            "centro_bbox_px": {"x": float(centros[0][0]), "y": float(centros[0][1])},
            "centroide_contorno_px": {"x": float(centroide_x), "y": float(centroide_y)},
            "centro_circulo_envolvente_px": {"x": float(circulo_x), "y": float(circulo_y)},
            "centro_medio_px": {"x": float(centro_medio[0]), "y": float(centro_medio[1])},
        },
        "pontos": approx.reshape(-1, 2).astype(int).tolist(),
    }


def detectar_circulos(gray):
    escala = min(1.0, 1200 / max(gray.shape))
    reduzida = cv2.resize(gray, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    suavizada = cv2.GaussianBlur(reduzida, (7, 7), 1.5)
    bordas = cv2.Canny(suavizada, 45, 110)
    distancia_borda = cv2.distanceTransform(255 - bordas, cv2.DIST_L2, 3)
    circulos = cv2.HoughCircles(
        suavizada, cv2.HOUGH_GRADIENT, 1.2, max(20, int(45 * escala)),
        param1=100, param2=55, minRadius=max(6, int(15 * escala)),
        maxRadius=max(20, int(min(gray.shape) * 0.18 * escala)),
    )
    parciais = cv2.HoughCircles(
        suavizada, cv2.HOUGH_GRADIENT, 1.2, max(20, int(45 * escala)),
        param1=100, param2=35, minRadius=max(6, int(15 * escala)),
        maxRadius=max(20, int(min(gray.shape) * 0.22 * escala)),
    )
    candidatos = [(*circulo, False) for circulo in circulos[0]] if circulos is not None else []
    candidatos += [(*circulo, True) for circulo in parciais[0]] if parciais is not None else []
    formas = []
    for x, y, raio, parcial in candidatos:
        angulos = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        xs = np.clip((x + np.cos(angulos) * raio).astype(int), 0, reduzida.shape[1] - 1)
        ys = np.clip((y + np.sin(angulos) * raio).astype(int), 0, reduzida.shape[0] - 1)
        apoios = (distancia_borda[ys, xs] <= 2).astype(float)
        suporte = float(apoios.mean())
        suporte_metade = float(np.convolve(np.r_[apoios, apoios[:89]], np.ones(90) / 90, mode="valid").max())
        if (parcial and (suporte > 0.55 or suporte_metade < 0.85)) or (
            not parcial and suporte < 0.50 and suporte_metade < 0.80
        ):
            continue
        x, y, raio = x / escala, y / escala, raio / escala
        if any(math.hypot(x - item[0], y - item[1]) < min(raio, item[2]) * 0.3
               and max(raio, item[2]) / min(raio, item[2]) < 1.25 for item in formas):
            continue
        contorno = np.asarray([
            [[round(x + math.cos(angulo) * raio), round(y + math.sin(angulo) * raio)]]
            for angulo in np.linspace(0, 2 * np.pi, 64, endpoint=False)
        ], dtype=np.int32)
        dados = classificar_forma(contorno, gray, "CIRCULO")
        if parcial and not dados["degrade"]:
            continue
        if dados["degrade"]:
            dados["forma"] = "MEIA_ESFERA"
        dados["confianca_deteccao"] = round(max(suporte, suporte_metade * 0.75), 3)
        formas.append((x, y, raio, dados))
    return [item[3] for item in formas]


def detectar_retangulos(gray):
    bordas = cv2.Canny(gray, 40, 120)
    bordas = cv2.morphologyEx(bordas, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contornos, _ = cv2.findContours(bordas, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        if not gray.size * 0.002 <= area <= gray.size * 0.25 or not perimetro:
            continue
        aproximado = cv2.approxPolyDP(contorno, 0.03 * perimetro, True)
        if len(aproximado) != 4 or not cv2.isContourConvex(aproximado):
            continue
        x, y, largura, altura = cv2.boundingRect(aproximado)
        retangularidade = area / (largura * altura)
        if retangularidade < 0.78 or min(largura, altura) < 30:
            continue
        bbox = np.asarray([x, y, x + largura, y + altura])
        escala = np.asarray([largura, altura, largura, altura])
        if any(np.max(np.abs(bbox - anterior[0]) / escala) < 0.08 for anterior in candidatos):
            continue
        forma = "QUADRADO" if 0.9 <= largura / altura <= 1.1 else "RETANGULO"
        dados = classificar_forma(aproximado, gray, forma)
        gradiente = detectar_gradiente_rampa(gray, aproximado)
        if dados["degrade"]:
            dados["forma"] = "RAMPA"
            dados["angulo_rampa_graus"] = gradiente if gradiente is not None else (0.0 if largura >= altura else 90.0)
        dados["confianca_deteccao"] = round(float(min(1, retangularidade)), 3)
        candidatos.append((bbox, dados))
    return [forma for _, forma in candidatos]


def detectar_gradiente_rampa(gray, contorno):
    mascara = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mascara, [contorno], -1, 255, -1)
    ys, xs = np.where((mascara > 0) & (gray > 10))
    if len(xs) < 500:
        return None
    if len(xs) > 30000:
        indices = np.linspace(0, len(xs) - 1, 30000).astype(int)
        xs, ys = xs[indices], ys[indices]
    valores = gray[ys, xs].astype(float)
    matriz = np.column_stack((xs - xs.mean(), ys - ys.mean(), np.ones(len(xs))))
    coeficientes = np.linalg.lstsq(matriz, valores, rcond=None)[0]
    previsto = matriz @ coeficientes
    total = np.sum((valores - valores.mean()) ** 2)
    r2 = 1 - np.sum((valores - previsto) ** 2) / max(total, 1)
    x, y, largura, altura = cv2.boundingRect(contorno)
    amplitude = abs(coeficientes[0]) * largura + abs(coeficientes[1]) * altura
    if r2 < 0.55 or amplitude < 45:
        return None
    return float(math.degrees(math.atan2(-coeficientes[1], coeficientes[0])))


def detectar_linhas(gray):
    bordas = cv2.Canny(gray, 50, 120)
    linhas = cv2.HoughLinesP(
        bordas, 1, np.pi / 180, 100,
        minLineLength=int(min(gray.shape) * 0.12), maxLineGap=15,
    )
    candidatas = []
    for linha in linhas.reshape(-1, 4) if linhas is not None else []:
        x1, y1, x2, y2 = map(float, linha)
        if (x2, y2) < (x1, y1):
            x1, y1, x2, y2 = x2, y2, x1, y1
        comprimento = math.hypot(x2 - x1, y2 - y1)
        angulo = math.atan2(y2 - y1, x2 - x1) % math.pi
        meio = ((x1 + x2) / 2, (y1 + y2) / 2)
        rho = -meio[0] * math.sin(angulo) + meio[1] * math.cos(angulo)
        if any(min(abs(angulo - item[0]), math.pi - abs(angulo - item[0])) < math.radians(3)
               and abs(rho - item[1]) < 8 for item in candidatas):
            continue
        normal = np.asarray([-math.sin(angulo), math.cos(angulo)]) * 3
        contorno = np.asarray([
            np.asarray([x1, y1]) + normal, np.asarray([x2, y2]) + normal,
            np.asarray([x2, y2]) - normal, np.asarray([x1, y1]) - normal,
        ], dtype=np.int32).reshape(-1, 1, 2)
        dados = classificar_forma(contorno, gray, "LINHA")
        dados["linha_px"] = {"start": [x1, y1], "end": [x2, y2]}
        dados["confianca_deteccao"] = round(float(min(1, comprimento / (min(gray.shape) * 0.3))), 3)
        candidatas.append((angulo, rho, comprimento, dados))
    return [item[3] for item in sorted(candidatas, key=lambda item: item[2], reverse=True)[:40]]


def analisar_imagem(caminho_imagem, metadados=None):
    gray, _ = preparar_imagem(caminho_imagem)
    formas = detectar_circulos(gray) + detectar_retangulos(gray) + detectar_linhas(gray)

    resultado = {
        "imagem": str(caminho_imagem),
        "formas": formas,
    }
    if metadados:
        resultado["z_inicial"] = metadados.get("z_inicial")
        resultado["z_final"] = metadados.get("z_final")
        for forma in formas:
            adicionar_coordenadas_reais(forma, metadados)
    return resultado


def carregar_metadados(pasta):
    registros = {}
    for caminho in Path(pasta).glob("*.json"):
        try:
            conteudo = json.loads(caminho.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for registro in conteudo.get("fatias", []) if isinstance(conteudo, dict) else []:
            if registro.get("arquivo"):
                registros[Path(registro["arquivo"]).name.lower()] = registro
    return registros


def adicionar_coordenadas_reais(forma, metadados):
    campos = ("x_min", "x_max", "y_min", "y_max", "pixels_x", "pixels_y")
    if not all(campo in metadados for campo in campos):
        return
    x_min, x_max = metadados["x_min"], metadados["x_max"]
    y_min, y_max = metadados["y_min"], metadados["y_max"]
    px_x, px_y = max(metadados["pixels_x"] - 1, 1), max(metadados["pixels_y"] - 1, 1)
    z = (metadados.get("z_inicial", 0) + metadados.get("z_final", 0)) / 2

    def converter(ponto):
        return {
            "x": float(x_min + ponto["x"] / px_x * (x_max - x_min)),
            "y": float(y_max - ponto["y"] / px_y * (y_max - y_min)),
            "z": float(z),
        }

    coordenada = forma["coordenada"]
    reais = {
        nome.removesuffix("_px"): converter(coordenada[nome])
        for nome in ("centro_bbox_px", "centroide_contorno_px", "centro_circulo_envolvente_px", "centro_medio_px")
    }
    reais["bbox"] = {
        "x_min": float(x_min + coordenada["x"] / px_x * (x_max - x_min)),
        "x_max": float(x_min + (coordenada["x"] + coordenada["largura"]) / px_x * (x_max - x_min)),
        "y_min": float(y_max - (coordenada["y"] + coordenada["altura"]) / px_y * (y_max - y_min)),
        "y_max": float(y_max - coordenada["y"] / px_y * (y_max - y_min)),
        "z": float(z),
    }
    forma["coordenadas_reais"] = reais
    if "linha_px" in forma:
        forma["linha_real"] = {
            nome: list(converter({"x": ponto[0], "y": ponto[1]}).values())
            for nome, ponto in forma["linha_px"].items()
        }


def chave_natural(caminho):
    return [(1, float(parte.replace("_", "."))) if re.fullmatch(r"\d+(?:_\d+)?", parte) else (0, parte.lower())
            for parte in re.split(r"(\d+(?:_\d+)?)", Path(caminho).stem)]


def agrupar_formas(fatias):
    grupos = []
    for indice, fatia in enumerate(fatias):
        for forma in fatia["formas"]:
            centro = forma["coordenada"]["centro_medio_px"]
            tamanho = max(forma["coordenada"]["largura"], forma["coordenada"]["altura"])
            candidatos = []
            for grupo in grupos:
                intervalo = indice - grupo["ultimo_indice"]
                ultimo = grupo["ocorrencias"][-1][1]
                familia = "REDONDA" if forma["forma"] in {"CIRCULO", "MEIA_ESFERA"} else "LINEAR" if forma["forma"] == "LINHA" else "RETANGULAR"
                familia_ultima = "REDONDA" if ultimo["forma"] in {"CIRCULO", "MEIA_ESFERA"} else "LINEAR" if ultimo["forma"] == "LINHA" else "RETANGULAR"
                if familia != familia_ultima:
                    continue
                ultimo_centro = ultimo["coordenada"]["centro_medio_px"]
                ultimo_tamanho = max(ultimo["coordenada"]["largura"], ultimo["coordenada"]["altura"])
                distancia = math.hypot(centro["x"] - ultimo_centro["x"], centro["y"] - ultimo_centro["y"])
                proporcao = max(tamanho, ultimo_tamanho) / max(1, min(tamanho, ultimo_tamanho))
                fator_distancia = {"LINEAR": 0.12, "REDONDA": 0.25, "RETANGULAR": 0.2}[familia]
                limite_distancia = max(10, ultimo_tamanho * fator_distancia)
                if familia == "LINEAR":
                    atual = forma["linha_px"]
                    anterior = ultimo["linha_px"]
                    angulo_atual = math.atan2(atual["end"][1] - atual["start"][1], atual["end"][0] - atual["start"][0]) % math.pi
                    angulo_anterior = math.atan2(anterior["end"][1] - anterior["start"][1], anterior["end"][0] - anterior["start"][0]) % math.pi
                    if min(abs(angulo_atual - angulo_anterior), math.pi - abs(angulo_atual - angulo_anterior)) > math.radians(8):
                        continue
                limite_proporcao = 1.6 if familia == "REDONDA" else 1.5
                if 1 <= intervalo <= 2 and distancia <= limite_distancia and proporcao <= limite_proporcao:
                    candidatos.append((distancia / max(ultimo_tamanho, 1) + intervalo * 0.1, grupo))
            if candidatos:
                grupo = min(candidatos, key=lambda item: item[0])[1]
            else:
                grupo = {"id": len(grupos) + 1, "ultimo_indice": indice, "ocorrencias": []}
                grupos.append(grupo)
            grupo["ultimo_indice"] = indice
            grupo["ocorrencias"].append((indice, forma))
    return grupos


def corrigir_indefinidas(grupo):
    ocorrencias = grupo["ocorrencias"]
    definidas = [forma["forma"] for _, forma in ocorrencias if forma["forma"] != "INDEFINIDA"]
    dominante = Counter(definidas).most_common(1)[0][0] if definidas else None
    for posicao, (_, forma) in enumerate(ocorrencias):
        forma["forma_detectada"] = forma["forma"]
        forma["forma_inferida"] = None
        forma["motivo_inferencia"] = None
        if forma["forma"] != "INDEFINIDA":
            forma["forma_final"] = forma["forma"]
            continue
        anterior = next((item[1]["forma"] for item in reversed(ocorrencias[:posicao]) if item[1]["forma"] != "INDEFINIDA"), None)
        posterior = next((item[1]["forma"] for item in ocorrencias[posicao + 1:] if item[1]["forma"] != "INDEFINIDA"), None)
        inferida = anterior if anterior == posterior and anterior else dominante
        if inferida:
            forma["forma_inferida"] = inferida
            forma["forma_final"] = inferida
            forma["motivo_inferencia"] = (
                f"As fatias vizinhas anterior e posterior sao {inferida}."
                if anterior == posterior else f"{inferida} e a forma predominante neste mesmo objeto."
            )
        else:
            forma["forma_final"] = "INDEFINIDA"


def tipo_geometria(formas):
    tipos = set(formas)
    if tipos == {"CIRCULO"}:
        return "CILINDRO" if len(formas) >= 2 else "CIRCULO"
    if tipos == {"MEIA_ESFERA"}:
        return "MEIA_ESFERA"
    if tipos <= {"QUADRADO", "RETANGULO"}:
        return "PRISMA_RETANGULAR" if len(formas) >= 2 else formas[0]
    if tipos == {"TRIANGULO"}:
        return "PRISMA_TRIANGULAR" if len(formas) >= 2 else "TRIANGULO"
    if tipos == {"LINHA"}:
        return "GEOMETRIA_LINEAR"
    if tipos == {"RAMPA"}:
        return "RAMPA"
    return "GEOMETRIA_COMPOSTA"


def tipo_geometria_grupo(grupo, formas):
    def converter_para_circulo(indices):
        for indice in indices:
            forma = grupo["ocorrencias"][indice][1]
            forma["forma_final"] = "CIRCULO"
            forma["ajuste_estrutural"] = "Degrade descartado: semiesfera so e valida no topo de um cilindro."
            formas[indice] = "CIRCULO"

    tipos = set(formas)
    if tipos == {"LINHA"}:
        return "TUBULACAO" if len(formas) < MIN_FATIAS_LINHA else "GEOMETRIA_LINEAR"
    if tipos == {"MEIA_ESFERA"}:
        raios = np.asarray([
            (forma["coordenada"]["largura"] + forma["coordenada"]["altura"]) / 4
            for _, forma in grupo["ocorrencias"]
        ])
        correlacao = np.corrcoef(np.arange(len(raios)), raios)[0, 1] if len(raios) >= 3 else 0
        valida = len(raios) >= 3 and raios[-1] < raios[0] * 0.8 and correlacao < -0.8
        if not valida:
            converter_para_circulo(range(len(formas)))
        return "MEIA_ESFERA" if valida else "CILINDRO"
    if tipos == {"CIRCULO", "MEIA_ESFERA"}:
        circulos = [i for i, forma in enumerate(formas) if forma == "CIRCULO"]
        semiesferas = [i for i, forma in enumerate(formas) if forma == "MEIA_ESFERA"]
        superior = [i for i in semiesferas if i > max(circulos)]
        raio_cilindro = np.median([
            (grupo["ocorrencias"][i][1]["coordenada"]["largura"] + grupo["ocorrencias"][i][1]["coordenada"]["altura"]) / 4
            for i in circulos
        ])
        raios_superiores = [
            (grupo["ocorrencias"][i][1]["coordenada"]["largura"] + grupo["ocorrencias"][i][1]["coordenada"]["altura"]) / 4
            for i in superior
        ]
        cap_valido = (
            len(raios_superiores) >= 2
            and raios_superiores[-1] <= raio_cilindro * 0.85
            and np.mean(np.diff(raios_superiores) <= raio_cilindro * 0.05) >= 0.7
        ) or (len(raios_superiores) == 1 and raios_superiores[0] <= raio_cilindro * 0.7)
        converter_para_circulo([i for i in semiesferas if not cap_valido or i not in superior])
        return "TANQUE_COM_SEMIESFERA_SUPERIOR" if cap_valido else "CILINDRO"
    if "RAMPA" in formas and any(forma in {"QUADRADO", "RETANGULO"} for forma in formas):
        rampas = [i for i, forma in enumerate(formas) if forma == "RAMPA"]
        paredes = [i for i, forma in enumerate(formas) if forma in {"QUADRADO", "RETANGULO"}]
        superiores = [i for i in rampas if i > max(paredes)]
        forma_base = Counter(formas[i] for i in paredes).most_common(1)[0][0]
        for indice in rampas:
            if indice in superiores:
                continue
            forma = grupo["ocorrencias"][indice][1]
            forma["forma_final"] = forma_base
            forma["ajuste_estrutural"] = "Rampa descartada: cobertura inclinada so e valida acima das paredes."
            formas[indice] = forma_base
        return "PRISMA_COM_TELHADO_DUAS_AGUAS" if superiores else "PRISMA_RETANGULAR"
    if tipos == {"RAMPA"}:
        return "TELHADO_DUAS_AGUAS"
    return tipo_geometria(formas)


def resumir_sequencia(formas):
    partes = []
    for forma in formas:
        if partes and partes[-1][0] == forma:
            partes[-1][1] += 1
        else:
            partes.append([forma, 1])
    return [{"forma": forma, "fatias_consecutivas": quantidade} for forma, quantidade in partes]


def regra_aplicada(tipo):
    regras = {
        "CILINDRO": "Duas ou mais fatias circulares consecutivas representam um cilindro.",
        "TANQUE_COM_SEMIESFERA_SUPERIOR": "Um cilindro com semiesfera somente na extremidade superior representa um tanque.",
        "MEIA_ESFERA": "Um circulo com degrade representa uma semiesfera.",
        "TELHADO_DUAS_AGUAS": "Duas rampas simetricas conectadas por uma cumeeira representam um telhado.",
        "PRISMA_COM_TELHADO_DUAS_AGUAS": "Um prisma retangular com duas rampas superiores representa uma edificacao com telhado.",
        "PRISMA_RETANGULAR": "Quadrados ou retangulos repetidos em fatias consecutivas representam um prisma retangular.",
        "PRISMA_TRIANGULAR": "Triangulos repetidos em fatias consecutivas representam um prisma triangular.",
        "TUBULACAO": "Uma linha que desaparece entre as fatias representa uma tubulacao.",
        "GEOMETRIA_COMPOSTA": "A mudanca de forma ao longo das fatias representa uma geometria composta.",
    }
    return regras.get(tipo, f"A forma predominante e a continuidade das fatias resultaram em {tipo}.")


def coordenadas_grupo(grupo, fatias):
    reais = any("coordenadas_reais" in forma for _, forma in grupo["ocorrencias"])
    amostras, valores = [], []
    for indice, forma in grupo["ocorrencias"]:
        if reais and "coordenadas_reais" not in forma:
            continue
        origem = forma.get("coordenadas_reais", forma["coordenada"])
        nomes = ("centro_bbox", "centroide_contorno", "centro_circulo_envolvente", "centro_medio") if reais else (
            "centro_bbox_px", "centroide_contorno_px", "centro_circulo_envolvente_px", "centro_medio_px")
        centros = {nome.removesuffix("_px"): origem[nome] for nome in nomes}
        pontos = list(centros.values())
        valores.extend([[p["x"], p["y"], p.get("z", indice)] for p in pontos])
        amostra = {
            "fatia": indice + 1,
            "imagem": fatias[indice]["imagem"],
            "forma_detectada": forma["forma_detectada"],
            "forma_final": forma["forma_final"],
            "centros_calculados": centros,
            "bbox": origem.get("bbox", {campo: forma["coordenada"][campo] for campo in ("x", "y", "largura", "altura")}),
        }
        if "linha_px" in forma:
            amostra["linha"] = forma.get("linha_real", forma["linha_px"])
        if "angulo_rampa_graus" in forma:
            amostra["angulo_rampa_graus"] = forma["angulo_rampa_graus"]
        amostras.append(amostra)
    matriz = np.asarray(valores, dtype=float)
    media, desvio = matriz.mean(axis=0), matriz.std(axis=0)
    return {
        "sistema": "coordenadas_da_nuvem_3d" if reais else "pixels_das_imagens; z corresponde ao indice da fatia",
        "amostras": amostras,
        "coordenada_ideal": {
            "metodo": "media aritmetica de todos os centros calculados em todas as fatias",
            "x": float(media[0]), "y": float(media[1]), "z": float(media[2]),
        },
        "desvio_padrao": {"x": float(desvio[0]), "y": float(desvio[1]), "z": float(desvio[2])},
    }


def confianca_grupo(grupo):
    ocorrencias = grupo["ocorrencias"]
    indices = [indice for indice, _ in ocorrencias]
    cobertura = len(indices) / (max(indices) - min(indices) + 1)
    centros = np.asarray([
        [forma["coordenada"]["centro_medio_px"]["x"], forma["coordenada"]["centro_medio_px"]["y"]]
        for _, forma in ocorrencias
    ])
    tamanhos = np.asarray([
        max(forma["coordenada"]["largura"], forma["coordenada"]["altura"])
        for _, forma in ocorrencias
    ])
    dispersao = np.mean(np.linalg.norm(centros - np.median(centros, axis=0), axis=1)) / max(tamanhos.mean(), 1)
    deteccao = np.mean([forma.get("confianca_deteccao", 0.7) for _, forma in ocorrencias])
    confianca = deteccao * (0.6 + 0.4 * cobertura) * max(0, 1 - dispersao * 2)
    return round(float(confianca), 3), round(float(cobertura), 3)


def montar_geometrias(fatias):
    geometrias = []
    for grupo in agrupar_formas(fatias):
        eh_linha = grupo["ocorrencias"][0][1]["forma"] == "LINHA"
        minimo = 1 if eh_linha else MIN_FATIAS_3D
        if len(grupo["ocorrencias"]) < minimo:
            continue
        if eh_linha and len(grupo["ocorrencias"]) < MIN_FATIAS_LINHA and np.mean([
            forma.get("confianca_deteccao", 0) for _, forma in grupo["ocorrencias"]
        ]) < MIN_CONFIANCA_TUBULACAO:
            continue
        corrigir_indefinidas(grupo)
        formas = [forma["forma_final"] for _, forma in grupo["ocorrencias"] if forma["forma_final"] != "INDEFINIDA"]
        if not formas:
            continue
        tipo = tipo_geometria_grupo(grupo, formas)
        inferidas = sum(forma["forma_inferida"] is not None for _, forma in grupo["ocorrencias"])
        indices = [indice + 1 for indice, _ in grupo["ocorrencias"]]
        sequencia = resumir_sequencia(formas)
        confianca, cobertura = confianca_grupo(grupo)
        geometrias.append({
            "id": len(geometrias) + 1,
            "geometria": tipo,
            "descricao": (
                f"{tipo.replace('_', ' ').title()} inferido pela continuidade de {len(formas)} fatias "
                f"entre as imagens {min(indices)} e {max(indices)}; {inferidas} classificacao(oes) indefinida(s) foi(ram) corrigida(s)."
            ),
            "confianca": confianca,
            "condicoes_avaliadas": [
                {"condicao": "continuidade espacial", "resultado": cobertura >= 0.6, "evidencia": f"Cobertura de {cobertura:.1%} entre as fatias {min(indices)} e {max(indices)}."},
                {"condicao": "formas indefinidas entre vizinhas", "resultado": inferidas > 0, "evidencia": f"{inferidas} forma(s) corrigida(s)."},
                {"condicao": "sequencia de formas", "resultado": True, "evidencia": sequencia},
                {"condicao": "regra geometrica", "resultado": tipo, "evidencia": regra_aplicada(tipo)},
            ],
            "fatias": {"indices": indices, "quantidade": len(indices), "sequencia": sequencia},
            "coordenadas": coordenadas_grupo(grupo, fatias),
        })
    return geometrias


def salvar_json(caminho, dados):
    caminho = Path(caminho)
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    temporario.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
    temporario.replace(caminho)


def gerar_json_3d(resultado, caminho_json=SAIDA_JSON_3D):
    objetos = {}

    def adicionar(tipo, dados):
        if tipo not in objetos:
            objetos[tipo] = dados
        elif isinstance(objetos[tipo], list):
            objetos[tipo].append(dados)
        else:
            objetos[tipo] = [objetos[tipo], dados]

    def amostras(geometria, formas=None):
        valores = geometria["coordenadas"]["amostras"]
        return [item for item in valores if formas is None or item["forma_final"] in formas]

    def centro(valores):
        pontos = []
        for item in valores:
            ponto = item["centros_calculados"]["centro_medio"]
            pontos.append([ponto["x"], ponto["y"], ponto.get("z", item["fatia"])])
        return [float(valor) for valor in np.mean(pontos, axis=0)]

    def dimensoes(valores):
        medidas = []
        for item in valores:
            bbox = item["bbox"]
            if "x_min" in bbox:
                medidas.append([abs(bbox["x_max"] - bbox["x_min"]), abs(bbox["y_max"] - bbox["y_min"])])
            else:
                medidas.append([bbox["largura"], bbox["altura"]])
        return np.mean(medidas, axis=0)

    def intervalo_z(valores):
        limites = []
        for item in valores:
            fatia = resultado["fatias"][item["fatia"] - 1]
            limites.append((fatia.get("z_inicial", item["fatia"]), fatia.get("z_final", item["fatia"])))
        return float(min(valor[0] for valor in limites)), float(max(valor[1] for valor in limites))

    def cilindro(valores):
        x, y, _ = centro(valores)
        largura, altura = dimensoes(valores)
        z_inicial, z_final = intervalo_z(valores)
        return {"start": [x, y, z_inicial], "end": [x, y, z_final], "radius": float((largura + altura) / 4)}

    def semiesfera(valores, normal):
        raios = [float(sum(dimensoes([item])) / 4) for item in valores]
        maior = int(np.argmax(raios))
        x, y, z = centro([valores[maior]])
        return {"center": [x, y, z], "radius": raios[maior], "normal": normal}

    def telhado_duas_aguas(valores, z_base=None):
        x, y, _ = centro(valores)
        largura, altura = dimensoes(valores)
        z_inicial, z_final = intervalo_z(valores)
        base = float(z_base if z_base is not None else z_inicial)
        menor, maior = float(min(largura, altura)), float(max(largura, altura))
        topo = max(float(z_final), base + menor * math.tan(math.radians(30)) / 2)
        if largura >= altura:
            inicios = ([x, y - altura / 2, base], [x, y + altura / 2, base])
        else:
            inicios = ([x - largura / 2, y, base], [x + largura / 2, y, base])
        fim = [x, y, topo]
        return [{"start": inicio, "end": fim, "width": maior} for inicio in inicios]

    for geometria in resultado["geometrias"]:
        tipo = geometria["geometria"]
        valores = amostras(geometria)
        x, y, z = centro(valores)
        largura, altura = dimensoes(valores)
        raio = float((largura + altura) / 4)
        if tipo in {"GEOMETRIA_LINEAR", "TUBULACAO"}:
            inicios, finais = [], []
            for item in valores:
                inicio, fim = item["linha"]["start"], item["linha"]["end"]
                inicios.append(inicio if len(inicio) == 3 else [*inicio, item["fatia"]])
                finais.append(fim if len(fim) == 3 else [*fim, item["fatia"]])
            inicio = np.mean(inicios, axis=0)
            fim = np.mean(finais, axis=0)
            z_inicial, z_final = intervalo_z(valores)
            if tipo == "TUBULACAO":
                z = (z_inicial + z_final) / 2
                adicionar("cylinder", {
                    "start": [float(inicio[0]), float(inicio[1]), z],
                    "end": [float(fim[0]), float(fim[1]), z],
                    "radius": max((z_final - z_inicial) / 2, 0.01),
                })
                continue
            base_inicio, base_fim = [float(inicio[0]), float(inicio[1]), z_inicial], [float(fim[0]), float(fim[1]), z_inicial]
            topo_inicio, topo_fim = [float(inicio[0]), float(inicio[1]), z_final], [float(fim[0]), float(fim[1]), z_final]
            for linha in (
                {"start": base_inicio, "end": base_fim}, {"start": topo_inicio, "end": topo_fim},
                {"start": base_inicio, "end": topo_inicio}, {"start": base_fim, "end": topo_fim},
            ):
                adicionar("line", linha)
        elif tipo == "QUADRADO":
            adicionar("square", {"center": [x, y, z], "size": float((largura + altura) / 2), "rotation": [0, 0, 0]})
        elif tipo == "RETANGULO":
            adicionar("rectangle", {"center": [x, y, z], "width": float(largura), "height": float(altura), "rotation": [0, 0, 0]})
        elif tipo == "CIRCULO":
            adicionar("circle", {"center": [x, y, z], "radius": raio, "normal": [0, 0, 1]})
        elif tipo == "MEIA_ESFERA":
            raios = [float(sum(dimensoes([item])) / 4) for item in valores]
            normal = [0, 0, -1 if raios[-1] > raios[0] else 1]
            adicionar("hemisphere", semiesfera(valores, normal))
        elif tipo == "TELHADO_DUAS_AGUAS":
            for rampa in telhado_duas_aguas(valores):
                adicionar("ramp", rampa)
        elif tipo == "PRISMA_COM_TELHADO_DUAS_AGUAS":
            paredes = amostras(geometria, {"QUADRADO", "RETANGULO"})
            cobertura = amostras(geometria, {"RAMPA"})
            px, py, _ = centro(paredes)
            pl, pa = dimensoes(paredes)
            z_inicial, z_final = intervalo_z(paredes)
            adicionar("box", {"center": [px, py, (z_inicial + z_final) / 2], "size": [float(pl), float(pa), z_final - z_inicial], "rotation": [0, 0, 0]})
            for rampa in telhado_duas_aguas(paredes[-3:] + cobertura, z_final):
                adicionar("ramp", rampa)
        elif tipo == "PRISMA_RETANGULAR":
            z_inicial, z_final = intervalo_z(valores)
            adicionar("box", {"center": [x, y, (z_inicial + z_final) / 2], "size": [float(largura), float(altura), z_final - z_inicial], "rotation": [0, 0, 0]})
        elif tipo == "CILINDRO":
            adicionar("cylinder", cilindro(valores))
        elif tipo == "TANQUE_COM_SEMIESFERA_SUPERIOR":
            circulos = amostras(geometria, {"CIRCULO"})
            semiesferas = amostras(geometria, {"MEIA_ESFERA"})
            corpo = cilindro(circulos)
            adicionar("cylinder", corpo)
            maior_circulo = max(item["fatia"] for item in circulos)
            superior = [item for item in semiesferas if item["fatia"] > maior_circulo]
            if superior:
                adicionar("hemisphere", {"center": corpo["end"], "radius": corpo["radius"], "normal": [0, 0, 1]})

    salvar_json(caminho_json, objetos)
    return objetos


def analisar_imagens(caminhos_imagens, caminho_json=SAIDA_JSON, metadados=None, caminho_json_3d=None):
    metadados = metadados or {}
    caminhos_imagens = sorted(caminhos_imagens, key=lambda caminho: (
        metadados.get(Path(caminho).name.lower(), {}).get("z_inicial", float("inf")), chave_natural(caminho)
    ))
    fatias = [analisar_imagem(caminho, metadados.get(Path(caminho).name.lower())) for caminho in caminhos_imagens]
    for indice, fatia in enumerate(fatias, 1):
        fatia["indice"] = indice
    geometrias = montar_geometrias(fatias)
    resultado = {
        "analise": {
            "total_fatias": len(fatias),
            "total_geometrias": len(geometrias),
            "criterio": f"Geometrias consistentes precisam aparecer em pelo menos {MIN_FATIAS_3D} fatias; linhas transitorias com confianca minima de {MIN_CONFIANCA_TUBULACAO:.0%} representam tubulacoes.",
            "regras_engenharia": {
                "tanque": "Semiesfera somente no topo, conectada ao cilindro com mesmo centro e raio.",
                "telhado": "Duas rampas simetricas conectadas pela mesma cumeeira e apoiadas sobre o prisma.",
                "linhas": "Linhas transitorias representam tubulacoes; segmentos persistentes permanecem independentes.",
            },
        },
        "fatias": fatias,
        "geometrias": geometrias,
    }

    salvar_json(caminho_json, resultado)
    gerar_json_3d(resultado, caminho_json_3d or Path(caminho_json).with_name(SAIDA_JSON_3D.name))

    return resultado


def analisar_pasta(caminho_pasta, caminho_json=SAIDA_JSON):
    pasta = Path(caminho_pasta)
    imagens = sorted(
        arquivo for arquivo in pasta.iterdir()
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_IMAGEM
    )
    return analisar_imagens(imagens, caminho_json, carregar_metadados(pasta))


if __name__ == "__main__":
    janela = Tk()
    janela.withdraw()
    pasta = filedialog.askdirectory(title="Selecione a pasta com as imagens")
    janela.destroy()

    if pasta:
        resultado = analisar_pasta(pasta)
        print(f"{resultado['analise']['total_fatias']} imagens analisadas.")
        print(f"{resultado['analise']['total_geometrias']} geometrias inferidas.")
        print(f"JSON salvo em: {SAIDA_JSON}")
        print(f"JSON 3D salvo em: {SAIDA_JSON_3D}")
    else:
        print("Nenhuma pasta selecionada.")
