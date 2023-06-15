import cv2
import numpy as np


def average(img):

    # Extrahieren der Breite und Höhe des Bildes
    height, width, channels = img.shape

    # Erstellen eines neuen Bildes für die Ausgabe mit numpy
    output_img = np.zeros((height, width, channels), np.uint8)

    # Schleife durch alle Spalten des Eingangsbildes und berechne den Mittelwert für jede Farbkomponente
    for x in range(width):
        column = img[:, x]
        r_avg = int(np.mean(column[:, 0]))
        g_avg = int(np.mean(column[:, 1]))
        b_avg = int(np.mean(column[:, 2]))
        # Setze den resultierenden Mittelwert als den Farbwert für alle Pixel in dieser Spalte im Ausgabebild
        output_img[:, x] = [r_avg, g_avg, b_avg]

    return output_img


def rgb_To_Yuv(img):

    # Konvertierung von RGB zu R'G'B'
    img_rg = np.power(img / 255.0, 1.0 / 2.2)

    # Konvertieren in YUV
    R = img_rg[:, :, 2]
    G = img_rg[:, :, 1]
    B = img_rg[:, :, 0]

    # Y U V - Werte berechnen
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = 0.493 * (B - Y)
    V = 0.877 * (R - Y)

    # Zusammenführen von Y, U und V zu einem Bild
    img_conv = np.dstack((Y, U, V))

    return img_conv


def shadow_detection(img):

    # Berechne T1
    U = img[:, :, 1]

    # Mittelwert und Standardabweichung von U
    mu_U = np.mean(U)
    std_U = np.std(U)
    T1 = np.zeros_like(U)
    T1[U > (mu_U + std_U)] = 255

    # Berechne T2
    V = img[:, :, 2]

    # Mittelwert und Standardabweichung von V
    mu_V = np.mean(V)
    std_V = np.std(V)
    T2 = np.zeros_like(V)
    T2[V < (mu_V - std_V)] = 255

    # Berechne die Schattenregion s als Kombination von T1 und T2
    s = (T1 * T2)

    return s


img_input = cv2.imread('p01_schatten.jpg')

# Bilder verarbeiten
img_average = average(img_input)
img_yuv = rgb_To_Yuv(img_input)
img_shadow = shadow_detection(img_yuv)

# Bilder anzeigen
cv2.imshow('Input', img_input)
cv2.imshow('Average', img_average)
cv2.imshow('YUV', img_yuv)
cv2.imshow('Shadow', img_shadow)

# Auf Eingabe warten
cv2.waitKey(0)
cv2.destroyAllWindows()
