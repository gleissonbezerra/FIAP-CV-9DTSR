import cv2
import numpy as np

# Abre o vídeo (substitua por 0 para webcam ao vivo)
cap = cv2.VideoCapture(1)

# Lê o primeiro frame do vídeo
ret, old_frame = cap.read()
if not ret:
    print("Erro ao ler o vídeo.")
    cap.release()
    exit()

# Converte o primeiro frame para escala de cinza
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detecta os cantos (pontos de interesse) com o algoritmo Shi-Tomasi
p0 = cv2.goodFeaturesToTrack(old_gray,                 # imagem
                             maxCorners=100,           # máximo de pontos
                             qualityLevel=0.3,         # qualidade mínima
                             minDistance=7,            # distância mínima entre pontos
                             blockSize=7)              # tamanho da janela para detecção

# Define os parâmetros do algoritmo de Lucas-Kanade
lk_params = dict(winSize=(15, 15),                     # janela de busca
                 maxLevel=2,                           # níveis da pirâmide
                 criteria=(cv2.TERM_CRITERIA_EPS |     # critério de parada:
                          cv2.TERM_CRITERIA_COUNT,     # ou número máximo de iterações
                          10, 0.03))                    # 10 iterações ou erro < 0.03

# Cria uma imagem preta (mesmo tamanho do vídeo) para desenhar as trilhas
mask = np.zeros_like(old_frame)

# Loop principal de leitura dos frames do vídeo
while True:
    ret, frame = cap.read()             # lê o próximo frame
    if not ret:
        break                           # termina se acabar o vídeo

    # Converte o novo frame para escala de cinza
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcula o fluxo óptico com base no frame anterior e atual
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,   # imagem anterior
                                           frame_gray, # imagem atual
                                           p0,          # pontos anteriores
                                           None,        # pontos novos serão calculados
                                           **lk_params)

    # Verifica se há pontos válidos para rastreamento
    if p1 is not None and st is not None:
        # Seleciona apenas os pontos que foram encontrados com sucesso (status == 1)
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Para cada par de pontos (anterior e atual), desenha a trilha
        for new, old in zip(good_new, good_old):
            a, b = new.ravel()             # novo ponto (coordenadas float)
            c, d = old.ravel()             # ponto anterior

            # Desenha uma linha verde do ponto anterior ao novo
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)

            # Desenha um pequeno círculo vermelho no novo ponto
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        # Combina a imagem atual com a máscara para mostrar as trilhas
        img = cv2.add(frame, mask)

        # Exibe o resultado
        cv2.imshow("Lucas-Kanade Optical Flow", img)

        # Atualiza os dados para o próximo loop
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)  # reorganiza os pontos para o próximo cálculo

    # Interrompe o loop se a tecla ESC for pressionada
    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()
