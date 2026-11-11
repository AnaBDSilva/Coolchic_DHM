import numpy as np
import cv2, os
from modules.processUtils import save_json, load_json

########################################################################################################################################################################
## ROI related function: choose standart and apply to a hologram                                                                                                      ##
########################################################################################################################################################################

def select_roi_opencv(image, scale=0.3, initial_coords=None):
    img = image.copy()

    # Normalizar para display
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img_norm = img_norm.astype(np.uint8)
    
    # Passar para formato a cores (BGR) para podermos desenhar o retângulo vermelho
    img_color = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)

    # Resize apenas para visualização
    h, w = img_norm.shape
    resized = cv2.resize(img_color, (int(w*scale), int(h*scale)))

    if initial_coords is None:
        # Desenhar, não se pode mexer 
        roi = cv2.selectROI("Desenha o ROI (Enter guarda, C cancela)", resized, showCrosshair=True)
        cv2.destroyAllWindows()

        rx, ry, rw, rh = roi
        if rw == 0 or rh == 0:
            return None, None, None, None

        # Converter coordenadas para a imagem original
        x = int(rx / scale)
        y = int(ry / scale)
        rw_orig = int(rw / scale)
        rh_orig = int(rh / scale)

        rmin, rmax = y, y + rh_orig
        cmin, cmax = x, x + rw_orig

        return rmin, rmax, cmin, cmax          

    else:
        # Tamanho trancado apenas se pode mover
        rmin_i, rmax_i, cmin_i, cmax_i = initial_coords
        
        # Converter coordenadas originais para a escala do ecrã
        x = int(cmin_i * scale)
        y = int(rmin_i * scale)
        w_box = int((cmax_i - cmin_i) * scale)
        h_box = int((rmax_i - rmin_i) * scale)

        dragging = False
        offset_x = 0
        offset_y = 0
        
        # Função para detetar o arrastar do rato
        # (x, y) -> canto esquerdo do retangulo, (mx, my) -> sitio do rato onde estamos a agarrar
        def mouse_callback(event, mx, my, flags, param):
            nonlocal x, y, dragging, offset_x, offset_y
            
            if event == cv2.EVENT_LBUTTONDOWN:
                # Se clicou dentro do retângulo vermelho, guarda a distancia entre esse ponto e o canto esquerdo
                if x <= mx <= x + w_box and y <= my <= y + h_box:
                    dragging = True
                    offset_x = mx - x
                    offset_y = my - y
            
            elif event == cv2.EVENT_MOUSEMOVE:
                # Aplica o offset para saber onde desenhar o canto esquerdo através da distância do canto ao ponto onde estamos a agarrar
                if dragging:
                    x = mx - offset_x
                    y = my - offset_y
                    
            elif event == cv2.EVENT_LBUTTONUP:
                dragging = False

        window_name = "ROI Trancado: Arrasta com o rato (Prime ENTER guarda, ESC cancela)"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        # Loop para manter a janela aberta a atualizar o arrasto
        while True:
            display = resized.copy()
            cv2.rectangle(display, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(20) & 0xFF
            if key == 13 or key == 32:  # 13=Enter, 32=Espaço -> Confirma
                break
            elif key == 27 or key == ord('c'):  # 27=Esc, 'c' -> Cancela
                cv2.destroyAllWindows()
                return None, None, None, None
                
        cv2.destroyAllWindows()

        # Converter de volta para a imagem original 
        orig_h = rmax_i - rmin_i
        orig_w = cmax_i - cmin_i
        
        new_cmin = int(x / scale)
        new_rmin = int(y / scale)
        
        return new_rmin, new_rmin + orig_h, new_cmin, new_cmin + orig_w

# loads the specific fiber intensity and then allows for choosing the ROI standart
def chooseStandartROI(fiber_name):
    # load the intensity to choose the roi
    filepath = f"reconstructions/ROI/intensity/{fiber_name}_intensity.npy"
    intensity = np.load(filepath)

    rmin, rmax, cmin, cmax = select_roi_opencv(intensity, scale=0.3)    
    if rmin is None:
        print("Seleção de ROI cancelada.")
        return
  
    #salvar os limites do recorte para aplicar o mesmo na compressão
    info = {
        "rmin": float(rmin),
        "rmax": float(rmax),
        "cmin": float(cmin),
        "cmax": float(cmax)
    }
        
    filepath = f"reconstructions/ROI/ROI_measure.json"
    save_json(filepath, info)

def applyStandartROI(fiber_name):
    filepath = f"reconstructions/ROI/intensity/{fiber_name}_intensity.npy"
    filepath_json = f"reconstructions/ROI/ROI_measure.json"

    intensity = np.load(filepath)

    initial_coords = None
    # Ler as medidas de ROI guardadas e utilizar 
    if os.path.exists(filepath_json):
        try:
            info = load_json(filepath_json)
            initial_coords = (info["rmin"], info["rmax"], info["cmin"], info["cmax"])
        except Exception as e:
            pass

        rmin, rmax, cmin, cmax = select_roi_opencv(intensity, 0.3, initial_coords=initial_coords)
        
        #salvar os limites do recorte para aplicar o mesmo na compressão
        info = {
            "rmin": float(rmin),
            "rmax": float(rmax),
            "cmin": float(cmin),
            "cmax": float(cmax)
        }
        
        jsonPath = filepath + "_infoROI.json"
        save_json(jsonPath, info)

    return rmin, rmax, cmin, cmax