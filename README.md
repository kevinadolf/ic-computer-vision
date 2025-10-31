# Pipeline de Estimativa de Pose 6D com Gêmeo Digital

Este projeto implementa um pipeline completo, escrito em Python, para estimar em tempo real a pose 6D (posição e orientação) de objetos vistos por uma câmera Intel RealSense D456. A arquitetura combina detecção 2D com YOLOv8, reconstrução 3D via nuvem de pontos e uma análise geométrica *model-free* que dispensa arquivos CAD. O resultado final é visualizado em um gêmeo digital interativo que reflete continuamente a cena real.

---

## 🧠 Visão Geral da Solução

1. **Aquisição RGB-D**
   - `RealSenseCamera` inicializa o pipeline da D456, alinha automaticamente profundidade e cor e expõe os parâmetros intrínsecos da câmera para o Open3D.
   - Cada frame retorna um par `(depth_image, color_image)` diretamente utilizável pelo pipeline.

2. **Reconstrução 3D da Cena**
   - As imagens RGB-D são convertidas em uma nuvem de pontos colorida (`PointCloud`) orientada no referencial da câmera.
   - Um corte de profundidade evita ruídos distantes e a nuvem é “desvirada” para alinhar eixos com o mundo real.

3. **Detecção 2D com YOLOv8**
   - `ObjectDetector` utiliza o modelo pré-treinado `yolov8n.pt` (baixado automaticamente pelo pacote `ultralytics`).
   - Cada detecção fornece classe, *bounding box* e confiança.

4. **Segmentação 3D por *Bounding Box***
   - Para cada detecção, o volume 3D correspondente é recortado da nuvem de pontos usando `SelectionPolygonVolume`.
   - Pequenos ruídos são reduzidos com *voxel downsampling*.

5. **Análise de Forma e Pose (Model-Free)**
   - `ShapeAnalyzer` obtém a `OrientedBoundingBox` do objeto, classifica a geometria em **caixa**, **cilindro**, **esfera** ou **bloco genérico** e extrai:
     - Pose 6D (matriz 4×4 construída com o centro e a rotação da OBB).
     - Dimensões (extents) do objeto.
   - A cor predominante é estimada a partir da nuvem segmentada.

6. **Gêmeo Digital Não-Bloqueante**
   - `DigitalTwinVisualizer` usa o `Visualizer` do Open3D em modo não bloqueante.
   - A cena renderizada inclui a nuvem completa e as primitivas geométricas sobrepostas com a pose inferida e cor média do objeto.
   - O loop principal atualiza o visualizador a cada novo frame, mantendo a aplicação responsiva.

---

## 📂 Estrutura do Repositório

```
.
├── requirements.txt          # Dependências Python do projeto
└── src/
    ├── main.py               # Pipeline completo de pose e gêmeo digital
    ├── ...                   # Outros scripts auxiliares de exploração
```

O fluxo oficial está concentrado em `src/main.py`. Os demais arquivos no diretório `src/` podem ser explorados para testes adicionais, mas não fazem parte do pipeline atual.

---

## 🧱 Componentes Principais (`src/main.py`)

- **RealSenseCamera**  
  Configura streaming sincronizado de profundidade e cor, realiza o alinhamento (depth → color) e retorna frames prontos para o Open3D.

- **ObjectDetector**  
  Abstrai o uso do YOLOv8, encapsulando o limiar de confiança e entregando uma lista de detecções por frame.

- **ShapeAnalyzer**  
  Converte a nuvem segmentada em uma bounding box orientada:
  - Compara extensões relativas para decidir entre “sphere”, “cylinder”, “box” ou “block”.
  - Retorna pose 6D, dimensões e a própria OBB.

- **PoseEstimationPipeline**  
  Orquestra as etapas de captura, reconstrução, deteção, segmentação e análise. Produz, para cada objeto, um dicionário com:
  - `class_name`, `confidence`, `shape`, `pose`, `dimensions`, `color`, `point_cloud`.

- **DigitalTwinVisualizer**  
  Mantém uma janela Open3D aberta, atualizando continuamente:
  - Nuvem de pontos completa para contexto.
  - Primitivas geométricas dimensionadas e transformadas pela pose 6D estimada.
  - Coloração uniforme baseada na cor média medida pela câmera.

---

## 🧩 Pré-requisitos

1. **Hardware**
   - Intel RealSense D456 (ou compatível com SDK 2.0).

2. **Software**
   - Python 3.10 ou superior recomendado.
   - [Intel RealSense SDK 2.0](https://github.com/IntelRealSense/librealsense) instalado com suporte à câmera (Linux: `apt`, Windows: instalador oficial).
   - Compilação do módulo `pyrealsense2` compatível com a versão do SDK e do Python (já listado em `requirements.txt`).
   pip install -r requirements.txt

3. **Sistema Operacional**
   - Linux (Ubuntu 20.04+ testado).

---

## ⚙️ Configurando o Ambiente

1. **Clonar o repositório**
   ```bash
   git clone https://github.com/kevinadolf/ic-computer-vision.git
   cd <seu_repositorio>
   ```

2. **Criar ambiente virtual (opcional, recomendado); instrução para Linux**
   ```
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Instalar dependências**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   > O pacote `ultralytics` baixa automaticamente o peso `yolov8n.pt` na primeira execução.

4. **Testar se o módulo RealSense responde**
   ```bash
   python - <<'PY'
   import pyrealsense2 as rs
   ctx = rs.context()
   print(f"Câmeras detectadas: {len(ctx.devices)}")
   PY
   ```

---

## ▶️ Executando o Pipeline

1. Conecte a câmera Intel RealSense D456.
2. Com o ambiente ativado, rode:
   ```bash
   python src/main.py
   ```
3. A janela “Digital Twin” abrirá automaticamente. A cada frame:
   - O terminal reporta objetos detectados, forma estimada, dimensões e centro.
   - A visualização mostra nuvem de pontos + primitivas alinhadas.

4. **Encerrar:** pressione `Ctrl+C` no terminal. O pipeline fecha a janela e libera a câmera.

---

## 🔧 Ajustes Importantes

- **Modelo YOLO**: altere `model_path` em `ObjectDetector` para usar variantes maiores (`yolov8s.pt`, `yolov8m.pt`, etc.) se a GPU suportar.
- **Limiar de confiança**: ajuste `confidence_threshold` caso ocorram muitas detecções falsas.
- **Segmentação**: `min_points`, `voxel_size` e profundidade máxima (`max_depth`) controlam o filtro de ruídos na nuvem de pontos.
- **Classificação geométrica**: refine `sphere_tol`, `base_tol` e `diff_tol` em `ShapeAnalyzer` de acordo com seus objetos reais.
- **Visualização**: modifique a cor padrão ou adicione eixos locais extras na classe `DigitalTwinVisualizer`.

---

## 🧪 Estratégias de Teste

- **Validação offline**: salve pares RGB-D (`np.savez`) e alimente o método `run_single_frame` com imagens gravadas para depuração sem hardware.
- **Inspeção de limites**: teste objetos não rígidos ou com geometrias complexas para avaliar a classificação “block”.
- **Performance**: monitore uso de CPU/GPU quando aumentar a resolução ou trocar o modelo YOLO.

---

## 🚀 Próximas Evoluções Sugeridas

1. **Rastreamento temporal**: introduzir previsão/associação entre frames (ex.: filtro de Kalman) para suavizar poses.
2. **Fusão multi-frame**: acumular nuvens de pontos ao longo do tempo para melhorar a robustez em ambientes com oclusões.
3. **Classificação híbrida**: treinar um classificador leve (ex.: PointNet) sobre as OBBs para diferenciar objetos com formas semelhantes.
4. **Integração ROS 2**: publicar `PoseStamped` e `MarkerArray` para uso direto em manipuladores robóticos.
5. **Persistência**: registrar históricos de poses e gerar estatísticas de confiança por classe.

---

## 📝 Créditos e Licença

- Autor: Kevin Adolfo Carvalho Koberstain de Araújo / 202110036511
- Licença: [MIT](LICENSE)
