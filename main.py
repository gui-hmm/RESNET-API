from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
from PIL import Image
import io
import torch

app = FastAPI()

# Adicionando o CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens (domínios)
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos os cabeçalhos
)

# Carregar o modelo pré-treinado
model = models.resnet50(pretrained=True)
model.eval()

# Carregar as classes do ImageNet
with open("imagenet_classes.txt") as f:
    imagenet_classes = [line.strip() for line in f.readlines()]

# Preprocessamento da imagem
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Ler e processar a imagem
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)

        # Fazer a previsão
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            class_name = imagenet_classes[predicted_idx.item()]

        return {"class_name": class_name}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
async def root():
    return {"message": "API Predictive ResnetImages"}
