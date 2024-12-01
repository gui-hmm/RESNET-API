# FastAPI Image Classification - ResNet50

Esta aplicação utiliza o FastAPI para construir uma API de classificação de imagens utilizando um modelo pré-treinado ResNet50, que foi treinado no dataset ImageNet. A aplicação permite o upload de imagens e retorna a classe prevista para a imagem fornecida.

## Tecnologias Utilizadas

- **FastAPI**: Framework para construir a API.
- **PyTorch**: Para carregar e executar o modelo pré-treinado ResNet50.
- **Torchvision**: Para pré-processamento de imagens e o modelo ResNet50.
- **Pillow**: Para manipulação e carregamento de imagens.
- **Uvicorn**: Servidor ASGI para rodar a aplicação FastAPI.

## Pré-requisitos

Certifique-se de ter o Python 3.7+ instalado no seu ambiente.

## Instalação

Clone o repositório:

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
