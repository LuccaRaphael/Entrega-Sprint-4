## **LeadTech - Estilista Inteligente: Sistema de Recomendações Personalizadas**

## **Objetivo Principal**
Desenvolver um **Sistema de Recomendação Personalizada** baseado em **Computer Vision**, que auxilie os clientes a encontrar produtos de moda de acordo com suas preferências e os produtos disponíveis na loja, aumentando assim as vendas ao oferecer recomendações específicas e direcionadas.

[Vídeo](https://youtu.be/VKAx2upQzrU)


## **Objetivos Específicos**
- **Demonstração do protótipo funcional**: Apresentar o estado atual do sistema, demonstrando as funcionalidades implementadas, como detecção de itens de moda e sistema de recomendação baseado em similaridade visual.
- **Detalhamento da arquitetura de IA**: Explicar a arquitetura utilizada para detecção de objetos e embeddings de imagens, justificando a escolha dos modelos e técnicas aplicadas.
- **Base de dados utilizada**: Detalhar os datasets usados no treinamento e teste do modelo, com links e descrições sobre como os dados foram preparados.

---

## **1. Demonstração do Protótipo Funcional**

O sistema de **Recomendações Personalizadas** permite que os usuários recebam sugestões de produtos similares baseadas em imagens. A interface oferece um mecanismo de busca visual para identificar produtos similares aos itens de interesse do cliente, ajudando na tomada de decisão de compra.

O protótipo conta com as seguintes funcionalidades:
- **Detecção de Objetos de Moda**: Utilizando o modelo **YOLOv5**, o sistema detecta e recorta itens de moda em uma imagem fornecida pelo usuário.
- **Embeddings de Imagem**: A partir dos objetos detectados, o sistema gera embeddings, que são representações vetoriais das características visuais dos itens. Esses embeddings são usados para encontrar produtos visualmente semelhantes no catálogo da loja.
- **Recomendações Personalizadas**: Com base nas preferências do cliente e nos itens disponíveis na loja, o sistema sugere produtos similares, ajudando o cliente a encontrar peças que correspondam ao seu gosto.
- **Apresentação via Streamlit**: O protótipo é apresentado usando **Streamlit**, que permite que os usuários façam upload de imagens e recebam recomendações visuais em tempo real.

### **Tecnologias Utilizadas**
- **Python**: Linguagem principal para o desenvolvimento.
- **YOLOv5**: Usado para detecção de objetos de moda em imagens.
- **AutoEncoder**: Rede neural utilizada para gerar embeddings visuais dos itens detectados.

## **2. Detalhamento da Arquitetura de IA**

A arquitetura do sistema de recomendações personalizadas foi desenvolvida em três fases interdependentes: **coleta e preparação de dados**, **modelagem de machine learning** e **fase de inferência**. Cada uma dessas fases desempenha um papel crucial no funcionamento do sistema, desde a coleta de dados até a recomendação final ao usuário, baseada em similaridade visual.

### **2.1 Coleta e Preparação dos Dados**
A primeira fase envolve a **coleta de dados** e sua **preparação** para treinar os modelos de detecção de objetos e de embeddings. Utilizamos datasets públicos com imagens de moda, sendo o principal deles o **Complete the Look Dataset**, que contém mais de **12.000 imagens de estilo** com caixas delimitadoras. Essas imagens são usadas para identificar e categorizar itens de moda, como roupas e acessórios.

#### **Etapas do processo:**
- **Limpeza e Transformação**: Os dados brutos coletados foram organizados utilizando a biblioteca **Pandas**, onde foram extraídas as colunas relevantes e removidos valores nulos ou strings vazias. Isso garante que o modelo seja treinado com dados consistentes e de alta qualidade.
- **Balanceamento de Classes**: Como em muitos datasets, havia uma desproporção entre as diferentes categorias de moda. Para garantir que o modelo não tenha viés em favor de certas categorias, aplicamos técnicas de balanceamento, resultando em um conjunto equilibrado de **15.657 objetos** distribuídos em **9.235 imagens**, com cerca de **1.000 imagens por categoria de moda**.
- **Correção de Imagens Corrompidas**: Utilizamos um script com **ImageMagick** para verificar e corrigir imagens corrompidas, evitando que o treinamento falhe ou sofra com entradas inválidas.
- **Criação dos Arquivos de Anotação**: Cada imagem recebeu um arquivo `.txt` associado, contendo as coordenadas de suas caixas delimitadoras e a classe dos objetos identificados. Esses arquivos são essenciais para treinar o modelo de detecção de objetos.

### **2.2 Modelo de Detecção de Objetos**
Com os dados preparados, a próxima fase é a construção e treinamento do **modelo de detecção de objetos**, responsável por identificar peças de moda em imagens. Utilizamos o **YOLOv5** pré-treinado, ajustado especificamente para detecção de roupas e acessórios.

#### **Funcionamento**:
- **Treinamento**: O modelo YOLOv5 foi ajustado com os dados anotados da fase anterior. Ele aprende a detectar objetos de moda em diferentes cenários e posições, gerando caixas delimitadoras que identificam e segmentam os itens nas imagens.
- **Resultados**: Após o treinamento, o modelo apresentou os seguintes resultados:
  - **Precisão (Precision)**: 54.8
  - **Recall**: 64.8
  - **mAP** (Mean Average Precision): 60.9

Esses resultados indicam que o modelo consegue detectar objetos com uma boa taxa de acerto, equilibrando precisão e recall.

### **2.3 Modelo de Embeddings**
Após a detecção dos objetos de moda, cada item é transformado em uma **representação vetorial** (embedding) por meio de um **AutoEncoder**. Esta representação captura as características visuais dos itens, como cor, textura e forma, permitindo que o sistema compare visualmente diferentes peças de moda.

#### **Funcionamento**:
- **Arquitetura do AutoEncoder**: O modelo de AutoEncoder é composto por um **encoder**, que comprime a imagem do item detectado em um vetor de 512 dimensões (embedding), e um **decoder**, que reconstrói a imagem a partir desse vetor.
- **Embedding de 512 Dimensões**: O "bottleneck" do AutoEncoder, onde ocorre a compressão, é o ponto onde a imagem é convertida em uma **representação vetorial** de 512 dimensões. Essa compactação permite uma busca eficiente de itens semelhantes com base em suas características visuais.
- **Comparação Visual**: Os embeddings gerados são utilizados para calcular a similaridade entre diferentes peças. Quanto mais próximo o vetor de um produto estiver de outro, mais semelhante ele é visualmente.

#### **Resultados do Modelo de Embeddings**:
- **Baseline (4096-d)**: Índice de tamanho 163.8 MB, mAP de 0.44.
- **+Layer (512-d)**: Índice de tamanho 25 MB, mAP de 0.53 (modelo utilizado no sistema).

A escolha pelo embedding de 512 dimensões foi feita para garantir um equilíbrio entre **precisão** e **eficiência computacional**, já que a redução no tamanho do vetor diminuiu o tempo de inferência e o espaço de armazenamento sem sacrificar significativamente a precisão.

### **2.4 Fase de Inferência e Recomendações**
Na fase final, o sistema já treinado utiliza os embeddings gerados para fornecer **recomendações personalizadas**. Quando o usuário faz upload de uma imagem, o modelo de detecção de objetos (YOLOv5) identifica as peças de moda na imagem e, em seguida, o modelo de embeddings compara essas peças com o catálogo da loja.

#### **Processo de Inferência**:
1. **Detecção do Item**: O YOLOv5 detecta e segmenta a peça de vestuário ou acessório.
2. **Geração do Embedding**: O AutoEncoder transforma o item detectado em um vetor de 512 dimensões.
3. **Busca por Similaridade**: O vetor gerado é comparado com os vetores já armazenados no sistema, e os itens mais semelhantes visualmente são retornados como sugestões.
4. **Apresentação ao Usuário**: As recomendações são apresentadas via aplicação React, permitindo uma navegação rápida e intuitiva.

#### **Justificativa da Arquitetura**:
A arquitetura foi desenhada para balancear **precisão** e **eficiência**. O uso do YOLOv5 garante detecção rápida e precisa, enquanto o AutoEncoder reduz a complexidade computacional na busca por similaridade, mantendo a alta precisão de recomendações.

Aqui está o passo a passo atualizado para ser adicionado ao seu README:

---

## **Passo a Passo de Instalação e Execução**

1. **Clone o projeto**  
   ```bash
   git clone https://github.com/LuccaRaphael/Entrega-Sprint-4.git
   ```

2. **Instale as dependências do Python**  
   No diretório raiz do projeto, instale as dependências necessárias para o backend:
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute o frontend (Estilista Inteligente)**  
   Navegue até a pasta do frontend e instale as dependências do Node:
   ```bash
   cd frontend
   cd estilistainteligente
   npm install
   ```
   Em seguida, inicie o frontend:
   ```bash
   npm start
   ```

4. **Execute o backend**  
   Volte para o diretório raiz (`ENTREGA-SPRINT-4`) e inicie o backend com:
   ```bash
   python home.py
   ```



<img src="src\modelo.png" alt="Texto Alternativo">
