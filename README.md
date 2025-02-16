# Desafio Pós-Graduação - Fase 04

Este repositório contém a implementação de um projeto desenvolvido como parte do Desafio da Pós-Graduação - Fase 04. O objetivo deste projeto é extrair, processar e enriquecer dados extraídos de vídeos para gerar resumos detalhados com base em transcrições e análises emocionais.

## Funcionalidades Principais

O sistema implementado possui as seguintes funcionalidades:

1. **Detecção de Emoções no Vídeo:**
   O projeto utiliza algoritmos de reconhecimento de emoções a partir da análise de áudio e imagens extraídas de um vídeo. Essas emoções são registradas com timestamps para associá-las aos eventos específicos do vídeo.

2. **Detecção de Atividades/Poses:**
   Analisando o vídeo definimos algumas poses para identificação das atividades de movimento dos braços e toque das mãos no rosto. Essas atividades são registradas com os respectivos frames e timestamp para associá-las aos eventos do vídeo.

3. **Transcrição de Áudio com Timestamps:**
   A partir do áudio do vídeo, é realizada a transcrição com a inclusão de timestamps para cada segmento de fala. O processo de transcrição é feito utilizando ferramentas de reconhecimento de fala em tempo real, o que permite a criação de transcrições detalhadas.

4. **Resumo de Conteúdo:**
   O sistema gera um resumo do conteúdo do vídeo com base nas transcrições. Esse resumo é então enriquecido com as informações extraídas das emoções detectadas no vídeo, criando um resumo mais completo e contextualizado.

5. **Integração com Modelos de Linguagem:**
   O resumo, juntamente com as transcrições e emoções, é enviado para uma IA local baseada no modelo **LLama 3.2** para gerar um resumo enriquecido. O modelo de IA é usado para refinar e melhorar o conteúdo gerado, incluindo o uso de títulos, palavras-chave e informações adicionais que podem ser úteis para uma melhor compreensão do vídeo.

## Objetivo do Projeto

O objetivo principal deste projeto é criar uma solução automatizada para enriquecer e contextualizar resumos de vídeos utilizando múltiplas fontes de dados, como emoções e atividades detectadas, transcrições com timestamps e resumos do conteúdo. Isso oferece uma nova abordagem para análise de vídeos, melhorando a compreensão por meio de resumos mais completos e precisos, com base em várias dimensões de dados.

## Tecnologias e Ferramentas Utilizadas

- **Reconhecimento de Emoções:** Algoritmos de processamento de áudio e imagem para detectar emoções.
- **Reconhecimento de Atividades:** Algoritmos de detecção de poses para identificação de atividades no vídeo.
- **Transcrição de Áudio:** Utilização de APIs para transcrição automática com timestamps.
- **LLama 3.2:** Modelo de linguagem utilizado para enriquecer os resumos com dados contextuais.
- **Python:** Linguagem de programação utilizada para desenvolver o projeto.
- **Ollama:** Ferramenta usada para interação com o modelo de linguagem local.

## Estrutura do Repositório

O repositório contém os seguintes módulos e arquivos principais:

### Módulos

- **Módulo main.py:** Módulo principal que realiza a orquestração chamando os demais módulos
- **Módulo desafio_fase_4.py:** Código responsável por extrair áudio do vídeo e realizar o reconhecimento de emoções e transcrição.
- **Módulo detect_pose.py:** Módulo responsável pela definição das poses e orquestração dos módulos de identificação das atividaes.
_ **Módulo is_arm_up.py:** Módulo responsável pela identificação da atividade de movimento dos braços. A Pose definida para este caso é termos o movimento do braço onde o cotovelo fique acima dos olhos.
- **Módulo is_hand_touching_face.py:** Módulo responsável pela identificação de toque da mão no rosto. 
- **Módulo movimentos_anomalos.py:** Módulo responsável pela identificação de movimentos anômalos.
- **Integração com LLM:** Código para interagir com o modelo de linguagem LLama, enviando os dados e recebendo os resumos enriquecidos.
- **Gerador de Resumo:** Algoritmo para gerar e enriquecer o resumo do vídeo com base nas transcrições e emoções.

### Arquivos

- **activity_log:** Arquivo que contém as atividades detectadas no vídeo
- **audio.wav:** Arquivo com o áudio extraído do vídeo
- **emotions_output.txt:** Arquivo que contém as emoções detectadas no vídeo
- **output_video_pose_detect_holistic.mp4:** Arquivo com o resultado do processamento de detecção de atividades/poses. Devido ao tamanho deste arquivo, não foi possível anexá-lo no git, para facilitar a visualização, deixamos o mesmo disponível [neste link](https://drive.google.com/file/d/1kD__cAogusetBaVsRi7HuB5xPwlyTkBk/view?usp=drive_link)
- **output_video_recognize.mp4:** Arquivo com o resultado do processamento de detecção de emoções.
- **requirements.txt:** Arquivo com as dependências do projeto
- **sumarize_llm.txt:** Arquivo contendo o resumo dos processamentos gerado pela llm.
- **transcriptions_timestamps.txt:** Arquivo com as transcrições e timestamps do vídeo.
- **Unlocking Facial Recognition_ Diverse Activities Analysis.mp4:** Vídeo utilizado para as detecções de atividaes e emoções.

## Como Rodar o Projeto

### Requisitos

1. **Python 3.x** (recomenda-se Python 3.8 ou superior).
2. **Dependências do projeto:** Todas as dependências necessárias estão listadas no arquivo `requirements.txt`. Para instalá-las, basta executar:
```
pip install -r requirements.txt
```
3. **Ollama Local:** É necessário ter o [**Ollama**]() rodando localmente para realizar as interações com o modelo LLama. Para este trabalho, utilizamos o modelo [llama3.2:latest](https://ollama.com/library/llama3.2). Para facilitar a instalação e excecução, sugerimos o [repositório oficial do Ollama no git](https://github.com/ollama/ollama).


### Passos para Execução

1. Dentro da raíz do projeto, execute
```
python main.py
````

#### Analise os arquivos de texto gerados

- activity_log.txt
- audio.wav
- emotions_output.txt
- sumarize_llm.txt
- sumarize.txt
- transcriptions_timestamps.txt

#### Analise os vídeos gerados

- output_video_pose_detect_holistic.mp4
- output_video_recognize.mp4


## Contribuições

Contribuições são bem-vindas! Se você tiver melhorias, correções ou novas funcionalidades que gostaria de adicionar ao projeto, sinta-se à vontade para enviar um *pull request*.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
