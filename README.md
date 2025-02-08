# Desafio Pós-Graduação - Fase 04

Este repositório contém a implementação de um projeto desenvolvido como parte do Desafio da Pós-Graduação - Fase 04. O objetivo deste projeto é extrair, processar e enriquecer dados extraídos de vídeos para gerar resumos detalhados com base em transcrições e análises emocionais.

## Funcionalidades Principais

O sistema implementado possui as seguintes funcionalidades:

1. **Detecção de Emoções no Vídeo:**
   O projeto utiliza algoritmos de reconhecimento de emoções a partir da análise de áudio e imagens extraídas de um vídeo. Essas emoções são registradas com timestamps para associá-las aos eventos específicos do vídeo.

2. **Transcrição de Áudio com Timestamps:**
   A partir do áudio do vídeo, é realizada a transcrição com a inclusão de timestamps para cada segmento de fala. O processo de transcrição é feito utilizando ferramentas de reconhecimento de fala em tempo real, o que permite a criação de transcrições detalhadas.

3. **Resumo de Conteúdo:**
   O sistema gera um resumo do conteúdo do vídeo com base nas transcrições. Esse resumo é então enriquecido com as informações extraídas das emoções detectadas no vídeo, criando um resumo mais completo e contextualizado.

4. **Integração com Modelos de Linguagem:**
   O resumo, juntamente com as transcrições e emoções, é enviado para uma IA local baseada no modelo **LLama 3.2** para gerar um resumo enriquecido. O modelo de IA é usado para refinar e melhorar o conteúdo gerado, incluindo o uso de títulos, palavras-chave e informações adicionais que podem ser úteis para uma melhor compreensão do vídeo.

## Objetivo do Projeto

O objetivo principal deste projeto é criar uma solução automatizada para enriquecer e contextualizar resumos de vídeos utilizando múltiplas fontes de dados, como emoções detectadas, transcrições com timestamps e resumos do conteúdo. Isso oferece uma nova abordagem para análise de vídeos, melhorando a compreensão por meio de resumos mais completos e precisos, com base em várias dimensões de dados.

## Tecnologias e Ferramentas Utilizadas

- **Reconhecimento de Emoções:** Algoritmos de processamento de áudio e imagem para detectar emoções.
- **Transcrição de Áudio:** Utilização de APIs para transcrição automática com timestamps.
- **LLama 3.2:** Modelo de linguagem utilizado para enriquecer os resumos com dados contextuais.
- **Python:** Linguagem de programação utilizada para desenvolver o projeto.
- **Ollama:** Ferramenta usada para interação com o modelo de linguagem local.

## Estrutura do Repositório

O repositório contém os seguintes módulos e arquivos principais:

- **Processamento de Áudio e Vídeo:** Código responsável por extrair áudio do vídeo e realizar o reconhecimento de emoções e transcrição.
- **Integração com LLM:** Código para interagir com o modelo de linguagem LLama, enviando os dados e recebendo os resumos enriquecidos.
- **Gerador de Resumo:** Algoritmo para gerar e enriquecer o resumo do vídeo com base nas transcrições e emoções.

## Como Rodar o Projeto

### Requisitos

1. **Python 3.x** (recomenda-se Python 3.8 ou superior).
2. **Dependências do projeto:** Todas as dependências necessárias estão listadas no arquivo `requirements.txt`. Para instalá-las, basta executar:
3. **Ollama Local:** É necessário ter o **Ollama** rodando localmente para realizar as interações com o modelo LLama.

### Passos para Execução

1. Extraia o áudio do vídeo e realize o processamento necessário para detectar emoções.
2. Transcreva o áudio com timestamps.
3. Envie os dados para o modelo de IA para gerar um resumo enriquecido.
4. O resultado será um resumo que inclui título, palavras-chave, informações adicionais e timestamps.

## Contribuições

Contribuições são bem-vindas! Se você tiver melhorias, correções ou novas funcionalidades que gostaria de adicionar ao projeto, sinta-se à vontade para enviar um *pull request*.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
