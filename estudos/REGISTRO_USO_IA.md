# Registro de Uso de IA — Repositório de Estudos

Registro no estilo de log de commits. Adicione uma entrada a cada sessão em que usar IA
(mais recente no topo). Modelo de entrada no fim do arquivo.

---

### 2026-10-28 — Criação da estrutura de estudos (padronização e clusterização)

- **Ferramenta:** Claude Code (Anthropic), sessão remota
- **Pedido:** criar artefatos de aprendizado sobre padronização de dados e principais algoritmos de clusterização, com base no Projeto 2 (workflow KNIME em `mineracao_de_dados/`), em um repositório de estudos que possa evoluir.
- **O que a IA fez:**
  - Analisou o workflow KNIME (SimpleKMeans k=7, distância euclidiana, t-SNE) e a base PNS usada nele.
  - Criou a pasta `estudos/` com roteiro, glossário, template de tópico e `requirements.txt`.
  - Escreveu a teoria de padronização (`01_padronizacao/README.md`) e de clusterização (`02_clusterizacao/*.md`: K-Means, hierárquico, DBSCAN/HDBSCAN, GMM, categóricos, avaliação, visualização).
  - Criou e executou 3 notebooks (padronização; comparação de algoritmos em dados sintéticos; aplicação na PNS) e as funções em `utils/`.
- **Resultados que vieram da execução (não da IA "achar"):** na PNS, silhueta ≈ 0,05–0,10, K-Means/K-Modes com k=7 instáveis entre sementes, k=2 mais estável. Esses números foram calculados pelo código e podem ser reproduzidos.
- **Pontos para verificação humana:**
  - Comportamento exato do nó SimpleKMeans/EuclideanDistance do Weka no KNIME (normalização e nominais) — conferir no diálogo do nó.
  - Se `Q00201` foi usada como atributo no K-Means do KNIME.
  - Ordem das categorias ordinais em `utils/pns.py` (definida pela IA a partir dos rótulos).
  - Referências bibliográficas citadas devem ser checadas antes de usar em trabalho formal.
- **Autoria:** estrutura, textos e código gerados pela IA; revisão, validação e uso acadêmico sob responsabilidade do aluno.

---

### Modelo de entrada

```
### AAAA-MM-DD — título curto

- **Ferramenta:** (ex.: Claude, ChatGPT, Copilot)
- **Pedido:** o que foi solicitado
- **O que a IA fez:** lista objetiva
- **O que eu fiz/alterei:** revisão, correções, decisões próprias
- **Pontos para verificação:** o que não foi conferido ainda
```
