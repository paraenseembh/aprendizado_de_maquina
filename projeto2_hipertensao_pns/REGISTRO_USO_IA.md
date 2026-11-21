# Registro de Uso de IA — Projeto 2 (Hipertensão / PNS 2019)

### 2026-11-21 — Organização do projeto no GitHub

- **Ferramenta:** Claude Code (Anthropic), sessão remota
- **Pedido:** colocar o projeto de Aprendizado de Máquina no GitHub com o artigo, a referência da base e um README.
- **O que a IA fez:**
  - Localizou o artigo final (`TP_final.pdf`) no Google Drive e copiou para `artigo/`, sem alterar o conteúdo.
  - Escreveu o `README.md` do projeto com base no texto do artigo e nos arquivos do repositório (workflow KNIME, CSV da PNS).
  - Escreveu `REFERENCIA_BASE.md`. Os números (linhas, distribuição do alvo, ausentes) foram calculados direto do CSV.
- **O que NÃO foi feito pela IA:** artigo, workflow KNIME, pré-processamento e resultados são trabalho do aluno. Nada disso foi alterado.
- **Pontos para verificação humana:**
  - Os links do IBGE/Fiocruz não puderam ser abertos de dentro da sessão (rede restrita). Confira se estão no ar.
  - As descrições das variáveis em `REFERENCIA_BASE.md` foram resumidas pela IA a partir do dicionário oficial (`dicionario_PNS_microdados_2019_23062023.xlsx`). Confira a regra exata de discretização das colunas `Status_*`, que a IA não conseguiu ver no CSV.
