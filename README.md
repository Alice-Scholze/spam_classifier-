# Naive_Bayes: Classificador de spam

Curso de Especialização de Inteligência Artificial Aplicada

Setor de Educação Profissional e Tecnológica - SEPT

Universidade Federal do Paraná - UFPR

---

**IAA003 - Linguagem de Programação Aplicada**

Prof. Alexander Robert Kutzke

# Exercício de implementação do algoritmo Naive Bayes

Altere o código do arquivo [spam_classifier.py](spam_classifier.py) para adicionar
algumas das seguintes funcionalidades:

- Utilizar a biblioteca NumPy se considerar pertinente;
- Utilizar a biblioteca Pandas se considerar pertinente;
- Analisar o conteúdo da mensagem e não apenas o Assunto;
- Considerar apenas palavras que aparecem um número mínimo de vezes 
  (`min_count`);
- Utilizar apenas radicais das palavras (pesquise por "Porter Stemmer");
- Considerar não apenas presença de palavras, mas outras características:
  - Por exemplo, se a mensagem possuí números:
    - A função `tokenizer` pode retornar *tokens* especiais para isso (por exemplo: 
      `contains:number`).

**Comente seu código indicando as alterações realizadas.**

Você pode, ainda, realizar testes de desempenho para cada uma das alterações realizadas (se for pertinente).
