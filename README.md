## Sobre o que é esse projeto?

O projeto tem em vista continuar o trabalho de Jéssica Costa com a predição da essencialidade de proteínas com a utilização de Machine Learning

## Estrutura principal

### Pasta Features de Contexto

Essa pasta contém todos os dados do cálculo das features de contexto dos organismos além de conter um arquivo chamado JuntarFeatures.py

### Pasta Features de Sequência

As features de sequência são calculadas no arquivo SequenceFeatures.py e Emboss.py utilizando como entrada os proteomas completos dos organismos modelos e organismo alvo

### Pasta Todas_Features

Contém os arquivos de cada organismo com as features de sequência e contexto

### STRING

Dados do banco de dados STRING para o cálculo de features

### Classificador

Possui as pastas com os resultados de cada fase, o CalculoFeatures.pyé o código responsável por calcular as features de contexto.
O arquivo EDAFeatures.py é responsável por fazer o processo de visualização de dados, treinamento dos modelos e cálculo dos resultados
