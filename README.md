# <img src="server/viewer/favicon.svg" alt="logo" width="128" height="128" align="middle"> SI2 - Space Invaders

A Space Invaders game implementation using the `ai-game-framework`.

## Features
- Real-time backend server.
- Web-based viewer with Canvas API.
- Dummy agent (random walker/shooter).
- Manual agent (terminal-based A/D/Space control).

## Setup & Running the Game

### 1. Prerequisites
- Python 3.10+ installed on your host.

### 2. Create and Activate Virtual Environment
Create a virtual environment (`venv`) to isolate dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install the required packages (this will install the local `ai-game-framework` package in editable mode and `numpy`):
```bash
pip install -r requirements.txt
```

### 4. Run the Game Server
Start the backend server (which also serves the frontend web viewer):
```bash
python3 -m server.server
```

### 5. Open the Viewer
Open your web browser and navigate to:
```
http://localhost:8765/
```

### 6. Run the Agents
In a separate terminal (ensure the virtual environment is activated):

- **Dummy Agent**:
  ```bash
  python3 -m agents.dummy_agent
  ```

- **Manual Agent (Terminal A/D/Space control)**:
  ```bash
  python3 -m agents.manual_agent
  ```

## Development
The project structure:
- `server/`: Game logic, server implementation, and visualizer assets (inside `server/viewer/`).
- `agents/`: Autonomous and manual agent implementations.
- `tests/`: Game unit tests.



# Relatório de Desenvolvimento: Agente Espacial (NEAT)

## 1. Arquitetura do Agente
O agente foi treinado utilizando o algoritmo **NEAT** (NeuroEvolution of Augmenting Topologies).
*   **Inputs (11):** Posição X do jogador, presença de laser, coordenadas (X, Y) do laser ativo, coordenadas do alien estático mais próximo, coordenadas do alien em mergulho (*dive*) mais próximo, densidade de aliens e o **Sensor de Cooldown da Arma** (essencial para evitar o bloqueio de movimento durante o disparo).
*   **Outputs (4):** Mover para Oeste, Mover para Este, Disparar, *Idle* (Ficar parado).
*   **Processamento:** Utilização de `ParallelEvaluator` com 3 processos em simultâneo para otimização do tempo de treino num CPU i7-1165G7.

## 2. Função Base de Recompensa (Fitness)
A função inicial foi desenhada para equilibrar sobrevivência e agressividade:
$$Fitness = (Score \times 5.0) + (Steps \times 0.1) - (VidasPerdidas \times 500)$$
*   **Score:** Incentiva a eliminação de aliens.
*   **Steps:** Recompensa a sobrevivência no tempo.
*   **Penalização de Morte:** Um peso negativo elevado para desencorajar comportamentos suicidas ou estáticos.

---

## 3. Histórico de Versões e Evolução

### Fase 0: Modelo Base (`winner_goated_behaviour.pkl`)
*   **Comportamento:** Modelo defensivo e estável.
*   **Destaque:** Aprendeu a "caçar" aliens que entram em modo *dive*, movendo-se para a sua coordenada X para interceptar o ataque.
*   **Problema Detetado:** O agente tendia a ficar parado no centro (*Camping*), pois os aliens acabavam sempre por passar pela sua linha de fogo.

### Fase 1: Introdução do Paradigma "Speedrun"
*   **Alteração:** Substituição da recompensa de tempo por uma penalização: $-(steps \times k)$.
*   **Objetivo:** Forçar o agente a limpar as *waves* o mais rápido possível para minimizar a perda de pontos por tempo decorrido. Foram testados dois coeficientes ($k=0.75$ e $k=1.0$).

### Fase 2: Expansão de Contexto e Modo Caçador
*   **Alterações:** 
    1. Aumento do limite de simulação para **6000 steps** (3 minutos) para testar a robustez contra o RNG.
    2. Introdução de bónus por alinhamento com aliens estáticos (Modo Caçador) quando não existem ameaças em *dive*.
*   **Resultado:** O agente tornou-se mais dinâmico, patrulhando a base para eliminar alvos antes de estes iniciarem o ataque.

### Fase 3: Reforço de Prioridades e o Problema do *Jitter*
*   **Alteração:** Aumento agressivo das recompensas de alinhamento ($0.4$ para alvos em *dive* e $1.0$ para alvos estáticos).
*   **Problema (Local Optimum):** O modelo de $k=0.75$ desenvolveu uma oscilação constante (*jitter*). Como a nave se move em coordenadas discretas (inteiros) e os aliens em coordenadas contínuas (floats), a IA tentava infinitamente ajustar a sua posição para um valor decimal impossível de atingir.

### Fase 4: Discretização de Alvos (Modelo Atual)
*   **Solução:** Implementação da função `round()` na posição X dos aliens para o cálculo de distância.
*   **Efeito:** Ao "discretizar" o alvo, o agente passou a considerar o alinhamento perfeito (distância = 0) assim que entra no mesmo bloco que o alien.
*   **Resultado Final:** Movimentos fluidos, eliminação do *jitter* e comportamento tático de alta performance, alternando perfeitamente entre defesa (evasão de *dives*) e ataque agressivo.

---

### Notas de Engenharia (Observações Extras)
*   **Sim2Real Gap:** A implementação de uma zona de segurança (Y < 4.0) onde o agente prioriza a evasão em vez do alinhamento foi crucial para mitigar a latência de rede (WebSockets) observada no servidor real.
*   **Mapa de Calor (Anti-Camping):** A imposição de uma penalização de 90% no fitness caso o agente passasse mais de 45% do tempo num único bloco X foi o "golpe final" que destruiu a tática de ficar parado no centro, forçando a exploração de todo o mapa.