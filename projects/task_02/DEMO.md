# Инструкция по запуску и демонстрации

## Быстрый старт

### Требования

```bash
python3 --version   # Python 3.10+
pip install torch torchvision   # ~2 ГБ, скачивается один раз
```

Все команды выполняются из директории `projects/task_02/`.

---

## Сценарий 1: Базовая демонстрация (standalone)

Минимальная конфигурация: один сервер, два клиента.

### Шаг 1 — запустить сервер

```bash
# Терминал 1
python3 -m server.main --rounds 5 --min-clients 2
```

Ожидаемый вывод:
```
2026-05-25 12:00:00 [INFO] server.main: default task created: mnist_mlp_a1b2c3
2026-05-25 12:00:00 [INFO] server.server: FL server listening on 0.0.0.0:9000
```

Запишите `task_id` из строки `default task created:` — он понадобится клиентам.

### Шаг 2 — узнать task_id (если не записали)

```bash
# Терминал 2 — запустить клиент без --task-id, получить список
python3 -m client.main --servers localhost:9000
```

Вывод:
```
2026-05-25 12:00:05 [INFO] client.main: available tasks:
2026-05-25 12:00:05 [INFO] client.main:   mnist_mlp_a1b2c3  [round 0/5, min_clients=2]
2026-05-25 12:00:05 [INFO] client.main: re-run with --task-id <id> to participate
```

### Шаг 3 — запустить двух клиентов

```bash
# Терминал 2 — Клиент 0
python3 -m client.main \
    --servers localhost:9000 \
    --task-id mnist_mlp_a1b2c3 \
    --client-id 0 \
    --num-clients 2
```

```bash
# Терминал 3 — Клиент 1
python3 -m client.main \
    --servers localhost:9000 \
    --task-id mnist_mlp_a1b2c3 \
    --client-id 1 \
    --num-clients 2
```

### Шаг 4 — наблюдать ход обучения

На сервере:
```
[INFO] server.server: new client from 127.0.0.1:54321
[INFO] server.server: new client from 127.0.0.1:54322
[INFO] task_manager: task mnist_mlp_a1b2c3 round 0: 1/2 updates received
[INFO] task_manager: task mnist_mlp_a1b2c3 round 0: 2/2 updates received
[INFO] task_manager: aggregating round 0 (2 clients)
[INFO] task_manager: round 0 complete, starting round 1
...
[INFO] task_manager: training complete after 5 rounds
```

На клиенте:
```
[INFO] client.main: === round 1/5 ===
[INFO] client.trainer: client 0: trained 1 epoch(s), 30000 samples, avg loss 0.8234
[INFO] client.main: weights accepted; waiting for other clients…
[INFO] client.main: round 1 complete (detected during poll).
[INFO] client.main: === round 2/5 ===
...
[INFO] client.main: training done after round 5. Exiting.
[INFO] client.main: client 0 finished; participated in 5 round(s).
```

---

## Сценарий 2: Несколько задач параллельно

Сервер поддерживает несколько задач одновременно.

```bash
# Терминал 1 — сервер уже запущен

# Терминал 2 — создать вторую задачу через первый клиент
# (нет команды создания задачи через CLI — сервер создаёт одну задачу при старте)
# Можно запустить второй сервер на другом порту:
python3 -m server.main --port 9001 --rounds 3 --min-clients 2
```

```bash
# Клиенты выбирают задачу через --task-id
python3 -m client.main --servers localhost:9001 \
    --task-id mnist_mlp_d4e5f6 --client-id 0 --num-clients 2

python3 -m client.main --servers localhost:9001 \
    --task-id mnist_mlp_d4e5f6 --client-id 1 --num-clients 2
```

---

## Сценарий 3: HA — демонстрация отказоустойчивости

Демонстрирует **Усложнение 3**: primary падает в середине обучения, backup принимает управление, клиенты продолжают без изменений.

### Шаг 1 — запустить primary

```bash
# Терминал 1
python3 -m server.main \
    --role primary \
    --port 9000 \
    --backup-repl-addr localhost:10100 \
    --rounds 10 \
    --min-clients 2
```

```
[INFO] server.main: default task created: mnist_mlp_a1b2c3
[INFO] replication: connected to backup localhost:10100  ← появится после шага 2
[INFO] server.server: FL server listening on 0.0.0.0:9000
```

### Шаг 2 — запустить backup

```bash
# Терминал 2
python3 -m server.main \
    --role backup \
    --port 9100 \
    --repl-port 10100
```

```
[INFO] server.main: backup mode: waiting for primary at repl-port 10100
[INFO] replication: BackupReplicationListener on port 10100 (timeout=15s)
[INFO] backup: replication connection from primary 127.0.0.1:XXXXX
[INFO] backup: state snapshot applied    ← через ~1 сек после старта primary
```

### Шаг 3 — запустить клиентов с двумя адресами

```bash
# Терминал 3 — Клиент 0
python3 -m client.main \
    --servers localhost:9000 localhost:9100 \
    --task-id mnist_mlp_a1b2c3 \
    --client-id 0 \
    --num-clients 2
```

```bash
# Терминал 4 — Клиент 1
python3 -m client.main \
    --servers localhost:9000 localhost:9100 \
    --task-id mnist_mlp_a1b2c3 \
    --client-id 1 \
    --num-clients 2
```

### Шаг 4 — убить primary в процессе обучения

После того как прошло 2-3 раунда, в терминале 1:

```
Ctrl+C
```

```
[INFO] server.main: server shutting down
```

### Шаг 5 — наблюдать промоцию backup

В терминале 2 (backup) через 15 секунд:

```
2026-05-25 12:05:15 [WARNING] backup: no contact from primary for 15.3s — promoting to primary
2026-05-25 12:05:15 [INFO] server.main: *** BACKUP PROMOTED TO PRIMARY — starting client server on port 9100 ***
2026-05-25 12:05:15 [INFO] server.server: FL server listening on 0.0.0.0:9100
```

### Шаг 6 — клиенты переподключаются автоматически

В терминале 3 (Клиент 0):

```
[WARNING] client.main: cannot reach localhost:9000 — [Errno 111] Connection refused
[INFO] client.main: connected to server localhost:9100   ← переключился на backup
[INFO] client.main: === round 4/10 ===
[INFO] client.trainer: client 0: trained 1 epoch(s), 30000 samples, avg loss 0.4521
...
```

Обучение продолжается с раунда 4 — backup имеет актуальное состояние (последний реплицированный снимок был после раунда 3).

---

## Сценарий 4: Демонстрация независимости от архитектуры (Усложнение 2)

Смысл: клиент не знает архитектуру модели — он получает её с сервера.

### Проверка вручную

```bash
# Запустить клиент с verbose-логами
python3 -c "
import torch, io, sys
sys.path.insert(0, '.')
from protocol.transport import send_msg, recv_msg
from protocol.messages import MsgType
import socket

sock = socket.socket()
sock.connect(('localhost', 9000))

# Присоединиться к задаче
send_msg(sock, MsgType.JOIN, {'task_id': 'mnist_mlp_a1b2c3'})
mt, data = recv_msg(sock)

arch_bytes = data['arch_bytes']
print(f'arch_bytes size: {len(arch_bytes):,} bytes')

# Загрузить модель БЕЗ импорта класса MNISTMLP
model = torch.jit.load(io.BytesIO(arch_bytes))
print('Model type:', type(model))
print('Parameters:', [(n, p.shape) for n, p in model.named_parameters()])
sock.close()
"
```

Ожидаемый вывод:
```
arch_bytes size: 832,456 bytes
Model type: <class 'torch.jit._script.RecursiveScriptModule'>
Parameters: [
  ('fc1.weight', torch.Size([256, 784])),
  ('fc1.bias',   torch.Size([256])),
  ('fc2.weight', torch.Size([10, 256])),
  ('fc2.bias',   torch.Size([10]))
]
```

Модель загружена, параметры доступны, класс `MNISTMLP` не импортировался.

---

## Полезные команды

### Просмотр справки

```bash
python3 -m server.main --help
python3 -m client.main --help
```

### Изменить число раундов и клиентов

```bash
python3 -m server.main --rounds 3 --min-clients 3 --hidden 128
```

### Участвовать только в N раундах

```bash
python3 -m client.main --servers localhost:9000 \
    --task-id <id> --client-id 0 --num-clients 2 --rounds 2
```

### Изменить гиперпараметры клиента

```bash
python3 -m client.main --servers localhost:9000 \
    --task-id <id> --client-id 0 --num-clients 2 \
    --epochs 3 \    # 3 эпохи локального обучения вместо 1
    --lr 0.001       # меньший learning rate
```

### Включить DEBUG-логи (видеть каждое сообщение протокола)

```bash
python3 -c "
import logging
logging.basicConfig(level=logging.DEBUG)
" 2>&1   # или установить уровень DEBUG в basicConfig внутри main.py
```

---

## Ожидаемые метрики обучения

На MNIST с 2 клиентами, 5 раундов, 1 эпоха локально, lr=0.01:

| Раунд | Примерный avg loss |
|-------|-------------------|
| 1     | 0.80 – 1.20       |
| 2     | 0.50 – 0.70       |
| 3     | 0.35 – 0.50       |
| 4     | 0.25 – 0.40       |
| 5     | 0.20 – 0.35       |

Точность на тесте после 5 раундов: ~90-93%.

---

## Возможные проблемы

| Проблема | Причина | Решение |
|----------|---------|---------|
| `ModuleNotFoundError: No module named 'torch'` | PyTorch не установлен | `pip install torch torchvision` |
| `Address already in use` | Порт занят | Подождать ~60 сек или сменить порт через `--port` |
| `Connection refused localhost:9000` | Сервер не запущен | Запустить `server.main` первым |
| Клиент зависает после `weights accepted` | Второй клиент не запущен | Запустить второй клиент с тем же `--task-id` |
| Backup не промотируется | Первичный старт backup слишком медленный | Подождать ещё 15 сек |
| `MNIST download` долго | Первая загрузка датасета | Подождать, скачивается один раз в `~/.fl_mnist_data` |
