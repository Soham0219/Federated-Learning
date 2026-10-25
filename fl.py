"""
Federated Learning for Next Word Prediction using LSTM.
Implements FedAvg with optional FedProx and Differential Privacy.
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ─── Reproducibility ────────────────────────────────────────────────────────
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ─── Model ──────────────────────────────────────────────────────────────────
def build_model(vocab_size, seq_len, embed_dim=32, lstm_units=64, lr=0.005):
    model = tf.keras.Sequential([
        Embedding(vocab_size, embed_dim, input_shape=(seq_len,)),
        LSTM(lstm_units),
        Dropout(0.2),
        Dense(vocab_size, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─── Data ───────────────────────────────────────────────────────────────────
def make_sequences(sentences, tokenizer, seq_len):
    """Convert sentences into (X, y) pairs for training."""
    X, y = [], []
    for s in sentences:
        tokens = tokenizer.texts_to_sequences([s])[0]
        for i in range(seq_len, len(tokens)):
            X.append(tokens[i - seq_len : i])
            y.append(tokens[i])
    if not X:
        return np.zeros((0, seq_len)), np.array([])
    return np.array(X), np.array(y)


def load_train_data():
    """100 sentences with strong repeating patterns for FL training."""
    return [
        # "I am going to the ___" pattern (10)
        "I am going to the store to buy some groceries",
        "I am going to the park to play with my friends",
        "I am going to the office to finish my work today",
        "I am going to the market to buy fresh vegetables",
        "I am going to the gym to exercise every morning",
        "I am going to the library to study for the exam",
        "I am going to the hospital to visit my friend",
        "I am going to the school to pick up my kids",
        "I am going to the beach to relax this weekend",
        "I am going to the cinema to watch a new movie",

        # "she is ___ in the ___" pattern (10)
        "she is reading a book in the library after class",
        "she is reading a novel about science and nature",
        "she is cooking dinner in the kitchen for the family",
        "she is playing the piano in the evening at home",
        "she is writing a letter to her best friend today",
        "she is studying in the library for her final exam",
        "she is working in the office on a new project",
        "she is teaching the students in the classroom today",
        "she is running in the park every morning for exercise",
        "she is painting a picture in her room this afternoon",

        # "he is ___ in the ___" pattern (10)
        "he is running in the park every single morning",
        "he is working on a new project at the office",
        "he is painting the walls of his new house today",
        "he is studying for his final exam at the library",
        "he is building a small house in the backyard today",
        "he is reading a book in the library after work",
        "he is playing soccer in the park with his friends",
        "he is cooking dinner in the kitchen for his family",
        "he is swimming in the pool every morning for exercise",
        "he is writing a story about his travels abroad",

        # "they are ___ in the ___" pattern (10)
        "they are planning a vacation to the mountains next week",
        "they are building a new shopping mall in the city",
        "they are playing soccer in the yard after school today",
        "they are watching a movie at the cinema tonight together",
        "they are discussing the project in the meeting room",
        "they are studying in the library for their exams today",
        "they are cooking dinner in the kitchen for the party",
        "they are working on a new project at the office",
        "they are planning a trip to the beach next month",
        "they are playing music in the park this afternoon",

        # "we are ___ in the ___" pattern (10)
        "we are preparing dinner for our friends this evening",
        "we are hiking in the forest this weekend with family",
        "we are watching a movie tonight at home together",
        "we are organizing a charity event next weekend afternoon",
        "we are planning a trip to the beach next month",
        "we are studying in the library for the final exam",
        "we are working on a new project at the office",
        "we are playing soccer in the park after school",
        "we are cooking dinner in the kitchen for the family",
        "we are building a new house in the city this year",

        # "the ___ is ___ in the ___" pattern (20)
        "the cat is sleeping on the couch in the living room",
        "the dog is barking at the strangers outside the gate",
        "the dog is running in the park every single morning",
        "the cat is playing with a ball in the living room",
        "the teacher is explaining the math problem to the students",
        "the teacher is teaching the students in the classroom today",
        "the chef is preparing a special meal for the guests",
        "the chef is cooking dinner in the kitchen for the party",
        "the team is preparing for the upcoming tournament this week",
        "the team is working on a new project at the office",
        "the children are playing soccer in the yard after school",
        "the children are studying in the library for their exams",
        "the doctor is helping the patients at the hospital today",
        "the doctor is working in the hospital every single day",
        "the students are studying in the library for their exams",
        "the students are working on a new project this week",
        "the musician is playing a new song for the audience",
        "the musician is practicing in the studio every morning",
        "the scientist is working on a new project in the lab",
        "the scientist is studying the data in the office today",

        # More "in the ___" / "at the ___" / "for the ___" (20)
        "I am reading a book in the library this afternoon",
        "I am working on a project at the office today",
        "I am playing with my friends in the park after school",
        "I am cooking dinner in the kitchen for the family",
        "I am studying for the exam at the library today",
        "she is preparing for the meeting at the office",
        "she is playing with her friends in the park today",
        "she is cooking a meal in the kitchen for the party",
        "she is studying for the exam in the library today",
        "she is helping the students in the classroom after school",
        "he is preparing for the meeting at the office today",
        "he is playing with his friends in the park after work",
        "he is cooking a meal in the kitchen for his family",
        "he is studying for the exam in the library this week",
        "he is helping the students in the classroom after school",
        "they are preparing for the meeting at the office today",
        "they are playing with their friends in the park today",
        "they are cooking a meal in the kitchen for the party",
        "we are preparing for the meeting at the office today",
        "we are playing with our friends in the park after work",
    ]


def load_test_data():
    """Test sentences using same vocabulary patterns as training data."""
    return [
        "I am going to the park to relax after work",
        "I am going to the store to buy some food",
        "she is reading a book in the library today",
        "she is cooking dinner in the kitchen tonight",
        "he is running in the park every morning",
        "he is working on a project at the office",
        "they are playing soccer in the yard today",
        "they are studying in the library for exams",
        "we are preparing dinner for our family tonight",
        "we are planning a trip to the park this week",
        "the cat is sleeping on the couch today",
        "the dog is running in the park this morning",
        "the teacher is helping the students after class",
        "the students are studying in the library today",
        "the chef is cooking dinner in the kitchen tonight",
    ]


# ─── Federated Server ──────────────────────────────────────────────────────
class FederatedServer:
    def __init__(self, vocab_size, seq_len, lr):
        self.model = build_model(vocab_size, seq_len, lr=lr)

    def get_weights(self):
        return self.model.get_weights()

    def aggregate(self, client_weights_list, sample_sizes, noise_sigma=0.0):
        """Weighted FedAvg aggregation. Optional DP noise."""
        total = sum(sample_sizes)
        new_w = [np.zeros_like(w) for w in client_weights_list[0]]
        for weights, n in zip(client_weights_list, sample_sizes):
            for j, w in enumerate(weights):
                new_w[j] += (n / total) * w
        # differential privacy noise
        if noise_sigma > 0:
            for j in range(len(new_w)):
                new_w[j] += np.random.normal(0, noise_sigma, size=new_w[j].shape)
        self.model.set_weights(new_w)

    def evaluate(self, X, y):
        if len(X) == 0:
            return 999.0, 0.0
        loss, acc = self.model.evaluate(X, y, verbose=0)
        return float(loss), float(acc)


# ─── Federated Client ──────────────────────────────────────────────────────
class FederatedClient:
    def __init__(self, sentences, vocab_size, seq_len, lr):
        self.sentences = sentences
        self.model = build_model(vocab_size, seq_len, lr=lr)

    def train(self, global_weights, tokenizer, seq_len, epochs=1, batch_size=16,
              mu=0.0):
        """
        Local training. If mu > 0, uses FedProx proximal term.
        """
        self.model.set_weights(global_weights)
        X, y = make_sequences(self.sentences, tokenizer, seq_len)
        if len(X) == 0:
            return self.model.get_weights(), 0

        if mu > 0:
            # FedProx: custom loop with proximal penalty
            dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(len(X)).batch(batch_size)
            optimizer = self.model.optimizer
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
            global_tensors = [tf.constant(w) for w in global_weights]

            for _ in range(epochs):
                for xb, yb in dataset:
                    with tf.GradientTape() as tape:
                        preds = self.model(xb, training=True)
                        ce = loss_fn(yb, preds)
                        prox = tf.add_n([
                            tf.reduce_sum(tf.square(w - gw))
                            for w, gw in zip(self.model.trainable_weights, global_tensors)
                            if w.shape == gw.shape
                        ])
                        total_loss = ce + (mu / 2.0) * prox
                    grads = tape.gradient(total_loss, self.model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        else:
            # Standard FedAvg
            self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

        return self.model.get_weights(), len(X)


# ─── Prediction ─────────────────────────────────────────────────────────────
def predict_next(model, tokenizer, text, seq_len, top_k=3):
    """Return top-k predicted next words with probabilities."""
    seq = tokenizer.texts_to_sequences([text])[0]
    if len(seq) > seq_len:
        seq = seq[-seq_len:]
    seq = np.pad(seq, (seq_len - len(seq), 0), "constant").reshape(1, seq_len)
    probs = model.predict(seq, verbose=0)[0]
    idxs = np.argsort(probs)[-top_k:][::-1]
    return [(tokenizer.index_word.get(int(i), "?"), float(probs[i])) for i in idxs]


def generate_text(model, tokenizer, seed, seq_len, n_words=6):
    """Auto-regressively generate n_words using greedy decoding."""
    text = seed
    for _ in range(n_words):
        seq = tokenizer.texts_to_sequences([text])[0]
        if len(seq) > seq_len:
            seq = seq[-seq_len:]
        seq = np.pad(seq, (seq_len - len(seq), 0), "constant").reshape(1, seq_len)
        probs = model.predict(seq, verbose=0)[0]
        idx = int(np.argmax(probs))
        text += " " + tokenizer.index_word.get(idx, "?")
    return text


# ─── Main FL Loop ───────────────────────────────────────────────────────────
def federated_learning(
    num_clients=3,
    num_rounds=30,
    seq_len=3,
    local_epochs=1,
    lr=0.005,
    mu=0.0,              # FedProx coefficient (0 = standard FedAvg)
    noise_sigma=0.0,     # DP noise (0 = disabled)
):
    set_seeds(42)

    train_data = load_train_data()
    test_data = load_test_data()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data + test_data)
    vocab_size = len(tokenizer.word_index) + 1

    # Split data across clients
    random.shuffle(train_data)
    splits = np.array_split(train_data, num_clients)
    clients = [FederatedClient(list(s), vocab_size, seq_len, lr) for s in splits]

    server = FederatedServer(vocab_size, seq_len, lr)

    # Pre-compute test set
    X_test, y_test = make_sequences(test_data, tokenizer, seq_len)

    algo = "FedProx" if mu > 0 else "FedAvg"
    print("=" * 60)
    print("  Federated Learning — Next Word Prediction")
    print("=" * 60)
    print(f"  Algorithm : {algo}" + (f" (μ={mu})" if mu > 0 else ""))
    print(f"  Clients   : {num_clients}  |  Rounds: {num_rounds}")
    print(f"  Seq len   : {seq_len}  |  Local epochs: {local_epochs}")
    print(f"  Vocab     : {vocab_size}  |  LR: {lr}")
    print(f"  Train     : {len(train_data)} sentences  |  Test: {len(test_data)}")
    if noise_sigma > 0:
        print(f"  DP noise  : σ={noise_sigma}")
    print("=" * 60)

    best_loss = float("inf")
    patience = 0

    for r in range(1, num_rounds + 1):
        all_w, all_n = [], []
        for c in clients:
            w, n = c.train(server.get_weights(), tokenizer, seq_len, local_epochs, mu=mu)
            all_w.append(w)
            all_n.append(n)

        server.aggregate(all_w, all_n, noise_sigma)
        loss, acc = server.evaluate(X_test, y_test)
        ppl = float(np.exp(min(loss, 20)))
        print(f"  Round {r:>3}/{num_rounds}  |  loss {loss:.4f}  |  acc {acc:.4f}  |  ppl {ppl:.1f}")

        if loss < best_loss - 1e-4:
            best_loss = loss
            patience = 0
        else:
            patience += 1
        if patience >= 10:
            print(f"\n  Early stopping at round {r}")
            break

    # ─── Final results ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    loss, acc = server.evaluate(X_test, y_test)
    print(f"  Final  |  loss {loss:.4f}  |  acc {acc:.4f}  |  ppl {np.exp(min(loss,20)):.1f}")

    print("\n  Next-word predictions:")
    for prompt in ["I am going", "she is", "he is", "they are", "the cat is"]:
        preds = predict_next(server.model, tokenizer, prompt, seq_len)
        s = ", ".join(f"{w} ({p:.1%})" for w, p in preds)
        print(f'    "{prompt}" → {s}')

    print("\n  Generated text:")
    for prompt in ["I am going", "she is", "they are"]:
        print(f'    "{generate_text(server.model, tokenizer, prompt, seq_len)}"')

    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    federated_learning(
        num_clients=3,
        num_rounds=50,
        seq_len=3,
        local_epochs=1,
        lr=0.005,
        # mu=0.01,         # uncomment to enable FedProx
        # noise_sigma=0.01 # uncomment to enable differential privacy
    )
"""
Federated Learning for Next Word Prediction using LSTM.

Implements FedAvg with optional FedProx regularization and differential privacy.
"""

import os
import pathlib
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Embedding,
    LayerNormalization,
)
from tensorflow.keras.preprocessing.text import Tokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FLConfig:
    """Central configuration for the federated learning experiment."""

    # Model
    embedding_dim: int = 64
    lstm_units: int = 128
    num_lstm_layers: int = 2
    dropout_rate: float = 0.3
    recurrent_dropout: float = 0.2

    # Training
    sequence_length: int = 4
    batch_size: int = 16
    local_epochs: int = 5
    learning_rate: float = 0.001
    clip_norm: float = 1.0          # gradient clipping

    # Federated
    num_clients: int = 5
    num_rounds: int = 30
    fraction_clients: float = 0.6
    mu: float = 0.0                 # FedProx proximal term (0 = standard FedAvg)

    # Differential privacy (set noise_multiplier > 0 to enable)
    noise_multiplier: float = 0.0
    l2_norm_clip: float = 1.0

    # Data
    data_source: str = "builtin"  # "builtin", "shakespeare", or a file path

    # Misc
    seed: int = 42
    top_k: int = 5
    temperature: float = 0.8


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class NextWordLSTM(tf.keras.Model):
    """Stacked LSTM model for next-word prediction."""

    def __init__(self, vocab_size: int, config: FLConfig):
        super().__init__()
        self.config = config

        self.embedding = Embedding(
            input_dim=vocab_size,
            output_dim=config.embedding_dim,
        )

        self.lstm_layers = []
        self.norm_layers = []
        self.dropout_layers = []
        for i in range(config.num_lstm_layers):
            return_seq = i < config.num_lstm_layers - 1
            self.lstm_layers.append(
                tf.keras.layers.LSTM(
                    config.lstm_units,
                    return_sequences=return_seq,
                    dropout=config.dropout_rate,
                    recurrent_dropout=config.recurrent_dropout,
                )
            )
            self.norm_layers.append(LayerNormalization())
            self.dropout_layers.append(Dropout(config.dropout_rate))

        self.fc = Dense(config.lstm_units, activation="relu")
        self.fc_norm = LayerNormalization()
        self.fc_drop = Dropout(config.dropout_rate)
        self.output_layer = Dense(vocab_size, activation="softmax")

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        for lstm, norm, drop in zip(self.lstm_layers, self.norm_layers, self.dropout_layers):
            x = lstm(x, training=training)
            x = norm(x)
            x = drop(x, training=training)
        x = self.fc(x)
        x = self.fc_norm(x)
        x = self.fc_drop(x, training=training)
        return self.output_layer(x)


def build_model(vocab_size: int, config: FLConfig) -> NextWordLSTM:
    """Build, compile, and initialize the model."""
    model = NextWordLSTM(vocab_size, config)
    dummy = np.zeros((1, config.sequence_length))
    model(dummy, training=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.learning_rate,
            clipnorm=config.clip_norm,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Data utilities
# ──────────────────────────────────────────────────────────────────────────────

def create_sequences(
    text_list: list[str],
    tokenizer: Tokenizer,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a list of sentences into (input_sequences, target_words)."""
    all_inputs, all_targets = [], []
    for sentence in text_list:
        tokens = tokenizer.texts_to_sequences([sentence])[0]
        for i in range(sequence_length, len(tokens)):
            all_inputs.append(tokens[i - sequence_length : i])
            all_targets.append(tokens[i])
    if not all_inputs:
        return np.array([]).reshape(0, sequence_length), np.array([])
    return np.array(all_inputs), np.array(all_targets)


def load_data_from_file(path: str) -> list[str]:
    """Load sentences from a plain-text file (one sentence per line)."""
    text = pathlib.Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def download_shakespeare() -> list[str]:
    """Download a subset of Shakespeare's text as a real-world corpus."""
    url = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
    path = tf.keras.utils.get_file("shakespeare.txt", url)
    raw = pathlib.Path(path).read_text(encoding="utf-8")
    # split into sentences on periods and newlines, keep reasonable lengths
    sentences = []
    for line in raw.splitlines():
        line = line.strip()
        if len(line.split()) >= 4:
            sentences.append(line)
    return sentences[:500]  # use first 500 lines to keep training feasible


def load_data(source: str = "builtin") -> list[str]:
    """
    Training corpus — each element is one sentence.

    Args:
        source: "builtin"      – large built-in corpus (200+ sentences)
                "shakespeare"  – download Shakespeare subset (~500 lines)
                <file path>    – load from a text file (one sentence per line)
    """
    if source == "shakespeare":
        return download_shakespeare()
    if source != "builtin" and pathlib.Path(source).is_file():
        return load_data_from_file(source)

    return [
        # ── Daily activities ─────────────────────────────────────────────
        "I am going to the store to buy some groceries.",
        "I am going to the park to relax after a long day.",
        "I am going to the office for work this morning.",
        "I am walking to the bus stop to catch the early bus.",
        "I am visiting the museum to see the new exhibit.",
        "She is reading a book in the library during her break.",
        "He is running in the park every morning before sunrise.",
        "They are planning a vacation to the mountains next summer.",
        "We are preparing dinner for our friends tonight.",
        "The cat is sleeping on the couch in the living room.",
        "He is fixing his car in the garage on the weekend.",
        "She is attending a yoga class in the evening to stay fit.",
        "The children are playing soccer in the yard after school.",
        "I am learning to play the guitar for the first time.",
        "We are watching a movie tonight at the cinema downtown.",
        "He is painting the walls of his new house this week.",
        "She is baking a cake for the birthday party tomorrow.",
        "I am traveling to Japan next month for a conference.",
        "He is writing a novel about his adventures abroad.",
        "She is gardening in her backyard every weekend in spring.",
        "The dog is barking at the strangers outside the gate.",
        "We are hiking in the forest this weekend with friends.",
        "He is swimming in the pool every morning for exercise.",

        # ── Work & education ─────────────────────────────────────────────
        "The teacher is explaining the math problem to the students clearly.",
        "The team is preparing for the upcoming tournament this Friday.",
        "They are launching a new app next year in the market.",
        "She is learning French for her trip to Paris next spring.",
        "He is practicing his speech for the conference on Tuesday.",
        "We are organizing a charity event next weekend at the hall.",
        "He is designing a new logo for the company website.",
        "They are discussing the project timeline in the meeting room.",
        "She is hosting a dinner party for her colleagues after work.",
        "He is training for a marathon next year in Boston.",
        "The scientist is researching a cure for the disease in the lab.",
        "The professor is teaching a course on machine learning this semester.",
        "Students are studying for their final exams at the library.",
        "The engineer is developing a new algorithm for data processing.",
        "She is presenting her research findings at the international symposium.",
        "The manager is reviewing the quarterly budget for the department.",
        "He is mentoring new employees during their first week at work.",
        "They are collaborating on a research paper about climate change.",
        "The intern is learning about software development best practices.",
        "She is completing her thesis on natural language processing.",
        "The team is deploying the new software update to production servers.",
        "He is debugging a critical issue in the payment system.",
        "The architect is designing a sustainable building for the campus.",
        "She is conducting interviews for the open engineering position.",
        "We are attending a workshop on artificial intelligence this afternoon.",
        "The company is hiring new developers for the mobile team.",
        "He is writing documentation for the new application programming interface.",
        "She is analyzing the data from the latest customer survey.",
        "The director is planning the next phase of the project carefully.",
        "They are setting up the new server infrastructure for the cloud.",

        # ── Technology ───────────────────────────────────────────────────
        "The new smartphone has an impressive camera with optical zoom.",
        "Artificial intelligence is transforming the healthcare industry rapidly.",
        "The software update includes several important security patches.",
        "Cloud computing enables businesses to scale their operations efficiently.",
        "Machine learning models require large amounts of training data.",
        "The robot is assembling car parts on the factory floor.",
        "Virtual reality headsets are becoming more affordable for consumers.",
        "The database needs to be optimized for faster query performance.",
        "Cybersecurity threats are increasing in both frequency and complexity.",
        "The algorithm processes millions of data points in seconds.",
        "Natural language processing helps computers understand human speech.",
        "The network infrastructure supports thousands of simultaneous connections.",
        "Blockchain technology ensures secure and transparent transactions online.",
        "The autonomous vehicle navigates through city traffic using sensors.",
        "Deep learning has achieved remarkable results in image recognition tasks.",
        "The application uses encryption to protect sensitive user information.",
        "Quantum computing promises to solve currently intractable problems.",
        "The startup is developing a platform for online education courses.",
        "Wearable technology tracks health metrics like heart rate and steps.",
        "The system automatically detects and flags suspicious login attempts.",

        # ── Science & nature ─────────────────────────────────────────────
        "The sun rises in the east and sets in the west.",
        "Water boils at one hundred degrees Celsius at sea level.",
        "The earth revolves around the sun once every year.",
        "Photosynthesis allows plants to convert sunlight into chemical energy.",
        "The rainforest is home to millions of different species.",
        "Climate change is causing glaciers to melt at alarming rates.",
        "The ocean covers more than seventy percent of the earth surface.",
        "Gravity keeps the moon in orbit around the earth.",
        "The atmosphere protects life on earth from harmful solar radiation.",
        "Volcanoes release gases and molten rock from deep underground.",
        "Coral reefs support a diverse ecosystem of marine life.",
        "The human brain contains approximately one hundred billion neurons.",
        "Fossils provide evidence of life that existed millions of years ago.",
        "The migration of birds follows seasonal patterns across continents.",
        "Renewable energy sources include solar wind and hydroelectric power.",
        "The telescope revealed distant galaxies billions of light years away.",
        "Evolution by natural selection drives the diversity of life on earth.",
        "The periodic table organizes chemical elements by atomic number.",
        "DNA carries the genetic instructions for growth and development.",
        "Earthquakes occur when tectonic plates shift beneath the surface.",

        # ── Health & lifestyle ───────────────────────────────────────────
        "Regular exercise improves both physical and mental health significantly.",
        "A balanced diet includes fruits vegetables proteins and whole grains.",
        "Getting enough sleep is essential for cognitive function and memory.",
        "Meditation helps reduce stress and improve emotional well being.",
        "Drinking plenty of water keeps the body hydrated and healthy.",
        "Vaccinations protect communities from the spread of infectious diseases.",
        "Walking thirty minutes a day can lower the risk of heart disease.",
        "Reading before bed helps the mind relax and improves sleep quality.",
        "Spending time in nature reduces anxiety and boosts your mood.",
        "Cooking meals at home is healthier than eating out frequently.",
        "Stretching before exercise helps prevent injuries and muscle soreness.",
        "Limiting screen time in the evening can improve your sleep cycle.",
        "Volunteering in the community gives a sense of purpose and belonging.",
        "Learning a musical instrument strengthens memory and concentration skills.",
        "Maintaining social connections is important for long term mental health.",

        # ── Travel & places ──────────────────────────────────────────────
        "The Eiffel Tower is one of the most visited landmarks in the world.",
        "Tokyo is known for its vibrant culture and delicious cuisine.",
        "The Grand Canyon attracts millions of visitors every single year.",
        "Venice is famous for its canals and beautiful architecture.",
        "The Great Wall of China stretches thousands of kilometers across the land.",
        "Sydney has stunning beaches and a world famous opera house.",
        "The Northern Lights are best seen in countries near the Arctic Circle.",
        "Machu Picchu sits high in the Andes mountains of Peru.",
        "New York City is often called the city that never sleeps.",
        "The Sahara Desert is the largest hot desert in the world.",
        "Iceland is known for its geysers glaciers and volcanic landscapes.",
        "The Amazon River flows through the largest tropical rainforest on earth.",
        "Rome is home to ancient ruins and rich historical heritage.",
        "The Maldives offers crystal clear waters and stunning coral reefs.",
        "Kyoto has beautiful temples and traditional Japanese gardens.",

        # ── Food & cooking ───────────────────────────────────────────────
        "She is making fresh pasta from scratch for the family dinner.",
        "The baker is preparing sourdough bread with organic whole wheat flour.",
        "He is grilling vegetables and chicken for a healthy summer barbecue.",
        "The restaurant serves authentic Italian pizza baked in a wood oven.",
        "She is blending fresh fruits to make a nutritious morning smoothie.",
        "The chef is seasoning the steak with salt pepper and rosemary.",
        "They are trying a new recipe for homemade chocolate chip cookies.",
        "He is brewing fresh coffee beans from a local roastery.",
        "The market sells organic produce from farms in the countryside.",
        "She is decorating cupcakes with colorful frosting for the school bake sale.",

        # ── Sports & entertainment ───────────────────────────────────────
        "The team won the championship after an exciting final match.",
        "She is practicing her ballet routine for the upcoming performance.",
        "He is learning to surf at the beach during summer vacation.",
        "The concert was attended by thousands of enthusiastic music fans.",
        "They are playing chess in the park on a sunny afternoon.",
        "The marathon runners crossed the finish line after many hours.",
        "She is training with her coach for the swimming competition.",
        "The film festival showcases independent movies from around the world.",
        "He is collecting vintage vinyl records from the local music store.",
        "The basketball game ended with a dramatic last second three pointer.",

        # ── Relationships & social ───────────────────────────────────────
        "She is calling her parents every weekend to catch up with them.",
        "He is writing a heartfelt letter to his best friend abroad.",
        "They are celebrating their wedding anniversary at a fancy restaurant.",
        "We are meeting old friends for brunch at the new cafe.",
        "She is helping her neighbor carry groceries to the apartment.",
        "He is teaching his younger sister how to ride a bicycle.",
        "They are throwing a surprise birthday party for their colleague.",
        "We are planning a family reunion for the holidays this year.",
        "She is volunteering at the local animal shelter every Saturday.",
        "He is coaching the youth soccer team on weekday evenings.",

        # ── Home & DIY ───────────────────────────────────────────────────
        "I am renovating my kitchen with new cabinets and countertops.",
        "She is organizing the closet to make more storage space.",
        "He is building custom shelves for the living room wall.",
        "They are painting the bedroom a calm shade of blue.",
        "We are installing solar panels on the roof of our house.",
        "She is rearranging the furniture to create a better layout.",
        "He is planting a vegetable garden in the backyard this spring.",
        "The plumber is fixing the leaking pipe in the bathroom.",
        "She is sewing curtains from fabric she bought at the craft store.",
        "He is assembling a new desk for his home office.",

        # ── Weather & seasons ────────────────────────────────────────────
        "The weather forecast predicts heavy rain for the rest of the week.",
        "Snow is covering the mountain peaks in a beautiful white blanket.",
        "The autumn leaves are turning shades of red orange and gold.",
        "Spring brings warmer temperatures and blooming flowers to the garden.",
        "The hot summer sun makes the beach a popular destination.",
        "A strong wind is blowing through the trees in the valley.",
        "The fog is rolling in from the coast this early morning.",
        "Winter nights are long and perfect for reading by the fireplace.",
        "The thunderstorm knocked out power in several neighborhoods last night.",
        "A rainbow appeared in the sky after the afternoon rain shower.",
    ]


def load_test_data() -> list[str]:
    """Held-out test sentences."""
    return [
        "The dog is running in the park with its owner.",
        "She is writing an article for the magazine about technology.",
        "They are rehearsing for their school play after classes.",
        "I am reading a book about artificial intelligence and robotics.",
        "He is building a treehouse in the backyard for the kids.",
        "The students are working on a group project for science class.",
        "We are planning a weekend trip to the countryside.",
        "She is learning to cook traditional dishes from her grandmother.",
        "The company is launching a new product next quarter.",
        "He is jogging along the river trail every evening after work.",
        "They are decorating the office for the holiday celebration.",
        "I am writing an email to the client about the proposal.",
        "The doctor is explaining the treatment options to the patient.",
        "She is practicing piano for two hours every single day.",
        "The weather is expected to be sunny and warm this weekend.",
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Federated Server
# ──────────────────────────────────────────────────────────────────────────────

class FederatedServer:
    """Central server that aggregates client updates (FedAvg / FedProx)."""

    def __init__(self, vocab_size: int, config: FLConfig):
        self.config = config
        self.vocab_size = vocab_size
        self.global_model = build_model(vocab_size, config)
        self.best_loss = float("inf")
        self.rounds_without_improvement = 0

    # ── aggregation ──────────────────────────────────────────────────────
    def aggregate(
        self,
        client_weights_list: list[list[np.ndarray]],
        sample_sizes: list[int],
    ) -> None:
        """Weighted FedAvg aggregation with optional DP noise."""
        total = sum(sample_sizes)
        new_weights = [np.zeros_like(w) for w in client_weights_list[0]]

        for weights, n in zip(client_weights_list, sample_sizes):
            fraction = n / total
            for j, w in enumerate(weights):
                new_weights[j] += fraction * w

        # optional differential privacy: add Gaussian noise
        if self.config.noise_multiplier > 0:
            for j in range(len(new_weights)):
                noise = np.random.normal(
                    0,
                    self.config.noise_multiplier * self.config.l2_norm_clip,
                    size=new_weights[j].shape,
                )
                new_weights[j] += noise / total

        self.global_model.set_weights(new_weights)

    # ── communication ────────────────────────────────────────────────────
    def broadcast_weights(self) -> list[np.ndarray]:
        return self.global_model.get_weights()

    # ── evaluation ───────────────────────────────────────────────────────
    def evaluate(
        self,
        tokenizer: Tokenizer,
        test_sentences: list[str],
    ) -> dict:
        """Evaluate global model; returns loss, accuracy, perplexity."""
        X, y = create_sequences(test_sentences, tokenizer, self.config.sequence_length)
        if len(X) == 0:
            return {"loss": float("inf"), "accuracy": 0.0, "perplexity": float("inf")}
        loss, acc = self.global_model.evaluate(X, y, verbose=0)
        perplexity = float(np.exp(loss))
        return {"loss": loss, "accuracy": acc, "perplexity": perplexity}

    def check_early_stop(self, loss: float, patience: int = 10) -> bool:
        """Track convergence; returns True when training should stop."""
        if loss < self.best_loss - 1e-4:
            self.best_loss = loss
            self.rounds_without_improvement = 0
        else:
            self.rounds_without_improvement += 1
        return self.rounds_without_improvement >= patience


# ──────────────────────────────────────────────────────────────────────────────
# Federated Client
# ──────────────────────────────────────────────────────────────────────────────

class FederatedClient:
    """A single FL participant that trains on local data."""

    def __init__(
        self,
        sentences: list[str],
        vocab_size: int,
        config: FLConfig,
    ):
        self.sentences = sentences
        self.config = config
        self.model = build_model(vocab_size, config)

    @property
    def num_samples(self) -> int:
        """Total number of training tokens across all local sentences."""
        return sum(len(s.split()) for s in self.sentences)

    def train(
        self,
        global_weights: list[np.ndarray],
        tokenizer: Tokenizer,
    ) -> list[np.ndarray]:
        """
        Perform local training rounds and return updated weights.

        Supports FedProx via a proximal term that penalises divergence
        from the global model (controlled by config.mu).
        """
        self.model.set_weights(global_weights)

        X, y = create_sequences(self.sentences, tokenizer, self.config.sequence_length)
        if len(X) == 0:
            return self.model.get_weights()

        if self.config.mu > 0:
            # FedProx: custom training loop with proximal term
            self._train_fedprox(X, y, global_weights)
        else:
            # Standard FedAvg local training
            self.model.fit(
                X, y,
                epochs=self.config.local_epochs,
                batch_size=self.config.batch_size,
                verbose=0,
            )

        return self.model.get_weights()

    def _train_fedprox(
        self,
        X: np.ndarray,
        y: np.ndarray,
        global_weights: list[np.ndarray],
    ) -> None:
        """Custom training loop with FedProx proximal regularisation."""
        dataset = (
            tf.data.Dataset.from_tensor_slices((X, y))
            .shuffle(len(X))
            .batch(self.config.batch_size)
        )
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.learning_rate,
            clipnorm=self.config.clip_norm,
        )
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        for _ in range(self.config.local_epochs):
            for x_batch, y_batch in dataset:
                with tf.GradientTape() as tape:
                    preds = self.model(x_batch, training=True)
                    ce_loss = loss_fn(y_batch, preds)
                    # proximal term
                    prox = sum(
                        tf.reduce_sum(tf.square(w - gw))
                        for w, gw in zip(self.model.trainable_weights, global_weights)
                        if w.shape == gw.shape
                    )
                    total_loss = ce_loss + (self.config.mu / 2.0) * prox

                grads = tape.gradient(total_loss, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))


# ──────────────────────────────────────────────────────────────────────────────
# Prediction utilities
# ──────────────────────────────────────────────────────────────────────────────

def predict_next_word(
    model: NextWordLSTM,
    tokenizer: Tokenizer,
    text: str,
    config: FLConfig,
) -> str:
    """Predict the single most likely next word."""
    probs = _get_next_word_probs(model, tokenizer, text, config)
    idx = int(np.argmax(probs))
    return tokenizer.index_word.get(idx, "<unknown>")


def predict_top_k(
    model: NextWordLSTM,
    tokenizer: Tokenizer,
    text: str,
    config: FLConfig,
) -> list[tuple[str, float]]:
    """Return top-k next-word candidates with their probabilities."""
    probs = _get_next_word_probs(model, tokenizer, text, config)
    top_indices = np.argsort(probs)[-config.top_k :][::-1]
    return [
        (tokenizer.index_word.get(int(i), "<unknown>"), float(probs[i]))
        for i in top_indices
    ]


def generate_text(
    model: NextWordLSTM,
    tokenizer: Tokenizer,
    seed_text: str,
    num_words: int,
    config: FLConfig,
) -> str:
    """Auto-regressively generate `num_words` words using temperature sampling."""
    current = seed_text
    for _ in range(num_words):
        probs = _get_next_word_probs(model, tokenizer, current, config)

        # temperature scaling
        probs = np.log(probs + 1e-10) / config.temperature
        probs = np.exp(probs) / np.sum(np.exp(probs))

        idx = np.random.choice(len(probs), p=probs)
        word = tokenizer.index_word.get(idx, "<unknown>")
        current += " " + word
    return current


def _get_next_word_probs(
    model: NextWordLSTM,
    tokenizer: Tokenizer,
    text: str,
    config: FLConfig,
) -> np.ndarray:
    """Tokenise, pad/truncate, and return softmax probabilities."""
    seq = tokenizer.texts_to_sequences([text])[0]
    if len(seq) > config.sequence_length:
        seq = seq[-config.sequence_length :]
    seq = np.pad(seq, (config.sequence_length - len(seq), 0), "constant")
    seq = seq.reshape(1, config.sequence_length)
    return model.predict(seq, verbose=0)[0]


# ──────────────────────────────────────────────────────────────────────────────
# Federated Learning loop
# ──────────────────────────────────────────────────────────────────────────────

def federated_learning(config: Optional[FLConfig] = None) -> None:
    """Run the full FL experiment."""
    if config is None:
        config = FLConfig()
    set_seeds(config.seed)

    # ── data ─────────────────────────────────────────────────────────────
    train_data = load_data(source=config.data_source)
    test_data = load_test_data()

    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(train_data + test_data)
    vocab_size = len(tokenizer.word_index) + 1

    # partition training data across clients  (IID random split)
    random.shuffle(train_data)
    splits = np.array_split(train_data, config.num_clients)
    clients = [
        FederatedClient(list(split), vocab_size, config)
        for split in splits
    ]

    server = FederatedServer(vocab_size, config)

    # ── header ───────────────────────────────────────────────────────────
    print("=" * 65)
    print("  Federated Learning — Next Word Prediction (LSTM)")
    print("=" * 65)
    print(f"  Clients        : {config.num_clients}")
    print(f"  Rounds         : {config.num_rounds}")
    print(f"  Selection frac : {config.fraction_clients}")
    print(f"  Sequence len   : {config.sequence_length}")
    print(f"  LSTM layers    : {config.num_lstm_layers} × {config.lstm_units}")
    print(f"  Vocab size     : {vocab_size}")
    print(f"  Train sentences: {len(train_data)}")
    print(f"  Test sentences : {len(test_data)}")
    print(f"  Data source    : {config.data_source}")
    if config.mu > 0:
        print(f"  FedProx μ      : {config.mu}")
    if config.noise_multiplier > 0:
        print(f"  DP noise σ     : {config.noise_multiplier}")
    print("=" * 65)

    # ── training rounds ──────────────────────────────────────────────────
    for rnd in range(1, config.num_rounds + 1):
        # select a subset of clients
        k = max(1, int(config.fraction_clients * config.num_clients))
        selected = random.sample(clients, k)

        all_weights = []
        all_sizes = []
        for client in selected:
            weights = client.train(server.broadcast_weights(), tokenizer)
            all_weights.append(weights)
            all_sizes.append(client.num_samples)

        server.aggregate(all_weights, all_sizes)

        # evaluate
        metrics = server.evaluate(tokenizer, test_data)
        print(
            f"  Round {rnd:>3d}/{config.num_rounds}  |  "
            f"loss {metrics['loss']:.4f}  |  "
            f"acc {metrics['accuracy']:.4f}  |  "
            f"ppl {metrics['perplexity']:.2f}"
        )

        if server.check_early_stop(metrics["loss"]):
            print(f"\n  Early stopping at round {rnd} (no improvement for 7 rounds)")
            break

    # ── final evaluation & predictions ───────────────────────────────────
    print("\n" + "=" * 65)
    print("  Final Results")
    print("=" * 65)

    final = server.evaluate(tokenizer, test_data)
    print(f"  Test loss       : {final['loss']:.4f}")
    print(f"  Test accuracy   : {final['accuracy']:.4f}")
    print(f"  Test perplexity : {final['perplexity']:.2f}")

    prompts = ["I am going", "She is", "The dog is", "They are"]
    print("\n  Next-word predictions:")
    for prompt in prompts:
        top = predict_top_k(server.global_model, tokenizer, prompt, config)
        top_str = ", ".join(f"{w} ({p:.2%})" for w, p in top[:3])
        print(f"    \"{prompt}\"  →  {top_str}")

    print("\n  Generated text (temperature={:.1f}):".format(config.temperature))
    for prompt in ["I am going", "She is"]:
        gen = generate_text(server.global_model, tokenizer, prompt, 6, config)
        print(f"    \"{gen}\"")

    print("=" * 65)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = FLConfig(
        num_clients=3,           # fewer clients = more data per client
        num_rounds=50,           # more rounds to converge
        fraction_clients=1.0,    # use all clients every round for stability
        sequence_length=3,       # shorter context = more training samples
        local_epochs=1,          # SINGLE epoch = minimal client drift (critical for FL)
        num_lstm_layers=1,       # simpler model for small data
        lstm_units=64,           # smaller hidden size
        dropout_rate=0.1,        # low dropout — small data needs to learn, not regularize
        recurrent_dropout=0.0,   # no recurrent dropout — helps convergence
        learning_rate=0.001,     # conservative LR
        mu=0.01,                 # FedProx — prevents clients from drifting apart
        # noise_multiplier=0.1,  # uncomment for differential privacy
    )
    federated_learning(cfg)
