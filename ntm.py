import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class NTMController(nn.Module):
    """
    Controller-Komponente des Neural Turing Machine.
    Verarbeitet die Eingabe und die gelesenen Speichervektoren.
    """
    def __init__(self, input_size, hidden_size, output_size, memory_word_size, num_heads, shift_range=3):
        super(NTMController, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.memory_word_size = memory_word_size
        self.num_heads = num_heads
        self.shift_range = shift_range
        
        # Controller-Netzwerk (LSTM)
        self.lstm = nn.LSTM(
            input_size=input_size + memory_word_size * num_heads,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # Berechne die Anzahl der Parameter für die Köpfe
        read_head_size = memory_word_size + 1 + 1 + shift_range + 1
        write_head_size = memory_word_size + 1 + 1 + shift_range + 1 + memory_word_size * 2
        total_head_params = num_heads * (read_head_size + write_head_size)
        
        # Ausgang für die NTM-Köpfe
        self.head_params = nn.Linear(hidden_size, total_head_params)
        
        # Ausgang für die Ausgabe
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, prev_state, read_vectors):
        """
        x: Eingabe mit Form [batch_size, seq_len, input_size]
        prev_state: Vorherige LSTM-Zustände (h, c)
        read_vectors: Liste von Lesevektoren von jedem Lesekopf [batch_size, seq_len, word_size]
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Konkateniere die Eingabe mit allen Lesevektoren
        if isinstance(read_vectors, list):
            read_vectors_tensor = torch.cat(read_vectors, dim=-1)
        else:
            read_vectors_tensor = read_vectors
            
        combined = torch.cat([x, read_vectors_tensor], dim=-1)
        
        # LSTM-Verarbeitung
        lstm_out, state = self.lstm(combined, prev_state)
        
        # Parameter für die Lese-/Schreibköpfe
        head_params = self.head_params(lstm_out)
        
        # Ausgabe
        output = self.output(lstm_out)
        
        return output, head_params, state

class NTMMemory(nn.Module):
    """
    Speichermodul des NTM.
    Speichert und erlaubt Operationen auf der Speichermatrix.
    """
    def __init__(self, memory_size, word_size):
        super(NTMMemory, self).__init__()
        self.memory_size = memory_size
        self.word_size = word_size
        
        # Register memory as a buffer (not a parameter)
        self.register_buffer('memory', torch.zeros(memory_size, word_size))
        self.reset_memory()
    
    def reset_memory(self):
        """Speicher initialisieren"""
        nn.init.kaiming_uniform_(self.memory)
        # Normalisiere Speicherzellen für stabile Kosinus-Ähnlichkeit
        self.memory = F.normalize(self.memory, p=2, dim=1)
    
    def size(self):
        return self.memory_size, self.word_size
    
    def read(self, weights):
        """
        Lesen aus dem Speicher mit Aufmerksamkeitsgewichten
        weights: [batch_size, memory_size]
        """
        # Anwenden der Aufmerksamkeitsgewichte auf die Speicherzellen
        # weights: [batch_size, memory_size]
        # memory: [memory_size, word_size]
        # return: [batch_size, word_size]
        return torch.matmul(weights, self.memory)
    
    def write(self, weights, erase_vector, add_vector):
        """
        Schreiben in den Speicher mit Gewichten, Löschen und Hinzufügen
        weights: [batch_size, memory_size]
        erase_vector: [batch_size, word_size]
        add_vector: [batch_size, word_size]
        """
        # Batch-Größe
        batch_size = weights.size(0)
        
        # Sicherstellen, dass alle Eingaben die richtigen Dimensionen haben
        weights = weights.view(batch_size, self.memory_size, 1)
        erase_vector = erase_vector.view(batch_size, 1, self.word_size)
        add_vector = add_vector.view(batch_size, 1, self.word_size)
        
        # Erzeugen der Lösch- und Additionsmasken für jede Batch-Element
        erase_matrix = torch.matmul(weights, erase_vector)  # [batch_size, memory_size, word_size]
        add_matrix = torch.matmul(weights, add_vector)      # [batch_size, memory_size, word_size]
        
        # Speicher für jedes Batch-Element aktualisieren
        # Wir nehmen den durchschnitt über alle Batch-Elemente für den Speicher
        memory_update = (1 - erase_matrix) * self.memory.unsqueeze(0) + add_matrix
        self.memory = memory_update.mean(dim=0)
        
        # Normalisieren der Speicherzellen für stabile Kosinus-Ähnlichkeit
        self.memory = F.normalize(self.memory, p=2, dim=1)

class NTMHead(nn.Module):
    """
    Basisklasse für Lese- und Schreibköpfe
    """
    def __init__(self, memory_size, word_size):
        super(NTMHead, self).__init__()
        self.memory_size = memory_size
        self.word_size = word_size
    
    def get_attention(self, k, beta, g, s, gamma, prev_weights, memory):
        """
        Adressierungsvorgang mit Inhaltsaufmerksamkeit und Positionsverschiebung
        k: Schlüsselvektor für Inhaltsadressierung [batch_size, word_size]
        beta: Schärfeparameter für Inhaltsadressierung [batch_size, 1]
        g: Interpolationsgate zwischen Inhalt und vorherigen Gewichten [batch_size, 1]
        s: Verschiebungsgewichtung [batch_size, shift_range]
        gamma: Schärfeparameter für Gewichte nach Verschiebung [batch_size, 1]
        prev_weights: Vorherige Aufmerksamkeitsgewichte [batch_size, memory_size]
        memory: Speichermodul
        """
        batch_size = k.size(0)
        
        # Inhaltsadressierung mit Kosinus-Ähnlichkeit
        # Normalisiere k für eine stabile Berechnung
        k = F.normalize(k, p=2, dim=1)
        
        # Berechne die Kosinus-Ähnlichkeit zwischen k und jedem Speicherwort
        # Expandieren und Umgestalten von k und Speicher für Batch-Operationen
        mem = memory.memory.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, memory_size, word_size]
        k = k.unsqueeze(1)  # [batch_size, 1, word_size]
        
        # Berechne die Kosinus-Ähnlichkeit
        content_similarity = F.cosine_similarity(k, mem, dim=2)  # [batch_size, memory_size]
        
        # Anwenden des Schärfeparameters beta
        content_weights = F.softmax(beta * content_similarity, dim=1)
        
        # Interpolation mit vorherigen Gewichten
        gated_weights = g * content_weights + (1 - g) * prev_weights
        
        # Zirkuläre Faltung für die Verschiebung
        # Wir verwenden eine vereinfachte Version mit einer festen Verschiebung
        # Hinweis: Eine vollständige Implementierung würde eine komplexere Faltungsoperation verwenden
        shift_weights = self._shift(gated_weights, s)
        
        # Anwenden des Schärfeparameters gamma
        sharpened_weights = shift_weights ** gamma
        
        # Normalisieren der Gewichte, um eine Verteilung zu erhalten
        attention_weights = sharpened_weights / (torch.sum(sharpened_weights, dim=1, keepdim=True) + 1e-8)
        
        return attention_weights
    
    def _shift(self, weights, shift):
        """
        Vereinfachte Verschiebungsfunktion
        Diese Funktion könnte durch eine vollständige zirkuläre Faltung ersetzt werden
        """
        # Hier nehmen wir an, dass shift eine Verschiebung um -1, 0, oder 1 darstellt
        shifted_weights = torch.zeros_like(weights)
        
        # Konvertieren zu normalisierten Gewichten
        shift = F.softmax(shift, dim=1)
        
        # Vereinfachte Shift-Implementierung:
        # Verschiebung nach links
        shifted_weights[:, 1:] += weights[:, :-1] * shift[:, 0].unsqueeze(1)
        # Keine Verschiebung
        shifted_weights += weights * shift[:, 1].unsqueeze(1)
        # Verschiebung nach rechts
        shifted_weights[:, :-1] += weights[:, 1:] * shift[:, 2].unsqueeze(1)
        
        return shifted_weights

class NTMReadHead(NTMHead):
    """
    Lesekopf für den Neural Turing Machine
    """
    def __init__(self, memory_size, word_size, shift_range=3):
        super(NTMReadHead, self).__init__(memory_size, word_size)
        self.shift_range = shift_range
    
    def forward(self, head_params, prev_weights, memory):
        """
        Lesen aus dem Speicher mit den gegebenen Parametern
        head_params: Parameter für den Lesekopf
        prev_weights: Vorherige Aufmerksamkeitsgewichte
        memory: Speichermodul
        """
        # Extrahiere Parameter aus dem Kopfausgang
        k = head_params[:, :self.word_size]
        beta = F.softplus(head_params[:, self.word_size:self.word_size+1])
        g = torch.sigmoid(head_params[:, self.word_size+1:self.word_size+2])
        s = F.softmax(head_params[:, self.word_size+2:self.word_size+2+self.shift_range], dim=1)
        gamma = 1.0 + F.softplus(head_params[:, self.word_size+2+self.shift_range:self.word_size+2+self.shift_range+1])
        
        # Berechne Aufmerksamkeitsgewichte
        weights = self.get_attention(k, beta, g, s, gamma, prev_weights, memory)
        
        # Lesen aus dem Speicher
        read_vector = memory.read(weights)
        
        return read_vector, weights

class NTMWriteHead(NTMHead):
    """
    Schreibkopf für den Neural Turing Machine
    """
    def __init__(self, memory_size, word_size, shift_range=3):
        super(NTMWriteHead, self).__init__(memory_size, word_size)
        self.shift_range = shift_range
    
    def forward(self, head_params, prev_weights, memory):
        """
        Schreiben in den Speicher mit den gegebenen Parametern
        head_params: Parameter für den Schreibkopf
        prev_weights: Vorherige Aufmerksamkeitsgewichte
        memory: Speichermodul
        """
        # Extrahiere Parameter aus dem Kopfausgang
        # Beachte, dass Schreibköpfe zusätzliche Parameter für das Löschen und Hinzufügen haben
        param_idx = 0
        
        # Schlüssel für Inhaltsadressierung
        k = head_params[:, param_idx:param_idx+self.word_size]
        param_idx += self.word_size
        
        # Schärfeparameter für Inhaltsadressierung
        beta = F.softplus(head_params[:, param_idx:param_idx+1])
        param_idx += 1
        
        # Interpolationsgate
        g = torch.sigmoid(head_params[:, param_idx:param_idx+1])
        param_idx += 1
        
        # Verschiebungsgewichtung
        s = F.softmax(head_params[:, param_idx:param_idx+self.shift_range], dim=1)
        param_idx += self.shift_range
        
        # Schärfeparameter nach Verschiebung
        gamma = 1.0 + F.softplus(head_params[:, param_idx:param_idx+1])
        param_idx += 1
        
        # Löschvektor
        erase = torch.sigmoid(head_params[:, param_idx:param_idx+self.word_size])
        param_idx += self.word_size
        
        # Additionsvektor
        add = torch.tanh(head_params[:, param_idx:param_idx+self.word_size])
        
        # Berechne Aufmerksamkeitsgewichte
        weights = self.get_attention(k, beta, g, s, gamma, prev_weights, memory)
        
        # Schreiben in den Speicher
        memory.write(weights, erase, add)
        
        return weights

class NeuralTuringMachine(nn.Module):
    """
    Neural Turing Machine Hauptklasse
    Kombiniert Controller, Speicher und Köpfe
    """
    def __init__(self, input_size, hidden_size, output_size, memory_size, word_size, num_heads=1, shift_range=3):
        super(NeuralTuringMachine, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_heads = num_heads
        self.shift_range = shift_range
        
        # Controller-Netzwerk
        self.controller = NTMController(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            memory_word_size=word_size,
            num_heads=num_heads,
            shift_range=shift_range
        )
        
        # Speicher
        self.memory = NTMMemory(memory_size, word_size)
        
        # Lese- und Schreibköpfe
        self.read_heads = nn.ModuleList([
            NTMReadHead(memory_size, word_size, shift_range) for _ in range(num_heads)
        ])
        self.write_heads = nn.ModuleList([
            NTMWriteHead(memory_size, word_size, shift_range) for _ in range(num_heads)
        ])
    
    def _split_head_params(self, head_params):
        """
        Teile die Kopfparameter für Lese- und Schreibköpfe auf
        head_params: [batch_size, seq_len, total_head_params]
        """
        # Parameter pro Kopftyp
        read_head_size = self.word_size + 1 + 1 + self.shift_range + 1
        write_head_size = self.word_size + 1 + 1 + self.shift_range + 1 + self.word_size * 2
        
        # Gesamtanzahl der Parameter
        total_param_size = read_head_size * self.num_heads + write_head_size * self.num_heads
        
        # Überprüfe, ob die Parameter-Dimensionen übereinstimmen
        if head_params.size(2) != total_param_size:
            raise ValueError(f"Expected head_params to have {total_param_size} features, got {head_params.size(2)}")
        
        # Aufteilen für individuelle Köpfe
        read_head_params = []
        write_head_params = []
        
        param_idx = 0
        
        # Lese-Köpfe
        for i in range(self.num_heads):
            start_idx = param_idx
            end_idx = start_idx + read_head_size
            read_head_params.append(head_params[:, :, start_idx:end_idx])
            param_idx = end_idx
        
        # Schreib-Köpfe
        for i in range(self.num_heads):
            start_idx = param_idx
            end_idx = start_idx + write_head_size
            write_head_params.append(head_params[:, :, start_idx:end_idx])
            param_idx = end_idx
        
        return read_head_params, write_head_params
    
    def reset(self, batch_size=1):
        """
        Zurücksetzen des Zustands für eine neue Sequenz
        """
        # Zurücksetzen des Speichers
        self.memory.reset_memory()
        
        # Zurücksetzen der LSTM-Zustände
        self.hidden = (
            torch.zeros(1, batch_size, self.hidden_size).to(next(self.parameters()).device),
            torch.zeros(1, batch_size, self.hidden_size).to(next(self.parameters()).device)
        )
        
        # Initialisierung der Kopfgewichte
        # Für die erste Zeitschritt, fokussieren wir auf die erste Speicherzelle
        self.read_weights = [
            F.one_hot(torch.zeros(batch_size, dtype=torch.long), self.memory_size).float().to(next(self.parameters()).device)
            for _ in range(self.num_heads)
        ]
        
        self.write_weights = [
            F.one_hot(torch.zeros(batch_size, dtype=torch.long), self.memory_size).float().to(next(self.parameters()).device)
            for _ in range(self.num_heads)
        ]
        
        # Erzeuge Nullvektoren als initiale Lesevektoren
        # Dies ist wichtig, da wir diese für den ersten Zeitschritt benötigen
        self.read_vectors = [
            torch.zeros(batch_size, 1, self.word_size).to(next(self.parameters()).device)
            for _ in range(self.num_heads)
        ]
    
    def forward(self, x, reset_state=True):
        """
        Vorwärtsdurchlauf für den NTM
        x: Eingabesequenz [batch_size, seq_len, input_size]
        reset_state: Ob der Zustand zurückgesetzt werden soll (für neue Sequenzen)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Zurücksetzen des Zustands für neue Sequenzen
        if reset_state:
            self.reset(batch_size)
        
        # Output für jede Zeitschritt
        outputs = []
        
        # Durchlaufen jeder Zeitschritt in der Sequenz
        for t in range(seq_len):
            # Eingabe für diesen Zeitschritt
            x_t = x[:, t:t+1, :]  # [batch_size, 1, input_size]
            
            # Konkatenierte Lesevektoren für den Controller
            # Expandieren für seq_len=1
            read_vectors_t = torch.cat([rv.expand(-1, 1, -1) for rv in self.read_vectors], dim=-1)
            
            # Durchlaufen des Controllers
            output_t, head_params_t, self.hidden = self.controller(x_t, self.hidden, read_vectors_t)
            
            # Aufteilen der Kopfparameter
            read_head_params, write_head_params = self._split_head_params(head_params_t)
            
            # Zuerst schreiben, dann lesen
            for i, head in enumerate(self.write_heads):
                # Schreiben in den Speicher
                self.write_weights[i] = head(write_head_params[i].squeeze(1), self.write_weights[i], self.memory)
            
            # Dann lesen
            for i, head in enumerate(self.read_heads):
                # Lesen aus dem Speicher
                read_vector, self.read_weights[i] = head(read_head_params[i].squeeze(1), self.read_weights[i], self.memory)
                self.read_vectors[i] = read_vector.unsqueeze(1)
            
            # Sammle die Ausgaben
            outputs.append(output_t)
        
        # Konkateniere alle Ausgaben
        return torch.cat(outputs, dim=1)

# Konfiguration für das Modell
class Config:
    def __init__(self):
        self.input_size = 10
        self.hidden_size = 100
        self.output_size = 10
        self.memory_size = 128
        self.word_size = 20
        self.num_heads = 1
        self.shift_range = 3
        self.batch_size = 10
        self.seq_length = 20
        self.num_epochs = 100
        self.learning_rate = 1e-3

# Beispiel für die Verwendung des NTM
def example_usage():
    config = Config()
    
    # Instanziiere das Modell
    ntm = NeuralTuringMachine(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        memory_size=config.memory_size,
        word_size=config.word_size,
        num_heads=config.num_heads,
        shift_range=config.shift_range
    )
    
    # Beispiel-Eingabe
    x = torch.randn(config.batch_size, config.seq_length, config.input_size)
    
    # Optimierer
    optimizer = optim.Adam(ntm.parameters(), lr=config.learning_rate)
    
    # Forward pass
    output = ntm(x)
    print(f"Output shape: {output.shape}")
    
    # Beispiel-Ziel
    target = torch.randn(config.batch_size, config.seq_length, config.output_size)
    
    # Verlustfunktion
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    
    # Backward pass und Optimierung
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item()}")

# Testproblem: Copy Task
def generate_copy_task(batch_size, seq_length, vector_size):
    # Eingabesequenz
    sequence = torch.rand(batch_size, seq_length, vector_size)
    
    # Flag für das Ende der Eingabe (für den Modelleingang)
    end_flag = torch.zeros(batch_size, 1, vector_size + 1)
    end_flag[:, :, -1] = 1  # Setze das letzte Bit als Endflag
    
    # Eingabe: Sequenz + Endflag, gepaddet mit Nullen für die erwartete Ausgabezeit
    input_sequence = torch.cat([
        torch.cat([sequence, torch.zeros(batch_size, seq_length, 1)], dim=2),
        end_flag,
        torch.zeros(batch_size, seq_length, vector_size + 1)
    ], dim=1)
    
    # Ausgabe: Gepaddet mit Nullen für die Eingabezeit + Endflag, dann die zu kopierende Sequenz
    target_sequence = torch.cat([
        torch.zeros(batch_size, seq_length + 1, vector_size),
        sequence
    ], dim=1)
    
    return input_sequence, target_sequence

# Trainingsschleife für die Copy Task
def train_copy_task():
    config = Config()
    
    # Anpassung der Konfiguration für die Copy Task
    config.input_size = 11  # 10 + 1 für das Endflag
    config.output_size = 10
    
    # Instanziiere das Modell
    ntm = NeuralTuringMachine(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        memory_size=config.memory_size,
        word_size=config.word_size,
        num_heads=config.num_heads,
        shift_range=config.shift_range
    )
    
    # Optimierer
    optimizer = optim.Adam(ntm.parameters(), lr=config.learning_rate)
    
    # Verlustfunktion
    criterion = nn.BCEWithLogitsLoss()
    
    # Trainingsschleife
    for epoch in range(config.num_epochs):
        # Generiere Daten für diese Epoche
        inputs, targets = generate_copy_task(
            batch_size=config.batch_size,
            seq_length=config.seq_length,
            vector_size=config.output_size
        )
        
        # Forward pass
        outputs = ntm(inputs)
        
        # Berechne Verlust (nur für die Ausgabezeit)
        loss = criterion(
            outputs[:, config.seq_length+1:, :],
            targets[:, config.seq_length+1:, :]
        )
        
        # Backward pass und Optimierung
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping um Explodieren zu vermeiden
        torch.nn.utils.clip_grad_value_(ntm.parameters(), 10)
        optimizer.step()
        
        # Gib den Fortschritt aus
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {loss.item():.4f}")
    
    return ntm

if __name__ == "__main__":
    example_usage()
    # Auskommentieren für die Copy Task
    # train_copy_task()
