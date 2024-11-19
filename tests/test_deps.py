import unittest


class TestDependencies(unittest.TestCase):
    def test_e2e(self):
        from tensorflow.keras import layers, models, datasets
        from tensorflow.keras.utils import to_categorical

        # Load MNIST dataset
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

        # Normalize the images to [0, 1] range
        train_images = train_images.astype('float32') / 255
        test_images = test_images.astype('float32') / 255

        # Flatten the images from 28x28 to 784
        train_images = train_images.reshape((train_images.shape[0], 28 * 28))
        test_images = test_images.reshape((test_images.shape[0], 28 * 28))

        # Convert labels to one-hot encoding
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)

        # Define the model architecture
        model = models.Sequential([
            layers.Dense(512, activation='relu', input_shape=(28 * 28,)),
            layers.Dropout(0.2),
            layers.Dense(10, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        model.fit(train_images, train_labels, epochs=10, batch_size=256, validation_split=0.1)

        # Evaluate the model on test data
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print(f'Test accuracy: {test_acc:.3f}')

        self.assertGreaterEqual(test_acc, 0.980)

    def test_pointer_networks(self):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import numpy as np

        class PointerNetwork(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super(PointerNetwork, self).__init__()
                self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
                self.pointer = nn.Linear(hidden_size, 1)  # Points to one of the encoder outputs

            def forward(self, x):
                batch_size = x.size(0)
                seq_len = x.size(1)

                encoder_out, (hidden, cell) = self.encoder(x)
                decoder_in = torch.zeros(batch_size, 1, encoder_out.size(2), device=x.device)  # Start token

                outputs = []
                for _ in range(seq_len):
                    _, (hidden, cell) = self.decoder(decoder_in, (hidden, cell))
                    query = hidden[-1].unsqueeze(1)  # [batch_size, 1, hidden_size]
                    logits = torch.bmm(query, encoder_out.transpose(1, 2)).squeeze(1)  # [batch_size, seq_len]
                    pointer = torch.softmax(logits, dim=1)
                    outputs.append(pointer.unsqueeze(1))

                    indices = pointer.argmax(1, keepdim=True)
                    decoder_in = torch.gather(encoder_out, 1, indices.unsqueeze(-1).repeat(1, 1, encoder_out.size(2)))

                return torch.cat(outputs, dim=1)

        def generate_data(batch_size, sequence_len, vector_dim):
            return torch.rand(batch_size, sequence_len, vector_dim)

        def loss_function(outputs, targets):
            batch_size = outputs.size(0)
            log_likelihood = torch.zeros(batch_size, device=outputs.device)
            for b in range(batch_size):
                # Gather the probabilities of the target sequence
                target_probs = outputs[b, torch.arange(outputs.size(1)), targets[b]]
                log_likelihood[b] = torch.log(target_probs).sum()  # Sum log probabilities
            return -log_likelihood.mean()

        model = PointerNetwork(input_size=2, hidden_size=256, num_layers=1)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        ### Step 3: Generate Sample Data

        # Assume 5 cities with 2D coordinates
        batch_size = 100
        sequence_len = 5  # Number of cities
        vector_dim = 2  # Dimension of coordinates

        data = generate_data(batch_size, sequence_len, vector_dim)
        targets_np = np.array([np.random.permutation(sequence_len) for _ in range(batch_size)])
        targets = torch.tensor(targets_np)

        ### Step 4: Train the Model

        num_epochs = 1000
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')

        def calculate_total_distance(tour, distances):
            total_distance = 0
            for i in range(len(tour) - 1):
                total_distance += distances[tour[i], tour[i + 1]]
            total_distance += distances[tour[-1], tour[0]]  # return to start
            return total_distance

        # Example usage
        distances = np.random.rand(5, 5)  # Random symmetric distance matrix
        tour = np.random.permutation(5)
        total_distance = calculate_total_distance(tour, distances)
        print("Total Distance of the tour:", total_distance)


if __name__ == '__main__':
    unittest.main()
