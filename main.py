import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe
from DataProcessor import DataProcessor
from intent_classifier_model import IntentModelArchitecture, IntentModelTrainer, IntentModelDataset, padding_fn

# once model is trained, set to True, allowing for model evaluations and inferences
model_is_saved = False
# set to True to compute evaluation metrics on saved/trained model to gain insights on performance of model
compute_eval_metrics = False

if __name__ == "__main__":

    # load json file to be parsed into different dataframes
    with open('data_full.json', 'r') as fp:
        data_full = json.load(fp)

    # combine all training, validation, and test datasets (using both oos and is prompts)
    for key in data_full.keys():
        if key == "oos_val":
            oos_val = pd.DataFrame(data_full[key], columns=['text', 'intent'])
            oos_val = oos_val.sample(n=20)
        elif key == "val":
            is_val = pd.DataFrame(data_full[key], columns=['text', 'intent'])
        elif key == "train":
            is_train = pd.DataFrame(data_full[key], columns=['text', 'intent'])
        elif key == "oos_test":
            oos_test = pd.DataFrame(data_full[key], columns=['text', 'intent'])
            oos_test = oos_test.sample(n=30)
        elif key == "test":
            is_test = pd.DataFrame(data_full[key], columns=['text', 'intent'])
        elif key == "oos_train":
            oos_train = pd.DataFrame(data_full[key], columns=['text', 'intent'])

    # concatenate dataframes for each training, validation, and testing dataset
    train_df = pd.concat([is_train, oos_train])
    validate_df = pd.concat([is_val, oos_val])
    test_df = pd.concat([is_test, oos_test])

    # preprocess training, validation, and testing data
    processed_train = DataProcessor(train_df)
    processed_validate = DataProcessor(validate_df)
    processed_test = DataProcessor(test_df)

    # clean, tokenize, remove stopwords, and lemmatize text of each dataset. Also, encode labels of each dataset
    processed_train.process_data()
    processed_validate.process_data()
    processed_test.process_data()

    # build and save vocabulary if training the model
    if not model_is_saved:
        # add each token to the vocabulary
        processed_train.build_vocab()
        # save vocabulary during training
        processed_train.save_vocab('vocab.json')

    # load vocabulary if evaluating or predicting with saved model
    if model_is_saved:
        # load vocabulary before loading the model
        processed_train.load_vocab("vocab.json")

    # instantiate CustomDataset objects to be loaded
    train_dataset = IntentModelDataset(processed_train.df, processed_train.vocab, processed_train.label_encoder)
    validate_dataset = IntentModelDataset(processed_validate.df, processed_train.vocab, processed_validate.label_encoder)
    test_dataset = IntentModelDataset(processed_test.df, processed_train.vocab, processed_test.label_encoder)

    # instantiate DataLoader objects to shuffle data to reduce bias, speed training process with batch size, padding, etc.
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=padding_fn)
    validate_loader = DataLoader(validate_dataset, batch_size=32, shuffle=True, collate_fn=padding_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=padding_fn)

    # define hyperparameters for IntentModelArchitecture object and instantiate pre-trained word embedding
    embedding_dim = 300
    hidden_dim = 500
    output_dim = 151  # 150 in scope intent class and 1 oos intent class
    glove = GloVe(name='6B', dim=embedding_dim)
    vocab_size = glove.vectors.shape[0]

    # create instance of IntentModelArchitecture
    model = IntentModelArchitecture(vocab_size, embedding_dim, hidden_dim, output_dim, glove.vectors)

    # define the loss function and optimization algo
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # learning rate scheduler that monitors the training process and dynamically adjusts the learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # start training process and find version of model with best validation accuracy through epochs, use patience value to find improvement
    if not model_is_saved:
        trainer = IntentModelTrainer(model, criterion, optimizer, label_encoder=processed_train.label_encoder,
                                     vocab=processed_train.vocab)
        epochs = 100  # set the maximum number of epochs
        early_stopping_patience = 10  # number of epochs to wait for improvement
        best_val_accuracy = 0.0
        epochs_since_last_improvement = 0

        for epoch in range(epochs):
            avg_loss, accuracy = trainer.train(train_loader)
            print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss}, Training Set Accuracy: {accuracy}')

            val_accuracy, val_precision, val_recall, val_f1, val_cm = trainer.validate(validate_loader)
            print(f'Validation Set Accuracy: {val_accuracy}')

            # check for improvement in validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_since_last_improvement = 0
                # save the model state when there is an improvement
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, 'best_intent_model.pth')
            else:
                epochs_since_last_improvement += 1

            # early stopping condition
            if epochs_since_last_improvement >= early_stopping_patience:
                print(f'No improvement in validation accuracy for {early_stopping_patience} epochs. Stopping early.')
                break

        print("Training complete.")

    # evaluate or predict text inputs using the intent classifier model that was saved after the training process above
    if model_is_saved:
        # load the checkpoint where the model had best validation accuracy
        loaded_checkpoint = torch.load('best_intent_model.pth')
        # instantiate the architecture class for the saved model
        loaded_model = IntentModelArchitecture(vocab_size, embedding_dim, hidden_dim, output_dim, glove.vectors)
        # load the state dictionary into the new model and load optimizer (i.e., further training or fine tuning)
        loaded_model.load_state_dict(loaded_checkpoint['model_state_dict'])
        loaded_optimizer = optim.Adam(loaded_model.parameters(), lr=0.005)
        loaded_optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
        # instantiate model trainer object with loaded model state
        trainer = IntentModelTrainer(loaded_model, criterion, loaded_optimizer, label_encoder=processed_train.label_encoder,
                                     vocab=processed_train.vocab)
        if compute_eval_metrics:
            # evaluate accuracy, precision, recall, f1_score, and confusion matrix of loaded model on validation dataset
            val_accuracy, val_precision, val_recall, val_f1, val_cm = trainer.validate(validate_loader)
            print(f'Validation Set Metrics; Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}, F1 Score: {val_f1}')
            # evaluate accuracy, precision, recall, f1_score, and confusion matrix of loaded model on test dataset
            test_accuracy, test_precision, test_recall, test_f1, test_cm = trainer.validate(test_loader)
            print(f'Test Set Metrics; Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1 Score: {test_f1}')
            # write the confusion matrices to text files
            with open("validation_confusion_matrix.txt", "w") as file:
                file.write("Validation Confusion Matrix:\n")
                file.write("\n".join(" ".join(map(str, row.astype(int))) for row in val_cm))

            with open("test_confusion_matrix.txt", "w") as file:
                file.write("Test Confusion Matrix:\n")
                file.write("\n".join(" ".join(map(str, row.astype(int))) for row in test_cm))

        # test a single text input to observe intent prediction
        print(trainer.predict("hi, how are you?"))