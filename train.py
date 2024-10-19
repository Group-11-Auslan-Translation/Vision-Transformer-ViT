import torch
from sklearn.metrics import accuracy_score, classification_report


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Trains the given model for a specified number of epochs and evaluates on the validation set after each epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): The loss function (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): The optimizer used for model parameter updates.
        num_epochs (int): Number of epochs for training.
        device (torch.device): The device (CPU or GPU) to perform computations.

    Returns:
        tuple: A tuple containing:
            - train_losses (list): List of average training losses for each epoch.
            - val_accuracies (list): List of validation accuracies for each epoch.
            - val_precisions (list): List of validation precisions for each epoch.
            - val_recalls (list): List of validation recalls for each epoch.
            - val_f1_scores (list): List of validation F1-scores for each epoch.
    """
    best_val_acc = 0.0  # Track the best validation accuracy
    train_losses = []  # Store training losses per epoch
    val_accuracies = []  # Store validation accuracies per epoch
    val_precisions = []  # Store validation precisions per epoch
    val_recalls = []  # Store validation recalls per epoch
    val_f1_scores = []  # Store validation F1-scores per epoch

    print(f"Using device: {device}")

    # Training loop over epochs
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0  # Accumulate training loss
        all_train_preds = []  # Store all predictions during training
        all_train_labels = []  # Store all true labels during training

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Starting Training...")

        # Loop over batches in the training data
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)  # Move data to device

            optimizer.zero_grad()  # Zero the gradients before the backward pass
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass (compute gradients)
            optimizer.step()  # Update model parameters

            running_loss += loss.item()  # Accumulate the loss

            _, preds = torch.max(outputs, 1)  # Get the predicted class
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

            # Calculate accuracy for the current batch
            batch_train_acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())

            # Print progress of the batch
            print(
                f"Training - Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {batch_train_acc:.4f}, Progress: {100 * (batch_idx + 1) / len(train_loader):.2f}%")

        avg_train_loss = running_loss / len(train_loader)  # Average loss for the epoch
        train_acc = accuracy_score(all_train_labels, all_train_preds)  # Accuracy for the entire epoch
        train_losses.append(avg_train_loss)  # Store the loss

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - Training Complete - Avg Loss: {avg_train_loss:.4f}, Avg Accuracy: {train_acc:.4f}")

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_preds = []  # Store validation predictions
        val_labels = []  # Store validation labels
        running_val_loss = 0.0  # Accumulate validation loss

        with torch.no_grad():  # Disable gradient calculations during evaluation
            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)  # Forward pass for validation
                loss = criterion(outputs, labels)  # Compute validation loss
                running_val_loss += loss.item()  # Accumulate validation loss
                _, preds = torch.max(outputs, 1)  # Get the predicted class
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

                # Calculate accuracy for this validation batch
                batch_val_acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
                print(
                    f"Validation - Batch [{batch_idx + 1}/{len(val_loader)}], Loss: {loss.item():.4f}, Accuracy: {batch_val_acc:.4f}, Progress: {100 * (batch_idx + 1) / len(val_loader):.2f}%")

        # Calculate validation metrics
        val_acc = accuracy_score(val_labels, val_preds)
        avg_val_loss = running_val_loss / len(val_loader)  # Average validation loss for the epoch
        classification_report_dict = classification_report(val_labels, val_preds, output_dict=True)

        # Store validation metrics
        val_accuracies.append(val_acc)
        val_precisions.append(classification_report_dict['macro avg']['precision'])
        val_recalls.append(classification_report_dict['macro avg']['recall'])
        val_f1_scores.append(classification_report_dict['macro avg']['f1-score'])

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - Validation Complete - Avg Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {classification_report_dict['macro avg']['precision']:.4f}, Recall: {classification_report_dict['macro avg']['recall']:.4f}, F1-Score: {classification_report_dict['macro avg']['f1-score']:.4f}")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_vit_model.pth')
            print("Best model saved.")

    return train_losses, val_accuracies, val_precisions, val_recalls, val_f1_scores


def evaluate_model(model, test_loader, device):
    """
    Evaluates the model on the test dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): The device (CPU or GPU) to perform computations.

    Returns:
        dict: A dictionary containing the test accuracy, average test loss, precision, recall, and F1-score.
    """
    model.eval()  # Set the model to evaluation mode
    y_true = []  # Store true labels
    y_pred = []  # Store predicted labels
    correct = 0  # Count correct predictions
    total = 0  # Total number of samples
    running_test_loss = 0.0  # Track total test loss

    with torch.no_grad():  # Disable gradient calculations during testing
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)  # Forward pass
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # Count correct predictions

            y_true.extend(labels.cpu().numpy())  # Store true labels
            y_pred.extend(predicted.cpu().numpy())  # Store predicted labels

            print(
                f"Testing - Batch [{batch_idx + 1}/{len(test_loader)}], Progress: {100 * (batch_idx + 1) / len(test_loader):.2f}%")

    # Calculate overall test accuracy
    accuracy = 100 * correct / total
    classification_report_dict = classification_report(y_true, y_pred, output_dict=True)

    print(f'Test Complete - Accuracy: {accuracy:.2f}%')
    print(f"Precision: {classification_report_dict['macro avg']['precision']:.2f}")
    print(f"Recall: {classification_report_dict['macro avg']['recall']:.2f}")
    print(f"F1-Score: {classification_report_dict['macro avg']['f1-score']:.2f}")

    return {
        "final_test_accuracy": accuracy,
        "precision": classification_report_dict["macro avg"]["precision"],
        "recall": classification_report_dict["macro avg"]["recall"],
        "f1-score": classification_report_dict["macro avg"]["f1-score"]
    }
