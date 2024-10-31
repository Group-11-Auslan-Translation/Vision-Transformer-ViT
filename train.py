import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

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
            - val_losses (list): List of validation losses for each epoch.
            - train_accuracies (list): List of training accuracies for each epoch.
            - val_accuracies (list): List of validation accuracies for each epoch.
            - train_precisions (list): List of training precisions for each epoch.
            - val_precisions (list): List of validation precisions for each epoch.
            - train_recalls (list): List of training recalls for each epoch.
            - val_recalls (list): List of validation recalls for each epoch.
            - train_f1_scores (list): List of training F1-scores for each epoch.
            - val_f1_scores (list): List of validation F1-scores for each epoch.
    """
    best_val_acc = 0.0  # Track the best validation accuracy
    train_losses = []  # Store training losses per epoch
    val_losses = []  # Store validation losses per epoch
    val_accuracies = []  # Store validation accuracies per epoch
    val_precisions = []  # Store validation precisions per epoch
    val_recalls = []  # Store validation recalls per epoch
    val_f1_scores = []  # Store validation F1-scores per epoch
    train_accuracies = []  # Store training accuracies per epoch
    train_precisions = []  # Store training precisions per epoch
    train_recalls = []  # Store training recalls per epoch
    train_f1_scores = []  # Store training F1-scores per epoch

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

            # Calculate accuracy for this training batch
            batch_train_acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
            print(
                f"Training - Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {batch_train_acc:.4f}, Progress: {100 * (batch_idx + 1) / len(train_loader):.2f}%"
            )

        # Calculate training metrics after all batches are processed
        avg_train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_precision = precision_score(all_train_labels, all_train_preds, average='macro', zero_division=0)
        train_recall = recall_score(all_train_labels, all_train_preds, average='macro', zero_division=0)
        train_f1 = f1_score(all_train_labels, all_train_preds, average='macro', zero_division=0)

        # Store training metrics
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1_scores.append(train_f1)

        # Calculate class-wise accuracies
        unique_classes = np.unique(all_train_labels)
        train_class_accuracies = {str(c): 0.0 for c in unique_classes}
        for c in unique_classes:
            class_indices = np.where(np.array(all_train_labels) == c)[0]
            if len(class_indices) > 0:
                train_class_accuracies[str(c)] = accuracy_score(
                    np.array(all_train_labels)[class_indices],
                    np.array(all_train_preds)[class_indices]
                )

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Class-wise Training Accuracies: {train_class_accuracies}")

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_preds = []  # Store validation predictions
        val_labels = []  # Store validation labels
        running_val_loss = 0.0  # Accumulate validation loss

        with torch.no_grad():  # Disable gradient calculations during evaluation
            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)  # Forward pass for validation
                val_loss = criterion(outputs, labels)  # Compute validation loss
                running_val_loss += val_loss.item()  # Accumulate validation loss
                _, preds = torch.max(outputs, 1)  # Get the predicted class
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

                # Calculate accuracy for this validation batch
                batch_val_acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
                print(
                    f"Validation - Batch [{batch_idx + 1}/{len(val_loader)}], Loss: {val_loss.item():.4f}, Accuracy: {batch_val_acc:.4f}, Progress: {100 * (batch_idx + 1) / len(val_loader):.2f}%")

        # Calculate validation metrics
        avg_val_loss = running_val_loss / len(val_loader)  # Average validation loss for the epoch
        val_acc = accuracy_score(val_labels, val_preds)
        classification_report_dict = classification_report(val_labels, val_preds, output_dict=True)

        # Store validation metrics
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        val_precisions.append(classification_report_dict['macro avg']['precision'])
        val_recalls.append(classification_report_dict['macro avg']['recall'])
        val_f1_scores.append(classification_report_dict['macro avg']['f1-score'])

        # Calculate class-wise validation accuracies
        unique_classes_val = np.unique(val_labels)
        val_class_accuracies = {str(c): 0.0 for c in unique_classes_val}
        for c in unique_classes_val:
            class_indices = np.where(np.array(val_labels) == c)[0]
            if len(class_indices) > 0:
                val_class_accuracies[str(c)] = accuracy_score(
                    np.array(val_labels)[class_indices],
                    np.array(val_preds)[class_indices]
                )

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Class-wise Validation Accuracies: {val_class_accuracies}")

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - Validation Complete - Avg Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {classification_report_dict['macro avg']['precision']:.4f}, Recall: {classification_report_dict['macro avg']['recall']:.4f}, F1-Score: {classification_report_dict['macro avg']['f1-score']:.4f}")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_vit_model.pth')
            print("Best model saved.")

    return train_losses, val_losses, train_accuracies, val_accuracies, train_precisions, val_precisions, train_recalls, val_recalls, train_f1_scores, val_f1_scores

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

    with torch.no_grad():
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
    classification_report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Calculate class-wise test accuracies
    unique_classes_test = np.unique(y_true)
    test_class_accuracies = {str(c): 0.0 for c in unique_classes_test}
    for c in unique_classes_test:
        class_indices = np.where(np.array(y_true) == c)[0]
        if len(class_indices) > 0:
            test_class_accuracies[str(c)] = accuracy_score(
                np.array(y_true)[class_indices],
                np.array(y_pred)[class_indices]
            )

    print(f'Test Complete - Accuracy: {accuracy:.2f}%')
    print(f"Precision: {classification_report_dict['macro avg']['precision']:.2f}")
    print(f"Recall: {classification_report_dict['macro avg']['recall']:.2f}")
    print(f"F1-Score: {classification_report_dict['macro avg']['f1-score']:.2f}")
    print(f"Class-wise Test Accuracies: {test_class_accuracies}")

    return {
        "final_test_accuracy": accuracy,
        "precision": classification_report_dict["macro avg"]["precision"],
        "recall": classification_report_dict["macro avg"]["recall"],
        "f1-score": classification_report_dict["macro avg"]["f1-score"],
        "class_accuracies": test_class_accuracies  # Add class-wise accuracies to return
    }
