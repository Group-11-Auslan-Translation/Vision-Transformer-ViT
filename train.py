import torch
from sklearn.metrics import accuracy_score, classification_report


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []

    print(f"Using device: {device}")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Starting Training...")
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

            # Calculate accuracy for this batch
            batch_train_acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())

            # Batch Progress and Loss
            print(
                f"Training - Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {batch_train_acc:.4f}, Progress: {100 * (batch_idx + 1) / len(train_loader):.2f}%")

        avg_train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_losses.append(avg_train_loss)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - Training Complete - Avg Loss: {avg_train_loss:.4f}, Avg Accuracy: {train_acc:.4f}")

        # Validation phase
        model.eval()
        val_preds = []
        val_labels = []
        running_val_loss = 0.0  # Track validation loss
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] - Validation - Batch [{batch_idx + 1}/{len(val_loader)}] - Fetching data...")

                images, labels = images.to(device), labels.to(device)

                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] - Validation - Batch [{batch_idx + 1}/{len(val_loader)}] - Data fetched.")

                outputs = model(images)
                loss = criterion(outputs, labels)  # Validation loss for each batch
                running_val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

                # Calculate accuracy for this validation batch
                batch_val_acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())

                # Batch-wise validation accuracy and loss
                print(
                    f"Validation - Batch [{batch_idx + 1}/{len(val_loader)}], Loss: {loss.item():.4f}, Accuracy: {batch_val_acc:.4f}, Progress: {100 * (batch_idx + 1) / len(val_loader):.2f}%")

        val_acc = accuracy_score(val_labels, val_preds)
        avg_val_loss = running_val_loss / len(val_loader)  # Avg validation loss for the epoch
        classification_report_dict = classification_report(val_labels, val_preds, output_dict=True)
        val_accuracies.append(val_acc)
        val_precisions.append(classification_report_dict['macro avg']['precision'])
        val_recalls.append(classification_report_dict['macro avg']['recall'])
        val_f1_scores.append(classification_report_dict['macro avg']['f1-score'])

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - Validation Complete - Avg Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {classification_report_dict['macro avg']['precision']:.4f}, Recall: {classification_report_dict['macro avg']['recall']:.4f}, F1-Score: {classification_report_dict['macro avg']['f1-score']:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_vit_model.pth')
            print("Best model saved.")

    return train_losses, val_accuracies, val_precisions, val_recalls, val_f1_scores


def evaluate_model(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    y_true = []
    y_pred = []
    correct = 0
    total = 0
    running_test_loss = 0.0  # To track the total test loss
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            print(f"Testing - Batch [{batch_idx + 1}/{len(test_loader)}] - Fetching data...")

            images, labels = images.to(device), labels.to(device)

            print(f"Testing - Batch [{batch_idx + 1}/{len(test_loader)}] - Data fetched.")

            outputs = model(images)
            loss = criterion(outputs, labels)  # Calculate loss for this batch
            running_test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # Count correct predictions

            y_true.extend(labels.cpu().numpy())  # Append true labels
            y_pred.extend(predicted.cpu().numpy())  # Append predicted labels

            # Calculate accuracy for this batch
            batch_test_acc = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())

            # Print batch-wise accuracy and loss
            print(
                f"Testing - Batch [{batch_idx + 1}/{len(test_loader)}], Loss: {loss.item():.4f}, Accuracy: {batch_test_acc:.4f}, Progress: {100 * (batch_idx + 1) / len(test_loader):.2f}%")

    # Calculate overall accuracy for the test set
    accuracy = 100 * correct / total
    avg_test_loss = running_test_loss / len(test_loader)  # Average test loss for the entire test set
    classification_report_dict = classification_report(y_true, y_pred, output_dict=True)

    # Print the final evaluation results
    print(f'Test Complete - Avg Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    print(f"Precision: {classification_report_dict['macro avg']['precision']:.2f}")
    print(f"Recall: {classification_report_dict['macro avg']['recall']:.2f}")
    print(f"F1-Score: {classification_report_dict['macro avg']['f1-score']:.2f}")

    return {
        "final_test_accuracy": accuracy,
        "avg_test_loss": avg_test_loss,
        "precision": classification_report_dict["macro avg"]["precision"],
        "recall": classification_report_dict["macro avg"]["recall"],
        "f1-score": classification_report_dict["macro avg"]["f1-score"]
    }
