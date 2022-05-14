import matplotlib.pyplot as plt

def visualize_accuracy(train_accuracy, val_accuracy):
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Accuracy")
    plt.plot(train_accuracy, label="train_acc")
    plt.plot(val_accuracy, label="val_acc")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    
def visualize_loss(train_loss, val_loss):
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(train_loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
def visualize_test_acc(test_acc):
    plt.figure(figsize=(10,5))
    plt.title("Test Accuracy")
    plt.plot(test_acc, label="test_acc")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    
def visualize_test_loss(test_loss):
    plt.figure(figsize=(10,5))
    plt.title("Test Loss")
    plt.plot(test_loss, label="test_loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()