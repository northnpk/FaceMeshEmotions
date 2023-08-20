import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def conf_plot(y_list, pred_list, class_name):
    cm = confusion_matrix(np.array(y_list), np.array(pred_list), normalize='all')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_name)
    disp.plot()
    plt.show()
    print(classification_report(y_list, pred_list, target_names=class_name))
    
def print_eval(epochs, train_backup, test_backup, val_backup):
    train_loss_backup, train_acc_backup = train_backup
    test_loss_backup, test_acc_backup = test_backup
    val_loss_backup, val_acc_backup = val_backup
    
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_loss_backup, label="train_loss")
    plt.plot(range(epochs), val_loss_backup, label="val_loss")
    plt.plot(range(epochs), test_loss_backup, label="test_loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_acc_backup, label="train_acc")
    plt.plot(range(epochs), val_acc_backup, label="val_acc")
    plt.plot(range(epochs), test_acc_backup, label="test_acc")
    plt.axis([None, None, 0, 1])
    plt.legend()
    
    plt.show()