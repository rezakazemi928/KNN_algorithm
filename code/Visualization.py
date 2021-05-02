import matplotlib.pyplot as plt

def dataVisualization(x_test, y_test, predicted_values):
    fig = plt.figure(figsize = (20, 20))
    w = 10
    h = 10

    for i in range(w * h):
        ax = fig.add_subplot(w, h, i + 1)
        plt.gray()
        ax.matshow(x_test[i].reshape(8, 8))
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(hspace = 1, wspace = 1)
        plt.title(f'y_true: {y_test[i]}\n y_pred: {int(predicted_values[i])}')
        
    plt.show() 